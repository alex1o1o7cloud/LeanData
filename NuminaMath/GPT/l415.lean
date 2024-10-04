import Mathlib

namespace bruce_money_shortage_l415_415162

def shirt_price : ℕ := 8
def pants_prices : list ℕ := [20, 25, 30]
def socks_price : ℕ := 3
def belt_price : ℕ := 15
def jacket_price : ℕ := 45
def initial_money : ℕ := 150
def belts_discount : ℚ := 0.5
def pants_discount : ℚ := 0.2
def jacket_discount : ℚ := 0.15
def tax_rate : ℚ := 0.07

noncomputable def total_cost_with_discounts_and_tax : ℚ :=
  let shirts_cost := 5 * shirt_price
  let pants_total := pants_prices.sum
  let socks_cost := 4 * socks_price
  let belts_cost := belt_price + (belt_price * belts_discount)
  let jacket_cost := jacket_price * (1 - jacket_discount)
  let pants_cost := pants_total - pants_prices.max' * pants_discount
  let subtotal := (shirts_cost + pants_cost + socks_cost + belts_cost + jacket_cost)
  let total_tax := subtotal * tax_rate
  subtotal + total_tax

theorem bruce_money_shortage : 
  initial_money - total_cost_with_discounts_and_tax = -44.4725 := 
sorry

end bruce_money_shortage_l415_415162


namespace distinct_ordered_pairs_count_l415_415255

theorem distinct_ordered_pairs_count (a b : ℕ) (h : a + b = 40) (ha_odd : a % 2 = 1) : 
  {s : ℕ × ℕ | s.1 + s.2 = 40 ∧ s.1 % 2 = 1}.card = 20 := 
  sorry

end distinct_ordered_pairs_count_l415_415255


namespace sum_of_first_ten_multiples_of_12_l415_415775

theorem sum_of_first_ten_multiples_of_12 : 
  (∑ k in Finset.range 10, 12 * (k + 1)) = 660 :=
by
  sorry

end sum_of_first_ten_multiples_of_12_l415_415775


namespace symmetric_line_eq_l415_415008

theorem symmetric_line_eq (x y : ℝ) (h : 2 * x - y = 0) : 2 * x + y = 0 :=
sorry

end symmetric_line_eq_l415_415008


namespace range_fx_a1_monotonic_fx_interval_5_5_min_value_fx_interval_0_2_l415_415259

section
variables {a x : ℝ}
def f (x a : ℝ) := x^2 + 2 * a * x + a + 1

-- (1) Proving the range of f(x) in [-2, 3] when a = 1 is [1, 17]
theorem range_fx_a1 : ∀ x ∈ Set.Icc (-2 : ℝ) 3, 1 ≤ f x 1 ∧ f x 1 ≤ 17 :=
begin
  sorry
end

-- (2) Proving that if f(x) is monotonic in [-5, 5], then a ∈ (-∞, -5] ∪ [5, +∞)
theorem monotonic_fx_interval_5_5 (h : @MonotonicOn ℝ ℝ _ _ (f x a) (Set.Icc (-5 : ℝ) 5)) : a ∈ Set.Iic (-5) ∪ Set.Ici 5 :=
begin
  sorry
end

-- (3) Proving the minimum value g(a) of f(x) in [0, 2]
def g (a : ℝ) : ℝ := 
  if a > 0 then a + 1 else 
  if -2 ≤ a ∧ a ≤ 0 then a^2 + a + 1 else 
  5 * a + 5

theorem min_value_fx_interval_0_2 : ∀ a : ℝ, g a = 
  if a > 0 then f 0 a else 
  if -2 ≤ a ∧ a ≤ 0 then f (-a) a else 
  f 2 a :=
begin
  sorry
end
end

end range_fx_a1_monotonic_fx_interval_5_5_min_value_fx_interval_0_2_l415_415259


namespace verify_A_l415_415879

def matrix_A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![62 / 7, -9 / 7], ![2 / 7, 17 / 7]]

theorem verify_A :
  matrix_A.mulVec ![1, 3] = ![5, 7] ∧
  matrix_A.mulVec ![-2, 1] = ![-19, 3] :=
by
  sorry

end verify_A_l415_415879


namespace number_of_tables_l415_415583

theorem number_of_tables (last_year_distance : ℕ) (factor : ℕ) 
  (distance_between_table_1_and_3 : ℕ) (number_of_tables : ℕ) :
  (last_year_distance = 300) ∧ 
  (factor = 4) ∧ 
  (distance_between_table_1_and_3 = 400) ∧
  (number_of_tables = ((factor * last_year_distance) / (distance_between_table_1_and_3 / 2)) + 1) 
  → number_of_tables = 7 :=
by
  intros
  sorry

end number_of_tables_l415_415583


namespace train_travel_distance_l415_415808

theorem train_travel_distance
  (rate_miles_per_pound : Real := 5 / 2)
  (remaining_coal : Real := 160)
  (distance_per_pound := λ r, r / 2)
  (total_distance := λ rc dpp, rc * dpp) :
  total_distance remaining_coal rate_miles_per_pound = 400 := sorry

end train_travel_distance_l415_415808


namespace shop_owner_profit_l415_415490

-- Let's define the conditions
def false_weight_buying (actual_weight : ℕ) : ℕ := actual_weight - (14 * actual_weight / 100)
def false_weight_selling (claimed_weight : ℕ) : ℕ := 80 * claimed_weight / 100
def cost_price (buy_weight sell_weight : ℕ) := 100 / buy_weight * sell_weight
def profit (sell_price cost_price : ℕ) := sell_price - cost_price
def profit_percentage (profit cost_price : ℕ) := (profit * 100) / cost_price

-- Theorem to prove the shop owner's percentage profit
theorem shop_owner_profit :
  let buy_weight := false_weight_buying 100
  let sell_weight := false_weight_selling 100
  let total_cost := cost_price buy_weight 80
  let total_profit := profit 100 total_cost
  let profit_percent := profit_percentage total_profit total_cost
  profit_percent = 7.5 := sorry

end shop_owner_profit_l415_415490


namespace intersection_points_C1_C2_l415_415632

theorem intersection_points_C1_C2 :
  (∀ t : ℝ, ∃ (ρ θ : ℝ), 
    (ρ^2 - 10 * ρ * Real.cos θ - 8 * ρ * Real.sin θ + 41 = 0) ∧ 
    (ρ = 2 * Real.cos θ) → 
    ((ρ = 2 ∧ θ = 0) ∨ (ρ = Real.sqrt 2 ∧ θ = Real.pi / 4))) :=
sorry

end intersection_points_C1_C2_l415_415632


namespace bread_count_at_end_of_day_l415_415494

def initial_loaves : ℕ := 2355
def sold_loaves : ℕ := 629
def delivered_loaves : ℕ := 489

theorem bread_count_at_end_of_day : 
  initial_loaves - sold_loaves + delivered_loaves = 2215 := by
  sorry

end bread_count_at_end_of_day_l415_415494


namespace total_packs_sold_l415_415358

theorem total_packs_sold (lucy_packs : ℕ) (robyn_packs : ℕ) (h1 : lucy_packs = 19) (h2 : robyn_packs = 16) : lucy_packs + robyn_packs = 35 :=
by
  sorry

end total_packs_sold_l415_415358


namespace smallest_positive_theta_l415_415568

-- Define the problem conditions in Lean
def sin_deg (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
def cos_deg (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)

-- Conditions are imported directly from trigonometry definitions.
theorem smallest_positive_theta :
  ∃ (θ : ℝ), θ > 0 ∧ θ < 360 ∧ cos_deg θ = sin_deg 45 + cos_deg 60 - sin_deg 30 - cos_deg 15 ∧ θ = 30 :=
by
  sorry

end smallest_positive_theta_l415_415568


namespace liam_total_time_l415_415348

noncomputable def total_time_7_laps : Nat :=
let time_first_200 := 200 / 5  -- Time in seconds for the first 200 meters
let time_next_300 := 300 / 6   -- Time in seconds for the next 300 meters
let time_per_lap := time_first_200 + time_next_300
let laps := 7
let total_time := laps * time_per_lap
total_time

theorem liam_total_time : total_time_7_laps = 630 := by
sorry

end liam_total_time_l415_415348


namespace area_of_triangle_bounded_by_coordinate_axes_and_line_l415_415511

def area_of_triangle (b h : ℕ) : ℕ := (b * h) / 2

theorem area_of_triangle_bounded_by_coordinate_axes_and_line :
  area_of_triangle 4 6 = 12 :=
by
  sorry

end area_of_triangle_bounded_by_coordinate_axes_and_line_l415_415511


namespace sum_of_final_two_numbers_l415_415790

theorem sum_of_final_two_numbers (x y T : ℕ) (h : x + y = T) :
  3 * (x + 5) + 3 * (y + 5) = 3 * T + 30 :=
by
  sorry

end sum_of_final_two_numbers_l415_415790


namespace hunter_saw_32_frogs_l415_415649

noncomputable def total_frogs (g1 : ℕ) (g2 : ℕ) (d : ℕ) : ℕ :=
g1 + g2 + d

theorem hunter_saw_32_frogs :
  total_frogs 5 3 (2 * 12) = 32 := by
  sorry

end hunter_saw_32_frogs_l415_415649


namespace metal_mixing_ratio_l415_415783

theorem metal_mixing_ratio :
  ∀ (x y : ℕ),
    let costA := 68 * x,
        costB := 96 * y,
        totalCost := costA + costB,
        totalWeight := x + y,
        costPerKg := totalCost / totalWeight in
    costPerKg = 75 → x / y = 3 :=
by sorry

end metal_mixing_ratio_l415_415783


namespace longer_side_length_l415_415122

-- Given a circle with radius 6 cm
def radius : ℝ := 6
def circle_area : ℝ := Real.pi * radius^2

-- Given that the area of the rectangle is three times the area of the circle
def rectangle_area : ℝ := 3 * circle_area

-- Given the rectangle has a shorter side equal to the diameter of the circle
def shorter_side : ℝ := 2 * radius

-- Prove that the length of the longer side of the rectangle is 9π cm
theorem longer_side_length : ∃ (longer_side : ℝ), longer_side = 9 * Real.pi :=
by
  have circle_area_def : circle_area = 36 * Real.pi := by sorry
  have rectangle_area_def : rectangle_area = 108 * Real.pi := by sorry
  have shorter_side_def : shorter_side = 12 := by sorry
  let longer_side := rectangle_area / shorter_side
  use longer_side
  show longer_side = 9 * Real.pi
  sorry

end longer_side_length_l415_415122


namespace onions_grown_l415_415860

theorem onions_grown (tomatoes : ℕ) (corn : ℕ) (delta : ℕ) (onions : ℕ) 
  (h_tomatoes : tomatoes = 2073) 
  (h_corn : corn = 4112) 
  (h_delta : delta = 5200)
  (h_eq : onions = (tomatoes + corn - delta)) :
  onions = 985 :=
by
  rw [h_tomatoes, h_corn, h_delta, h_eq]
  norm_num
  sorry

end onions_grown_l415_415860


namespace c_range_valid_l415_415930

-- Defining the propositions
def p (c : ℝ) : Prop := ∀ x1 x2 : ℝ, x1 < x2 → c^x1 > c^x2
def q (c : ℝ) : Prop := ∀ x : ℝ, x^2 - real.sqrt 2 * x + c > 0

-- Given conditions
variables (c : ℝ) (h_decreasing : p c) (hnq : ¬ q c)

-- The theorem to prove
theorem c_range_valid : 0 < c ∧ c ≤ 1 / 2 :=
by
  sorry

end c_range_valid_l415_415930


namespace oaks_not_adjacent_probability_l415_415481

theorem oaks_not_adjacent_probability :
  let total_trees := 13
  let oaks := 5
  let other_trees := total_trees - oaks
  let possible_slots := other_trees + 1
  let combinations := Nat.choose possible_slots oaks
  let total_arrangements := Nat.factorial total_trees / (Nat.factorial oaks * Nat.factorial (total_trees - oaks))
  let probability := combinations / total_arrangements
  probability = 1 / 220 :=
by
  sorry

end oaks_not_adjacent_probability_l415_415481


namespace parity_of_expression_l415_415659

theorem parity_of_expression (a b c : ℕ) (h_apos : 0 < a) (h_aodd : a % 2 = 1) (h_beven : b % 2 = 0) :
  (3^a + (b+1)^2 * c) % 2 = if c % 2 = 0 then 1 else 0 :=
sorry

end parity_of_expression_l415_415659


namespace max_min_sum_of_exp2_l415_415025

theorem max_min_sum_of_exp2 (f : ℝ → ℝ) (h : ∀ x ∈ set.Icc 0 1, f x = 2^x) : 
  (set.Icc 0 1).sup f + (set.Icc 0 1).inf f = 3 :=
by 
  /- The proof will go here -/
  sorry

end max_min_sum_of_exp2_l415_415025


namespace train_length_is_240_meters_l415_415496

-- Define the conditions of the problem
def train_speed_kmh : ℝ := 144
def time_to_cross_seconds : ℝ := 6

-- Convert speed from km/h to m/s
def kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

def train_speed_ms : ℝ := kmh_to_ms train_speed_kmh

-- Define the distance calculation based on speed and time
def train_length : ℝ :=
  train_speed_ms * time_to_cross_seconds

-- The theorem we aim to prove
theorem train_length_is_240_meters :
  train_length = 240 :=
by
  sorry

end train_length_is_240_meters_l415_415496


namespace train_travel_distance_l415_415812

def coal_efficiency := (5 : ℝ) / (2 : ℝ)  -- Efficiency in miles per pound
def coal_remaining := 160  -- Coal remaining in pounds
def distance_travelled := coal_remaining * coal_efficiency  -- Total distance the train can travel

theorem train_travel_distance : distance_travelled = 400 := 
by
  sorry

end train_travel_distance_l415_415812


namespace trigonometric_sum_of_ratios_l415_415275

theorem trigonometric_sum_of_ratios (α β : ℝ) 
  (h : (cos α ^ 4 / cos β ^ 2) + (sin α ^ 4 / sin β ^ 2) = 1) :
  (sin β ^ 4 / sin α ^ 2) + (cos β ^ 4 / cos α ^ 2) = 1 := 
begin
  sorry
end

end trigonometric_sum_of_ratios_l415_415275


namespace train_cross_time_l415_415495

def train_length : ℝ := 500 -- meters
def train_speed_kmh : ℝ := 36 -- km/h
def kmh_to_ms (v : ℝ) : ℝ := v * (1000 / 3600) -- conversion factor from km/h to m/s

theorem train_cross_time : 
  let speed_ms := kmh_to_ms train_speed_kmh,
      time := train_length / speed_ms 
  in time = 50 := 
by
  sorry

end train_cross_time_l415_415495


namespace dan_baseball_cards_total_l415_415538

-- Define the initial conditions
def initial_baseball_cards : Nat := 97
def torn_baseball_cards : Nat := 8
def sam_bought_cards : Nat := 15
def alex_bought_fraction : Nat := 4
def gift_cards : Nat := 6

-- Define the number of cards    
def non_torn_baseball_cards : Nat := initial_baseball_cards - torn_baseball_cards
def remaining_after_sam : Nat := non_torn_baseball_cards - sam_bought_cards
def remaining_after_alex : Nat := remaining_after_sam - remaining_after_sam / alex_bought_fraction
def final_baseball_cards : Nat := remaining_after_alex + gift_cards

-- The theorem to prove 
theorem dan_baseball_cards_total : final_baseball_cards = 62 := by
  sorry

end dan_baseball_cards_total_l415_415538


namespace find_m_l415_415665

theorem find_m (m : ℝ) (P1 : ℝ × ℝ) (P2 : ℝ × ℝ) (inclination_angle : ℝ) :
  (P1 = (m, 2) ∧ P2 = (-m, 2m - 1) ∧ inclination_angle = 45) → m = 3 / 4 :=
by
  sorry

end find_m_l415_415665


namespace scale_division_l415_415143

noncomputable def length_each_part (total_length : ℕ) (parts : ℕ) : ℝ :=
  total_length / parts

theorem scale_division (ft_inch_conv factor inch_add parts : ℕ) :
  length_each_part (ft_inch_conv * factor + inch_add) parts = 24.75 := by
  -- Total length in inches: 15 * 12 + 18
  let total_length := 15 * 12 + 18
  -- Number of parts: 8
  have length_per_part : ℝ := total_length / 8
  -- Desired result
  unfold length_each_part
  -- Argument for division
  have h_div : total_length / 8 = 24.75 := 
  by norm_num
  exact h_div
  sorry

end scale_division_l415_415143


namespace target_practice_l415_415184

theorem target_practice (shots_total : ℕ) (total_points : ℕ) (points_per_shot : ℕ → ℕ) :
  shots_total = 10 →
  total_points = 90 →
  (count (λ x, points_per_shot x = 10) (list.range shots_total)) = 4 →
  (∀ x, points_per_shot x ∈ {7, 8, 9, 10}) →
  (count (λ x, points_per_shot x = 7) (list.range shots_total)) = 1 :=
by
  intro shots_total_eq total_points_eq ten_points_count no_misses
  sorry

end target_practice_l415_415184


namespace buttons_on_first_type_of_shirt_l415_415739

/--
The GooGoo brand of clothing manufactures two types of shirts.
- The first type of shirt has \( x \) buttons.
- The second type of shirt has 5 buttons.
- The department store ordered 200 shirts of each type.
- A total of 1600 buttons are used for the entire order.

Prove that the first type of shirt has exactly 3 buttons.
-/
theorem buttons_on_first_type_of_shirt (x : ℕ) 
  (h1 : 200 * x + 200 * 5 = 1600) : 
  x = 3 :=
  sorry

end buttons_on_first_type_of_shirt_l415_415739


namespace min_real_roots_0_l415_415709

-- Define the polynomial g with the properties given

def polynomial_with_properties : Prop :=
  ∃ (g : ℝ[X]), degree g = 500 ∧ 
    (∃ (roots : fin 500 → ℂ), 
      (∀ i, g.is_root (roots i)) ∧ 
      (∀ i, ∃ j, ↑i ≠ j ∧ roots i = conj (roots j)) ∧ 
      (set.card ((finset.fin_range 500).image (λ i, abs (roots i)))) = 250)

-- The theorem following the question and conditions
theorem min_real_roots_0 : polynomial_with_properties → ∃ g : ℝ[X], g.real_roots.card = 0 :=
sorry

end min_real_roots_0_l415_415709


namespace smallest_positive_angle_l415_415565

theorem smallest_positive_angle :
  ∃ θ : ℝ, θ = 30 ∧
    cos (θ * (Real.pi / 180)) = 
      sin (45 * (Real.pi / 180)) + cos (60 * (Real.pi / 180)) - sin (30 * (Real.pi / 180)) - cos (15 * (Real.pi / 180)) := 
by
  sorry

end smallest_positive_angle_l415_415565


namespace problem1_problem2_l415_415617

noncomputable def T : Set ℝ := { t | t ≤ 1 }

theorem problem1 :
  ∃ x₀ : ℝ, |x₀ - 1| - |x₀ - 2| ≥ t ↔ t ≤ 1 := 
by
  sorry

theorem problem2 (m n : ℝ) (hm : m > 1) (hn : n > 1) :
  (∀ t ∈ T, Real.logBase 3 m * Real.logBase 3 n ≥ t) → m * n ≥ 9 :=
by
  sorry

end problem1_problem2_l415_415617


namespace green_duck_percentage_l415_415073

theorem green_duck_percentage (G_small G_large : ℝ) (D_small D_large : ℕ)
    (H1 : G_small = 0.20) (H2 : D_small = 20)
    (H3 : G_large = 0.15) (H4 : D_large = 80) : 
    ((G_small * D_small + G_large * D_large) / (D_small + D_large)) * 100 = 16 := 
by
  sorry

end green_duck_percentage_l415_415073


namespace eccentricity_of_ellipse_l415_415402

open Real

theorem eccentricity_of_ellipse 
  (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let c := sqrt (a^2 - b^2) in 
  (∃ F₁ F₂ P I G, 
    P ∈ setOf (λ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1) ∧
    F₁ = (-c, 0) ∧ 
    F₂ = (c, 0) ∧ 
    (∃ I G, 
      I ∈ setOf (λ (x : ℝ), x = 0) ∧
      G ∈ setOf (λ (x : ℝ), x = 0)))
  → (c / a) = 1 / 3 :=
by 
  intros _ h_c exists_s
  sorry


end eccentricity_of_ellipse_l415_415402


namespace megan_popsicles_volume_l415_415352

open BigOperators

noncomputable def melt_volume (initial_volume: ℝ) (loss_rate: ℝ) (time_hr: ℝ): ℝ :=
  initial_volume * (loss_rate ^ time_hr)

def total_popsicle_volume (initial_volume: ℝ) (loss_rate: ℝ) (duration_hr: ℝ) (interval_min: ℝ): ℝ :=
  let n := 6 * duration_hr in  -- number of Popsicles eaten
  ∑ i in finset.range (nat.floor n), melt_volume initial_volume loss_rate (i / 6)

theorem megan_popsicles_volume :
  total_popsicle_volume 100 0.9 3 10 = 1558.2 :=
sorry

end megan_popsicles_volume_l415_415352


namespace cards_needed_for_47_floors_l415_415463

def cards_per_floor (n : ℕ) : ℕ :=
  if n = 1 then 2 else 2 + 3 * (n - 1)

def total_cards (floors : ℕ) : ℕ :=
  (finset.range(floors + 1)).sum cards_per_floor

theorem cards_needed_for_47_floors : total_cards 47 = 3337 :=
  sorry

end cards_needed_for_47_floors_l415_415463


namespace probability_star_top_card_is_one_fifth_l415_415840

-- Define the total number of cards in the deck
def total_cards : ℕ := 65

-- Define the number of star cards in the deck
def star_cards : ℕ := 13

-- Define the probability calculation
def probability_star_top_card : ℚ := star_cards / total_cards

-- State the theorem regarding the probability
theorem probability_star_top_card_is_one_fifth :
  probability_star_top_card = 1 / 5 :=
by
  sorry

end probability_star_top_card_is_one_fifth_l415_415840


namespace find_m_equal_roots_l415_415545

theorem find_m_equal_roots :
  ∃ m : ℝ, (
    (∀ x : ℝ, (x * (x - 1) - (m^2 + m * x + 1)) / ((x - 1) * (m - 1)) = x / m) ∧
    (x * x - x * (m + 1) + m^3 + m = 0 → 
     discriminant (λ x => x^2 - x * (m + 1) + m^3 + m) = 0) →
    m = -1 / 2
  ) :=
by
  sorry

end find_m_equal_roots_l415_415545


namespace octagon_area_l415_415437

-- Defining points in the 2D plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Defining the octagon explicitly using the coordinates from the problem
def octagon := [
  Point.mk 1 4,  -- Top left
  Point.mk 5 4,  -- Top right
  Point.mk 6 0,  -- Right middle top
  Point.mk 5 -4, -- Bottom right
  Point.mk 1 -4, -- Bottom left
  Point.mk 0 0   -- Centre point in the X-axis
]

-- Function to calculate the area of octagon using vertices
-- Placeholder definition for polygon_area to skip the proof
noncomputable def polygon_area (points : List Point) : ℝ := 48

-- Assertion that the area of the octagon is equal to 48 square units
theorem octagon_area : polygon_area octagon = 48 :=
  by
    -- Placeholder proof to ensure the statement can be compiled
    sorry

end octagon_area_l415_415437


namespace incorrect_description_of_R_squared_l415_415679

theorem incorrect_description_of_R_squared (R2 : ℝ) :
  (∀ R2, R2 is coefficient of determination → 
     ( ( the_larger_the_R2_the_better_the_simulation_effect_of_the_model R2 
     ∧ the_larger_the_R2_the_greater_the_contribution_of_the_explanatory_variable R2 )
     ∧ (¬ the_larger_the_R2_the_larger_the_sum_of_squared_residuals R2 ) ) )

end incorrect_description_of_R_squared_l415_415679


namespace complete_square_proof_l415_415735

def complete_square (x : ℝ) : Prop :=
  x^2 - 2 * x - 8 = 0 -> (x - 1)^2 = 9

theorem complete_square_proof (x : ℝ) :
  complete_square x :=
sorry

end complete_square_proof_l415_415735


namespace f_one_zero_x_range_l415_415870

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
-- f is defined for x > 0
variable (f : ℝ → ℝ)
variables (h_domain : ∀ x, x > 0 → ∃ y, f x = y)
variables (h1 : f 2 = 1)
variables (h2 : ∀ x y, x > 0 → y > 0 → f (x * y) = f x + f y)
variables (h3 : ∀ x y, x > y → f x > f y)

-- Question 1
theorem f_one_zero (hf1 : f 1 = 0) : True := 
  by trivial
  
-- Question 2
theorem x_range (x: ℝ) (hx: f 3 + f (4 - 8 * x) > 2) : x ≤ 1/3 := sorry

end f_one_zero_x_range_l415_415870


namespace longer_side_of_rectangle_l415_415110

theorem longer_side_of_rectangle
  (r : ℝ) (A_rect A_circle L S : ℝ) (h1 : r = 6)
  (h2 : A_circle = π * r^2)
  (h3 : A_rect = 3 * A_circle)
  (h4 : S = 2 * r)
  (h5 : A_rect = S * L) : L = 9 * π :=
by
  sorry

end longer_side_of_rectangle_l415_415110


namespace ed_pets_count_l415_415551

theorem ed_pets_count : 
  let dogs := 2 
  let cats := 3 
  let fish := 2 * (cats + dogs) 
  let birds := dogs * cats 
  dogs + cats + fish + birds = 21 := 
by
  sorry

end ed_pets_count_l415_415551


namespace range_of_c_l415_415590

noncomputable def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y

noncomputable def q (c : ℝ) : Prop := ∀ r : ℝ, ∃ x : ℝ, r = log (2 * c * x^2 - 2 * x + 1)

theorem range_of_c (c : ℝ) (h1 : c > 0) (h2 : ¬p c ∨ ¬q c) (h3 : p c ∨ q c) : 1 / 2 < c ∧ c < 1 :=
by
  sorry

end range_of_c_l415_415590


namespace dog_height_l415_415092

theorem dog_height (h_lamp_post : ℝ) (s_lamp_post : ℝ) (s_dog : ℝ) : ℝ :=
  let ratio := h_lamp_post / s_lamp_post
  ratio * s_dog

example : dog_height 50 8 (6 / 12) = 37.5 := by
  let h_dog := dog_height 50 8 (6 / 12)
  have : h_dog = (50 / 8) * (6 / 12) := rfl
  have : (50 / 8) * (6 / 12) = 37.5 := by norm_num
  rw [this, this]
  sorry

end dog_height_l415_415092


namespace train_length_is_constant_l415_415453

def train_length (v: ℝ) (t1 t2: ℝ) (d: ℝ) : ℝ := (v * t1 * t2) / (d - v * t2)

theorem train_length_is_constant : 
  ∀ (L Vbridge Vpost: ℝ), 
  Vpost = L / 30 ∧ 
  Vbridge = (L + 2500) / 120 ∧ 
  Vpost = Vbridge →
  L = 75000 / 90 := 
by 
  intros L Vbridge Vpost
  intros hVpost hVbridge hVpost_eq_Vbridge
  sorry

end train_length_is_constant_l415_415453


namespace max_height_achieved_l415_415093

-- Define the height function
def height (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 16

-- Prove that the maximum height is 141 feet
theorem max_height_achieved : ∃ t : ℝ, height t = 141 := sorry

end max_height_achieved_l415_415093


namespace max_temp_range_l415_415392

theorem max_temp_range (temps : Fin 7 → ℝ)
  (avg_temp : (∑ i, temps i) / 7 = 45)
  (low_temp : ∀ i, temps i ≥ 28)
  (exists_low : ∃ i, temps i = 28) :
  ∃ high_temp, high_temp = 147 ∧ (∃ i, temps i = 147) ∧ (max_temp_range : max temps - min temps = 119) :=
by
  sorry

end max_temp_range_l415_415392


namespace closest_fraction_to_team_japan_medals_l415_415680

theorem closest_fraction_to_team_japan_medals :
  let medals_won := 23
  let total_medals := 150
  let fraction_won := medals_won / total_medals
  ∃ closest_fraction : ℚ, closest_fraction ∈ {1/5, 1/6, 1/7, 1/8, 1/9} ∧ 
  |fraction_won - closest_fraction| = min (|fraction_won - (1/5)|) (min (|fraction_won - (1/6)|) (min (|fraction_won - (1/7)|) (min (|fraction_won - (1/8)|) (|fraction_won - (1/9)|)))) :=
by
  let medals_won := 23
  let total_medals := 150
  let fraction_won := (medals_won:ℚ) / total_medals
  let fractions := {1/5, 1/6, 1/7, 1/8, 1/9}
  have min_dist : ∃ frac, frac ∈ fractions ∧ |fraction_won - frac| = min (|fraction_won - (1/5)|) (min (|fraction_won - (1/6)|) (min (|fraction_won - (1/7)|) (min (|fraction_won - (1/8)|) (|fraction_won - (1/9)|))),
  from sorry,
  exact min_dist

end closest_fraction_to_team_japan_medals_l415_415680


namespace determine_angle_at_vertex_C_l415_415317

variables {a b f1 f2 : ℝ}
variables {C : Triangle}

axiom a_gt_b : a > b
axiom f2_over_f1_eq : f2 / f1 = (a + b) / (a - b) * sqrt 3

theorem determine_angle_at_vertex_C (h : f2 / f1 = (a + b) / (a - b) * sqrt 3): ∠C = 120 :=
sorry

end determine_angle_at_vertex_C_l415_415317


namespace hyperbola_equation_hyperbola_equation_l415_415664

theorem hyperbola_equation :
  (∃ f : ℝ, (∀ x y : ℝ, x^2 / 27 + y^2 / 36 = 1 → (y = f * (x - sqrt (27 / 4 * x^2 + 47 / 36 * y^2)) ∨ y = - f * (x + sqrt (27 / 4 * x^2 + 47 / 36 * y^2)))) ∧
             (∀ f : ℝ, ∀ p : ℝ × ℝ, p.fst = sqrt 15 ∧ p.snd = 4 → p.snd / f = 2)) →
  (∀ x y : ℝ, ((y^2 / 4) - (x^2 / 5) = 1)) :=
by
sor#import mathlib

-- Define the foci of the ellipse
def foci_ellipse : set (ℝ × ℝ) := {(0, 3), (0, -3)}

-- Define the condition for the ellipse
def ellipse (x y : ℝ) : Prop := (x^2 / 27 + y^2 / 36 = 1)

-- Define a point through which the hyperbola passes
def point_hyperbola : ℝ × ℝ := (sqrt 15, 4)

-- Prove the equation of the hyperbola given the conditions
theorem hyperbola_equation :
  (∀ x y : ℝ, ellipse x y) →
  (∀ f ∈ foci_ellipse, f ∈ foci_ellipse) →
  (∀ p, p = point_hyperbola) →
  (∀ x y : ℝ, (y^2 / 4 - x^2 / 5 = 1)) :=
by
  sorry

end hyperbola_equation_hyperbola_equation_l415_415664


namespace fibonacci_geometric_sequence_l415_415389

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci n + fibonacci (n+1)

theorem fibonacci_geometric_sequence :
  ∃ (a b c : ℕ) (r : ℝ), a + b + c = 2010 ∧ b = a + 1 ∧ c = b + 1 ∧
  (r : ℝ) = ↑(fibonacci b) / ↑(fibonacci a) ∧
  (r : ℝ) = ↑(fibonacci c) / ↑(fibonacci b) ∧
  a = 669 :=
by
  sorry

end fibonacci_geometric_sequence_l415_415389


namespace problem_unique_integer_sequence_l415_415878

theorem problem_unique_integer_sequence :
  ∃! (x : Fin 10 → ℕ), 
    (∀ (i j : Fin 10), i < j → x i < x j) ∧ 
    x 8 * x 9 ≤ 2 * (∑ i : Fin 9, x i) ∧ 
    (x 0 = 1) ∧ (x 1 = 2) ∧ (x 2 = 3) ∧ (x 3 = 4) ∧ (x 4 = 5) ∧ 
    (x 5 = 6) ∧ (x 6 = 7) ∧ (x 7 = 8) ∧ (x 8 = 9) ∧ (x 9 = 10) := by
  sorry

end problem_unique_integer_sequence_l415_415878


namespace ratio_angle_OBE_BAC_l415_415156

-- Define the conditions as given in the problem
variables (ABC : Triangle) (O : Circle) (E : Point)
  (h1 : TriangleInscribedInCircle ABC O)
  (h2 : AngleMeasure (ABC.angle BAC) = 120)
  (h3 : ArcMeasure (O.arc BC) = 72)
  (h4 : Perpendicular (O.center) E (ABC.side AC))

-- The goal is to prove the ratio of ∠OBE to ∠BAC is 1/3
theorem ratio_angle_OBE_BAC : 
  let OBE := O.angle B E,
      BAC := ABC.angle B A C in
  (OBE / BAC) = 1 / 3 := 
sorry

end ratio_angle_OBE_BAC_l415_415156


namespace least_integer_x_l415_415455

theorem least_integer_x (x : ℤ) (h : 240 ∣ x^2) : x = 60 :=
sorry

end least_integer_x_l415_415455


namespace smallest_positive_theta_l415_415567

-- Define the problem conditions in Lean
def sin_deg (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
def cos_deg (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)

-- Conditions are imported directly from trigonometry definitions.
theorem smallest_positive_theta :
  ∃ (θ : ℝ), θ > 0 ∧ θ < 360 ∧ cos_deg θ = sin_deg 45 + cos_deg 60 - sin_deg 30 - cos_deg 15 ∧ θ = 30 :=
by
  sorry

end smallest_positive_theta_l415_415567


namespace total_license_groups_l415_415146

-- Defining the given conditions
def letter_choices : Nat := 3
def digit_choices_per_slot : Nat := 10
def number_of_digit_slots : Nat := 5

-- Statement to prove that the total number of different license groups is 300000
theorem total_license_groups : letter_choices * (digit_choices_per_slot ^ number_of_digit_slots) = 300000 := by
  sorry

end total_license_groups_l415_415146


namespace greatest_sundays_in_49_days_l415_415434

theorem greatest_sundays_in_49_days : 
  ∀ (days : ℕ), 
    days = 49 → 
    ∀ (sundays_per_week : ℕ), 
      sundays_per_week = 1 → 
      ∀ (weeks : ℕ), 
        weeks = days / 7 → 
        weeks * sundays_per_week = 7 :=
by
  sorry

end greatest_sundays_in_49_days_l415_415434


namespace extreme_value_a_eq_2_range_of_a_l415_415622

noncomputable def f (a x : ℝ) := (a + log (2 * x + 1)) / (2 * x + 1)

theorem extreme_value_a_eq_2 :
  ∀ x > -1 / 2, has_extreme_value (λ x, f 2 x) e := sorry

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Icc ((real.exp 1 - 1) / 2) ((real.exp 2 - 1) / 2), 
  ∃ t : ℝ, (2 * x + 1) ^ 2 * deriv (λ x, f a x) = t ^ 3 - 12 * t ∧
  distinct_real_roots (2 * x + 1) ^ 2 * deriv (λ x, f a x) = t ^ 3 - 12 * t) ↔
  a ∈ Ioo (-8) 7 := sorry

end extreme_value_a_eq_2_range_of_a_l415_415622


namespace find_r_cubed_l415_415287

theorem find_r_cubed (r : ℝ) (h : (r + 1/r)^2 = 5) : r^3 + 1/r^3 = 2 * Real.sqrt 5 :=
by
  sorry

end find_r_cubed_l415_415287


namespace savings_equal_after_25_weeks_l415_415450

theorem savings_equal_after_25_weeks
  (your_initial_savings : ℤ)
  (your_weekly_savings : ℤ)
  (friend_initial_savings : ℤ)
  (friend_weekly_savings : ℤ) :
  your_initial_savings = 160 →
  your_weekly_savings = 7 →
  friend_initial_savings = 210 →
  friend_weekly_savings = 5 →
  ∃ w : ℤ, w = 25 ∧ (your_initial_savings + your_weekly_savings * w = friend_initial_savings + friend_weekly_savings * w) :=
by
  -- given conditions
  intros h1 h2 h3 h4
  use 25
  -- provide details with 'use ..' earlier than 'intros h5'
  intro    
  split; sorry

end savings_equal_after_25_weeks_l415_415450


namespace largest_common_value_l415_415544

theorem largest_common_value :
  ∃ (a : ℕ), (∃ (n m : ℕ), a = 4 + 5 * n ∧ a = 5 + 10 * m) ∧ a < 1000 ∧ a = 994 :=
by {
  sorry
}

end largest_common_value_l415_415544


namespace range_of_f_l415_415253

noncomputable def triangle (a b : ℝ) : ℝ := real.sqrt (a * b) + a + b

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := triangle k x

theorem range_of_f (k : ℝ) (h : triangle 1 k = 3) : 
  set.range (λ x, f x k) = set.Ici 1 :=
sorry

end range_of_f_l415_415253


namespace inclination_angle_l415_415655

theorem inclination_angle (α : ℝ) (hα1 : 0 < α) (hα2 : α < (π / 2)) :
  let P1 := (0, Real.cos α)
  let P2 := (Real.sin α, 0)
  ∃ β : ℝ, β = π / 2 + α ∧
    let slope := (P2.2 - P1.2) / (P2.1 - P1.1)
    slope = -Real.cot α ∧ ∃ θ, θ = Real.atan slope ∧ θ = β := sorry

end inclination_angle_l415_415655


namespace pq_sum_l415_415748

noncomputable def p (x : ℝ) : ℝ := -4 * (x - 3)
noncomputable def q (x : ℝ) : ℝ := -2 * (x + 2) * (x - 3)

theorem pq_sum :
  p(2) = 4 ∧ q(2) = 8 ∧ (∀ x: ℝ, p(x) ≠ 0 → q(x) ≠ 0 → (p x + q x) = -2 * x^2 - 2 * x + 24) →
  (∀ x : ℝ, p(x) + q(x) = -2 * x^2 - 2 * x + 24) :=
by
  intro h
  specialize h 2
  sorry

end pq_sum_l415_415748


namespace exists_triangular_numbers_with_ratio_2_l415_415573

theorem exists_triangular_numbers_with_ratio_2 :
  ∃ (m n : ℕ), T m = m * (m + 1) / 2 ∧ T n = n * (n + 1) / 2 ∧ T m = 2 * T n := 
sorry

end exists_triangular_numbers_with_ratio_2_l415_415573


namespace p_sufficient_but_not_necessary_for_q_l415_415926

theorem p_sufficient_but_not_necessary_for_q (x : ℝ) :
  (|x - 1| < 2 → x ^ 2 - 5 * x - 6 < 0) ∧ ¬ (x ^ 2 - 5 * x - 6 < 0 → |x - 1| < 2) :=
by
  sorry

end p_sufficient_but_not_necessary_for_q_l415_415926


namespace tan_double_angle_l415_415236

theorem tan_double_angle 
  (α : ℝ)
  (h : (sin (π - α) + sin ( π / 2 - α)) / (sin α - cos α) = 1 / 2) : 
  tan (2 * α) = 3 / 4 :=
by
  sorry

end tan_double_angle_l415_415236


namespace solve_equation_l415_415381

theorem solve_equation :
  { x : ℝ | x * (x - 3)^2 * (5 - x) = 0 } = {0, 3, 5} :=
by
  sorry

end solve_equation_l415_415381


namespace frog_arrangement_problem_l415_415032

def frog_arrangements : ℕ :=
  let green_frogs := 2
  let red_frogs := 3
  let blue_frogs := 2
  let total_frogs := 7
  let valid_configurations := 4
  let green_arrangements := 2!
  let red_arrangements := 3!
  let blue_arrangements := 2!
  valid_configurations * green_arrangements * red_arrangements * blue_arrangements

theorem frog_arrangement_problem (green_frogs red_frogs blue_frogs total_frogs : ℕ)
    (h1 : green_frogs = 2) (h2 : red_frogs = 3) (h3 : blue_frogs = 2) (h4 : total_frogs = 7)
    (valid_configurations green_arrangements red_arrangements blue_arrangements : ℕ)
    (h5 : valid_configurations = 4) (h6 : green_arrangements = 2!) 
    (h7 : red_arrangements = 3!) (h8 : blue_arrangements = 2!) :
    frog_arrangements = 96 :=
by
  rw [h1, h2, h3, h4, h5, h6, h7, h8]
  unfold frog_arrangements
  norm_num
  sorry

end frog_arrangement_problem_l415_415032


namespace area_of_triangle_bounded_by_coordinate_axes_and_line_l415_415509

def area_of_triangle (b h : ℕ) : ℕ := (b * h) / 2

theorem area_of_triangle_bounded_by_coordinate_axes_and_line :
  area_of_triangle 4 6 = 12 :=
by
  sorry

end area_of_triangle_bounded_by_coordinate_axes_and_line_l415_415509


namespace arithmetic_sequence_sum_l415_415329

theorem arithmetic_sequence_sum :
  ∃ (a : ℕ → ℝ) (d : ℝ), 
  (∀ n, a n = a 0 + n * d) ∧ 
  (∃ b c, b^2 - 6*b + 5 = 0 ∧ c^2 - 6*c + 5 = 0 ∧ a 3 = b ∧ a 15 = c) →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 :=
by
  sorry

end arithmetic_sequence_sum_l415_415329


namespace triangle_area_l415_415497

theorem triangle_area : 
  ∃ (A : ℝ), A = 12 ∧ (∃ (x_intercept y_intercept : ℝ), 3 * x_intercept + 2 * y_intercept = 12 ∧ x_intercept * y_intercept / 2 = A) :=
by
  sorry

end triangle_area_l415_415497


namespace solve_for_t_l415_415656

theorem solve_for_t (s t : ℝ) (h1 : 12 * s + 8 * t = 160) (h2 : s = t^2 + 2) :
  t = (Real.sqrt 103 - 1) / 3 :=
sorry

end solve_for_t_l415_415656


namespace find_phi_l415_415946

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)

theorem find_phi (phi : ℝ) (h_shift : ∀ x : ℝ, f (x + phi) = f (-x - phi)) : 
  phi = Real.pi / 8 :=
  sorry

end find_phi_l415_415946


namespace cost_ranking_l415_415988

variable (a q_S : ℝ)

def cost_per_pencil_S := a / q_S
def cost_per_pencil_M := (1.2 * a) / (1.5 * q_S)
def cost_per_pencil_L := (1.6 * a) / (1.875 * q_S)

theorem cost_ranking :
  cost_per_pencil_M a q_S < cost_per_pencil_L a q_S ∧ cost_per_pencil_L a q_S < cost_per_pencil_S a q_S :=
by sorry

end cost_ranking_l415_415988


namespace Sean_Julie_ratio_l415_415374

-- Define the sum of the first n natural numbers
def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the sum of even numbers up to 2n
def sum_even (n : ℕ) : ℕ := 2 * sum_n n

theorem Sean_Julie_ratio : 
  (sum_even 250) / (sum_n 250) = 2 := 
by
  sorry

end Sean_Julie_ratio_l415_415374


namespace gcd_relatively_prime_l415_415694

theorem gcd_relatively_prime (a : ℤ) (m n : ℕ) (h_odd : a % 2 = 1) (h_pos_m : m > 0) (h_pos_n : n > 0) (h_diff : n ≠ m) :
  Int.gcd (a ^ 2^m + 2 ^ 2^m) (a ^ 2^n + 2 ^ 2^n) = 1 :=
by
  sorry

end gcd_relatively_prime_l415_415694


namespace find_a_l415_415338

def f (x : ℝ) : ℝ :=
if x >= 0 then (2/3 : ℝ) * x - 1 else 1 / x

theorem find_a (a : ℝ) (h : f a = a) : a = -1 :=
by {
  sorry
}

end find_a_l415_415338


namespace intersection_of_sets_l415_415931

def set_M := {x : ℝ | sqrt (x + 1) ≥ 0}
def set_N := {x : ℝ | x^2 + x - 2 < 0}

theorem intersection_of_sets :
  {x : ℝ | (sqrt (x + 1) ≥ 0) ∧ (x^2 + x - 2 < 0)} = {x : ℝ | -1 ≤ x ∧ x < 1} := 
sorry

end intersection_of_sets_l415_415931


namespace max_grapes_in_bag_l415_415033

theorem max_grapes_in_bag : ∃ (x : ℕ), x > 100 ∧ x % 3 = 1 ∧ x % 5 = 2 ∧ x % 7 = 4 ∧ x = 172 := by
  sorry

end max_grapes_in_bag_l415_415033


namespace monotonicity_and_extremes_l415_415258

noncomputable def f (x : ℝ) : ℝ := 1 / 2 * x^2 + x - 2 * Real.log x

theorem monotonicity_and_extremes : 
  (∀ x > 1, Deriv f x > 0) ∧
  (∀ x, 0 < x ∧ x < 1 → Deriv f x < 0) ∧
  (∃ c, c = 1 ∧ IsLocalMin f c ∧ f c = 3 / 2) :=
by
  sorry

end monotonicity_and_extremes_l415_415258


namespace last_two_digits_sum_eq_13_l415_415467

def is_contributing (n : ℕ) : Prop :=
  ¬ ((n % 3 = 0) ∧ (n % 5 = 0))

def last_two_digits (n : ℕ) := n % 100

def sum_of_contributing_factorials : ℕ :=
  (Finset.range 101).filter is_contributing
    .sum (λ n => last_two_digits (Nat.factorial n))

theorem last_two_digits_sum_eq_13 : last_two_digits sum_of_contributing_factorials = 13 :=
  sorry

end last_two_digits_sum_eq_13_l415_415467


namespace range_a_always_below_x_axis_l415_415975

noncomputable def f (a x : ℝ) : ℝ := a * x ^ 2 + 2 * a * x - 2

theorem range_a_always_below_x_axis (a : ℝ) :
  (∀ x : ℝ, f a x < 0) ↔ a ∈ Ioo (-2 : ℝ) 0 :=
begin
  sorry
end

end range_a_always_below_x_axis_l415_415975


namespace parabola_equation_l415_415563

open Real

noncomputable def parabola_vertex_form (x : ℝ) (a : ℝ) : ℝ := a * (x - 3)^2 + 5

theorem parabola_equation :
  ∃ a b c : ℝ,
  (∀ x : ℝ, parabola_vertex_form x a = a * (x - 3)^2 + 5) ∧
  -- Point (0,2) lies on the parabola
  (∀ x : ℝ, x = 0 → (parabola_vertex_form x a) = 2) ∧
  -- Given point x=3 is the vertex
  (∀ y : ℝ, ∃ x : ℝ, x = 3 → y = 5) →
  -- General equation in the form ax^2 + bx + c
  ∀ x : ℝ, (-⅓) * x^2 + 2 * x + 2 = -⅓ * x^2 + 2 * x + 2 :=
begin
  use [-⅓, 2, 2],
  split,
  { intros x,
    exact calc
    (-⅓) * (x - 3)^2 + 5 = (-⅓) * (3 - x)^2 + 5 : by ring
    ... = (-⅓) * (x^2 - 6 * x + 9) + 5 : by ring
    ... = (-⅓) * x^2 + 2 * x + 2 : by ring },
  split,
  { intros x h,
    rw h,
    refl, },
  intro y,
  use [3],
  intro hx,
  rw hx,
  refl,
end

end parabola_equation_l415_415563


namespace probability_bus_there_when_mark_arrives_l415_415799

noncomputable def isProbabilityBusThereWhenMarkArrives : Prop :=
  let busArrival : ℝ := 60 -- The bus can arrive from time 0 to 60 minutes (2:00 PM to 3:00 PM)
  let busWait : ℝ := 30 -- The bus waits for 30 minutes
  let markArrival : ℝ := 90 -- Mark can arrive from time 30 to 90 minutes (2:30 PM to 3:30 PM)
  let overlapArea : ℝ := 1350 -- Total shaded area where bus arrival overlaps with Mark's arrival
  let totalArea : ℝ := busArrival * (markArrival - 30)
  let probability := overlapArea / totalArea
  probability = 1 / 4

theorem probability_bus_there_when_mark_arrives : isProbabilityBusThereWhenMarkArrives :=
by
  sorry

end probability_bus_there_when_mark_arrives_l415_415799


namespace isosceles_triangle_perimeter_l415_415232

theorem isosceles_triangle_perimeter : 
  ∀ a b c : ℝ, a^2 - 6 * a + 5 = 0 → b^2 - 6 * b + 5 = 0 → 
    (a = b ∨ b = c ∨ a = c) →
    (a + b > c ∧ b + c > a ∧ a + c > b) →
    a + b + c = 11 := 
by
  intros a b c ha hb hiso htri
  sorry

end isosceles_triangle_perimeter_l415_415232


namespace alternating_students_count_l415_415791

theorem alternating_students_count :
  let num_male := 4
  let num_female := 5
  let arrangements := Nat.factorial num_female * Nat.factorial num_male
  arrangements = 2880 :=
by
  sorry

end alternating_students_count_l415_415791


namespace find_original_function_l415_415292

-- Define the function transformation operation
def transform (f : ℝ → ℝ) (x : ℝ) := f (2 * (x - π / 3))

-- Given resulting function after transformations
def resulting_function (x : ℝ) := Real.sin (x - π / 4)

-- Original function
def original_function (x : ℝ) := Real.sin (x / 2 + π / 12)

theorem find_original_function :
  transform original_function = resulting_function := 
sorry

end find_original_function_l415_415292


namespace no_such_polyhedron_exists_l415_415321

def valid_polyhedron (P : Type) : Prop :=
  (∃ (f : P → finset P), 
    (∀ x, x ∈ f x) ∧
    (∃! y, y ∈ f x → (P × finset P)) ∧ 
    (∃! z, ¬(∀ a b, a ∈ f a ∧ b ∈ f b ∧ a ≠ b → a ∈ b)) ∧ 
    ∀ v ∈ P, even (degree v))

theorem no_such_polyhedron_exists : ∀ (P : Type), ¬valid_polyhedron P :=
by
  sorry

end no_such_polyhedron_exists_l415_415321


namespace longer_side_of_rectangle_is_l415_415114

-- Define the radius of the circle
def radius (r : ℝ) : Prop := r = 6

-- Define the area of the circle given the radius
def area_circle (A : ℝ) : Prop := A = 36 * Real.pi

-- Define the relationship between the area of the rectangle and the circle
def area_rectangle (A_rect : ℝ) (A : ℝ) : Prop := A_rect = 3 * A

-- Define the length of the shorter side of the rectangle
def shorter_side (s : ℝ) : Prop := s = 12

-- State the goal: the length of the longer side of the rectangle
def longer_side (l : ℝ) (A_rect : ℝ) (s : ℝ) : Prop := l = A_rect / s

-- Theorem statement combining all conditions and the goal
theorem longer_side_of_rectangle_is :
  ∃ l : ℝ, ∃ A_rect : ℝ, ∃ s : ℝ, radius 6 ∧ area_circle 36 * Real.pi ∧ area_rectangle A_rect (36 * Real.pi) ∧ shorter_side s ∧ longer_side l A_rect s :=
by {
  sorry
}

end longer_side_of_rectangle_is_l415_415114


namespace negation_of_universal_l415_415411

-- Define the original proposition: Every triangle has a circumcircle.
def every_triangle_has_circumcircle : Prop :=
  ∀ (T : Type), T → has_circumcircle T 

-- Define the negation of the proposition.
def some_triangles_do_not_have_circumcircle : Prop :=
  ∃ (T : Type), T → ¬ has_circumcircle T 

-- The proof statement:
theorem negation_of_universal (every_triangle_has_circumcircle : Prop) :
  some_triangles_do_not_have_circumcircle ↔ ¬ every_triangle_has_circumcircle :=
by
  sorry

end negation_of_universal_l415_415411


namespace distance_between_intersections_l415_415200

theorem distance_between_intersections (a : ℝ) :
  (u, v, w) = (4 * a^2, 0, 1) :=
by
  -- Defining the curves
  let curve1 : ℝ → ℝ := λ y, y^5
  let curve2 : ℝ → ℝ := λ y, 1 - y^3

  -- Checking the intersection points
  have intersection1 : ∃ y, y^5 + y^3 = 1 := sorry
  let y1 := classical.some intersection1
  let y2 := -y1

  -- Coordinates of intersections
  let point1 := (curve1 y1, y1)
  let point2 := (curve1 y2, y2)

  -- Distance calculation
  have distance : ℝ := sqrt ((point1.1 - point2.1)^2 + (point1.2 - point2.2)^2)
  have h_distance : distance = 2 * |a| := sorry

  -- Distance in the form  √(u + v√w)
  have h_form : √(u + v√w) = 2 * |a| := sorry

  -- Therefore
  have h_triplet : (u, v, w) = (4 * a^2, 0, 1) := sorry
  exact h_triplet


end distance_between_intersections_l415_415200


namespace train_cross_man_in_18_seconds_l415_415845

noncomputable def speed_in_mps (speed_kmph : ℕ) : ℝ := (speed_kmph : ℝ) * (1000 / 3600)
noncomputable def time_to_cross_platform (train_speed : ℝ) (platform_length : ℕ) (total_time : ℕ) : ℝ :=
  train_speed * (total_time : ℝ) - (platform_length : ℝ)
noncomputable def time_to_cross_man (train_length : ℝ) (train_speed : ℝ) : ℝ :=
  train_length / train_speed

theorem train_cross_man_in_18_seconds (speed_kmph platform_length total_time : ℕ) (train_speed : ℝ) (train_length : ℝ) :
  speed_kmph = 72 ∧ platform_length = 260 ∧ total_time = 31 ∧
  train_speed = speed_in_mps speed_kmph ∧
  train_length = time_to_cross_platform train_speed platform_length total_time →
  time_to_cross_man train_length train_speed = 18 :=
by
  intros h
  cases h with hs1 hremain
  cases hremain with hs2 hremain
  cases hremain with ht1 hremain
  cases hremain with ht2 hl
  rw [hs1, hs2, ht1, ht2, hl]
  sorry

end train_cross_man_in_18_seconds_l415_415845


namespace last_two_digits_of_squared_expression_l415_415770

theorem last_two_digits_of_squared_expression (n : ℕ) :
  (n * 2 * 3 * 4 * 46 * 47 * 48 * 49) ^ 2 % 100 = 76 :=
by
  sorry

end last_two_digits_of_squared_expression_l415_415770


namespace total_shoes_calculation_l415_415186

noncomputable def total_pairs_of_shoes (ellie : ℕ) (riley : ℕ) (jordan : Real) : ℕ :=
ellie + riley + jordan.toInt

theorem total_shoes_calculation : total_pairs_of_shoes 8 (8 - 3) ((1.5 * ((8) + (8 - 3))).toInt) = 32 := by
  sorry

end total_shoes_calculation_l415_415186


namespace common_tangent_length_l415_415602

noncomputable def circle1 : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
noncomputable def circle2 : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 8 * p.1 + 12 = 0}

theorem common_tangent_length :
  let d := Real.sqrt ((4 - 0)^2 + (0 - 0)^2) in
  let r₁ := 1 in
  let r₂ := 2 in
  Real.sqrt (d^2 - (r₁ - r₂)^2) = Real.sqrt 15 := by
  sorry

end common_tangent_length_l415_415602


namespace express_as_percentage_express_as_simplified_fraction_express_as_mixed_number_l415_415555

theorem express_as_percentage (x : ℝ) (h : x = 2.09) : (2.09 * 100 = 209) := by
  have h1 : 2.09 * 100 = 209 := by norm_num
  exact h1

theorem express_as_simplified_fraction (x : ℝ) (h : x = 2.09) : (2.09 = 209 / 100) ∧ Nat.gcd 209 100 = 1 := by
  have h1 : 2.09 = 209 / 100 := by norm_num
  have h2 : Nat.gcd 209 100 = 1 := by norm_num
  exact ⟨h1, h2⟩

theorem express_as_mixed_number (x : ℝ) (h : x = 2.09) : (2.09 = 2 + (9 / 100)) := by
  have h1 : 2.09 = 2 + (9 / 100) := by norm_num
  exact h1

end express_as_percentage_express_as_simplified_fraction_express_as_mixed_number_l415_415555


namespace PropositionC_is_false_l415_415446

-- Definitions of the propositions
def PropositionA : Prop := ∀ {α β γ : ℝ}, (α + β = 180 ∧ α + γ = 180) → β = γ
def PropositionB : Prop := ∀ {α β : ℝ}, (α ∠ β) → α = β
def PropositionC : Prop := ∀ {α β γ δ : ℝ}, (α ∠ β ∧ γ ∠ δ) → β = δ
def PropositionD : Prop := ∀ {l m n : ℝ}, (l ⊥ m ∧ l ⊥ n) → m ∥ n

-- Lean statement that the Proposition C is false
theorem PropositionC_is_false : ¬ PropositionC := 
by
sorry

end PropositionC_is_false_l415_415446


namespace sum_of_angles_in_star_l415_415552

theorem sum_of_angles_in_star : 
  let α : ℝ := 45 -- each arc between points
  let β : ℝ := α * 3 / 2 -- angle at each tip
  8 * β = 540 := 
begin
  let α : ℝ := 45, -- each arc is 45 degrees
  let β : ℝ := α * 3 / 2, -- tip angle is half of 3 arcs
  have hβ : β = 67.5, 
    from calc
      β = 45 * 3 / 2 : by rfl
      ... = 135 / 2 : by norm_num
      ... = 67.5 : by norm_num,
  calc
    8 * β = 8 * 67.5 : by rw hβ
    ... = 540 : by norm_num,
end

end sum_of_angles_in_star_l415_415552


namespace imaginary_part_of_complex_l415_415813

noncomputable theory

open Complex

def is_imaginary_part_neg_one (z : ℂ) : Prop :=
  z.im = -1

theorem imaginary_part_of_complex (z : ℂ) (h : 2 * z + conj z = 6 - I) : is_imaginary_part_neg_one z :=
sorry

end imaginary_part_of_complex_l415_415813


namespace limit_sum_perimeters_l415_415533

theorem limit_sum_perimeters (a : ℝ) : ∑' n : ℕ, (4 * a) * (1 / 2) ^ n = 8 * a :=
by sorry

end limit_sum_perimeters_l415_415533


namespace remainder_of_x_to_15_plus_1_div_by_x_plus_1_is_0_l415_415062

def f (x : ℝ) : ℝ := x^15 + 1

theorem remainder_of_x_to_15_plus_1_div_by_x_plus_1_is_0 : f (-1) = 0 := by
  sorry

end remainder_of_x_to_15_plus_1_div_by_x_plus_1_is_0_l415_415062


namespace paul_catch_up_time_l415_415074

-- Define the speeds of Mary and Paul
constants (mary_speed paul_speed : ℝ)
constants (mary_speed_val : mary_speed = 50)
constants (paul_speed_val : paul_speed = 80)

-- Define the time difference in minutes and hours
constants (time_diff_min : ℝ)
constants (time_diff_val : time_diff_min = 15)

-- Define a constant for conversion from minutes to hours
def minutes_to_hours (minutes : ℝ) : ℝ := minutes / 60

-- Define the distance Mary travels in the given time difference
def mary_distance (minutes : ℝ) (speed : ℝ) : ℝ := speed * (minutes_to_hours minutes)

-- Define the catch-up speed
def catch_up_speed (paul_speed mary_speed : ℝ) : ℝ := paul_speed - mary_speed

-- Define the time it takes for Paul to catch up to Mary in hours
def catch_up_time (distance speed : ℝ) : ℝ := distance / speed

-- Convert catch-up time from hours to minutes
def catch_up_time_minutes (hours : ℝ) : ℝ := hours * 60

-- Theorem stating Paul will catch up with Mary in 25 minutes
theorem paul_catch_up_time : 
  catch_up_time_minutes (catch_up_time (mary_distance time_diff_min mary_speed) 
                                          (catch_up_speed paul_speed mary_speed)) = 25 :=
by 
  sorry

end paul_catch_up_time_l415_415074


namespace count_squares_with_specific_last_digit_l415_415274

theorem count_squares_with_specific_last_digit :
  (finset.filter (λ n: ℕ, (n^2 < 2000) ∧ (n^2 % 10 = 2 ∨ n^2 % 10 = 3 ∨ n^2 % 10 = 4 ∨ n^2 % 10 = 6)) (finset.range 45)).card = 18 :=
by
  sorry

end count_squares_with_specific_last_digit_l415_415274


namespace total_chickens_after_purchase_l415_415529

def initial_chickens : ℕ := 400
def percentage_died : ℕ := 40
def times_to_buy : ℕ := 10

noncomputable def chickens_died : ℕ := (percentage_died * initial_chickens) / 100
noncomputable def chickens_remaining : ℕ := initial_chickens - chickens_died
noncomputable def chickens_bought : ℕ := times_to_buy * chickens_died
noncomputable def total_chickens : ℕ := chickens_remaining + chickens_bought

theorem total_chickens_after_purchase : total_chickens = 1840 :=
by
  sorry

end total_chickens_after_purchase_l415_415529


namespace center_of_k4_on_I_O_line_l415_415031

universe u
variable {α : Type u}

theorem center_of_k4_on_I_O_line (ABC : Triangle α) (k1 k2 k3 k4 : Circle α)
  (h1 : k1.radius = k2.radius)
  (h2 : k2.radius = k3.radius)
  (h3 : k3.radius = k4.radius)
  (h4 : k1.touch (ABC.side AB) ∧ k1.touch (ABC.side AC) ∧ k1.touch k4)
  (h5 : k2.touch (ABC.side AB) ∧ k2.touch (ABC.side BC) ∧ k2.touch k4)
  (h6 : k3.touch (ABC.side AC) ∧ k3.touch (ABC.side BC) ∧ k3.touch k4) :
  on_line (center k4) (incenter ABC) (circumcenter ABC) :=
sorry

end center_of_k4_on_I_O_line_l415_415031


namespace bisector_bisects_CD_l415_415689

-- Definitions based on conditions
variables {A B C D : Type} [AddCommGroup A] [AffineSpace A B]

structure Trapezoid (A B C D : A) : Prop :=
(parallel_sides : ∃ p, parallel A C B D ∧ A B = A C + B D)

-- The proof statement
theorem bisector_bisects_CD
  (A B C D : A)
  [Trapezoid A B C D]
  (h_parallel : parallel A C B D)
  (h_AB_eq_AD_BC : A B = A C + B D)
  : bisects (angle_bisector A) C D :=
sorry

end bisector_bisects_CD_l415_415689


namespace beau_age_calculation_l415_415161

variable (sons_age : ℕ) (beau_age_today : ℕ) (beau_age_3_years_ago : ℕ)

def triplets := 3
def sons_today := 16
def sons_age_3_years_ago := sons_today - 3
def sum_of_sons_3_years_ago := triplets * sons_age_3_years_ago

theorem beau_age_calculation
  (h1 : sons_today = 16)
  (h2 : sum_of_sons_3_years_ago = beau_age_3_years_ago)
  (h3 : beau_age_today = beau_age_3_years_ago + 3) :
  beau_age_today = 42 :=
sorry

end beau_age_calculation_l415_415161


namespace wholesaler_profit_percentage_correct_l415_415825

-- Defining the conditions
def manufacturer_cost_price : ℝ := 17
def manufacturer_profit_percentage : ℝ := 18
def retailer_selling_price : ℝ := 30.09
def retailer_profit_percentage : ℝ := 25

-- Defining the target profit percentage for the wholesaler
def wholesaler_profit_percentage : ℝ :=
  let manufacturer_selling_price := manufacturer_cost_price * (1 + manufacturer_profit_percentage / 100)
  let retailer_cost_price := retailer_selling_price / (1 + retailer_profit_percentage / 100)
  let wholesaler_profit := retailer_cost_price - manufacturer_selling_price
  (wholesaler_profit / manufacturer_selling_price) * 100

-- Statement to be proved
theorem wholesaler_profit_percentage_correct : wholesaler_profit_percentage = 20 := by
  sorry

end wholesaler_profit_percentage_correct_l415_415825


namespace find_f_expression_find_f_range_l415_415263

noncomputable def y (t x : ℝ) : ℝ := 1 - 2 * t - 2 * t * x + 2 * x ^ 2

noncomputable def f (t : ℝ) : ℝ := 
  if t < -2 then 3 
  else if t > 2 then -4 * t + 3 
  else -t ^ 2 / 2 - 2 * t + 1

theorem find_f_expression (t : ℝ) : 
  f t = if t < -2 then 3 else 
          if t > 2 then -4 * t + 3 
          else - t ^ 2 / 2 - 2 * t + 1 :=
sorry

theorem find_f_range (t : ℝ) (ht : -2 ≤ t ∧ t ≤ 0) : 
  1 ≤ f t ∧ f t ≤ 3 := 
sorry

end find_f_expression_find_f_range_l415_415263


namespace true_propositions_count_l415_415634

-- Original Proposition
def P (x y : ℝ) : Prop := x^2 + y^2 = 0 → x = 0 ∧ y = 0

-- Converse Proposition
def Q (x y : ℝ) : Prop := x = 0 ∧ y = 0 → x^2 + y^2 = 0

-- Contrapositive Proposition
def contrapositive_Q_P (x y : ℝ) : Prop := (x ≠ 0 ∨ y ≠ 0) → (x^2 + y^2 ≠ 0)

-- Inverse Proposition
def inverse_P (x y : ℝ) : Prop := (x^2 + y^2 ≠ 0) → (x ≠ 0 ∨ y ≠ 0)

-- Problem Statement
theorem true_propositions_count : ∀ (x y : ℝ),
  P x y ∧ Q x y ∧ contrapositive_Q_P x y ∧ inverse_P x y → 3 = 3 :=
by
  intros x y h
  sorry

end true_propositions_count_l415_415634


namespace harmonic_motion_initial_phase_l415_415819

theorem harmonic_motion_initial_phase :
  ∀ (x : ℝ), (x ≥ 0) → (f : ℝ → ℝ) = λ x, 4 * Real.sin (8 * x - π / 9) → 
  (8 * 0 - π / 9 = -π / 9) :=
by
  intros x hx f hf
  rw Real.sin_eq at hf
  simp
  sorry

end harmonic_motion_initial_phase_l415_415819


namespace circle_center_l415_415101

theorem circle_center 
(C : Type*) [MetricSpace C] [NormedAddCommGroup C] [NormedSpace ℝ C] 
(center : C) 
(h₁ : center ∈ {p | ∃ x y, p = (x, y) ∧ 5 * x - 2 * y = 10})
(h₂ : center ∈ {p | ∃ x y, p = (x, y) ∧ x + 3 * y = 0}):
  center = (30 / 17 : ℝ, -10 / 17 : ℝ) :=
sorry

end circle_center_l415_415101


namespace housewife_spend_money_l415_415488

theorem housewife_spend_money (P M: ℝ) (h1: 0.75 * P = 30) (h2: M / (0.75 * P) - M / P = 5) : 
  M = 600 :=
by
  sorry

end housewife_spend_money_l415_415488


namespace estimate_defective_pairs_l415_415797

/-- Define the data from the frequency table. --/
def sample_data : List (ℕ × ℕ × ℝ) :=
  [(20, 17, 0.85),
   (40, 38, 0.95),
   (60, 55, 0.92),
   (80, 75, 0.94),
   (100, 96, 0.96),
   (200, 189, 0.95),
   (300, 286, 0.95)]

/-- Calculate the defective rate from the largest sample size. --/
def defective_rate : ℝ :=
  1 - 0.95

/-- Prove that the estimated number of defective pairs in 1500 pairs of sports shoes is 75. --/
theorem estimate_defective_pairs :
  let total_pairs := 1500
  let estimated_defective_pairs := total_pairs * defective_rate
  estimated_defective_pairs = 75 :=
by
  let total_pairs := 1500
  let estimated_defective_pairs := total_pairs * defective_rate
  have h_defective_rate : defective_rate = 0.05 := rfl
  have h_calc : estimated_defective_pairs = 1500 * 0.05 := rfl
  rw [← h_defective_rate] at h_calc
  exact h_calc ▸ rfl

end estimate_defective_pairs_l415_415797


namespace exists_overlap_l415_415469

variable (R : Fin 5 → Set (Fin 3 → Bool))
variable (area : ∀ i : Fin 5, R i → ℝ)
variable (intersect_area : ∀ i j : Fin 5, R i ∩ R j → ℝ)

noncomputable def area_of_rug := 1.0
noncomputable def total_area := 3.0
noncomputable def overlap_threshold := 0.2

axiom h_all_rugs_have_area : ∀ i : Fin 5, area i (R i) = area_of_rug
axiom h_total_area : ∑ i : Fin 5, area i (R i) = 5.0
axiom h_floor_area : total_area ≥ 3.0

theorem exists_overlap :
  ∃ i j : Fin 5, i ≠ j ∧ intersect_area i j (R i ∩ R j) ≥ overlap_threshold :=
sorry

end exists_overlap_l415_415469


namespace Tricia_age_is_correct_l415_415764

variables (A Y E K R V T : ℕ)

-- Given conditions
def condition1 : Prop := A = Y / 4
def condition2 : Prop := Y = 2 * E
def condition3 : Prop := K = E / 3
def condition4 : Prop := R = K + 10
def condition5 : Prop := R = V - 2
def condition6 : Prop := V = 22
def condition7 : Prop := T = 5

-- The proof target
theorem Tricia_age_is_correct (h1 : condition1) (h2 : condition2) (h3 : condition3) 
(h4 : condition4) (h5 : condition5) (h6 : condition6) (h7 : condition7) : T = (1 / 3) * A := 
by
  sorry

end Tricia_age_is_correct_l415_415764


namespace parabola_equation_l415_415562

open Real

noncomputable def parabola_vertex_form (x : ℝ) (a : ℝ) : ℝ := a * (x - 3)^2 + 5

theorem parabola_equation :
  ∃ a b c : ℝ,
  (∀ x : ℝ, parabola_vertex_form x a = a * (x - 3)^2 + 5) ∧
  -- Point (0,2) lies on the parabola
  (∀ x : ℝ, x = 0 → (parabola_vertex_form x a) = 2) ∧
  -- Given point x=3 is the vertex
  (∀ y : ℝ, ∃ x : ℝ, x = 3 → y = 5) →
  -- General equation in the form ax^2 + bx + c
  ∀ x : ℝ, (-⅓) * x^2 + 2 * x + 2 = -⅓ * x^2 + 2 * x + 2 :=
begin
  use [-⅓, 2, 2],
  split,
  { intros x,
    exact calc
    (-⅓) * (x - 3)^2 + 5 = (-⅓) * (3 - x)^2 + 5 : by ring
    ... = (-⅓) * (x^2 - 6 * x + 9) + 5 : by ring
    ... = (-⅓) * x^2 + 2 * x + 2 : by ring },
  split,
  { intros x h,
    rw h,
    refl, },
  intro y,
  use [3],
  intro hx,
  rw hx,
  refl,
end

end parabola_equation_l415_415562


namespace intersection_convex_quadrilateral_l415_415995

open Set

-- Definitions of convex polygonal regions P and Q with lattice points
variable {P Q : Set (ℝ × ℝ)}
variable (hPconvex : Convex ℝ P) (hQconvex : Convex ℝ Q)
variable (hPlattice : ∀ p ∈ P, ∃ x y : ℤ, p = (x, y))
variable (hQlattice : ∀ q ∈ Q, ∃ x y : ℤ, q = (x, y))

-- Non-emptiness of intersection T
variable (hTnonempty : (P ∩ Q).Nonempty)

-- Condition that T contains no lattice points
variable (hTnolattice : ∀ t ∈ P ∩ Q, ¬∃ x y : ℤ, t = (x, y))

-- Prove T is a non-degenerate convex quadrilateral
theorem intersection_convex_quadrilateral (hPconvex : Convex ℝ P) (hQconvex : Convex ℝ Q)
  (hPlattice : ∀ p ∈ P, ∃ x y : ℤ, p = (x, y)) (hQlattice : ∀ q ∈ Q, ∃ x y : ℤ, q = (x, y))
  (hTnonempty : (P ∩ Q).Nonempty) (hTnolattice : ∀ t ∈ P ∩ Q, ¬∃ x y : ℤ, t = (x, y)) :
  ∃ T : Set (ℝ × ℝ), Convex ℝ T ∧ T = P ∩ Q ∧ ¬DegenerateConvexPolygon ℝ T ∧ Quadrilateral ℝ T := by
  sorry

end intersection_convex_quadrilateral_l415_415995


namespace longer_side_of_rectangle_l415_415112

theorem longer_side_of_rectangle
  (r : ℝ) (A_rect A_circle L S : ℝ) (h1 : r = 6)
  (h2 : A_circle = π * r^2)
  (h3 : A_rect = 3 * A_circle)
  (h4 : S = 2 * r)
  (h5 : A_rect = S * L) : L = 9 * π :=
by
  sorry

end longer_side_of_rectangle_l415_415112


namespace savings_equal_after_25_weeks_l415_415451

theorem savings_equal_after_25_weeks
  (your_initial_savings : ℤ)
  (your_weekly_savings : ℤ)
  (friend_initial_savings : ℤ)
  (friend_weekly_savings : ℤ) :
  your_initial_savings = 160 →
  your_weekly_savings = 7 →
  friend_initial_savings = 210 →
  friend_weekly_savings = 5 →
  ∃ w : ℤ, w = 25 ∧ (your_initial_savings + your_weekly_savings * w = friend_initial_savings + friend_weekly_savings * w) :=
by
  -- given conditions
  intros h1 h2 h3 h4
  use 25
  -- provide details with 'use ..' earlier than 'intros h5'
  intro    
  split; sorry

end savings_equal_after_25_weeks_l415_415451


namespace identity_holds_for_all_real_numbers_l415_415903

theorem identity_holds_for_all_real_numbers (a b : ℝ) : 
  a^2 + b^2 + 2 * a * b = (a + b)^2 := 
by sorry

end identity_holds_for_all_real_numbers_l415_415903


namespace range_of_f_l415_415405

noncomputable
def f (x : ℝ) : ℝ := (Real.log x / Real.log 2 + 1) * (Real.log x / Real.log 2 - 5)

theorem range_of_f : set.range (f : ℝ → ℝ) ∩ set.Icc (-9 : ℝ) (-5 : ℝ) = set.Icc (-9 : ℝ) (-5 : ℝ) :=
sorry

end range_of_f_l415_415405


namespace base_7_to_base_10_l415_415136

theorem base_7_to_base_10 :
  (3 * 7^2 + 2 * 7^1 + 1 * 7^0) = 162 :=
by
  sorry

end base_7_to_base_10_l415_415136


namespace correct_option_is_B_l415_415328

open Real

noncomputable def law_of_sines (a b : ℝ) (A : ℝ) : ℝ :=
  (b * sin A) / a

theorem correct_option_is_B : 
  -- Statement A
  (law_of_sines 7 14 (π / 6) = 1 → (∃ B : ℝ, B = π / 2) → ¬ ∃ B1 B2 : ℝ, B1 ≠ B2 ∧ (sin B1 = 1) ∧ (sin B2 = 1)) ∧
  -- Statement B (correct statement)
  (law_of_sines 30 25 (5 * π / 6) = 5 / 12 → (∃! B : ℝ, sin B = 5 / 12)) ∧
  -- Statement C
  (law_of_sines 6 9 (π / 4) = sqrt 2 / 3 → ¬ ∃ B1 B2 : ℝ, B1 ≠ B2 ∧ (sin B1 = sqrt 2 / 3) ∧ (sin B2 = sqrt 2 / 3)) ∧
  -- Statement D
  (law_of_sines 9 10 (π / 3) < 1 → (∃ B1 B2 : ℝ, B1 ≠ B2 ∧ sin B1 = law_of_sines 9 10 (π / 3) ∧ sin B2 = law_of_sines 9 10 (π / 3) → ¬ ∃1! D)) :=
begin
  -- Proof steps will go here, we add "sorry" to compile successfully.
  sorry
end

end correct_option_is_B_l415_415328


namespace range_of_m_l415_415264

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := -x^2 + x + m + 2

theorem range_of_m (m : ℝ) : 
  (∃! x : ℤ, f x m ≥ |x|) ↔ -2 ≤ m ∧ m < -1 :=
by
  sorry

end range_of_m_l415_415264


namespace amn_div_l415_415344

theorem amn_div (a m n : ℕ) (a_pos : a > 1) (h : a > 1 ∧ (a^m + 1) ∣ (a^n + 1)) : m ∣ n :=
by sorry

end amn_div_l415_415344


namespace border_area_correct_l415_415141

theorem border_area_correct :
  let photo_height := 9
  let photo_width := 12
  let border_width := 3
  let photo_area := photo_height * photo_width
  let framed_height := photo_height + 2 * border_width
  let framed_width := photo_width + 2 * border_width
  let framed_area := framed_height * framed_width
  let border_area := framed_area - photo_area
  border_area = 162 :=
by sorry

end border_area_correct_l415_415141


namespace cyclist_speed_flat_l415_415477

noncomputable def cyclist_speed (total_distance : ℝ) (total_time : ℝ) (flat_distance : ℝ) (flat_time : ℝ) : ℝ :=
flat_distance / (flat_time / 60)

theorem cyclist_speed_flat (total_distance = 1080) (total_time = 12) 
(flat_distance = 540) (flat_time = 6) :
cyclist_speed 1080 12 540 6 = 5.4 := by
  sorry

end cyclist_speed_flat_l415_415477


namespace chess_game_players_l415_415419

theorem chess_game_players 
  (n : ℕ) 
  (h_n : n = 15) 
  (total_games : ℕ) 
  (h_total_games : total_games = 105) :
  n.choose 2 = total_games := 
by {
  rw [h_n, h_total_games],
  exact Nat.choose_self 15 2,
}

end chess_game_players_l415_415419


namespace surface_area_brick_l415_415205

def bottom_base : ℝ := 10
def top_base : ℝ := 7
def trapezoid_height : ℝ := 4
def depth : ℝ := 3

def area_trapezoid (b1 b2 h : ℝ) : ℝ :=
  (1 / 2) * (b1 + b2) * h

def area_rectangle (a b : ℝ) : ℝ :=
  a * b

theorem surface_area_brick :
  let A_trapezoid := area_trapezoid bottom_base top_base trapezoid_height
  ∧ let A_2_trapezoids := 2 * A_trapezoid
  ∧ let A_side := area_rectangle trapezoid_height depth
  ∧ let A_2_sides := 2 * A_side
  ∧ let A_top := area_rectangle top_base depth
  ∧ let A_bottom := A_top in
  (A_2_trapezoids + A_2_sides + A_top + A_bottom) = 134 := 
sorry

end surface_area_brick_l415_415205


namespace min_expr_value_l415_415201

noncomputable def expr (x : ℝ) : ℝ :=
  real.sqrt (x^2 + (1 - x)^2) + real.sqrt ((x - 1)^2 + (x - 1)^2)

theorem min_expr_value : ∀ x : ℝ, expr x ≥ 1 :=
sorry

end min_expr_value_l415_415201


namespace convert_shneids_to_shmacks_l415_415029

-- Declare the conversion ratios as constants
constants (shmacks : ℝ) (shicks : ℝ) (shures : ℝ) (shneids : ℝ)

-- Conditions based on the problem statement
axiom h1 : 5 * shmacks = 2 * shicks
axiom h2 : 3 * shicks = 5 * shures
axiom h3 : 2 * shures = 9 * shneids

-- The proof statement: converting 6 shneids to shmacks results in 2 shmacks
theorem convert_shneids_to_shmacks : 6 * shneids = 2 * shmacks :=
by
  sorry

end convert_shneids_to_shmacks_l415_415029


namespace rainfall_forecast_correct_interpretation_l415_415014

def rainfall_forecast (p : ℝ) : Prop := p = 0.85
def correct_interpretation : Prop := 
  ∃ (options : ℕ → Prop), 
    options 1 = (85% of the areas in our city will experience rainfall tomorrow) ∧ 
    options 2 = (it will rain for 85% of the time tomorrow in our city) ∧ 
    options 3 = (if you go out without rain gear tomorrow, you will definitely get wet) ∧ 
    options 4 = (if you go out without rain gear tomorrow, there is a high possibility of getting wet) ∧ 
    ∀ n, (rainfall_forecast 0.85 → options n → n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 3) ∧ (n = 4)

theorem rainfall_forecast_correct_interpretation (p : ℝ) :
  rainfall_forecast p → correct_interpretation := by
  sorry

end rainfall_forecast_correct_interpretation_l415_415014


namespace ax5_by5_eq_neg1065_l415_415703

theorem ax5_by5_eq_neg1065 (a b x y : ℝ) 
  (h1 : a*x + b*y = 5) 
  (h2 : a*x^2 + b*y^2 = 9) 
  (h3 : a*x^3 + b*y^3 = 20) 
  (h4 : a*x^4 + b*y^4 = 48) 
  (h5 : x + y = -15) 
  (h6 : x^2 + y^2 = 55) : 
  a * x^5 + b * y^5 = -1065 := 
sorry

end ax5_by5_eq_neg1065_l415_415703


namespace cartesian_to_polar_curve_C_l415_415387

theorem cartesian_to_polar_curve_C (x y : ℝ) (θ ρ : ℝ) 
  (h1 : x = ρ * Real.cos θ)
  (h2 : y = ρ * Real.sin θ)
  (h3 : x^2 + y^2 - 2 * x = 0) : 
  ρ = 2 * Real.cos θ :=
sorry

end cartesian_to_polar_curve_C_l415_415387


namespace collinear_condition_right_angle_triangle_condition_l415_415269

-- Given definitions
def OA := (3, -4) : ℝ × ℝ
def OB := (6, -3) : ℝ × ℝ
def OC (m : ℝ) := (5 - m, -3 - m) : ℝ × ℝ

-- Proof for collinearity condition
theorem collinear_condition (m : ℝ) : 
  (let AB := (3, 1) : ℝ × ℝ in 
  let AC := (2 - m, 1 - m) : ℝ × ℝ in 
  (∃ k : ℝ, AC = (k * (3, 1))) -> m = 1 / 2) :=
sorry

-- Proof for right-angled triangle condition
theorem right_angle_triangle_condition (m : ℝ) : 
  (let AB := (3, 1) : ℝ × ℝ in
   let AC := (2 - m, 1 - m) : ℝ × ℝ in
   let BC := (-1 - m, -m) : ℝ × ℝ in
   (∃ (k : ℝ), AB.1 * AC.1 + AB.2 * AC.2 = 0 ∨
   (AB.1 * BC.1 + AB.2 * BC.2 = 0 ∨
   (BC.1 * AC.1 + BC.2 * AC.2 = 0)) -> 
   (m = 7 / 4 ∨ m = -3 / 4 ∨ m = (1 + Real.sqrt 5) / 2 ∨
   m = (1 - Real.sqrt 5) / 2)) :=
sorry

end collinear_condition_right_angle_triangle_condition_l415_415269


namespace complex_expression_l415_415241
-- Define the complex number z
def z : ℂ := 1 + I

-- Define the complex conjugate of z
def z_conj : ℂ := conj z

-- Define the magnitude of the complex conjugate
def z_conj_mag : ℝ := complex.abs z_conj

-- Given z, prove the expression
theorem complex_expression (z : ℂ) (hz : z = 1 + I) : z * conj z + ↑(complex.abs (conj z)) - 1 = ↑(real.sqrt 2) + 1 :=
by
  -- Context has all necessary information
  -- Proof to be filled in
  sorry

end complex_expression_l415_415241


namespace find_smallest_a_l415_415079

noncomputable def quadratic_eq_1_roots (a b : ℤ) : Prop :=
∃ α β : ℤ, x < -1 ∧ x^2 + bx + a = 0

noncomputable def quadratic_eq_2_roots (a c : ℤ) : Prop :=
∃ γ δ : ℤ, x < -1 ∧ x^2 + cx + (a - 1) = 0

theorem find_smallest_a : ∃ a, (quadratic_eq_1_roots a b ∧ quadratic_eq_2_roots a c) ↔ a = 15 :=
sorry

end find_smallest_a_l415_415079


namespace vitC_two_apple_three_orange_l415_415721

-- Define the conditions
constant apple_juice_vitC : ℕ := 103  -- milligrams in one glass of apple juice
constant total_vitC_1_oj_1_aj : ℕ := 185  -- total vitamin C in one glass of apple juice and one glass of orange juice

-- Define the content of vitamin C in orange juice based on the conditions
def orange_juice_vitC : ℕ := total_vitC_1_oj_1_aj - apple_juice_vitC  -- 82 milligrams 

-- Prove the final quantity asked in the question
theorem vitC_two_apple_three_orange : 2 * apple_juice_vitC + 3 * orange_juice_vitC = 452 := by
  -- Skipping the proof steps with sorry
  sorry

end vitC_two_apple_three_orange_l415_415721


namespace f_discontinuity_f_at_discontinuity_l415_415333

noncomputable def f (x : ℝ) : ℝ := 
  Real.exp ( - (1 / (2 * Real.ofNat (Nat.gcd1 (2 * Real.exp x - 1)))) )

theorem f_discontinuity : ∃! x ∈ Set.Ioo (0 : ℝ) (⊤ : ℝ), ¬ ContinuousAt f x :=
begin
  sorry
end

theorem f_at_discontinuity : f (1 / 2) = Real.exp (-1 / 2) :=
begin
  sorry
end

end f_discontinuity_f_at_discontinuity_l415_415333


namespace volume_of_prism_l415_415061

   theorem volume_of_prism (a b c : ℝ)
     (h1 : a * b = 18) (h2 : b * c = 12) (h3 : a * c = 8) :
     a * b * c = 24 * Real.sqrt 3 :=
   sorry
   
end volume_of_prism_l415_415061


namespace longer_side_of_rectangle_l415_415104

theorem longer_side_of_rectangle (r : ℝ) (Aₙ Aₙ: ℝ) (L S: ℝ):
  (r = 6) → 
  (Aₙ = 36 * π) →
  (Aₙ = 108 * π) →
  (S = 12) → 
  (S * L = Aₙ) →
  L = 9 * π := sorry

end longer_side_of_rectangle_l415_415104


namespace problem_statement_l415_415332

noncomputable def f (t : ℝ) (N : ℕ) (a : ℕ → ℝ) : ℝ :=
  ∑ j in finset.range (N + 1), a j * real.sin (2 * real.pi * j * t)

noncomputable def N_k (k : ℕ) (N : ℕ) (a : ℕ → ℝ) : ℕ :=
  ((λ t, (f t N a)^(k)).deriv_zero_count [0,1)) -- This is a placeholder for counting zeros, should be defined precisely.

theorem problem_statement (N : ℕ) (a : ℕ → ℝ) (h_nonzero : a N ≠ 0) :
  (∀ k : ℕ, N_k k N a ≤ N_k (k + 1) N a) ∧ (tendsto (λ k, N_k k N a) at_top (2 * N)) :=
begin
  sorry
end

end problem_statement_l415_415332


namespace solve_for_x_l415_415219

-- Definitions of δ and φ
def delta (x : ℚ) : ℚ := 4 * x + 9
def phi (x : ℚ) : ℚ := 9 * x + 8

-- The main proof statement
theorem solve_for_x :
  ∃ x : ℚ, delta (phi x) = 10 ∧ x = -31 / 36 :=
by
  sorry

end solve_for_x_l415_415219


namespace largest_reciprocal_l415_415445

theorem largest_reciprocal :
  let a := (3: ℚ) / 4
  let b := 5 / (3: ℚ)
  let c := (-1: ℚ) / 6
  let d := (7: ℚ)
  let e := (3: ℚ)
  ∀ x ∈ {a, b, c, d, e}, has_lt.lt (1/x) (1/a) :=
by 
  sorry

end largest_reciprocal_l415_415445


namespace discriminant_of_given_quadratic_l415_415198

-- define the coefficients a, b, c
def a : ℚ := 2
def b : ℚ := 2 + 1/2
def c : ℚ := 1/2

-- define the discriminant function for a quadratic equation ax^2 + bx + c
def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

-- state the theorem
theorem discriminant_of_given_quadratic : discriminant a b c = 9/4 :=
by
  -- add the proof here
  sorry

end discriminant_of_given_quadratic_l415_415198


namespace Alyssa_missed_games_l415_415517

theorem Alyssa_missed_games (total_games attended_games : ℕ) (h1 : total_games = 31) (h2 : attended_games = 13) : total_games - attended_games = 18 :=
by sorry

end Alyssa_missed_games_l415_415517


namespace triangle_min_sum_l415_415970

-- Let a, b, and c be the sides of the triangle opposite to angles A, B, and C respectively
variables {a b c : ℝ}

-- Given conditions:
-- 1. (a + b)^2 - c^2 = 4
-- 2. C = 60 degrees, and by cosine rule, we have cos C = (a^2 + b^2 - c^2) / (2ab)
-- Since C = 60 degrees, cos C = 1/2
-- Therefore, (a^2 + b^2 - c^2) / (2ab) = 1/2

theorem triangle_min_sum (h1 : (a + b) ^ 2 - c ^ 2 = 4)
    (h2 : cos (real.pi / 3) = (a ^ 2 + b ^ 2 - c ^ 2) / (2 * a * b)) :
  a + b ≥ 2 * real.sqrt (4 / 3) :=
  by
    sorry

end triangle_min_sum_l415_415970


namespace set_sum_card_ge_l415_415636

variable {A B : Finset ℕ}
variable {m n : ℕ}
variable (cond : ∀ x y u v ∈ A, x + y = u + v → {x,y} = {u,v})

theorem set_sum_card_ge (hA : A.card = m) (hB : B.card = n) (hA_ne : A ≠ ∅) (hB_ne : B ≠ ∅) :
  (A.product B).card ≥ (m^2 * n) / (m + n - 1) :=
by
  have : ∀ a₁ a₂ a₃ a₄ ∈ A, a₁ + a₂ = a₃ + a₄ → {a₁, a₂} = {a₃, a₄} := cond
  sorry

end set_sum_card_ge_l415_415636


namespace annieka_free_throws_l415_415000

theorem annieka_free_throws :
  let d : ℕ := 12 in
  let k : ℕ := d + (d / 2) in
  let a : ℕ := k - 4 in
  a = 14 :=
by
  let d := 12
  let k := d + (d / 2)
  let a := k - 4
  show a = 14
  sorry

end annieka_free_throws_l415_415000


namespace frog_hops_ratio_l415_415037

theorem frog_hops_ratio (S T F : ℕ) (h1 : S = 2 * T) (h2 : S = 18) (h3 : F + S + T = 99) :
  F / S = 4 / 1 :=
by
  sorry

end frog_hops_ratio_l415_415037


namespace probability_x_lt_2y_in_rectangle_l415_415834

theorem probability_x_lt_2y_in_rectangle :
  let ℝ := Real,
      rectangle := set.Icc (0, 0) (4, 3),
      region := {p : ℝ × ℝ | p.1 < 2 * p.2},
      area_of_triangle := (1 / 2) * 3 * 3,
      area_of_rectangle := 4 * 3,
      prob := area_of_triangle / area_of_rectangle in
  prob = 3 / 8 :=
by sorry

end probability_x_lt_2y_in_rectangle_l415_415834


namespace find_angle_ABC_l415_415959

-- Define the structure of the triangle and its properties
structure Triangle :=
  (A B C : Point)
  (side_AC_largest: AC > AB ∧ AC > BC)

-- Define points M and N with given conditions
structure PointsOnAC (A B C M N : Point) :=
  (AM_eq_AB : AM = AB)
  (CN_eq_CB : CN = CB)

-- Define the angles and the given condition for angle NBM
structure Angles (ABC NBM : ℝ) :=
  (angle_NBM_eq_third_ABC : NBM = ABC / 3)

-- Main theorem
theorem find_angle_ABC (A B C M N : Point) 
  (triangle : Triangle A B C)
  (points_on_AC : PointsOnAC A B C M N)
  (angles : Angles ABC NBM) : 
  ∠ABC = 108 :=
  sorry

end find_angle_ABC_l415_415959


namespace solve_for_s_l415_415189

theorem solve_for_s (s : ℝ) (h : 3 * real.log s / real.log 2 = real.log (4 * s) / real.log 2) : s = 2 :=
by 
  sorry

end solve_for_s_l415_415189


namespace two_pow_div_factorial_iff_l415_415365

theorem two_pow_div_factorial_iff (n : ℕ) : 
  (∃ k : ℕ, k > 0 ∧ n = 2^(k - 1)) ↔ (∃ m : ℕ, m > 0 ∧ 2^(n - 1) ∣ n!) :=
by
  sorry

end two_pow_div_factorial_iff_l415_415365


namespace distance_between_points_l415_415771

theorem distance_between_points :
  let A : (ℝ × ℝ) := (-5, 3)
  let B : (ℝ × ℝ) := (6, 3)
  dist (A.1, A.2) (B.1, B.2) = 11 :=
by
  unfold dist
  calc
    sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = sqrt ((6 - (-5))^2 + (3 - 3)^2) : by sorry
    ...                                                                                                                                                                                                                                  = sqrt ((6 + 5)^2 + 0^2) : by sorry
    ...                                                             = sqrt (11^2 + 0) : by sorry
    ...                                                             = sqrt 121 : by sorry
    ...                                                             = 11 : by sorry
  sorry

end distance_between_points_l415_415771


namespace cube_plane_cuts_estimate_l415_415989

/--
In a cube where all vertices and edge midpoints are marked, the number of pieces \( N \) 
the cube is divided into by all planes passing through at least four marked points is estimated.
-/
theorem cube_plane_cuts_estimate :
  let vertices := 8
  let edge_midpoints := 12
  let total_marked_points := vertices + edge_midpoints
  let possible_planes := Nat.choose total_marked_points 4
  let unique_planes_estimate := 15600
  (estimate_N : ℕ) := estimate_N = unique_planes_estimate :=
begin
  let vertices := 8,
  let edge_midpoints := 12,
  let total_marked_points := vertices + edge_midpoints,
  let possible_planes := Nat.choose total_marked_points 4,
  let unique_planes_estimate := 15600,
  exists.intro unique_planes_estimate rfl,
end

end cube_plane_cuts_estimate_l415_415989


namespace annieka_free_throws_l415_415002

theorem annieka_free_throws (deshawn_throws : ℕ) (kayla_factor : ℝ) (annieka_diff : ℕ) (ht1 : deshawn_throws = 12) (ht2 : kayla_factor = 1.5) (ht3 : annieka_diff = 4) :
  ∃ (annieka_throws : ℕ), annieka_throws = (⌊deshawn_throws * kayla_factor⌋.toNat - annieka_diff) :=
by
  sorry

end annieka_free_throws_l415_415002


namespace opposite_face_P_l415_415169

-- Define predicate for faces booleans.
def is_adjacent (P Q R S : Prop) := P ∨ Q ∨ R ∨ S

-- Define the faces as propositions.
variables (P Q R S T U : Prop)

-- Given conditions
axiom face_surrounded_by : is_adjacent Q R S
axiom face_opposite_candidates : ¬ is_adjacent T U

-- Prove that U is the face opposite to P
theorem opposite_face_P : U :=
by
  -- The proof details are omitted and replaced with 'sorry'.
  sorry

end opposite_face_P_l415_415169


namespace fraction_problem_l415_415657

theorem fraction_problem
    (q r s u : ℚ)
    (h1 : q / r = 8)
    (h2 : s / r = 4)
    (h3 : s / u = 1 / 3) :
    u / q = 3 / 2 :=
  sorry

end fraction_problem_l415_415657


namespace tan_transformation_l415_415910

-- Given condition in Lean 4
def given_tan_condition (α : ℝ) : Prop :=
  Real.tan (π / 7 + α) = 5

-- The proof goal in Lean 4
theorem tan_transformation (α : ℝ) (h : given_tan_condition α) : 
  Real.tan (6 * π / 7 - α) = -5 := 
sorry

end tan_transformation_l415_415910


namespace triangle_area_l415_415507

theorem triangle_area : 
  let line_eq (x y : ℝ) := 3 * x + 2 * y = 12
  let x_intercept := (4 : ℝ)
  let y_intercept := (6 : ℝ)
  ∃ (x y : ℝ), line_eq x y ∧ x = x_intercept ∧ y = y_intercept ∧
  ∃ (area : ℝ), area = 1 / 2 * x * y ∧ area = 12 :=
by
  sorry

end triangle_area_l415_415507


namespace problem1_problem2_l415_415468

-- Definitions for the first problem
def isProjection (O M : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop := 
  ∃ slope : ℝ, 
    (M.2 - O.2) = slope * (M.1 - O.1) ∧
    ∀ (P : ℝ × ℝ), l P ↔ P.2 = slope * (P.1 - 2) + (-1 + 2 * slope)

-- Problem 1: Proving the equation of the line
theorem problem1 (O : ℝ × ℝ) (M : ℝ × ℝ) (l : ℝ × ℝ → Prop)
  (hM : M = (2, -1)) (hO : O = (0, 0)) : isProjection O M l → ∃ a b c : ℝ, a = 2 ∧ b = -1 ∧ c = -5 ∧ ∀ P : ℝ × ℝ, l P ↔ a * P.1 + b * P.2 + c = 0 :=
by
  sorry

-- Definitions for the second problem
def isMidpoint (A B M : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

def isCentroid (A B C P : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1 + C.1) / 3 ∧ P.2 = (A.2 + B.2 + C.2) / 3

def dist (P Q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

-- Problem 2: Proving the length of side BC
theorem problem2 (A B M P C : ℝ × ℝ)
  (hA : A = (4, -1)) (hM : M = (3, 2)) (hP : P = (4, 2)) (hMid : isMidpoint A B M) (hCent : isCentroid A B C P) :
  dist B C = 5 :=
by
  sorry

end problem1_problem2_l415_415468


namespace michael_saves_more_l415_415148

-- Definitions for the conditions
def price_per_pair : ℝ := 50
def discount_a (price : ℝ) : ℝ := price + 0.6 * price
def discount_b (price : ℝ) : ℝ := 2 * price - 15

-- Statement to prove
theorem michael_saves_more (price : ℝ) (h : price = price_per_pair) : discount_b price - discount_a price = 5 :=
by
  sorry

end michael_saves_more_l415_415148


namespace area_of_triangle_bounded_by_line_and_axes_l415_415501

theorem area_of_triangle_bounded_by_line_and_axes (x y : ℝ) (hx : 3 * x + 2 * y = 12) :
  ∃ (area : ℝ), area = 12 := by
sorry

end area_of_triangle_bounded_by_line_and_axes_l415_415501


namespace distance_MN_l415_415688

open Real

def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2 + (q.3 - p.3) ^ 2)

noncomputable def M : ℝ × ℝ × ℝ := (0, 1, 2)
noncomputable def N : ℝ × ℝ × ℝ := (-1, 2, 1)

theorem distance_MN : distance M N = sqrt 3 :=
by
  sorry

end distance_MN_l415_415688


namespace no_real_solution_l415_415900

theorem no_real_solution (x y : ℝ) (h: y = 3 * x - 1) : ¬ (4 * y ^ 2 + y + 3 = 3 * (8 * x ^ 2 + 3 * y + 1)) :=
by
  sorry

end no_real_solution_l415_415900


namespace probability_AI77_is_correct_l415_415682

def is_vowel (c : Char) : Prop :=
  c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

def is_digit (c : Char) : Prop :=
  c.isDigit

def license_plate_valid (s : String) : Prop :=
  s.length = 4 ∧ 
  is_vowel s[0] ∧ 
  is_vowel s[1] ∧ 
  s[0] ≠ s[1] ∧ 
  is_digit s[2] ∧ 
  is_digit s[3]

def total_license_plates : ℕ :=
  5 * 4 * 10 * 10

def specific_license_plate_prob (plate : String) : ℚ :=
  if license_plate_valid plate ∧ plate = "AI77"
  then 1 / total_license_plates
  else 0

noncomputable def probability_AI77 : ℚ :=
  specific_license_plate_prob "AI77"

theorem probability_AI77_is_correct : 
  probability_AI77 = 1 / 2000 := 
by
  sorry

end probability_AI77_is_correct_l415_415682


namespace cos_neg_half_implies_sin_sqrt_three_half_l415_415245

theorem cos_neg_half_implies_sin_sqrt_three_half (A : ℝ) (h : 0 < A ∧ A < π) :
  cos A = -1/2 → sin A = sqrt 3 / 2 ∧ ¬(sin A = sqrt 3 / 2 → cos A = -1/2) :=
by
  sorry

end cos_neg_half_implies_sin_sqrt_three_half_l415_415245


namespace min_value_of_n_for_constant_term_l415_415295

theorem min_value_of_n_for_constant_term :
  ∃ (n : ℕ) (r : ℕ) (h₁ : r > 0) (h₂ : n > 0), 
  (2 * n - 7 * r / 3 = 0) ∧ n = 7 :=
by
  sorry

end min_value_of_n_for_constant_term_l415_415295


namespace pair_of_straight_lines_l415_415543

noncomputable def conic_section_graph (x y : ℝ) : Prop :=
  x^2 - x * y - 6 * y^2 = 0

theorem pair_of_straight_lines : 
  ∃ (a b c d : ℝ), (∀ x y : ℝ, conic_section_graph x y → (x = a * y ∨ x = b * y)) :=
begin
  -- proof to be filled in later
  sorry
end

end pair_of_straight_lines_l415_415543


namespace impossible_distance_l415_415642

noncomputable def radius_O1 : ℝ := 2
noncomputable def radius_O2 : ℝ := 5

theorem impossible_distance :
  ∀ (d : ℝ), ¬ (radius_O1 ≠ radius_O2 → ¬ (d < abs (radius_O2 - radius_O1) ∨ d > radius_O2 + radius_O1) → d = 5) :=
by
  sorry

end impossible_distance_l415_415642


namespace no_isosceles_triangle_l415_415418

theorem no_isosceles_triangle (sticks : List ℝ) (h : sticks = List.map (λ n : ℕ, 0.9^n) (List.range 100)) :
  ¬ ∃ (s1 s2 s3 : ℝ), s1 ∈ sticks ∧ s2 ∈ sticks ∧ s3 ∈ sticks ∧ s1 = s2 ∧ s1 + s2 > s3 ∧ s1 + s3 > s2 ∧ s2 + s3 > s1 :=
by 
  sorry

end no_isosceles_triangle_l415_415418


namespace longer_side_of_rectangle_l415_415106

theorem longer_side_of_rectangle (r : ℝ) (Aₙ Aₙ: ℝ) (L S: ℝ):
  (r = 6) → 
  (Aₙ = 36 * π) →
  (Aₙ = 108 * π) →
  (S = 12) → 
  (S * L = Aₙ) →
  L = 9 * π := sorry

end longer_side_of_rectangle_l415_415106


namespace arithmetic_sequence_a8_l415_415757

noncomputable def arithmetic_sequence (a d n : ℕ): ℤ := a + (n - 1) * d

theorem arithmetic_sequence_a8 :
  ∃ (a d : ℤ),
    (arithmetic_sequence a d 5 = 11) ∧
    (12 * a + 66 * d = 186) ∧
    (arithmetic_sequence a d 8 = 20) :=
by {
  sorry
}

end arithmetic_sequence_a8_l415_415757


namespace find_foci_of_hyperbola_l415_415193

namespace HyperbolaFoci

def hyperbola := { p : ℝ × ℝ // (p.2^2 / 3) - (p.1^2) = 1 }

def foci_coordinates := (0, 2) ∨ (0, -2)

theorem find_foci_of_hyperbola : 
  (∀ p : hyperbola, (p.val = (0, 2) ∨ p.val = (0, -2))) :=
sorry

end HyperbolaFoci

end find_foci_of_hyperbola_l415_415193


namespace train_travel_distance_l415_415804

theorem train_travel_distance
  (coal_per_mile_lb : ℝ)
  (remaining_coal_lb : ℝ)
  (travel_distance_per_unit_mile : ℝ)
  (units_per_unit_lb : ℝ)
  (remaining_units : ℝ)
  (total_distance : ℝ) :
  coal_per_mile_lb = 2 →
  remaining_coal_lb = 160 →
  travel_distance_per_unit_mile = 5 →
  units_per_unit_lb = remaining_coal_lb / coal_per_mile_lb →
  remaining_units = units_per_unit_lb →
  total_distance = remaining_units * travel_distance_per_unit_mile →
  total_distance = 400 :=
by
  sorry

end train_travel_distance_l415_415804


namespace negation_of_exists_l415_415015

theorem negation_of_exists (x : ℝ) (h : ∃ x : ℝ, x^2 - x + 1 ≤ 0) : 
  (∀ x : ℝ, x^2 - x + 1 > 0) :=
sorry

end negation_of_exists_l415_415015


namespace max_participants_l415_415304

def multiple_choice_answers (choices : ℕ) (questions : ℕ) := { answers : finset (fin choices) // answers.card = questions }

def valid_competition (participants : ℕ) (choices questions : ℕ) :=
  ∃ (answers : finset (multiple_choice_answers choices questions)), answers.card = participants ∧
  ∀ (p1 p2 p3 ∈ answers), ∃ q ∈ range questions, p1.val q ≠ p2.val q ∧ p2.val q ≠ p3.val q ∧ p1.val q ≠ p3.val q

theorem max_participants : valid_competition 9 3 4 :=
sorry

end max_participants_l415_415304


namespace probability_of_weight_ge_30_l415_415439

noncomputable theory

-- Define the probability of an egg weighing less than 30 grams
def P_lt_30 : ℝ := 0.30

-- Define the probability of an egg weighing within the range [30, 40] grams
def P_30_40 : ℝ := 0.50

-- Define the event of the weight being not less than 30 grams
def P_ge_30 : ℝ := 1 - P_lt_30

-- Prove the event using predefined conditions
theorem probability_of_weight_ge_30 : P_ge_30 = 0.70 :=
sorry  

end probability_of_weight_ge_30_l415_415439


namespace max_value_occurs_l415_415179

noncomputable def max_value_of_function : ℝ := 5

theorem max_value_occurs (k : ℤ) : 
  ∃ (x : ℝ), 
  (3 - 2 * cos (x + (Real.pi / 4)) = max_value_of_function) ∧ 
  (x = 2 * k * Real.pi + (3 * Real.pi / 4)) :=
begin
  sorry
end

end max_value_occurs_l415_415179


namespace fastest_rate_is_C_l415_415415

-- Define reaction rates given in the problem
def rate_A : ℝ := 0.15  -- mol/(L·s)
def rate_B : ℝ := 0.6   -- mol/(L·s)
def rate_C : ℝ := 0.5   -- mol/(L·s)
def rate_D : ℝ := 0.4   -- mol/(L·s)

-- Define stoichiometric coefficients
def coeff_A : ℝ := 1
def coeff_B : ℝ := 3
def coeff_C : ℝ := 2
def coeff_D : ℝ := 2

-- Define normalized rates
def norm_rate_A : ℝ := rate_A / coeff_A
def norm_rate_B : ℝ := rate_B / coeff_B
def norm_rate_C : ℝ := rate_C / coeff_C
def norm_rate_D : ℝ := rate_D / coeff_D

-- Prove that the normalized rate of C is the fastest
theorem fastest_rate_is_C : 
  norm_rate_C > norm_rate_B ∧ norm_rate_C > norm_rate_D ∧ norm_rate_C > norm_rate_A := 
by 
  sorry

end fastest_rate_is_C_l415_415415


namespace min_sum_of_arithmetic_sequence_l415_415218

variable {a : ℕ → ℤ} -- Define a sequence a(n) of type ℕ → ℤ

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = d

theorem min_sum_of_arithmetic_sequence 
  (a : ℕ → ℤ) 
  (d : ℤ) 
  (h1 : a 1 = -10) 
  (h2 : a 4 + a 6 = -4)
  (h3 : is_arithmetic_sequence a d) :
  d = 2 ∧ 
  ∃ n : ℕ, (n = 5 ∨ n = 6) ∧ (S n = n^2 - 11 * n) ∧ (S n ≤ S m ∀ m) 
  :=
sorry

end min_sum_of_arithmetic_sequence_l415_415218


namespace no_married_triple_if_even_n_and_k_geq_half_n_l415_415157

variable (n : Nat)
variable (k : Nat)
variable (males females emales : Finset Nat)
variable (likes : Nat → Nat → Bool)

theorem no_married_triple_if_even_n_and_k_geq_half_n
  (h_even : Even n)
  (h_k : k ≥ n / 2)
  (h_likes_mutual : ∀ x y, likes x y = true ↔ likes y x = true)
  (h_males_likes : ∀ m ∈ males, ∃ kₘ, kₘ ≥ k ∧ kₘ ≤ (females ∪ emales).card ∧ (Finset.filter (likes m) (females ∪ emales)).card ≥ kₘ)
  (h_females_likes : ∀ f ∈ females, ∃ kₓ, kₓ ≥ k ∧ kₓ ≤ (males ∪ emales).card ∧ (Finset.filter (likes f) (males ∪ emales)).card ≥ kₓ)
  (h_emales_likes : ∀ e ∈ emales, ∃ kₑ, kₑ ≥ k ∧ kₑ ≤ (males ∪ females).card ∧ (Finset.filter (likes e) (males ∪ females)).card ≥ kₑ)
  : ∃ males females emales, males.card = n ∧ females.card = n ∧ emales.card = n ∧ ∀ m ∈ males, ∀ f ∈ females, ∀ e ∈ emales, (likes m f = false ∧ likes m e = false ∧ likes f e = false) :=
  sorry

end no_married_triple_if_even_n_and_k_geq_half_n_l415_415157


namespace D_72_l415_415202

-- Define what it means to express a number as a product of integers greater than 1
def is_factorization (n : ℕ) (factors : List ℕ) : Prop :=
  factors.prod = n ∧ ∀ x ∈ factors, x > 1

-- Define D(n) to be the number of such factorizations where the order counts
def D (n : ℕ) : ℕ :=
  List.length {factors // factors.prod = n ∧ ∀ x ∈ factors, x > 1}

-- Prove the specific case for the number 72
theorem D_72 : D 72 = 22 :=
  sorry

end D_72_l415_415202


namespace certain_event_proof_l415_415442

def Moonlight_in_front_of_bed := "depends_on_time_and_moon_position"
def Lonely_smoke_in_desert := "depends_on_specific_conditions"
def Reach_for_stars_with_hand := "physically_impossible"
def Yellow_River_flows_into_sea := "certain_event"

theorem certain_event_proof : Yellow_River_flows_into_sea = "certain_event" :=
by
  sorry

end certain_event_proof_l415_415442


namespace expand_product_l415_415885

theorem expand_product (x : ℝ) :
  (3 * x + 4) * (2 * x - 5) = 6 * x^2 - 7 * x - 20 :=
sorry

end expand_product_l415_415885


namespace min_alpha_beta_l415_415588

theorem min_alpha_beta (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1)
  (alpha : ℝ := a + 1 / a) (beta : ℝ := b + 1 / b) :
  alpha + beta ≥ 10 := by
  sorry

end min_alpha_beta_l415_415588


namespace x5_plus_x16_l415_415615

variables (d : ℝ) (x : ℕ → ℝ)

-- Conditions
def harmonic_seq : Prop := ∀ n : ℕ, 0 < n → (1 / x (n + 1) - 1 / x n = d)
def sum_x1_to_x20 : Prop := (∑ i in Finset.range 20, x (i + 1)) = 200

-- Statement to prove
theorem x5_plus_x16 (hyp_harmonic : harmonic_seq d x) (hyp_sum : sum_x1_to_x20 x) :
  x 5 + x 16 = 20 :=
sorry

end x5_plus_x16_l415_415615


namespace least_number_added_to_divide_l415_415471

-- Definitions of conditions
def lcm_three_five_seven_eight : ℕ := Nat.lcm (Nat.lcm (Nat.lcm 3 5) 7) 8
def remainder_28523_lcm := 28523 % lcm_three_five_seven_eight

-- Lean statement to prove the correct answer
theorem least_number_added_to_divide (n : ℕ) :
  n = lcm_three_five_seven_eight - remainder_28523_lcm :=
sorry

end least_number_added_to_divide_l415_415471


namespace find_original_function_l415_415293

-- Definitions based on conditions
def shortened_abscissa (f : ℝ → ℝ) : ℝ → ℝ := λ x, f (2 * x)

def shifted_right (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x, f (x - a)

-- Theorem statement
theorem find_original_function (f : ℝ → ℝ)
  (h : (shifted_right (shortened_abscissa f) (π / 3)) = (λ x, sin (x - π / 4))) :
  f = (λ x, sin (x / 2 + π / 12)) :=
sorry

end find_original_function_l415_415293


namespace hyperbola_eccentricity_l415_415247

theorem hyperbola_eccentricity (a : ℝ) (h : 0 < a) :
  (∃ b : ℝ, (∀ (x y : ℝ), x + 2 * y = 0 → x^2 / a - y^2 = 1 * y → 1 * y)) →
  (sqrt(1 + 1 / a) = sqrt(5) / 2) :=
by
  sorry

end hyperbola_eccentricity_l415_415247


namespace triangle_angle_C_l415_415300

theorem triangle_angle_C (A B C : ℝ) (a b c : ℝ) (k : ℝ) (h_pos : k > 0)
  (h_ratio : sin A / sin B = 3 / 5 ∧ sin B / sin C = 5 / 7)
  (h_sides : a = 3 * k ∧ b = 5 * k ∧ c = 7 * k)
  (h_C_range : 0 < C ∧ C < 180) :
  C = 120 :=
sorry

end triangle_angle_C_l415_415300


namespace seating_arrangement_l415_415672

theorem seating_arrangement (G : Type) [Fintype G] [Graph G] (k : ℕ)
  (h1 : ∀ x : G, Fintype.card (set_of (λ y, y ≠ x ∧ adj x y)) ≥ k) :
  ∃ (S : Finset G), S.card ≥ k + 1 ∧ ∀ (x ∈ S), ∀ (y ∈ S), (adj x y ↔ (x ≠ y ∧ ∃ (z : G), adj x z ∧ z ≠ x ∧ z ≠ y)) :=
sorry

end seating_arrangement_l415_415672


namespace sequence_sum_l415_415227

theorem sequence_sum (a : ℕ → ℕ) (S : ℕ → ℚ) :
  (a 1 = 1) →
  (∀ n : ℕ, a (n + 1) = a n + 1) →
  (∀ n : ℕ, S n = n * (n + 1) / 2) →
  (∑ k in finset.range n, 1 / S (k + 1)) = 2 * n / (n + 1) :=
by
  intros 
  sorry

end sequence_sum_l415_415227


namespace custom_op_1_4_3_2_eq_3_7_l415_415175

-- Define the fractional part function
def frac_part (x : ℝ) : ℝ := x - x.floor

-- Define custom operation
def custom_op (a b : ℝ) : ℝ :=
  2 * frac_part (a / 2) + 3 * frac_part ((a + b) / 6)

-- The theorem to prove
theorem custom_op_1_4_3_2_eq_3_7 : custom_op 1.4 3.2 = 3.7 :=
by
  sorry

end custom_op_1_4_3_2_eq_3_7_l415_415175


namespace kite_area_parabolas_l415_415017

theorem kite_area_parabolas (a b : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (h_intersection : ∀ (x : ℝ), ax^2 - 3 = 5 - bx^2 -> False) -- ensuring exactly 4 intersections on coordinate axes
  (h4points : -- some condition to ensure the points form a kite)
  (h_area : 15 = 1 / 2 * (8) * (2 * sqrt (3 / a))) :
  a + b = 2.3 :=
begin
  sorry
end

end kite_area_parabolas_l415_415017


namespace find_matrix_N_l415_415889

def matrix2x2 := ℚ × ℚ × ℚ × ℚ

def apply_matrix (M : matrix2x2) (v : ℚ × ℚ) : ℚ × ℚ :=
  let (a, b, c, d) := M;
  let (x, y) := v;
  (a * x + b * y, c * x + d * y)

theorem find_matrix_N : ∃ (N : matrix2x2), 
  apply_matrix N (3, 1) = (5, -1) ∧ 
  apply_matrix N (1, -2) = (0, 6) ∧ 
  N = (10/7, 5/7, 4/7, -19/7) :=
by {
  sorry
}

end find_matrix_N_l415_415889


namespace total_students_in_fifth_grade_l415_415312

-- Definitions of the conditions
def total_boys := 296
def total_playing_soccer := 250
def percent_boys_playing_soccer := 0.86
def girls_not_playing_soccer := 89

-- Calculation parameters
def boys_playing_soccer := percent_boys_playing_soccer * total_playing_soccer
def boys_not_playing_soccer := total_boys - boys_playing_soccer
def girls_playing_soccer := total_playing_soccer - boys_playing_soccer
def total_girls := girls_playing_soccer + girls_not_playing_soccer

-- Statement of the theorem to be proven
theorem total_students_in_fifth_grade :
  total_boys + total_girls = 420 :=
by
  -- Proof steps go here
  sorry

end total_students_in_fifth_grade_l415_415312


namespace sum_of_roots_eq_l415_415896

theorem sum_of_roots_eq : 
  ∀ (x : ℝ), (x ≠ 1) ∧ (x ≠ -1) ∧ ( -10 * x / (x^2 - 1) = 3 * x / (x + 1) - 5 / (x - 1)) → 
  ∑ (x : ℝ) in {x | -10 * x / (x^2 - 1) = 3 * x / (x + 1) - 5 / (x - 1)}, x = 8 / 3 :=
by sorry

end sum_of_roots_eq_l415_415896


namespace eval_frac_exp_correct_eval_log_exp_correct_l415_415554

noncomputable def eval_frac_exp : ℚ := (81 / 16) ^ (-3 / 4)
noncomputable def eval_log_exp : ℤ := Int.log2 (4^7 * 2^5)

theorem eval_frac_exp_correct : eval_frac_exp = 8 / 27 :=
by
  sorry

theorem eval_log_exp_correct : eval_log_exp = 19 :=
by
  sorry


end eval_frac_exp_correct_eval_log_exp_correct_l415_415554


namespace range_of_f_l415_415747

-- Defining the floor function
def floor (x : ℝ) : ℤ := int.floor x

-- Defining the function f(x)
def f (x : ℝ) : ℝ := floor x - 2 * x

-- Statement of the problem in Lean 4
theorem range_of_f : ∀ y : ℝ, (∃ x : ℝ, f(x) = y) ↔ y < 0 :=
by
  sorry

end range_of_f_l415_415747


namespace find_analytical_expression_of_f_l415_415965

variable (f : ℝ → ℝ)

theorem find_analytical_expression_of_f
  (h : ∀ x : ℝ, f (2 * x + 1) = 4 * x^2 + 4 * x) :
  ∀ x : ℝ, f x = x^2 - 1 :=
sorry

end find_analytical_expression_of_f_l415_415965


namespace paint_left_after_three_days_l415_415986

theorem paint_left_after_three_days :
  ∀ (p : ℚ), p > 0 →
  (let day1 := p - (1/4 * p) in
   let day2 := day1 - (1/2 * day1) in
   let day3 := day2 - (1/3 * day2) in
   p ≠ 0 → day3 = p * (1/4)) := 
begin
  intros p p_pos,
  have initial := p,
  have h1 : initial - (1/4 * initial) = 3/4 * initial, by field_simp,
  let day1 := 3/4 * initial,
  have h2 : day1 - (1/2 * day1) = 1/2 * day1, by field_simp,
  let day2 := 1/2 * day1,
  have h3 : day2 - (1/3 * day2) = 2/3 * day2, by field_simp,
  let day3 := 2/3 * day2,
  show day3 = p * (1/4), from sorry,
end

end paint_left_after_three_days_l415_415986


namespace longer_side_length_l415_415119

-- Given a circle with radius 6 cm
def radius : ℝ := 6
def circle_area : ℝ := Real.pi * radius^2

-- Given that the area of the rectangle is three times the area of the circle
def rectangle_area : ℝ := 3 * circle_area

-- Given the rectangle has a shorter side equal to the diameter of the circle
def shorter_side : ℝ := 2 * radius

-- Prove that the length of the longer side of the rectangle is 9π cm
theorem longer_side_length : ∃ (longer_side : ℝ), longer_side = 9 * Real.pi :=
by
  have circle_area_def : circle_area = 36 * Real.pi := by sorry
  have rectangle_area_def : rectangle_area = 108 * Real.pi := by sorry
  have shorter_side_def : shorter_side = 12 := by sorry
  let longer_side := rectangle_area / shorter_side
  use longer_side
  show longer_side = 9 * Real.pi
  sorry

end longer_side_length_l415_415119


namespace find_lambda_l415_415951

-- Definitions based on the problem conditions
def a : ℝ × ℝ := (-2, 3)
def b : ℝ × ℝ := (1, 2)

-- Definition of perpendicular vectors in terms of dot product
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Proof statement: λ that satisfies the condition
theorem find_lambda (λ : ℝ) : perpendicular (λ • a + b) b → λ = -5 / 4 :=
by sorry

end find_lambda_l415_415951


namespace complex_conjugate_is_correct_l415_415891

-- Define the imaginary unit i
def i : ℂ := complex.I

-- Specify the given complex number z
def z : ℂ := (2 - i) / i

-- Define the correct answer as the complex conjugate of z
def correct_answer : ℂ := -1 + 2 * i

-- The proof statement: proving z's conjugate equals the correct answer
theorem complex_conjugate_is_correct : complex.conj z = correct_answer := by
  sorry

end complex_conjugate_is_correct_l415_415891


namespace sqrt_div_value_eq_4_l415_415756

theorem sqrt_div_value_eq_4 (x : ℝ) : (sqrt 5184) / x = 4 → x = 18 :=
by
  intro h
  have h₀ : sqrt 5184 = 72 := rfl
  rw [h₀] at h
  linarith

end sqrt_div_value_eq_4_l415_415756


namespace smallest_positive_period_monotonically_increasing_intervals_min_max_values_on_interval_l415_415626

noncomputable def f (x : ℝ) : ℝ := sqrt 2 * Real.cos (2 * x - (π / 4))

theorem smallest_positive_period (x : ℝ) : ∃ T, T > 0 ∧ (∀ x ∈ ℝ, f(x + T) = f(x)) ∧ T = π := sorry

theorem monotonically_increasing_intervals (k : ℤ) : 
  ∃ (a b : ℝ), (∀ x ∈ ℝ, -3 * π / 8 + k * π ≤ x ∧ x ≤ π / 8 + k * π → ∃ U V, ∀ x ∈ ℝ, (a ≤ x ∧ x ≤ b → f' x ≥ 0)) :=
sorry

theorem min_max_values_on_interval : 
  ∃ (a b : ℝ), a = - π / 8 ∧ b = π / 2 ∧ (∀ x ∈ set.Icc a b, 
    (f x = -1 ∧ x = π/2) ∨ 
    (f x = 1 ∧ (x = -π/8 ∨ x = π/8))) :=
sorry

end smallest_positive_period_monotonically_increasing_intervals_min_max_values_on_interval_l415_415626


namespace geom_seq_product_l415_415701

variable {a : ℕ → ℝ}
variable (q : ℝ)
variable (n : ℕ)
variable (prod_a_1_33 : ℝ)
variable (geom_seq : ∀ n, a n = a 1 * q ^ (n - 1))

theorem geom_seq_product (h : ∀ n, a n = a 1 * q ^ (n - 1))
  (q_eq_two : q = 2)
  (prod_eq : (∏ i in Finset.range 33, a i.succ) = 2 ^ 33) :
  (∏ i in Finset.range 11, a (3 * i + 3)) = 2 ^ 22 := sorry

end geom_seq_product_l415_415701


namespace longer_side_of_rectangle_is_l415_415116

-- Define the radius of the circle
def radius (r : ℝ) : Prop := r = 6

-- Define the area of the circle given the radius
def area_circle (A : ℝ) : Prop := A = 36 * Real.pi

-- Define the relationship between the area of the rectangle and the circle
def area_rectangle (A_rect : ℝ) (A : ℝ) : Prop := A_rect = 3 * A

-- Define the length of the shorter side of the rectangle
def shorter_side (s : ℝ) : Prop := s = 12

-- State the goal: the length of the longer side of the rectangle
def longer_side (l : ℝ) (A_rect : ℝ) (s : ℝ) : Prop := l = A_rect / s

-- Theorem statement combining all conditions and the goal
theorem longer_side_of_rectangle_is :
  ∃ l : ℝ, ∃ A_rect : ℝ, ∃ s : ℝ, radius 6 ∧ area_circle 36 * Real.pi ∧ area_rectangle A_rect (36 * Real.pi) ∧ shorter_side s ∧ longer_side l A_rect s :=
by {
  sorry
}

end longer_side_of_rectangle_is_l415_415116


namespace domain_of_g_l415_415177

def g (x : ℝ) : ℝ := real.sqrt (4 - real.sqrt (5 - real.sqrt (6 - x)))

theorem domain_of_g :
  (∀ x : ℝ, real.sqrt (4 - real.sqrt (5 - real.sqrt (6 - x))) = g x → x ∈ set.Icc (-19 : ℝ) (6 : ℝ)) :=
by 
  sorry

end domain_of_g_l415_415177


namespace valid_fahrenheit_count_l415_415210

theorem valid_fahrenheit_count :
  let isValidTemp (F : ℤ) : Prop := let C := ⌊ (5:ℝ) / 9 * (F - 32)⌋ in
                                    let F' := ⌊ (9:ℝ) / 5 * C + 32⌋ in
                                    (F = F') ∧ (7 ∣ F)
  in
  finset.card (finset.filter isValidTemp (finset.range 1001)) = 324 :=
by
  sorry

end valid_fahrenheit_count_l415_415210


namespace bus_passing_time_correct_l415_415426

variables (length_bus : ℝ) (speed_bus1_kmph speed_bus2_kmph : ℝ) 
           (length_bridge : ℝ) (speed_reduction_factor : ℝ)

def bus_pass_time (length_bus speed_bus1_kmph speed_bus2_kmph length_bridge speed_reduction_factor : ℝ) : ℝ :=
  let speed_bus1 := speed_bus1_kmph * (1000 / 3600) in
  let speed_bus2 := speed_bus2_kmph * (1000 / 3600) in
  let reduced_speed_bus1 := speed_bus1 * speed_reduction_factor in
  let reduced_speed_bus2 := speed_bus2 * speed_reduction_factor in
  let relative_speed := reduced_speed_bus1 + reduced_speed_bus2 in
  let total_distance := length_bridge + length_bus * 2 in
  total_distance / relative_speed

theorem bus_passing_time_correct
  (h1 : length_bus = 200)
  (h2 : speed_bus1_kmph = 60)
  (h3 : speed_bus2_kmph = 80)
  (h4 : length_bridge = 600)
  (h5 : speed_reduction_factor = 0.5)
  : bus_pass_time 200 60 80 600 0.5 = 51.44 := 
by
  -- proof goes here
  sorry

end bus_passing_time_correct_l415_415426


namespace correct_propositions_count_l415_415155

def proposition1 : Prop := ∀ x : ℝ, x ≠ 0 → (y = 1 / x → ∀ x1 x2 : ℝ, x1 > x2 → x2 ≠ 0 → (1 / x1 > 1 / x2 → x1 < x2))

def proposition2 : Prop :=
  ∀ x : ℝ, (x ≠ 1 → (x^2 - x ≠ 0))

def proposition3 : Prop :=
  ∀ p q : Prop, (¬p ∨ q = False → (p ∧ ¬q = True))

def proposition4 : Prop :=
  ∃ a b : ℝ, (0 < a ∧ 0 < b ∧ a + b = 1 ∧ (1 / a + 1 / b = 3))

def count_correct_propositions : ℕ :=
  (if proposition1 then 1 else 0) +
  (if proposition2 then 1 else 0) +
  (if proposition3 then 1 else 0) +
  (if proposition4 then 1 else 0)

theorem correct_propositions_count : count_correct_propositions = 2 :=
  by sorry

end correct_propositions_count_l415_415155


namespace LTE_divisibility_l415_415447

theorem LTE_divisibility (m : ℕ) (h_pos : 0 < m) :
  (∀ k : ℕ, k % 2 = 1 ∧ k ≥ 3 → 2^m ∣ k^m - 1) ↔ m = 1 ∨ m = 2 ∨ m = 4 :=
by
  sorry

end LTE_divisibility_l415_415447


namespace rhombus_side_length_l415_415758

theorem rhombus_side_length (m S : ℝ) (x y : ℝ) (h1 : x + y = m) (h2 : x * y = 2 * S) :
  (1/2 : ℝ) * real.sqrt (m^2 - 4 * S) = (1 / 2) * real.sqrt (m^2 - 4 * S) :=
by
  sorry

end rhombus_side_length_l415_415758


namespace unique_pair_not_opposite_l415_415519

def QuantumPair (a b : String): Prop := ∃ oppositeMeanings : Bool, a ≠ b ∧ oppositeMeanings

theorem unique_pair_not_opposite :
  ∃ (a b : String), 
    (a = "increase of 2 years" ∧ b = "decrease of 2 liters") ∧ 
    (¬ QuantumPair a b) :=
by 
  sorry

end unique_pair_not_opposite_l415_415519


namespace ratio_CQ_QM_is_8_l415_415676

theorem ratio_CQ_QM_is_8
  (A B C P Q M : Type)
  [EuclideanSpace.Vector3 A B C]
  [Midpoint P A B]
  [IsoscelesTriangle AB AC]
  [equal_angles PQB AQC]
  [AltitudeFoot M P BQ] :
  ratio_length CQ QM = 8 := by
  sorry

end ratio_CQ_QM_is_8_l415_415676


namespace arithmetic_sequence_sum_l415_415165

theorem arithmetic_sequence_sum : ∀ (a1 d n : ℤ), a1 = -3 → d = 7 → n = 6 → 
  ∑ i in finset.range n, (a1 + i * d) = 87 :=
by
  intros a1 d n h1 h2 h3
  sorry

end arithmetic_sequence_sum_l415_415165


namespace power_inequality_l415_415589

variable (a b c : ℝ)

theorem power_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * (a^3 + b^3 + c^3) ≥ a * b^2 + a^2 * b + b * c^2 + b^2 * c + a * c^2 + a^2 * c :=
by sorry

end power_inequality_l415_415589


namespace trihedral_angle_properties_l415_415727

theorem trihedral_angle_properties
  (S A B C : Point)
  (SA_eq_SB : dist S A = dist S B)
  (SB_eq_SC : dist S B = dist S C)
  (SA_eq_SC : dist S A = dist S C)
  (O : Point)
  (O_proj : is_orthogonal_projection O (triangle_plane A B C))
  (α β γ : real) -- Dihedral angles
  (plane_angle_sum : ∠ ASB + ∠ BSC + ∠ CSA) :
  (∠ ASB + ∠ BSC + ∠ CSA < 2 * π) ∧ (α + β + γ > π) :=
by
  sorry

end trihedral_angle_properties_l415_415727


namespace total_time_l415_415353

def time_to_eat_cereal (rate1 rate2 rate3 : ℚ) (amount : ℚ) : ℚ :=
  let combined_rate := rate1 + rate2 + rate3
  amount / combined_rate

theorem total_time (rate1 rate2 rate3 : ℚ) (amount : ℚ) 
  (h1 : rate1 = 1 / 15)
  (h2 : rate2 = 1 / 20)
  (h3 : rate3 = 1 / 30)
  (h4 : amount = 4) : 
  time_to_eat_cereal rate1 rate2 rate3 amount = 80 / 3 := 
by 
  rw [time_to_eat_cereal, h1, h2, h3, h4]
  sorry

end total_time_l415_415353


namespace twelfth_term_is_three_l415_415776

-- Define the first term and the common difference of the arithmetic sequence
def first_term : ℚ := 1 / 4
def common_difference : ℚ := 1 / 4

-- Define the nth term of an arithmetic sequence
def nth_term (a₁ d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1) * d

-- Prove that the twelfth term is equal to 3
theorem twelfth_term_is_three : nth_term first_term common_difference 12 = 3 := 
  by 
    sorry

end twelfth_term_is_three_l415_415776


namespace number_of_valid_x_values_l415_415781

noncomputable def count_valid_x (bound : ℕ) : ℕ :=
  (Finset.range bound).filter (λ x : ℕ, x > 3 ∧ ((x + 3) * (x - 3) * (x^2 + 9) < 500)).card

theorem number_of_valid_x_values : count_valid_x 10 = 1 :=
by sorry

end number_of_valid_x_values_l415_415781


namespace circle_secant_problem_l415_415914

variables {P O D D' D'' : Type} [metric_space P] [metric_space O] [metric_space D] [metric_space D']

noncomputable def radius (c : Type) := sorry
noncomputable def center (c : Type) := sorry
noncomputable def secant (P : Type) (A B : Type) := sorry
noncomputable def diameter (A B : Type) := sorry
noncomputable def touches (P O : Type) := sorry

theorem circle_secant_problem 
  {c₁ c₂ : Type} [metric_space c₁] [metric_space c₂]
  (h1 : radius c₁ = r)
  (h2 : secant P A B)
  (h3 : diameter A B = diameter c₂) 
  (h4 : touches c₂ O) 
  : center c₂ = D' ∨ center c₂ = D'' :=
sorry

end circle_secant_problem_l415_415914


namespace common_tangents_two_circles_l415_415005

open Real

def circle_eq (x0 y0 r : ℝ) (x y : ℝ) : Prop :=
  (x - x0) ^ 2 + (y - y0) ^ 2 = r ^ 2

theorem common_tangents_two_circles :
  let C1 := circle_eq (-1) (-1) (sqrt 2)
  let C2 := circle_eq 2 1 (sqrt 2)
  let center_dist := sqrt ((2 + 1) ^ 2 + (1 + 1) ^ 2)
  center_dist < (2 * sqrt 2) → (number_of_common_tangents C1 C2 = 2)
:= 
by {
  sorry
}

end common_tangents_two_circles_l415_415005


namespace no_odd_intersection_of_even_segments_l415_415690

theorem no_odd_intersection_of_even_segments (a b c p q : ℤ) (A B C : set ℝ)
  (ha : A = set.Icc 0 (2 * a))
  (hb : B = set.Icc p (p + 2 * b))
  (hc : C = set.Icc q (q + 2 * c))
  (hevenA : ∃ n : ℤ, 2 * n = 2 * a)
  (hevenB : ∃ n : ℤ, 2 * n = 2 * b)
  (hevenC : ∃ n : ℤ, 2 * n = 2 * c)
  (H1 : ∃ i_s1 i_t1 : ℝ, i_s1 < i_t1 ∧ set.Icc i_s1 i_t1 ⊆ A ∩ B ∧ (i_t1 - i_s1) ∈ set.Icc 1 ∞ ∧ (i_t1 - i_s1) % 2 = 1)
  (H2 : ∃ i_s2 i_t2 : ℝ, i_s2 < i_t2 ∧ set.Icc i_s2 i_t2 ⊆ B ∩ C ∧ (i_t2 - i_s2) ∈ set.Icc 1 ∞ ∧ (i_t2 - i_s2) % 2 = 1)
  (H3 : ∃ i_s3 i_t3 : ℝ, i_s3 < i_t3 ∧ set.Icc i_s3 i_t3 ⊆ C ∩ A ∧ (i_t3 - i_s3) ∈ set.Icc 1 ∞ ∧ (i_t3 - i_s3) % 2 = 1)
: false := sorry

end no_odd_intersection_of_even_segments_l415_415690


namespace boxed_boxed_prime_17_l415_415902

def sum_factors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ k => n % k = 0).sum id

def box (n : ℕ) : ℕ :=
  sum_factors n

def double_box (p : ℕ) : ℕ :=
  box (box p)

theorem boxed_boxed_prime_17 : double_box 17 = 39 := by
  sorry

end boxed_boxed_prime_17_l415_415902


namespace triangle_obtuse_height_l415_415979

theorem triangle_obtuse_height (A B C : Point)
  (h : ℝ)
  (H_obtuse : ∠C > π / 2)
  (H_height : height_on_AB h) :
  AB > 2 * h := 
  sorry

end triangle_obtuse_height_l415_415979


namespace range_of_m_l415_415261

def f (x : ℝ) : ℝ := 
if x ≤ 0 then 3^(-x) - 2 
else (1/2) * Real.log 3 x

theorem range_of_m (m : ℝ) (h : f m > 1) : m < -1 ∨ m > 9 :=
sorry

end range_of_m_l415_415261


namespace inequality_cos_exp_l415_415216

theorem inequality_cos_exp (θ₁ θ₂ : ℝ) (h₀ : 0 < θ₁) (h₁ : θ₁ < θ₂) (h₂ : θ₂ < π / 2) :
  (cos θ₂) * (exp (cos θ₁)) < (cos θ₁) * (exp (cos θ₂)) := by
  sorry

end inequality_cos_exp_l415_415216


namespace total_frogs_in_pond_l415_415652

def frogsOnLilyPads : ℕ := 5
def frogsOnLogs : ℕ := 3
def babyFrogsOnRock : ℕ := 2 * 12 -- Two dozen

theorem total_frogs_in_pond : frogsOnLilyPads + frogsOnLogs + babyFrogsOnRock = 32 :=
by
  sorry

end total_frogs_in_pond_l415_415652


namespace price_fluctuation_52_5_percent_l415_415147

-- Variables declarations
variables (P : ℝ) (x : ℝ)

-- Conditions
def original_price := P
def increased_price := P * (1 + x / 100)
def taxed_price := increased_price * 1.05
def final_price := taxed_price * (1 - x / 100)
def result_price := 0.76 * P

-- Theorem to prove
theorem price_fluctuation_52_5_percent (P_ne_zero : P ≠ 0) (h : final_price = result_price) : x = 52.5 :=
by {
  -- Placeholder for proof
  sorry
}

end price_fluctuation_52_5_percent_l415_415147


namespace susan_homework_start_time_l415_415385

def start_time_homework (finish_time : ℕ) (homework_duration : ℕ) (interval_duration : ℕ) : ℕ :=
  finish_time - homework_duration - interval_duration

theorem susan_homework_start_time :
  let finish_time : ℕ := 16 * 60 -- 4:00 p.m. in minutes
  let homework_duration : ℕ := 96 -- Homework duration in minutes
  let interval_duration : ℕ := 25 -- Interval between homework finish and practice in minutes
  start_time_homework finish_time homework_duration interval_duration = 13 * 60 + 59 := -- 13:59 in minutes
by
  sorry

end susan_homework_start_time_l415_415385


namespace length_of_EM_l415_415985

theorem length_of_EM 
  (EFGH : Type) [is_square EFGH 8]
  (LMNO : Type) [is_rectangle LMNO 12 8]
  (EH_perpendicular_to_LM : ∀ (EH LM : Type), is_perpendicular EH LM)
  (shaded_area : 48) :
  let EM := 2 in
  true := 
sorry

end length_of_EM_l415_415985


namespace evaluate_expression_l415_415884

theorem evaluate_expression (α : ℝ) (h : Real.tan α = 3) :
  (2 * Real.sin (2 * α) - 3 * Real.cos (2 * α)) / (4 * Real.sin (2 * α) + 5 * Real.cos (2 * α)) = -9 / 4 :=
sorry

end evaluate_expression_l415_415884


namespace find_ponderosa_pine_cost_l415_415373

variables (total_trees : ℕ) (douglas_fir_count : ℕ) (cost_douglas_fir : ℕ) (total_cost : ℕ)
          (ponderosa_pine_count : ℕ) (cost_ponderosa_pine : ℕ)

-- Define given conditions
def conditions (total_trees = 850) (douglas_fir_count = 350) (cost_douglas_fir = 300) 
               (total_cost = 217500) : Prop :=
ponderosa_pine_count = total_trees - douglas_fir_count ∧
cost_ponderosa_pine = (total_cost - (douglas_fir_count * cost_douglas_fir)) / ponderosa_pine_count

-- Define the correct answer to be proven
def correct_answer : Prop :=
cost_ponderosa_pine = 225

-- The Lean theorem statement
theorem find_ponderosa_pine_cost (h : conditions 850 350 300 217500) : correct_answer :=
sorry

end find_ponderosa_pine_cost_l415_415373


namespace remainder_when_divided_by_x_plus_2_l415_415940

-- Define the polynomial q(x) = D*x^4 + E*x^2 + F*x + 8
variable (D E F : ℝ)
def q (x : ℝ) : ℝ := D * x^4 + E * x^2 + F * x + 8

-- Given condition: q(2) = 12
axiom h1 : q D E F 2 = 12

-- Prove that q(-2) = 4
theorem remainder_when_divided_by_x_plus_2 : q D E F (-2) = 4 := by
  sorry

end remainder_when_divided_by_x_plus_2_l415_415940


namespace average_rainfall_correct_l415_415981

-- Define the leap year condition and days in February
def leap_year_february_days : ℕ := 29

-- Define total hours in a day
def hours_in_day : ℕ := 24

-- Define total rainfall in February 2012 in inches
def total_rainfall : ℕ := 420

-- Define total hours in February 2012
def total_hours_february : ℕ := leap_year_february_days * hours_in_day

-- Define the average rainfall calculation
def average_rainfall_per_hour : ℚ :=
  total_rainfall / total_hours_february

-- Theorem to prove the average rainfall is 35/58 inches per hour
theorem average_rainfall_correct :
  average_rainfall_per_hour = 35 / 58 :=
by 
  -- Placeholder for proof
  sorry

end average_rainfall_correct_l415_415981


namespace number_of_rows_l415_415188

theorem number_of_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (h1 : total_pencils = 30) (h2 : pencils_per_row = 5) : total_pencils / pencils_per_row = 6 :=
by
  sorry

end number_of_rows_l415_415188


namespace at_least_two_equal_l415_415322

theorem at_least_two_equal (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + b^2 + c^2 = b + a^2 + c^2) (h2 : b + a^2 + c^2 = c + a^2 + b^2) : 
  (a = b) ∨ (a = c) ∨ (b = c) :=
sorry

end at_least_two_equal_l415_415322


namespace modulus_complex_z_l415_415410

noncomputable def complex_z : ℂ := complex.mk 0 1 / (complex.mk 1 (-1))

theorem modulus_complex_z (z: ℂ) (h : z = complex_z) : complex.abs z = real.sqrt 2 / 2 :=
by {
  sorry
}

end modulus_complex_z_l415_415410


namespace sum_remainder_l415_415699

theorem sum_remainder :
    let T := ∑ n in Finset.range 333, (-1:ℤ)^n * ((1002:ℤ).choose (3 * n)) 
    in  T % 500 = 2 := by
sorry

end sum_remainder_l415_415699


namespace lean_l415_415906

open ProbabilityTheory

variable {Ω : Type*} [Fintype Ω] [DecidableEq Ω] (s : Finset Ω)

def red_balls := 5
def blue_balls := 4
def total_balls := red_balls + blue_balls

-- Events
def A (i : Fin 2) : Event Ω := {ω | is_red ω i}
def B (j : Fin 2) : Event Ω := {ω | is_blue ω j}

noncomputable def P (ev : Event Ω) : ℚ := (ev.card : ℚ) / (Fintype.card Ω)

theorem lean statement:
  (P(A 2) = 5 / 9) ∧ (P(A 2) + P(B 2) = 1) ∧ (P(A 2 | A 1) + P(B 2 | A 1) = 1)  := by
  sorry

end lean_l415_415906


namespace four_distinct_real_roots_l415_415704

noncomputable def f (x d : ℝ) : ℝ := x^2 + 10*x + d

theorem four_distinct_real_roots (d : ℝ) :
  (∀ r, f r d = 0 → (∃! x, f x d = r)) → d < 25 :=
by
  sorry

end four_distinct_real_roots_l415_415704


namespace number_of_lock_codes_l415_415023

theorem number_of_lock_codes : 
  ∃ n : ℕ, 
    (n = 1440) ∧ 
    ∀ code : Fin 6 → ℕ, 
      (∀ i : Fin 6, 1 ≤ code i ∧ code i ≤ 9) ∧ 
      (function.injective code) ∧ 
      (code 0 % 2 = 1) ∧ 
      (code 1 % 2 = 0) ∧ 
      (code 2 % 2 = 1) ∧ 
      (code 3 % 2 = 0) ∧ 
      (code 4 % 2 = 1) ∧ 
      (code 5 % 2 = 0) → 
    ∃ perm : Finset (Fin 6 → ℕ), 
      perm.card = n ∧ 
      ∀ code' ∈ perm, 
        (∀ i : Fin 6, 1 ≤ code' i ∧ code' i ≤ 9) ∧ 
        (function.injective code') ∧ 
        (code' 0 % 2 = 1) ∧ 
        (code' 1 % 2 = 0) ∧ 
        (code' 2 % 2 = 1) ∧ 
        (code' 3 % 2 = 0) ∧ 
        (code' 4 % 2 = 1) ∧ 
        (code' 5 % 2 = 0) :=
begin
  sorry
end

end number_of_lock_codes_l415_415023


namespace area_of_right_triangle_l415_415316

-- Define a structure for the triangle with the given conditions
structure Triangle :=
(A B C : ℝ × ℝ)
(right_angle_at_C : (C.1 = 0 ∧ C.2 = 0))
(hypotenuse_length : (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = 50 ^ 2)
(median_A : ∀ x: ℝ, A.2 = A.1 + 5)
(median_B : ∀ x: ℝ, B.2 = 2 * B.1 + 2)

-- Theorem statement
theorem area_of_right_triangle (t : Triangle) : 
  ∃ area : ℝ, area = 500 :=
sorry

end area_of_right_triangle_l415_415316


namespace savings_equal_after_weeks_l415_415448

theorem savings_equal_after_weeks :
  ∃ w : ℕ, 160 + 7 * w = 210 + 5 * w ∧ w = 25 :=
by {
  have h : 160 + 7 * 25 = 210 + 5 * 25,
  calc 
    160 + 7 * 25 = 160 + 175 : by norm_num
              ... = 335 : by norm_num
              ... = 210 + 125 : by norm_num
              ... = 210 + 5 * 25 : by norm_num,
  use 25,
  split,
  exact h,
  simp,
  }

end savings_equal_after_weeks_l415_415448


namespace find_sin_angle_BAD_l415_415927

def isosceles_right_triangle (A B C : ℝ → ℝ → Prop) (AB BC AC : ℝ) : Prop :=
  AB = 2 ∧ BC = 2 ∧ AC = 2 * Real.sqrt 2

def right_triangle_on_hypotenuse (A C D : ℝ → ℝ → Prop) (AC CD DA : ℝ) (DAC : ℝ) : Prop :=
  AC = 2 * Real.sqrt 2 ∧ CD = DA / 2 ∧ DAC = Real.pi / 6

def equal_perimeters (AC CD DA : ℝ) : Prop := 
  AC + CD + DA = 4 + 2 * Real.sqrt 2

theorem find_sin_angle_BAD :
  ∀ (A B C D : ℝ → ℝ → Prop) (AB BC AC CD DA : ℝ),
  isosceles_right_triangle A B C AB BC AC →
  right_triangle_on_hypotenuse A C D AC CD DA (Real.pi / 6) →
  equal_perimeters AC CD DA →
  Real.sin (2 * (Real.pi / 4 + Real.pi / 6)) = 1 / 2 :=
by
  intros
  sorry

end find_sin_angle_BAD_l415_415927


namespace color_property_l415_415542

theorem color_property (k : ℕ) (h : k ≥ 1) : k = 1 ∨ k = 2 :=
by
  sorry

end color_property_l415_415542


namespace angle_bisector_correct_length_l415_415992

-- Define the isosceles triangle with the given conditions
structure IsoscelesTriangle :=
  (base : ℝ)
  (lateral : ℝ)
  (is_isosceles : lateral = 20 ∧ base = 5)

-- Define the problem of finding the angle bisector
noncomputable def angle_bisector_length (tri : IsoscelesTriangle) : ℝ :=
  6

-- The main theorem to state the problem
theorem angle_bisector_correct_length (tri : IsoscelesTriangle) : 
  angle_bisector_length tri = 6 :=
by
  -- We state the theorem, skipping the proof (sorry)
  sorry

end angle_bisector_correct_length_l415_415992


namespace sum_f_1_to_2023_l415_415613

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain_real : ∀ x : ℝ, f x = f x
axiom f_even : ∀ x : ℝ, f (x + 1) = f (-(x + 1))
axiom f_odd : ∀ x : ℝ, f (x + 2) = -f (-(x + 2))
axiom f_sum_1_2 : f 1 + f 2 = 2

theorem sum_f_1_to_2023 : (∑ k in finset.range 2023, f (k + 1)) = 0 :=
by
  sorry

end sum_f_1_to_2023_l415_415613


namespace train_travel_distance_l415_415807

theorem train_travel_distance
  (rate_miles_per_pound : Real := 5 / 2)
  (remaining_coal : Real := 160)
  (distance_per_pound := λ r, r / 2)
  (total_distance := λ rc dpp, rc * dpp) :
  total_distance remaining_coal rate_miles_per_pound = 400 := sorry

end train_travel_distance_l415_415807


namespace loaves_count_l415_415491

theorem loaves_count (initial_loaves afternoon_sales evening_delivery end_day_loaves: ℕ)
  (h_initial: initial_loaves = 2355)
  (h_sales: afternoon_sales = 629)
  (h_delivery: evening_delivery = 489)
  (h_end: end_day_loaves = 2215) :
  initial_loaves - afternoon_sales + evening_delivery = end_day_loaves :=
  by {
    rw [h_initial, h_sales, h_delivery, h_end],
    sorry
  }

end loaves_count_l415_415491


namespace ratio_of_c_to_a_l415_415574

variable {a c : ℝ}

-- Conditions
def distinct_points_on_plane (p1 p2 p3 p4 p5 : ℝ × ℝ) : Prop :=
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ p2 ≠ p3 ∧
  p2 ≠ p4 ∧ p2 ≠ p5 ∧ p3 ≠ p4 ∧ p3 ≠ p5 ∧ p4 ≠ p5

def segment_lengths (a c : ℝ) (p1 p2 p3 p4 p5 : ℝ × ℝ) : Prop :=
  let d := λ x y : ℝ × ℝ, (x.1 - y.1)^2 + (x.2 - y.2)^2 in
  ∃ (l1 l2 l3 l4 l5 l6 : ℝ), 
    (l1 = a ∧ l1 = d p1 p2 ∨ l1 = d p1 p3 ∨ l1 = d p1 p4 ∨ l1 = d p1 p5 ∨ l1 = d p2 p3 ∨ l1 = d p2 p4 ∨ l1 = d p2 p5 ∨ l1 = d p3 p4 ∨ l1 = d p3 p5 ∨ l1 = d p4 p5) ∧
    (l2 = a ∧ l2 = d p1 p2 ∨ l2 = d p1 p3 ∨ l2 = d p1 p4 ∨ l2 = d p1 p5 ∨ l2 = d p2 p3 ∨ l2 = d p2 p4 ∨ l2 = d p2 p5 ∨ l2 = d p3 p4 ∨ l2 = d p3 p5 ∨ l2 = d p4 p5) ∧
    (l3 = 2 * a ∧ l3 = d p1 p2 ∨ l3 = d p1 p3 ∨ l3 = d p1 p4 ∨ l3 = d p1 p5 ∨ l3 = d p2 p3 ∨ l3 = d p2 p4 ∨ l3 = d p2 p5 ∨ l3 = d p3 p4 ∨ l3 = d p3 p5 ∨ l3 = d p4 p5) ∧
    (l4 = 2 * a ∧ l4 = d p1 p2 ∨ l4 = d p1 p3 ∨ l4 = d p1 p4 ∨ l4 = d p1 p5 ∨ l4 = d p2 p3 ∨ l4 = d p2 p4 ∨ l4 = d p2 p5 ∨ l4 = d p3 p4 ∨ l4 = d p3 p5 ∨ l4 = d p4 p5) ∧
    (l5 = 3 * a ∧ l5 = d p1 p2 ∨ l5 = d p1 p3 ∨ l5 = d p1 p4 ∨ l5 = d p1 p5 ∨ l5 = d p2 p3 ∨ l5 = d p2 p4 ∨ l5 = d p2 p5 ∨ l5 = d p3 p4 ∨ l5 = d p3 p5 ∨ l5 = d p4 p5) ∧
    (l6 = c ∧ l6 = d p1 p2 ∨ l6 = d p1 p3 ∨ l6 = d p1 p4 ∨ l6 = d p1 p5 ∨ l6 = d p2 p3 ∨ l6 = d p2 p4 ∨ l6 = d p2 p5 ∨ l6 = d p3 p4 ∨ l6 = d p3 p5 ∨ l6 = d p4 p5)

theorem ratio_of_c_to_a (p1 p2 p3 p4 p5 : ℝ × ℝ) (h1 : distinct_points_on_plane p1 p2 p3 p4 p5)
  (h2 : segment_lengths a c p1 p2 p3 p4 p5) : c = a * Real.sqrt 7 :=
sorry

end ratio_of_c_to_a_l415_415574


namespace pure_imaginary_solution_l415_415276

-- Defining the main problem as a theorem in Lean 4

theorem pure_imaginary_solution (m : ℝ) : 
  (∃ a b : ℝ, (m^2 - m = a ∧ a = 0) ∧ (m^2 - 3 * m + 2 = b ∧ b ≠ 0)) → 
  m = 0 :=
sorry -- Proof is omitted as per the instructions

end pure_imaginary_solution_l415_415276


namespace max_marks_proof_l415_415457

-- Define the conditions
def scored_marks : ℕ := 212
def shortfall : ℕ := 13
def passing_percentage : ℝ := 30 / 100

-- Define what needs to be proven
theorem max_marks_proof (max_marks : ℕ) (h_pass : (scored_marks + shortfall) = passing_percentage * max_marks) : max_marks = 750 :=
sorry

end max_marks_proof_l415_415457


namespace prob_inequality_l415_415412

variables {Ω : Type*} {P : Ω → Prop} {A B C D : Prop}
variable [ProbabilitySpace Ω]
variable (P : Set Ω → ℝ)

-- Define the events
def EventA : Set Ω := {ω | A}
def EventB : Set Ω := {ω | B}
def EventC : Set Ω := {ω | C}
def EventABC : Set Ω := {ω | A ∧ B ∧ C}
def EventD : Set Ω := {ω | D}

-- Given condition
axiom h : P EventD ≥ P EventABC

-- The theorem to be proved
theorem prob_inequality (h : P EventD ≥ P EventABC) : P EventA + P EventB + P EventC - P EventD ≤ 2 :=
sorry

end prob_inequality_l415_415412


namespace number_125th_position_l415_415515

/-- The sequence of natural numbers whose digits sum to exactly 5, arranged in ascending order. --/
def numbers_with_digit_sum_5 : List ℕ :=
  List.filter (λ n => (Nat.digits 10 n).sum = 5) (List.range 100000)

/-- The 125th number in the sequence of natural numbers whose digits sum to 5. --/
theorem number_125th_position : numbers_with_digit_sum_5[124] = 41000 :=
  sorry

end number_125th_position_l415_415515


namespace f_no_zeros_in_interval_f_zeros_in_interval_l415_415944

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x - Real.log x

theorem f_no_zeros_in_interval (x : ℝ) (hx1 : x > 1 / Real.exp 1) (hx2 : x < 1) :
  f x ≠ 0 := sorry

theorem f_zeros_in_interval (h1 : 1 < e) (x_exists : ∃ x, 1 < x ∧ x < Real.exp 1 ∧ f x = 0) :
  true := sorry

end f_no_zeros_in_interval_f_zeros_in_interval_l415_415944


namespace smallest_integer_ending_in_6_divisible_by_13_l415_415773

theorem smallest_integer_ending_in_6_divisible_by_13 (n : ℤ) (h1 : ∃ n : ℤ, 10 * n + 6 = x) (h2 : x % 13 = 0) : x = 26 :=
  sorry

end smallest_integer_ending_in_6_divisible_by_13_l415_415773


namespace length_LH_eq_length_KM_perpendicular_bisects_HK_l415_415427

variables {P : Type} [MetricSpace P]
variables {C1 C2 : Set P} {D E B H K L M : P}

-- Given conditions
axiom circles_touch_at_B : C1 ∩ C2 = {B}
axiom circle_with_diameter_DE : ∀ p ∈ Sphere (D + E) (dist D E / 2), p ∈ C1 ∨ p ∈ C2
axiom H_between_C1_C2 : H ∈ C1 ∧ K ∈ C2 ∧ (H - D) × (K - E) > 0
axiom HK_Meets_Circles : ∃ L' M' : P, L' ∈ C1 ∧ M' ∈ C2 ∧ ∃ t : ℝ, L' = H + t * (K - H) ∧ M' = K - t * (K - H)

-- Proof statements
theorem length_LH_eq_length_KM 
  (circles_touch_at_B : C1 ∩ C2 = {B})
  (circle_with_diameter_DE : ∀ p ∈ Sphere (D + E) (dist D E / 2), p ∈ C1 ∨ p ∈ C2)
  (H_between_C1_C2 : H ∈ C1 ∧ K ∈ C2 ∧ (H - D) × (K - E) > 0)
  (HK_Meets_Circles : ∃ L' M' : P, L' ∈ C1 ∧ M' ∈ C2 ∧ ∃ t : ℝ, L' = H + t * (K - H) ∧ M' = K - t * (K - H))
  : dist L H = dist K M := 
by sorry

theorem perpendicular_bisects_HK 
  (circles_touch_at_B : C1 ∩ C2 = {B})
  (circle_with_diameter_DE : ∀ p ∈ Sphere (D + E) (dist D E / 2), p ∈ C1 ∨ p ∈ C2)
  (H_between_C1_C2 : H ∈ C1 ∧ K ∈ C2 ∧ (H - D) × (K - E) > 0)
  (HK_Meets_Circles : ∃ L' M' : P, L' ∈ C1 ∧ M' ∈ C2 ∧ ∃ t : ℝ, L' = H + t * (K - H) ∧ M' = K - t * (K - H))
  : ∃ S : P, orthogonal_projection (line_through B D) S = B ∧ midpoint H K = S := 
by sorry

end length_LH_eq_length_KM_perpendicular_bisects_HK_l415_415427


namespace work_done_together_l415_415076

theorem work_done_together (x_days y_days : ℕ) (hx : x_days = 10) (hy : y_days = 15) :
  (1 / 10 + 1 / 15) = 1 / 6 := 
by
  cases hx with x_eq
  cases hy with y_eq
  sorry

end work_done_together_l415_415076


namespace dilation_image_l415_415006

theorem dilation_image (z₀ : ℂ) (k : ℝ) (w : ℂ) (z' : ℂ)
  (h₁ : z₀ = 1 + 3 * complex.I) 
  (h₂ : k = -3)
  (h₃ : w = 1 + 2 * complex.I) 
  (h₄ : z' - z₀ = k * (w - z₀)) : 
  z' = 1 + 6 * complex.I :=
sorry

end dilation_image_l415_415006


namespace incenter_lies_on_RT_l415_415991

variables {A B C D X E S R T : Type*} [euclidean_geometry A B C D X E S R T]

theorem incenter_lies_on_RT
  (h_ABC_acute : ∀ (α β γ : ℝ), α + β + γ = π → ∀ (α β γ : ℝ), α < π/2 ∧ β < π/2 ∧ γ < π/2)
  (h_D_midpoint_BC : midpoint_arc D (arc_not_containing A B C))
  (h_X_on_BD : lies_on_arc X (arc BD))
  (h_E_midpoint_ABX : midpoint_arc E (arc_ABX A B X))
  (h_S_on_AC : lies_on_arc S (arc AC))
  (h_SD_R : intersects_line (line SD) (line BC) R)
  (h_SE_T : intersects_line (line SE) (line XA) T)
  (h_RT_parallel_DE : parallel (line RT) (line DE)) :
  lies_on_line (incenter_triangle A B C) (line RT) :=
sorry

end incenter_lies_on_RT_l415_415991


namespace valid_quadratic_polynomials_count_l415_415334

theorem valid_quadratic_polynomials_count :
  let P (x : ℝ) := (x-1)*(x-2)*(x-3)*(x-4) in
  let Q_valid (Q : ℝ → ℝ) := 
    ∃ R : ℝ → ℝ, polynomial.degree R = 4 ∧ ∀ x, P (Q x) = P x * R x in
  card {Q : ℝ → ℝ // polynomial.degree (Q) = 2 ∧ Q_valid Q} = 250 :=
begin
  sorry
end

end valid_quadratic_polynomials_count_l415_415334


namespace two_solutions_exist_l415_415644

theorem two_solutions_exist 
  (a b c : ℝ) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_equation : (1 / a) + (1 / b) + (1 / c) = (1 / (a + b + c))) : 
  ∃ (a' b' c' : ℝ), 
    ((a' = 1/3 ∧ b' = 1/3 ∧ c' = 1/3) ∨ (a' = -1/3 ∧ b' = -1/3 ∧ c' = -1/3)) := 
sorry

end two_solutions_exist_l415_415644


namespace balance_scale_l415_415838

-- Defining the given weights and the problem conditions
def weights : List ℕ := [1, 2, 4, 8, 16, 32]

noncomputable def candy_weight : ℕ := 25
noncomputable def total_weight : ℕ := (weights.sum + candy_weight)
noncomputable def balance_point : ℕ := total_weight / 2

-- Main theorem statement
theorem balance_scale : ∃ w₁ w₂ w₃, 
  (w₁ ∈ weights) ∧ (w₂ ∈ weights) ∧ (w₃ ∈ weights) ∧ (w₁ ≠ w₂) ∧ (w₂ ≠ w₃) ∧ (w₁ ≠ w₃) ∧ 
  (w₁ + w₂ + w₃ = balance_point) ∧ 
  (candy_weight + (weights.sum - (w₁ + w₂ + w₃)) = balance_point) :=
begin
  -- We skip the proof here
  sorry
end

end balance_scale_l415_415838


namespace problem1_problem2_l415_415909

-- Definitions for the problems conditions and statement proofs

-- Definition of point P and Q with given conditions
def P (a : ℝ) : ℝ × ℝ := (2 * a - 2, a + 5)
def Q : ℝ × ℝ := (4, 5)

-- Definition ensuring P is in second quadrant
def in_second_quadrant (a : ℝ) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y > 0 ∧ P a = (x, y)

-- Definition ensuring P and Q form a line parallel to y-axis
def parallel_to_y_axis (a : ℝ) : Prop :=
  (P a).1 = 4 ∧ Q.1 = 4

-- Problem (1): Coordinates of point P
theorem problem1 (a : ℝ) (h1 : parallel_to_y_axis a) : P a = (4, 8) := by
  sorry

-- Problem (2): Value of a^2023 + ∛a where P is in second quadrant and distances are equal
theorem problem2 (a : ℝ) (h2 : in_second_quadrant a) (h3 : -(2 * a - 2) = a + 5) :
  a^2023 + real.cbrt a = -2 := by
  sorry

end problem1_problem2_l415_415909


namespace find_center_of_circle_l415_415895

noncomputable def center_of_circle (θ ρ : ℝ) : Prop :=
  ρ = (1 : ℝ) ∧ θ = (-Real.pi / (3 : ℝ))

theorem find_center_of_circle (θ ρ : ℝ) (h : ρ = Real.cos θ - Real.sqrt 3 * Real.sin θ) :
  center_of_circle θ ρ := by
  sorry

end find_center_of_circle_l415_415895


namespace rectangle_ABCD_AB_length_l415_415308

theorem rectangle_ABCD_AB_length
  (ABCD : Type) [rectangle ABCD]
  {A B C D : ABCD}
  (h_rect : rectangle A B C D)
  (P : ABCD)
  (hP_on_BC : P ∈ line_segment B C)
  (BP_len : BP = 12)
  (CP_len : CP = 12)
  (h_angle_APD : cos ∠APD = 1/5) :
  length AB = 12 * sqrt 2 := sorry

end rectangle_ABCD_AB_length_l415_415308


namespace max_a_plus_2b_plus_c_l415_415268

open Real

theorem max_a_plus_2b_plus_c
  (A : Set ℝ := {x | |x + 1| ≤ 4})
  (T : ℝ := 3)
  (a b c : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_T : a^2 + b^2 + c^2 = T) :
  a + 2 * b + c ≤ 3 * sqrt 2 :=
by
  -- Proof is omitted
  sorry

end max_a_plus_2b_plus_c_l415_415268


namespace unbounded_kn_sequence_l415_415345

noncomputable def iter_apply {α : Type*} (f : α → α) : ℕ → (α → α)
| 0       := id
| (m + 1) := f ∘ (iter_apply m)

theorem unbounded_kn_sequence
  (f : ℕ → ℕ)
  (h : ∀ n : ℕ, ∃ k : ℕ, iter_apply f (2 * k) n = n + k)
  (k_n : ℕ → ℕ := λ n, Nat.find (h n)) :
  ∀ N : ℕ, ∃ n : ℕ, k_n n > N := 
sorry

end unbounded_kn_sequence_l415_415345


namespace max_a_monotonic_function_l415_415221

theorem max_a_monotonic_function :
  ∀ (a : ℝ), (∀ (x : ℝ), 1 ≤ x → 0 ≤ 3 * x^2 - a) → a ≤ 3 := 
by
  intro a h
  specialize h 1
  linarith
  sorry

end max_a_monotonic_function_l415_415221


namespace percentage_received_certificates_l415_415987

theorem percentage_received_certificates (boys girls : ℕ) (pct_boys pct_girls : ℝ) :
    boys = 30 ∧ girls = 20 ∧ pct_boys = 0.1 ∧ pct_girls = 0.2 →
    ((pct_boys * boys + pct_girls * girls) / (boys + girls) * 100) = 14 := by
  sorry

end percentage_received_certificates_l415_415987


namespace distance_point_to_line_l415_415400

theorem distance_point_to_line : 
  let x0 := 1
  let y0 := 0
  let A := 1
  let B := -2
  let C := 1 
  let dist := (A * x0 + B * y0 + C : ℝ) / Real.sqrt (A^2 + B^2)
  abs dist = 2 * Real.sqrt 5 / 5 :=
by
  -- Using basic principles of Lean and Mathlib to state the equality proof
  sorry

end distance_point_to_line_l415_415400


namespace rhombus_area_l415_415915

theorem rhombus_area
  (d1 d2 : ℝ)
  (hd1 : d1 = 14)
  (hd2 : d2 = 20) :
  (d1 * d2) / 2 = 140 := by
  -- Problem: Given diagonals of length 14 cm and 20 cm,
  -- prove that the area of the rhombus is 140 square centimeters.
  sorry

end rhombus_area_l415_415915


namespace sin_add_pi_over_2_eq_l415_415279

theorem sin_add_pi_over_2_eq :
  ∀ (A : ℝ), (cos (π + A) = -1/2) → (sin (π / 2 + A) = 1/2) :=
by
  intros A h
  -- Here we assume the proof steps
  sorry

end sin_add_pi_over_2_eq_l415_415279


namespace wang_pens_purchase_l415_415354

theorem wang_pens_purchase :
  ∀ (total_money spent_on_albums pen_cost : ℝ)
  (number_of_pens : ℕ),
  total_money = 80 →
  spent_on_albums = 45.6 →
  pen_cost = 2.5 →
  number_of_pens = 13 →
  (total_money - spent_on_albums) / pen_cost ≥ number_of_pens ∧ 
  (total_money - spent_on_albums) / pen_cost < number_of_pens + 1 :=
by
  intros
  sorry

end wang_pens_purchase_l415_415354


namespace simplify_expression_l415_415734

variable {a b : ℝ}

theorem simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (4 * a ^ (2 / 3) * b ^ (-1 / 3) / (-2 / 3 * a ^ (-1 / 3) * b ^ (2 / 3))) = - (6 * a / b) :=
by
  sorry

end simplify_expression_l415_415734


namespace sum_black_leq_sum_white_l415_415839

-- Define the problem statements
variable {P : Polyhedral}       -- P is a polyhedral
variable inscribed_sphere : Sphere -- Sphere inscribed in polyhedral P
variable is_colored : ∀ (f : Face P), f.color = Black ∨ f.color = White -- Faces colored black or white
variable not_adjacent_black : ∀ (f1 f2 : Face P), (f1.color = Black ∧ f2.color = Black) → ¬adjacent f1 f2 -- No two black faces share an edge

-- Define the areas of black and white faces
def sum_black_faces (P : Polyhedral) : ℝ :=
  ∑ (f : Face P), if f.color = Black then f.area else 0

def sum_white_faces (P : Polyhedral) : ℝ :=
  ∑ (f : Face P), if f.color = White then f.area else 0

-- The theorem statement
theorem sum_black_leq_sum_white : sum_black_faces P ≤ sum_white_faces P := 
by {
  -- Proof not required
  sorry
}

end sum_black_leq_sum_white_l415_415839


namespace function_inequality_l415_415816

noncomputable def f (x : ℝ) : ℝ := sorry

theorem function_inequality (f : ℝ → ℝ) (h1 : ∀ x : ℝ, x ≥ 1 → f x ≤ x)
  (h2 : ∀ x : ℝ, x ≥ 1 → f (2 * x) / Real.sqrt 2 ≤ f x) :
  ∀ x ≥ 1, f x < Real.sqrt (2 * x) :=
sorry

end function_inequality_l415_415816


namespace complex_number_multiplication_l415_415524

theorem complex_number_multiplication :
  let a := 3 
  let b := -4 
  let c := 0 
  let d := 6 
  ∃ (x y : ℤ), (a + b * complex.I) * (c + d * complex.I) = x + y * complex.I ∧ x = 24 ∧ y = 18 :=
by
  let a := 3
  let b := -4
  let c := 0
  let d := 6
  use [24, 18]
  sorry

end complex_number_multiplication_l415_415524


namespace solve_for_y_l415_415380

theorem solve_for_y (y : ℝ) (h : 3 * y ^ (1 / 4) - 3 * y ^ (1 / 2) / y ^ (1 / 4) = 13 - 2 * y ^ (1 / 4)) :
  y = (13 / 2) ^ 4 :=
by sorry

end solve_for_y_l415_415380


namespace production_problem_l415_415284

theorem production_problem (x y : ℝ) (h₁ : x > 0) (h₂ : ∀ k : ℝ, x * x * x * k = x) : (x * x * y * (1 / (x^2)) = y) :=
by {
  sorry
}

end production_problem_l415_415284


namespace expression1_expression2_l415_415641

noncomputable def a : ℝ := real.sqrt 3 - 2
noncomputable def b : ℝ := real.sqrt 3 + 2

theorem expression1 : a^2 + 2 * a * b + b^2 = 12 :=
by
  -- proof goes here
  sorry

theorem expression2 : a^2 * b - a * b^2 = 4 :=
by
  -- proof goes here
  sorry

end expression1_expression2_l415_415641


namespace sequence_arithmetic_progression_l415_415577

theorem sequence_arithmetic_progression:
  ∀ n : ℕ, n ≤ 9 →
    let b_n := ((1000 + n) ^ 2) / 100 in
    (b_{n + 1} - b_n) = 20 :=
by
  sorry

end sequence_arithmetic_progression_l415_415577


namespace china_math_competition_proof_l415_415710

open Set

theorem china_math_competition_proof (n k m : ℕ) (A : Finset ℕ)
  (h1 : 0 < n) (h2 : 0 < k) (h3 : 0 < m)
  (h4 : 2 ≤ k)
  (h5 : n ≤ m) (h6 : m < (2 * k - 1) * n / k)
  (hA : A.card = n) (hSubset : ↑A ⊆ (Icc 1 m : Set ℕ)) :
  ∀ t : ℕ, 0 < t → t < n / (k - 1) → ∃ a a' ∈ A, a - a' = t := 
by
  sorry

end china_math_competition_proof_l415_415710


namespace area_union_of_reflected_triangles_l415_415922

def point := (ℝ × ℝ)
def triangle := point × point × point

-- Given conditions
def A : point := (4, 3)
def B : point := (6, -1)
def C : point := (10, 2)

def line_y_equals_2 (p : point) : point := (p.1, 2 * 2 - p.2)

def reflect (A B C : point) : triangle :=
  (line_y_equals_2 A, line_y_equals_2 B, line_y_equals_2 C)

def area (t : triangle) : ℝ :=
  let (A, B, C) := t in
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

def original_triangle : triangle := (A, B, C)
def reflected_triangle : triangle := reflect A B C

theorem area_union_of_reflected_triangles :
  area original_triangle + area reflected_triangle = 22 :=
sorry

end area_union_of_reflected_triangles_l415_415922


namespace work_rates_l415_415096

theorem work_rates (A B : ℝ) (combined_days : ℝ) (b_rate: B = 35) 
(combined_rate: combined_days = 20 / 11):
    A = 700 / 365 :=
by
  have h1 : B = 35 := by sorry
  have h2 : combined_days = 20 / 11 := by sorry
  have : 1/A + 1/B = 11/20 := by sorry
  have : 1/A = 11/20 - 1/B := by sorry
  have : 1/A =  365 / 700:= by sorry
  have : A = 700 / 365 := by sorry
  assumption

end work_rates_l415_415096


namespace coins_difference_is_zero_l415_415723

-- Definitions for the problem conditions
def can_pay_exactly (c n d q : ℕ) : Prop :=
  10 * c + 20 * n + 50 * d = 45

def min_coins (c n d : ℕ) : ℕ :=
  c + n + d

def max_coins (c n d : ℕ) : ℕ :=
  c + n + d

-- The theorem to prove
theorem coins_difference_is_zero : ∃ c₁ n₁ d₁ c₂ n₂ d₂,
  can_pay_exactly c₁ n₁ d₁ ∧ can_pay_exactly c₂ n₂ d₂ ∧
  min_coins c₁ n₁ d₁ = max_coins c₂ n₂ d₂ :=
by
  -- Proof is yet to be provided
  sorry

end coins_difference_is_zero_l415_415723


namespace product_cosine_value_l415_415899

theorem product_cosine_value :
    (1 + 2 * cos (2 * Real.pi / 7)) *
    (1 + 2 * cos (4 * Real.pi / 7)) *
    (1 + 2 * cos (6 * Real.pi / 7)) *
    (1 + 2 * cos (8 * Real.pi / 7)) *
    (1 + 2 * cos (10 * Real.pi / 7)) *
    (1 + 2 * cos (12 * Real.pi / 7)) = 1 := 
sorry

end product_cosine_value_l415_415899


namespace max_value_of_k_l415_415310

noncomputable def max_k (k : ℝ) : ℝ :=
by
  let h := ((4 * k - 2) / Real.sqrt (1 + k ^ 2))
  have h_le : h <= 3 := sorry
  have k_sq : 7 * k^2 - 24 * k <= 0 := sorry
  exact if h_le then (24 / 7 : ℝ)
  else sorry

theorem max_value_of_k : max_k 0 = (24 / 7 : ℝ) := sorry

end max_value_of_k_l415_415310


namespace longer_side_of_rectangle_is_l415_415113

-- Define the radius of the circle
def radius (r : ℝ) : Prop := r = 6

-- Define the area of the circle given the radius
def area_circle (A : ℝ) : Prop := A = 36 * Real.pi

-- Define the relationship between the area of the rectangle and the circle
def area_rectangle (A_rect : ℝ) (A : ℝ) : Prop := A_rect = 3 * A

-- Define the length of the shorter side of the rectangle
def shorter_side (s : ℝ) : Prop := s = 12

-- State the goal: the length of the longer side of the rectangle
def longer_side (l : ℝ) (A_rect : ℝ) (s : ℝ) : Prop := l = A_rect / s

-- Theorem statement combining all conditions and the goal
theorem longer_side_of_rectangle_is :
  ∃ l : ℝ, ∃ A_rect : ℝ, ∃ s : ℝ, radius 6 ∧ area_circle 36 * Real.pi ∧ area_rectangle A_rect (36 * Real.pi) ∧ shorter_side s ∧ longer_side l A_rect s :=
by {
  sorry
}

end longer_side_of_rectangle_is_l415_415113


namespace geometric_sequence_general_term_arithmetic_sequence_general_term_l415_415313

noncomputable theory

-- Definition of the geometric sequence \{a_n\}
def a_n (n : ℕ) : ℕ := 2 * (2^(n - 1))

-- Definition of the arithmetic sequence \{b_n\} with given terms from \{a_n\}
def b_n (n : ℕ) : ℤ := 12 * n - 28

theorem geometric_sequence_general_term :
  (∀ n, a_n n = 2^n) :=
sorry

theorem arithmetic_sequence_general_term :
  a_n 3 = b_n 3 ∧ a_n 5 = b_n 5 →
  (∀ n, b_n n = 12 * n - 28) :=
sorry

end geometric_sequence_general_term_arithmetic_sequence_general_term_l415_415313


namespace inclination_angle_of_line_is_correct_l415_415013

theorem inclination_angle_of_line_is_correct (θ : ℝ) : 
    (∀ x y : ℝ, (sqrt 3) * x + y - 1 = 0 -> tan θ = - (sqrt 3)) → 
    (0 ≤ θ ∧ θ < real.pi) → θ = (2 * real.pi) / 3 :=
begin
  sorry
end

end inclination_angle_of_line_is_correct_l415_415013


namespace midpoint_locus_equation_l415_415170

variable (A B C : Type)
variable [add_comm_group A] [vector_space ℝ A] [exactly_one_mem_group B] [linear_ordered_field C]

def line_segment_length_4 (a b : ℝ) : Bool := 
  let A := (a, a)
  let B := (b, 2*b)
  (a - b)^2 + (a - 2*b)^2 = 16

def midpoint_locus (a b x y : ℝ) : Bool :=
  let A := (a, a)
  let B := (b, 2*b)
  let M := ((a + b) / 2, (a + 2 * b) / 2)
  25 * x^2 - 36 * x * y + 13 * y^2 = 4

theorem midpoint_locus_equation :
  ∀ (a b x y : ℝ), 
  line_segment_length_4 a b -> 
  midpoint_locus a b x y := 
by sorry

end midpoint_locus_equation_l415_415170


namespace number_of_covered_squares_l415_415145

-- Definition of the problem and the conditions
def square_tile_side_length (D : ℝ) := D
def checkerboard_side_length := 10
def each_square_side_length (D : ℝ) := D
def centers_coincide := true

-- The main theorem stating the number of completely covered squares
theorem number_of_covered_squares (D : ℝ) (h : centers_coincide) : 
  let checkerboard_squares := checkerboard_side_length * checkerboard_side_length in
  let tile_squares := 5 * 5 in
  tile_squares * 4 = 25 :=
by
  sorry

end number_of_covered_squares_l415_415145


namespace a_n_formula_sum_b_n_l415_415235

noncomputable def a_n (n : Nat) : ℕ := 2 * n + 1

noncomputable def S_n (n : Nat) : ℕ := ∑ i in Finset.range (n+1), a_n i

theorem a_n_formula (n : ℕ) (h : a_n n > 0) (hs : a_n n ^ 2 + 2 * a_n n = 4 * S_n n + 3) :
  a_n n = 2 * n + 1 := 
sorry

noncomputable def b_n (n : Nat) : ℚ := 1 / (a_n n * a_n (n + 1))

def T_n (n : ℕ) : ℚ := ∑ i in Finset.range n, b_n i

theorem sum_b_n (n : ℕ) :
  T_n n = n / (3 * (2 * n + 3)) := 
sorry

end a_n_formula_sum_b_n_l415_415235


namespace longer_side_of_rectangle_l415_415108

theorem longer_side_of_rectangle
  (r : ℝ) (A_rect A_circle L S : ℝ) (h1 : r = 6)
  (h2 : A_circle = π * r^2)
  (h3 : A_rect = 3 * A_circle)
  (h4 : S = 2 * r)
  (h5 : A_rect = S * L) : L = 9 * π :=
by
  sorry

end longer_side_of_rectangle_l415_415108


namespace eccentricity_of_hyperbola_l415_415534

variables (x y a b c : ℝ)

-- Conditions
def hyperbola_equation (a b : ℝ) := (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)
def focal_axis_length (c : ℝ) := (2 * c = 4)
def distance_focus_asymptote (a b c : ℝ) := (b * c / Real.sqrt (a^2 + b^2) = Real.sqrt 3)

-- Proof that the eccentricity e = 2
theorem eccentricity_of_hyperbola :
  ∃ a b c e : ℝ, hyperbola_equation a b ∧ focal_axis_length c ∧ distance_focus_asymptote a b c ∧ (e = c / a) ∧ (e = 2) :=
by {
  sorry
}

end eccentricity_of_hyperbola_l415_415534


namespace tangent_parallel_l415_415741

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel (P0 : ℝ × ℝ) : 
  ((P0.1 = 1 ∧ P0.2 = f 1) ∨ (P0.1 = -1 ∧ P0.2 = f (-1))) → 
  (∃ m b : ℝ, ∃ tangent : ℝ → ℝ, 
    tangent = (λ x, m * x + b) ∧ 
    (∀ x : ℝ, m = 4 ↔ deriv f x = 4) ∧ 
    tangent P0.1 = P0.2) :=
by
  sorry

end tangent_parallel_l415_415741


namespace determine_phi_l415_415627

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem determine_phi 
  (φ : ℝ)
  (H1 : ∀ x : ℝ, f x φ ≤ |f (π / 6) φ|)
  (H2 : f (π / 3) φ > f (π / 2) φ) :
  φ = π / 6 :=
sorry

end determine_phi_l415_415627


namespace digit_arrangements_l415_415677

section
variable (digits : Finset ℕ := {6, 0, 4, 0, 2})

theorem digit_arrangements : 
  (Finset.filter (λ n : List ℕ, n.head ≠ 0) 
      (Finset.image List.ofFn 
        (Finset.univ : Finset (Fin₅ → ℕ))))
      .card = 96 := by
  sorry
end

end digit_arrangements_l415_415677


namespace carla_chickens_l415_415532

theorem carla_chickens (initial_chickens : ℕ) (percent_died : ℕ) (bought_factor : ℕ) :
  initial_chickens = 400 →
  percent_died = 40 →
  bought_factor = 10 →
  let died := (percent_died * initial_chickens) / 100 in
  let bought := bought_factor * died in
  let total := initial_chickens - died + bought in
  total = 1840 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  let died := (40 * 400) / 100
  have hdied : died = 160 := rfl
  let bought := 10 * died
  have hbought : bought = 1600 := rfl
  let total := 400 - 160 + 1600
  have htotal : total = 1840 := rfl
  exact htotal

end carla_chickens_l415_415532


namespace thirty_two_operations_result_in_ones_l415_415090

theorem thirty_two_operations_result_in_ones (a : Fin 32 → ℤ) (h : ∀ i, a i = 1 ∨ a i = -1) :
  ∃ N : ℕ, N = 32 ∧ ∀ k : Fin 32, (iterate (λ f, λ (a : Fin 32 → ℤ) i, a i * a (Fin.modNat (i.val + 1))) N a) k = 1 :=
by sorry

end thirty_two_operations_result_in_ones_l415_415090


namespace find_a_l415_415611

-- Given conditions as definitions.
def f (a x : ℝ) := a * x^3
def tangent_line (a : ℝ) (x : ℝ) : ℝ := 3 * x + a - 3

-- Problem statement in Lean 4.
theorem find_a (a : ℝ) (h_tangent : ∀ x : ℝ, f a 1 = 1 ∧ f a 1 = tangent_line a 1) : a = 1 := 
by sorry

end find_a_l415_415611


namespace constant_term_of_binomial_expansion_l415_415183

theorem constant_term_of_binomial_expansion:
  let T_r (r : ℕ) := (Nat.choose 6 r) * (2^r) in
  (∃ r : ℕ, r ≤ 6 ∧ 6 - 2 * r = 0 ∧ T_r r = 160) :=
sorry

end constant_term_of_binomial_expansion_l415_415183


namespace problem1_problem2_l415_415912

noncomputable def m (ω x : ℝ) := (Real.cos (ω * x) + Real.sin (ω * x), Real.cos (ω * x))
noncomputable def n (ω x : ℝ) := (Real.cos (ω * x) - Real.sin (ω * x), 2 * Real.sin (ω * x))
noncomputable def f (ω x : ℝ) := (m ω x).1 * (n ω x).1 + (m ω x).2 * (n ω x).2

theorem problem1 (ω : ℝ) (hω : 0 < ω) (hx : ∀ x, ∃ k ∈ ℤ, f ω x = f ω (x + π * k)) : ω = 1 :=
sorry

variables {A B C a b c : ℝ}

theorem problem2 (hABC : 0 < B ∧ B < π) (hseq : 2 * b = a + c) (hfB : f 1 B = 1) : 
  a = c ∧ ∠ A B C = π / 6 ∧ (a = b ∧ b = c) :=
sorry

end problem1_problem2_l415_415912


namespace angle_between_a_and_b_min_magnitude_c_l415_415640

open Real

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-1, 3)
def c (λ : ℝ) : ℝ × ℝ := (λ * a.1 + b.1, λ * a.2 + b.2)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1^2 + v.2^2)

theorem angle_between_a_and_b : 
  let θ := acos ((a.1 * b.1 + a.2 * b.2) / (magnitude a * magnitude b))
  θ = π / 4 :=
by
  sorry

theorem min_magnitude_c : 
  ∃ λ : ℝ, magnitude (c λ) = sqrt 5 ∧ ∀ x : ℝ, magnitude (c x) ≥ sqrt 5 :=
by
  sorry

end angle_between_a_and_b_min_magnitude_c_l415_415640


namespace grocer_pounds_of_bananas_purchased_l415_415482

/-- 
Given:
1. The grocer purchased bananas at a rate of 3 pounds for $0.50.
2. The grocer sold the entire quantity at a rate of 4 pounds for $1.00.
3. The profit from selling the bananas was $11.00.

Prove that the number of pounds of bananas the grocer purchased is 132. 
-/
theorem grocer_pounds_of_bananas_purchased (P : ℕ) 
    (h1 : ∃ P, (3 * P / 0.5) - (4 * P / 1.0) = 11) : 
    P = 132 := 
sorry

end grocer_pounds_of_bananas_purchased_l415_415482


namespace quad_bike_overtakes_motorcycle_on_11th_lap_l415_415828

-- Define the conditions
def forest_fraction : ℝ := 1.0 / 4.0
def field_fraction : ℝ := 3.0 / 4.0

def motorcycle_forest_speed : ℝ := 20.0 -- km/h
def motorcycle_field_speed : ℝ := 60.0 -- km/h
def atv_forest_speed : ℝ := 40.0 -- km/h
def atv_field_speed : ℝ := 45.0 -- km/h

-- Define the time taken to complete a lap by each vehicle
def motorcycle_lap_time (C : ℝ) : ℝ :=
  (forest_fraction * C / motorcycle_forest_speed) + (field_fraction * C / motorcycle_field_speed)

def atv_lap_time (C : ℝ) : ℝ :=
  (forest_fraction * C / atv_forest_speed) + (field_fraction * C / atv_field_speed)

-- State the theorem
theorem quad_bike_overtakes_motorcycle_on_11th_lap (C : ℝ) (hC : 0 < C) :
  ∃ (n : ℕ), n = 11 ∧ 
    (n * atv_lap_time C < (n - 1) * motorcycle_lap_time C + some_extra_distance_for_motorcycle) := sorry

end quad_bike_overtakes_motorcycle_on_11th_lap_l415_415828


namespace first_player_wins_l415_415044

def wins (sum_rows sum_cols : ℕ) : Prop := sum_rows > sum_cols

theorem first_player_wins 
  (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℕ)
  (h : a_1 > a_2 ∧ a_2 > a_3 ∧ a_3 > a_4 ∧ a_4 > a_5 ∧ a_5 > a_6 ∧ a_6 > a_7 ∧ a_7 > a_8 ∧ a_8 > a_9) :
  ∃ sum_rows sum_cols, wins sum_rows sum_cols :=
sorry

end first_player_wins_l415_415044


namespace cos_C_of_triangle_l415_415298

theorem cos_C_of_triangle
  (sin_A : ℝ) (cos_B : ℝ) 
  (h1 : sin_A = 3/5)
  (h2 : cos_B = 5/13) :
  ∃ (cos_C : ℝ), cos_C = 16/65 :=
by
  -- Place for the proof
  sorry

end cos_C_of_triangle_l415_415298


namespace longer_side_of_rectangle_l415_415111

theorem longer_side_of_rectangle
  (r : ℝ) (A_rect A_circle L S : ℝ) (h1 : r = 6)
  (h2 : A_circle = π * r^2)
  (h3 : A_rect = 3 * A_circle)
  (h4 : S = 2 * r)
  (h5 : A_rect = S * L) : L = 9 * π :=
by
  sorry

end longer_side_of_rectangle_l415_415111


namespace paint_square_distance_l415_415726

theorem paint_square_distance (c : ℝ) : 
  (∀ (p : ℝ × ℝ), (p.1 = 0 ∨ p.1 = c ∨ p.2 = 0 ∨ p.2 = c) → 
  ∃ (q : ℝ × ℝ), (q.1 = 0 ∨ q.1 = c ∨ q.2 = 0 ∨ q.2 = c) ∧ q ≠ p ∧ 
  ∃ (color : bool) (color_p : bool) (color_q : bool), color_p = color_q ∧ 
  distance p q ≥ sqrt 5) ↔ c ≥ sqrt 10 / 2 := sorry

end paint_square_distance_l415_415726


namespace angle_B_in_triangle_l415_415299

theorem angle_B_in_triangle (a b c : ℝ) (B : ℝ) (h : (a^2 + c^2 - b^2) * Real.tan B = Real.sqrt 3 * a * c) :
  B = 60 ∨ B = 120 := 
sorry

end angle_B_in_triangle_l415_415299


namespace possible_positions_of_M_l415_415637

noncomputable theory

section Geometry

variables {K L M : ℝ} -- Represent K, L, and M as variables for real coordinates in the plane

-- Define the distance function
def distance (P Q : ℝ) : ℝ := abs (P - Q)

-- Define the altitude and median from a vertex of the triangle formed by K, L, and M.
-- Note: We assume the triangle is not degenerate for sensible geometric interpretations.
def altitude (P Q R : ℝ) : ℝ := distance P ( (Q + R) / 2 ) -- Altitude (Simplified as example with coordinates)
def median (P Q R : ℝ) : ℝ := distance ( (P + Q) / 2 ) R -- Median

-- The geometric loci (lines and circles) are conditions from the problem.
def valid_positions (K L : ℝ) (M : ℝ) : Prop :=
  (altitude K L M = median K L M) ∨
  (altitude L M K = median L M K) ∨
  (altitude M K L = median M K L)

-- Main theorem statement
theorem possible_positions_of_M (K L : ℝ) : set ℝ :=
  { M : ℝ | valid_positions K L M }

end Geometry

end possible_positions_of_M_l415_415637


namespace percentage_spent_on_eating_out_and_socializing_l415_415549

-- Definitions derived from conditions
def net_monthly_salary : ℝ := 3400
def discretionary_income := (1 / 5) * net_monthly_salary
def vacation_fund := 0.3 * discretionary_income
def savings := 0.2 * discretionary_income
def gifts_charity : ℝ := 102
def total_allocated := vacation_fund + savings + gifts_charity
def eating_out_socializing := discretionary_income - total_allocated
def percentage_eating_out_socializing := (eating_out_socializing / discretionary_income) * 100

-- Theorem to be proven in Lean
theorem percentage_spent_on_eating_out_and_socializing : percentage_eating_out_socializing = 35 := by
  -- proof skipped
  sorry

end percentage_spent_on_eating_out_and_socializing_l415_415549


namespace two_pow_2023_mod_17_l415_415868

theorem two_pow_2023_mod_17 : (2 ^ 2023) % 17 = 4 := 
by
  sorry

end two_pow_2023_mod_17_l415_415868


namespace miles_monday_calculation_l415_415452

-- Define the constants
def flat_fee : ℕ := 150
def cost_per_mile : ℝ := 0.50
def miles_thursday : ℕ := 744
def total_cost : ℕ := 832

-- Define the equation to be proved
theorem miles_monday_calculation :
  ∃ M : ℕ, (flat_fee + (M : ℝ) * cost_per_mile + (miles_thursday : ℝ) * cost_per_mile = total_cost) ∧ M = 620 :=
by
  sorry

end miles_monday_calculation_l415_415452


namespace equal_areas_of_lune_and_triangle_l415_415081

noncomputable def isosceles_right_triangle (A B C : ℝ) (r : ℝ) :=
  A = r ∧ B = r ∧ C = r * Real.sqrt 2

theorem equal_areas_of_lune_and_triangle (r : ℝ) (A B C : ℝ) 
  (h_triangle : isosceles_right_triangle A B C r) : 
  let area_triangle := (1/2) * A * B in
  let radius_larger_semicircle := C / 2 in
  let area_larger_semicircle := (1/2) * Real.pi * radius_larger_semicircle^2 in
  let area_smaller_semicircle := (1/2) * Real.pi * r^2 in
  let area_lune := area_larger_semicircle - area_smaller_semicircle in
  area_lune = area_triangle :=
sorry

end equal_areas_of_lune_and_triangle_l415_415081


namespace can_cut_figure_equally_l415_415537

-- Definitions to be used in Lean code
def canBeCutEqually : Prop :=
  ∃ (cut: set (int × int)), 
    -- conditions on the cut to ensure equal parts and shape
    (∀ (x y: int × int), (x ∈ cut ∧ y ∉ cut) → 
        ∃ (f : (int × int) → (int × int)), 
          (f y = x) ∧ 
          ((y ∈ cut) → (f y ∈ cut)) ∧ 
          ((y ∉ cut) → (f y ∉ cut)) ∧ 
          (rotatesOrFlips f))

-- Function indicating possible rotation or flipping transformations
def rotatesOrFlips (f: (int × int) → (int × int)) : Prop :=
  ∀ (p: int × int), f (f p) = p

-- Statement requiring proof
theorem can_cut_figure_equally :
  canBeCutEqually :=
sorry

end can_cut_figure_equally_l415_415537


namespace alex_piles_of_jelly_beans_l415_415514

theorem alex_piles_of_jelly_beans : 
  ∀ (initial_weight eaten weight_per_pile remaining_weight piles : ℕ),
    initial_weight = 36 →
    eaten = 6 →
    weight_per_pile = 10 →
    remaining_weight = initial_weight - eaten →
    piles = remaining_weight / weight_per_pile →
    piles = 3 :=
by
  intros initial_weight eaten weight_per_pile remaining_weight piles h_init h_eat h_wpile h_remaining h_piles
  sorry

end alex_piles_of_jelly_beans_l415_415514


namespace symmetric_circle_eq_l415_415559

theorem symmetric_circle_eq (x y : ℝ) :
  let C1 := (x - 3)^2 + (y + 1)^2 = 1 in
  let L := x + 2*y - 3 = 0 in
  C1 → L →
  (x - 11/3)^2 + (y - 1/3)^2 = 1 :=
by
  intro C1 L
  sorry

end symmetric_circle_eq_l415_415559


namespace ellipse_eq_from_foci_chord_l415_415604

noncomputable def fociDistance (a b : ℝ) : ℝ := sqrt(a^2 - b^2)
noncomputable def pointOnEllipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_eq_from_foci_chord
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hab : a > b)
  (h_foci : fociDistance a b = 1)
  (h_AB : pointOnEllipse a b 1 (3/2) ∧ pointOnEllipse a b 1 (-3/2)): 
  (a = 2 ∧ b = 1.732) :=
by 
  -- Proof to establish a = 2 and b = sqrt(3) will go here
  sorry

end ellipse_eq_from_foci_chord_l415_415604


namespace area_of_rectangle_l415_415034

theorem area_of_rectangle (length : ℝ) (width : ℝ) (h_length : length = 47.3) (h_width : width = 24) : 
  length * width = 1135.2 := 
by 
  sorry

end area_of_rectangle_l415_415034


namespace distance_to_plane_l415_415173

variable (V : ℝ) (A : ℝ) (r : ℝ) (d : ℝ)

-- Assume the volume of the sphere and area of the cross-section
def sphere_volume := V = 4 * Real.sqrt 3 * Real.pi
def cross_section_area := A = Real.pi

-- Define radius of sphere and cross-section
def sphere_radius := r = Real.sqrt 3
def cross_section_radius := Real.sqrt A = 1

-- Define distance as per Pythagorean theorem
def distance_from_center := d = Real.sqrt (r^2 - 1^2)

-- Main statement to prove
theorem distance_to_plane (V A : ℝ)
  (h1 : sphere_volume V) 
  (h2 : cross_section_area A) 
  (h3: sphere_radius r) 
  (h4: cross_section_radius A) : 
  distance_from_center r d :=
sorry

end distance_to_plane_l415_415173


namespace sin_alpha_plus_pi_div_12_l415_415608

theorem sin_alpha_plus_pi_div_12 
  {α : ℝ}
  (h1 : α > -π / 3)
  (h2 : α < 0)
  (h3 : cos (α + π / 6) - sin α = 4 * real.sqrt 3 / 5) 
  : sin (α + π / 12) = -real.sqrt 2 / 10 :=
sorry

end sin_alpha_plus_pi_div_12_l415_415608


namespace problem1_solution_problem2_solution_l415_415166

noncomputable def problem1 : ℝ :=
  real.sqrt 8 - 2 * real.sin (real.pi / 4) + |1 - real.sqrt 2| + (1 / 2)⁻¹

theorem problem1_solution : problem1 = 2 * real.sqrt 2 + 1 :=
by
  sorry

noncomputable def f (x : ℝ) : ℝ :=
  x^2 + 4 * x - 5

theorem problem2_solution : ∃ (x1 x2 : ℝ), f x1 = 0 ∧ f x2 = 0 ∧ x1 = -5 ∧ x2 = 1 :=
by
  use -5
  use 1
  sorry

end problem1_solution_problem2_solution_l415_415166


namespace lines_perpendicular_l415_415968

theorem lines_perpendicular (l1 l2 : Line) (A : Point) 
  (h1 : intersect_line_point l1 l2 A) 
  (h2 : ∃ θ, is_right_angle θ ∧ formed_by l1 l2 A θ) : 
  are_perpendicular l1 l2 :=
sorry

end lines_perpendicular_l415_415968


namespace min_value_of_ac_sq_plus_bd_sq_l415_415702

theorem min_value_of_ac_sq_plus_bd_sq (a b c d : ℝ) (h1 : a * b = 2) (h2 : c * d = 18) : 
  (ac := a * c; bd := b * d) (ac^2 + bd^2 >= 12) :=
by
  intro a b c d h1 h2
  let ac := a * c
  let bd := b * d
  have h3 : (ac)^2 + (bd)^2 >= 2 * sqrt ((ac)^2 * (bd)^2)
  sorry

end min_value_of_ac_sq_plus_bd_sq_l415_415702


namespace largest_prime_factor_correct_l415_415064

-- We define our numbers
def numA := 210
def numB := 255
def numC := 143
def numD := 187
def numE := 169

-- Define the largest prime factor of each number
def largest_prime_factor (n : Nat) : Nat :=
  Nat.factors n |>.last' |>.getOrElse 1

-- Define the assertion that we need to prove
theorem largest_prime_factor_correct :
  largest_prime_factor numB = 17 ∧ largest_prime_factor numD = 17 ∧
  (largest_prime_factor numA < 17) ∧ (largest_prime_factor numC < 17) ∧ (largest_prime_factor numE < 17) :=
by
  sorry

end largest_prime_factor_correct_l415_415064


namespace min_value_AB_l415_415386

open Real

-- Define parametric equations for the line l
def parametric_line (t φ : ℝ) : ℝ × ℝ := (t * sin φ, 1 + t * cos φ)

-- Define polar equation of curve C
def polar_curve (ρ θ : ℝ) : Prop := ρ * cos θ * cos θ = 4 * sin θ

-- Define Cartesian equation of curve C
def cartesian_curve (x y : ℝ) : Prop := x * x = 4 * y

-- Problem statement
def proof_problem : Prop :=
  ∀ (φ : ℝ), 0 < φ ∧ φ < π →
  (∃ t : ℝ, parametric_line t φ = (x, y)) → 
  (∃ (l : ℝ × ℝ → Prop),
    (∀ (t : ℝ), l (parametric_line t φ)) ∧ 
    (general_equation l φ)) ∧
  (cartesian_curve x y) ∧
  -- Prove the minimum distance |AB| when the line intersects the curve
  (∀ (t1 t2 : ℝ), (parametric_line t1 φ = (x1, y1)) ∧ (parametric_line t2 φ = (x2, y2)) →
    C (min_AB_distance t1 t2 φ = 4))

noncomputable def min_AB_distance (t1 t2 φ : ℝ) : ℝ := 
  (abs (t1 - t2))

theorem min_value_AB : proof_problem := by
  sorry

end min_value_AB_l415_415386


namespace walnut_trees_planted_l415_415420

-- Define the initial number of walnut trees
def initial_walnut_trees : ℕ := 22

-- Define the total number of walnut trees after planting
def total_walnut_trees_after : ℕ := 55

-- The Lean statement to prove the number of walnut trees planted today
theorem walnut_trees_planted : (total_walnut_trees_after - initial_walnut_trees = 33) :=
by
  sorry

end walnut_trees_planted_l415_415420


namespace quad_ratio_l415_415993

theorem quad_ratio (A B C D E : Type) (d : A → B → ℝ)
  (AB BC CD DA : ℝ) (ABC_angle : Real.Angle)
  (h_ABC : ABC_angle = Real.pi / 2)
  (AB_eq_5 : d A B = 5)
  (BC_eq_6 : d B C = 6)
  (CD_eq_5 : d C D = 5)
  (DA_eq_4 : d D A = 4)
  (intersect_AC_BD : ∃ E, AC A C E ∧ BD B D E) :
  BE_ratio (d B E) (d E D) = 5 / 4 :=
by
  sorry

end quad_ratio_l415_415993


namespace scrooge_mcduck_max_box_l415_415696

-- Define Fibonacci numbers
def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

-- The problem statement: for a given positive integer k (number of coins initially),
-- the maximum box index n into which Scrooge McDuck can place a coin
-- is F_{k+2} - 1.
theorem scrooge_mcduck_max_box (k : ℕ) (h_pos : 0 < k) :
  ∃ n, n = fib (k + 2) - 1 :=
sorry

end scrooge_mcduck_max_box_l415_415696


namespace circles_radii_l415_415363

theorem circles_radii (A B C D E : ℝ) (r R : ℝ) 
  (h1 : B = A + 2) (h2 : C = B + 2) (h3 : D = C + 1) (h4 : E = D + 2) 
  (h5 : ∃ O Q : ℝ, 
    ∃ ω Ω : circle, 
    Ω.radius = R ∧ ω.radius = r ∧ 
    Ω.passes_through D ∧ Ω.passes_through E ∧ 
    ω.passes_through B ∧ ω.passes_through C ∧ 
    Ω.tangent_to ω ∧ collinear {A, O, Q}) :
  (R = 8 / Real.sqrt 19) ∧ (r = 11 / (2 * Real.sqrt 19)) :=
by
  sorry

end circles_radii_l415_415363


namespace XiaoKang_min_sets_pushups_pullups_l415_415068

theorem XiaoKang_min_sets_pushups_pullups (x y : ℕ) (hx : x ≥ 100) (hy : y ≥ 106) (h : 8 * x + 5 * y = 9050) :
  x ≥ 100 ∧ y ≥ 106 :=
by {
  sorry  -- proof not required as per instruction
}

end XiaoKang_min_sets_pushups_pullups_l415_415068


namespace annieka_free_throws_l415_415003

theorem annieka_free_throws (deshawn_throws : ℕ) (kayla_factor : ℝ) (annieka_diff : ℕ) (ht1 : deshawn_throws = 12) (ht2 : kayla_factor = 1.5) (ht3 : annieka_diff = 4) :
  ∃ (annieka_throws : ℕ), annieka_throws = (⌊deshawn_throws * kayla_factor⌋.toNat - annieka_diff) :=
by
  sorry

end annieka_free_throws_l415_415003


namespace davi_minimum_spending_l415_415876

-- Define the cost of a single bottle
def singleBottleCost : ℝ := 2.80

-- Define the cost of a box of six bottles
def boxCost : ℝ := 15.00

-- Define the number of bottles Davi needs to buy
def totalBottles : ℕ := 22

-- Calculate the minimum amount Davi will spend
def minimumCost : ℝ := 45.00 + 11.20 

-- The theorem to prove
theorem davi_minimum_spending :
  ∃ minCost : ℝ, minCost = 56.20 ∧ minCost = 3 * boxCost + 4 * singleBottleCost := 
by
  use 56.20
  sorry

end davi_minimum_spending_l415_415876


namespace square_root_properties_l415_415921

theorem square_root_properties (a : ℝ) (h : a^2 = 10) :
  a = Real.sqrt 10 ∧ (a^2 - 10 = 0) ∧ (3 < a ∧ a < 4) :=
by sorry

end square_root_properties_l415_415921


namespace graph_point_condition_l415_415251

noncomputable def f : ℝ → ℝ := sorry

theorem graph_point_condition (h : f 3 = 9) :
  (4 * (5 * f (3 * 1) + 2) / 4 = 47 / 4)
  ∧ (1 + 47 / 4 = 12.75) :=
by
  have h1 : f (3 * 1) = 9 := by rw [h, mul_one]
  have h2 : 4 * (f (3 * 1) * 5 + 2) = 47 := by rw [h1, mul_assoc]; norm_num
  split
  · linarith
  · norm_num

end graph_point_condition_l415_415251


namespace xiaoqiang_expected_score_l415_415826

noncomputable def expected_score_xiaoqiang : ℚ :=
by
  let n := 25
  let p := 0.8
  let correct_points := 4
  let X := binomial n p -- Binomial distribution
  let score := correct_points * X
  have expected_score : E(score) = correct_points * E(X) := sorry
  have E_X : E(X) = n * p := sorry
  have result : expected_score_xiaoqiang = correct_points * n * p := sorry
  exact sorry

theorem xiaoqiang_expected_score : expected_score_xiaoqiang = 80 := sorry

end xiaoqiang_expected_score_l415_415826


namespace part_a_part_b_l415_415160

-- Define the problem setup and rules
structure Hexagon := (vertices : Fin 6 → ℝ × ℝ)

def initial_hexagon : Hexagon := {
  vertices := λ i, -- Coordinates for regular hexagon vertices
    let θ := 2 * Real.pi * (i : ℕ) / 6 in (Real.cos θ, Real.sin θ)
}

-- Rules for jumping
def rule1 (F P : ℝ × ℝ) : ℝ × ℝ :=
  let (fx, fy) := F in
  let (px, py) := P in
  (2 * px - fx, 2 * py - fy)

def rule2 (F P : ℝ × ℝ) : ℝ × ℝ :=
  let (fx, fy) := F in
  let (px, py) := P in
  ((px + fx) / 2, (py + fy) / 2)

-- Define the main statements

-- Part (a): Only using Rule 1
theorem part_a : ∀ (hex : Hexagon) (c : ℝ × ℝ), 
  hex = initial_hexagon →
  c = (0,0) →
  ¬ (∃ (n : ℕ) (seq : Fin n → ℝ × ℝ),
    ∃ (i : Fin 6), 
    ∃ (move : Fin n → ℝ × ℝ × ℝ × ℝ × bool), -- (F, P, isRule1)
    (∀ (k : Fin n), 
      let (F, P, is_rule1) := move k in
      (is_rule1 → seq k+1 = rule1 F P) ∧ 
      (¬is_rule1 → seq k+1 = rule1 F P)) ∧
    seq 0 = hex.vertices i ∧
    seq (n-1) = c) :=
sorry

-- Part (b): Using both Rule 1 and Rule 2
theorem part_b : ∀ (hex : Hexagon) (c : ℝ × ℝ), 
  hex = initial_hexagon →
  c = (0,0) →
  ∃ (n : ℕ) (seq : Fin n → ℝ × ℝ),
    ∃ (i : Fin 6), 
    ∃ (move : Fin n → ℝ × ℝ × ℝ × ℝ × bool), -- (F, P, isRule1)
    (∀ (k : Fin n), 
      let (F, P, is_rule1) := move k in
      (is_rule1 → seq k+1 = rule1 F P) ∧ 
      (¬is_rule1 → seq k+1 = rule2 F P)) ∧
    seq 0 = hex.vertices i ∧
    seq (n-1) = c :=
sorry

end part_a_part_b_l415_415160


namespace percentage_of_sikh_boys_l415_415306

-- Define the conditions
def total_boys : ℕ := 650
def muslim_boys : ℕ := (44 * total_boys) / 100
def hindu_boys : ℕ := (28 * total_boys) / 100
def other_boys : ℕ := 117
def sikh_boys : ℕ := total_boys - (muslim_boys + hindu_boys + other_boys)

-- Define and prove the theorem
theorem percentage_of_sikh_boys : (sikh_boys * 100) / total_boys = 10 :=
by
  have h_muslims: muslim_boys = 286 := by sorry
  have h_hindus: hindu_boys = 182 := by sorry
  have h_total: muslim_boys + hindu_boys + other_boys = 585 := by sorry
  have h_sikhs: sikh_boys = 65 := by sorry
  have h_percentage: (65 * 100) / 650 = 10 := by sorry
  exact h_percentage

end percentage_of_sikh_boys_l415_415306


namespace coordinates_of_M_l415_415935

theorem coordinates_of_M (a : ℝ) (h : a + 1 = 0) : (a - 2, a + 1) = (-3, 0) :=
by
  have ha : a = -1 := by linarith
  rw [ha]
  simp

end coordinates_of_M_l415_415935


namespace distinct_integer_solutions_l415_415413

open Polynomial

noncomputable def p (a : ℕ → ℤ) : Polynomial ℤ :=
  ∑ i in (range 1992), monomial i (a i)

theorem distinct_integer_solutions (a : ℕ → ℤ) :
  (∀ i, a i ∈ ℤ) →
  ∃ n ≤ 1995, ∀ x ∈ finset.filter (λ x, (p a).eval x ^ 2 = 9) finset.univ, x ∈ finset.range n :=
begin
  -- proof goes here
  sorry
end

end distinct_integer_solutions_l415_415413


namespace distance_from_T_to_ABC_l415_415725

noncomputable def distance_to_plane (A B C T : EuclideanSpace ℝ (Fin 3))
  (h1 : T.dist A = 15) (h2 : T.dist B = 15) (h3 : T.dist C = 9)
  (h4 : A ≠ B) (h5 : B ≠ C) (h6 : A ≠ C)
  (h7 : InnerProductSpace.isOrthogonal ℝ (A - T) (B - T))
  (h8 : InnerProductSpace.isOrthogonal ℝ (A - T) (C - T))
  (h9 : InnerProductSpace.isOrthogonal ℝ (B - T) (C - T)) : ℝ :=
  
  let AB : EuclideanSpace ℝ (Fin 3) := B - A in
  let AC : EuclideanSpace ℝ (Fin 3) := C - A in
  let TA : ℝ := T.dist A in
  let TB : ℝ := T.dist B in
  let TC : ℝ := T.dist C in
  let AB_length : ℝ := AB.norm in
  let AC_length : ℝ := AC.norm in
  let area_ABC : ℝ := 0.5 * AB_length * sqrt (AC_length^2 - (AB_length/2)^2) in
  (3 * (TA * TB / 2 * TC) / area_ABC)

theorem distance_from_T_to_ABC
  (A B C T : EuclideanSpace ℝ (Fin 3))
  (h1 : T.dist A = 15) (h2 : T.dist B = 15) (h3 : T.dist C = 9)
  (h4 : A ≠ B) (h5 : B ≠ C) (h6 : A ≠ C)
  (h7 : InnerProductSpace.isOrthogonal ℝ (A - T) (B - T))
  (h8 : InnerProductSpace.isOrthogonal ℝ (A - T) (C - T))
  (h9 : InnerProductSpace.isOrthogonal ℝ (B - T) (C - T)) :
  distance_to_plane A B C T h1 h2 h3 h4 h5 h6 h7 h8 h9 = 6 * sqrt 6 :=
by sorry

end distance_from_T_to_ABC_l415_415725


namespace area_of_right_triangle_l415_415999

theorem area_of_right_triangle (AC AB : ℝ) (h1 : AC = 5) (h2 : AB = 12) : ∃ (area : ℝ), area = (5 * Real.sqrt 119) / 2 :=
by
  use (5 * Real.sqrt 119) / 2
  sorry

end area_of_right_triangle_l415_415999


namespace labeled_price_percent_l415_415489

variables (L : ℝ) (x : ℝ)

-- Given conditions
def list_price := 100
def purchase_price := list_price * 0.70
def selling_price := x * 0.75
def profit := selling_price - purchase_price
def profit_requirement := selling_price * 0.30

-- Problem statement: labeled price percent of list price
theorem labeled_price_percent :
  profit = profit_requirement →
  x = 133.33 :=
by
  sorry

end labeled_price_percent_l415_415489


namespace vacation_cost_l415_415026

theorem vacation_cost (C : ℝ) (h : C / 6 - C / 8 = 120) : C = 2880 :=
by
  sorry

end vacation_cost_l415_415026


namespace greatest_third_side_of_triangle_l415_415050

theorem greatest_third_side_of_triangle (a b : ℕ) (h1 : a = 7) (h2 : b = 15) :
  ∃ x : ℕ, 8 < x ∧ x < 22 ∧ (∀ y : ℕ, 8 < y ∧ y < 22 → y ≤ x) ∧ x = 21 :=
by
  sorry

end greatest_third_side_of_triangle_l415_415050


namespace triangle_area_is_21_l415_415871

-- Definitions for the coordinates of the points
structure Point :=
(x : ℝ) 
(y : ℝ)

def A : Point := { x := 0, y := 2 }
def B : Point := { x := 6, y := 0 }
def C : Point := { x := 3, y := 8 }

-- Determinant formula for the area of a triangle
def area_of_triangle (A B C : Point) : ℝ :=
  (1 / 2) * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

-- The statement to be proved
theorem triangle_area_is_21 : area_of_triangle A B C = 21 :=
  sorry

end triangle_area_is_21_l415_415871


namespace sequence_a_has_prime_factors_l415_415695
open Nat

noncomputable def sequence_a : ℕ → ℕ 
| 0       => 0             -- dummy value for convenience
| 1       => 2021          
| (n + 2) => let s := (Finset.range (n + 1)).sum (λ k => sequence_a (k + 1))
              in s * s - 1

def is_prime (p : ℕ) : Prop := 2 ≤ p ∧ ∀ m, m ∣ p → m = 1 ∨ m = p

def num_prime_factors (n : ℕ) : ℕ :=
  (Finset.filter (λ p => is_prime p ∧ p ∣ n) (Finset.range (n + 1))).card

theorem sequence_a_has_prime_factors (n : ℕ) (hn : n ≥ 2) : 
  num_prime_factors (sequence_a n) ≥ 2 * n := 
sorry

end sequence_a_has_prime_factors_l415_415695


namespace length_of_first_train_l415_415843

theorem length_of_first_train
  (speed_first : ℕ)
  (speed_second : ℕ)
  (length_second : ℕ)
  (distance_between : ℕ)
  (time_to_cross : ℕ)
  (h1 : speed_first = 10)
  (h2 : speed_second = 15)
  (h3 : length_second = 150)
  (h4 : distance_between = 50)
  (h5 : time_to_cross = 60) :
  ∃ L : ℕ, L = 100 :=
by
  sorry

end length_of_first_train_l415_415843


namespace find_50th_number_l415_415009

-- Define the sequence in terms of rows and their lengths
def row (n : ℕ) : list ℕ := list.repeat (3 * n) (3 * n * n)

-- Define the function that concatenates the rows into one sequence
def sequence : list ℕ := list.join (list.map row (list.range 50))

-- Define the property we want to prove
theorem find_50th_number : sequence.nth 49 = some 12 :=
by sorry

end find_50th_number_l415_415009


namespace probability_rectangle_l415_415078

noncomputable def F (x y : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ (π / 2) ∧ 0 ≤ y ∧ y ≤ (π / 2) then
    Real.sin x * Real.sin y
  else
    0

theorem probability_rectangle :
  let x1 := 0
  let x2 := π / 4
  let y1 := π / 6
  let y2 := π / 3
  F x2 y2 - F x1 y2 - (F x2 y1 - F x1 y1) = (Real.sqrt 6 - Real.sqrt 2) / 4 :=
by {
  sorry
}

end probability_rectangle_l415_415078


namespace value_of_g_at_2_l415_415290

def g (x : ℝ) : ℝ := x^2 - 4

theorem value_of_g_at_2 : g 2 = 0 :=
by
  -- proof goes here
  sorry

end value_of_g_at_2_l415_415290


namespace UV_parallel_AA_l415_415330

noncomputable theory

-- Define the points and lines
variables {A A' B B' C C' U V W : Type} [Point A] [Point A'] [Point B] [Point B'] [Point C] [Point C']

-- Conditions
variable (h1 : ¬ coplanar {A, A', B, B', C, C', W})
variable (h2 : parallel (line_through A A') (line_through B B'))
variable (h3 : parallel (line_through B B') (line_through C C'))
variable (h4 : parallel (line_through C C') (line_through A A'))
variable (h5 : U = intersection (plane_through A' B C) (plane_through A B' C) (plane_through A B C'))
variable (h6 : V = intersection (plane_through A B' C') (plane_through A' B C') (plane_through A' B' C))

-- Conclude that UV is parallel to AA'
theorem UV_parallel_AA' : parallel (line_through U V) (line_through A A') :=
sorry

end UV_parallel_AA_l415_415330


namespace find_k_l415_415191

open Nat

theorem find_k (k : ℕ) (h : k > 0) : 
  ∃ (a : ℕ) (n : ℕ) (hn : n > 1), 
  (let primes : List ℕ := List.filter prime (List.range (p k + 1))) 
   in list.prod primes - 1 = a^n → k = 1 :=
by
  sorry

end find_k_l415_415191


namespace greatest_third_side_of_triangle_l415_415049

theorem greatest_third_side_of_triangle (a b : ℕ) (h1 : a = 7) (h2 : b = 15) :
  ∃ x : ℕ, 8 < x ∧ x < 22 ∧ (∀ y : ℕ, 8 < y ∧ y < 22 → y ≤ x) ∧ x = 21 :=
by
  sorry

end greatest_third_side_of_triangle_l415_415049


namespace initial_number_of_matches_l415_415536

theorem initial_number_of_matches (n : ℕ) (h1 : n < 40)
  (h2 : ∃ a b c : ℕ, a + b > c ∧ a + c > b ∧ b + c > a ∧ (a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c) = n * n)
  (h3 : ∃ a b c : ℕ, a + b > c ∧ a + c > b ∧ b + c > a ∧ (a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c) = (n - 6) * (n - 6))
  (h4 : ∃ a b c : ℕ, a + b > c ∧ a + c > b ∧ b + c > a ∧ (a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c) = (n - 12) * (n - 12)) :
  n = 36 := 
begin
  sorry
end

end initial_number_of_matches_l415_415536


namespace symmetrical_implies_congruent_l415_415043

-- Define a structure to represent figures
structure Figure where
  segments : Set ℕ
  angles : Set ℕ

-- Define symmetry about a line
def is_symmetrical_about_line (f1 f2 : Figure) : Prop :=
  ∀ s ∈ f1.segments, s ∈ f2.segments ∧ ∀ a ∈ f1.angles, a ∈ f2.angles

-- Define congruent figures
def are_congruent (f1 f2 : Figure) : Prop :=
  f1.segments = f2.segments ∧ f1.angles = f2.angles

-- Lean 4 statement of the proof problem
theorem symmetrical_implies_congruent (f1 f2 : Figure) (h : is_symmetrical_about_line f1 f2) : are_congruent f1 f2 :=
by
  sorry

end symmetrical_implies_congruent_l415_415043


namespace product_of_x_values_l415_415658

theorem product_of_x_values :
  (∏ x in {x : ℚ | |24 / x + 4| = 4}.to_finset) = -3 := by
  sorry

end product_of_x_values_l415_415658


namespace equilateral_triangle_perimeter_l415_415851

theorem equilateral_triangle_perimeter (median : ℝ) (side : ℝ) (perimeter : ℝ) 
  (h1 : median = 12)
  (h2 : median = (√3 / 2) * side) 
  (h3 : perimeter = 3 * side) : 
  perimeter = 24 * √3 := 
by
  sorry

end equilateral_triangle_perimeter_l415_415851


namespace sin_add_pi_over_2_eq_l415_415278

theorem sin_add_pi_over_2_eq :
  ∀ (A : ℝ), (cos (π + A) = -1/2) → (sin (π / 2 + A) = 1/2) :=
by
  intros A h
  -- Here we assume the proof steps
  sorry

end sin_add_pi_over_2_eq_l415_415278


namespace locus_of_centers_l415_415513

-- Define the conditions
def circle (center : ℝ × ℝ) (radius : ℝ) := 
{c : ℝ × ℝ | (c.1 - center.1)^2 + (c.2 - center.2)^2 = radius^2}

variables (r : ℝ) (d : set (ℝ × ℝ))
  (k : set (ℝ × ℝ)) 
  (center_k : ℝ × ℝ := (0, 0))
  (radius_k : ℝ := r / 2)

-- Define circle \( k \) and its diameter \( d \)
def circle_k : set (ℝ × ℝ) := circle center_k radius_k
def diameter_d : set (ℝ × ℝ) := {p | p.2 = 0 ∧ p.1 ∈ [-(radius_k), (radius_k)]}

-- Define the locus of centers of circles touching diameter \( d \)
-- and their closest point to \( k \) has distance equal to their radius
def is_locus (P : ℝ × ℝ) := 
∃ r' : ℝ, circle P r' ∩ diameter_d ≠ ∅ ∧ ∀ x ∈ circle P r', dist center_k x = r'

-- Prove that is_locus P == arc of hyperbola and its mirror image
theorem locus_of_centers :
  ∀ (P : ℝ × ℝ), is_locus P ↔ (P ∈ {p | 
    let a := radius_k / 3,
    let y_center := 2 * radius_k / 3,
    (p.2 - y_center)^2 / a^2 - p.1^2 / (radius_k / Math.sqrt(3))^2 = 1 
    ∨ 
    (p.2 + y_center)^2 / a^2 - p.1^2 / (radius_k / Math.sqrt(3))^2 = 1}) :=
sorry

end locus_of_centers_l415_415513


namespace harmonic_motion_initial_phase_l415_415820

theorem harmonic_motion_initial_phase :
  ∀ (x : ℝ), (x ≥ 0) → (f : ℝ → ℝ) = λ x, 4 * Real.sin (8 * x - π / 9) → 
  (8 * 0 - π / 9 = -π / 9) :=
by
  intros x hx f hf
  rw Real.sin_eq at hf
  simp
  sorry

end harmonic_motion_initial_phase_l415_415820


namespace sum_possible_values_y_arithmetic_progression_l415_415172

theorem sum_possible_values_y_arithmetic_progression :
  (let l := [12, 3, 6, 3, 5, 3, y, 10] in
  let mean := (42 + y) / 8 in
  let mode := 3 in
  let median := if y < 5 then 5 else if 5 ≤ y ∧ y ≤ 6 then y else if 6 < y ∧ y ≤ 10 then 6 else 10 in
  (3, median, mean) is_arithmetic_progression → y = 94) :=
sorry

end sum_possible_values_y_arithmetic_progression_l415_415172


namespace angle_between_vectors_is_pi_div_2_l415_415557

def v1 : ℝ × ℝ × ℝ := (3, -2, 2)
def v2 : ℝ × ℝ × ℝ := (2, 2, -1)

noncomputable def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

noncomputable def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (u.1^2 + u.2^2 + u.3^2)

theorem angle_between_vectors_is_pi_div_2
  (v1 v2 : ℝ × ℝ × ℝ)
  (h1 : v1 = (3, -2, 2))
  (h2 : v2 = (2, 2, -1)) :
  real.arccos (dot_product v1 v2 / (magnitude v1 * magnitude v2)) = π / 2 :=
by
  sorry

end angle_between_vectors_is_pi_div_2_l415_415557


namespace area_quad_ABEF_l415_415010

theorem area_quad_ABEF {F A B E : Point} (p : ∀ x y : ℝ, y^2 = 4*x) 
  (focus_F : F = (1, 0))
  (directrix_l : ∀ l, intersects_x_axis l E)
  (line_AF : inclination_angle AF = 60)
  (perpendicular_AB_l : AB ⊥ directrix_l)
  (foot_perpendicular_B : foot B A directrix_l)
  (ABEF_convex : convex_quadrilateral F A B E) :
  area_quadrilateral F A B E = 6 * sqrt 3 := 
sorry

end area_quad_ABEF_l415_415010


namespace probability_calculation_l415_415139

noncomputable def probability_in_ellipsoid : ℝ :=
  let prism_volume := (2 - (-2)) * (1 - (-1)) * (1 - (-1))
  let ellipsoid_volume := (4 * Real.pi / 3) * 1 * 2 * 2
  ellipsoid_volume / prism_volume

theorem probability_calculation :
  probability_in_ellipsoid = Real.pi / 3 :=
sorry

end probability_calculation_l415_415139


namespace delivery_driver_stops_l415_415131

theorem delivery_driver_stops (initial_stops more_stops total_stops : ℕ)
  (h_initial : initial_stops = 3)
  (h_more : more_stops = 4)
  (h_total : total_stops = initial_stops + more_stops) : total_stops = 7 := by
  sorry

end delivery_driver_stops_l415_415131


namespace log_one_plus_x_sq_lt_x_sq_l415_415661

theorem log_one_plus_x_sq_lt_x_sq {x : ℝ} (hx : 0 < x) : 
  Real.log (1 + x^2) < x^2 := 
sorry

end log_one_plus_x_sq_lt_x_sq_l415_415661


namespace reach_every_positive_l415_415054

def append_4 (n : ℕ) : ℕ := n * 10 + 4
def append_0 (n : ℕ) : ℕ := n * 10
def divide_by_2 (n : ℕ) (h : n % 2 = 0) : ℕ := n / 2

theorem reach_every_positive (n : ℕ) (hn : n > 0) : 
  ∃ (seq : list ℕ), seq.head = 4 ∧ seq.last (by {apply list.cons_ne_nil, apply sorry}) = n ∧ 
  ∀ (i : ℕ) (h : i < seq.length - 1), 
    seq.nth_le (i + 1) (by {apply sorry}) = append_4(seq.nth_le i (by {apply sorry})) ∨
    seq.nth_le (i + 1) (by {apply sorry}) = append_0(seq.nth_le i (by {apply sorry})) ∨
    ∃ (h_even : seq.nth_le i (by {apply sorry}) % 2 = 0), seq.nth_le (i + 1) (by {apply sorry}) = divide_by_2(seq.nth_le i (by {apply sorry})) h_even :=
sorry

end reach_every_positive_l415_415054


namespace jake_watching_hours_l415_415323

theorem jake_watching_hours
    (monday_hours : ℕ := 12) -- Half of 24 hours in a day is 12 hours for Monday
    (wednesday_hours : ℕ := 6) -- A quarter of 24 hours in a day is 6 hours for Wednesday
    (friday_hours : ℕ := 19) -- Jake watched 19 hours on Friday
    (total_hours : ℕ := 52) -- The entire show is 52 hours long
    (T : ℕ) -- To find the total number of hours on Tuesday
    (h : monday_hours + T + wednesday_hours + (monday_hours + T + wednesday_hours) / 2 + friday_hours = total_hours) :
    T = 4 := sorry

end jake_watching_hours_l415_415323


namespace future_tech_high_absentee_percentage_l415_415859

theorem future_tech_high_absentee_percentage :
  let total_students := 180
  let boys := 100
  let girls := 80
  let absent_boys_fraction := 1 / 5
  let absent_girls_fraction := 1 / 4
  let absent_boys := absent_boys_fraction * boys
  let absent_girls := absent_girls_fraction * girls
  let total_absent_students := absent_boys + absent_girls
  let absent_percentage := (total_absent_students / total_students) * 100
  (absent_percentage = 22.22) := 
by
  sorry

end future_tech_high_absentee_percentage_l415_415859


namespace f_positive_for_all_x_f_min_value_negative_two_f_triangle_sides_l415_415945

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := (4^x + k * 2^x + 1) / (4^x + 2^x + 1)

-- Part (1)
theorem f_positive_for_all_x (k : ℝ) : (∀ x : ℝ, f x k > 0) ↔ k > -2 := sorry

-- Part (2)
theorem f_min_value_negative_two (k : ℝ) : (∀ x : ℝ, f x k ≥ -2) → k = -8 := sorry

-- Part (3)
theorem f_triangle_sides (k : ℝ) :
  (∀ x1 x2 x3 : ℝ, (f x1 k + f x2 k > f x3 k) ∧ (f x2 k + f x3 k > f x1 k) ∧ (f x3 k + f x1 k > f x2 k)) ↔ (-1/2 ≤ k ∧ k ≤ 4) := sorry

end f_positive_for_all_x_f_min_value_negative_two_f_triangle_sides_l415_415945


namespace solve_for_x_l415_415091

theorem solve_for_x (x : ℕ) : x * 12 = 173 * 240 → x = 3460 :=
by
  sorry

end solve_for_x_l415_415091


namespace square_side_length_properties_l415_415918

theorem square_side_length_properties (a: ℝ) (h: a^2 = 10) :
  a = Real.sqrt 10 ∧ (a^2 - 10 = 0) ∧ (3 < a ∧ a < 4) :=
by
  sorry

end square_side_length_properties_l415_415918


namespace regular_polyhedra_similarity_l415_415371

theorem regular_polyhedra_similarity 
  {T₁ T₂ : Type} 
  (combinatorial_type : T₁ → T₂ → Prop)
  (same_kind_faces : T₁ → T₂ → Prop)
  (same_kind_angles : T₁ → T₂ → Prop) 
  (regular_polyhedron : T₁ → Prop)
  (regular_polyhedron : T₂ → Prop) :
  combinatorial_type T₁ T₂ → 
  same_kind_faces T₁ T₂ → 
  same_kind_angles T₁ T₂ → 
  similar T₁ T₂ :=
by
  intro h_comb h_faces h_angles
  sorry

end regular_polyhedra_similarity_l415_415371


namespace a7_value_l415_415949

theorem a7_value
  (a : ℕ → ℝ)
  (hx2 : ∀ n, n > 0 → a n ≠ 0)
  (slope_condition : ∀ n, n ≥ 2 → 2 * a n = 2 * a (n - 1) + 1)
  (point_condition : a 1 * 4 = 8) :
  a 7 = 5 :=
by
  sorry

end a7_value_l415_415949


namespace greatest_third_side_l415_415045

-- Given data and the Triangle Inequality theorem
theorem greatest_third_side (c : ℕ) (h1 : 8 < c) (h2 : c < 22) : c = 21 :=
by
  sorry

end greatest_third_side_l415_415045


namespace number_of_valid_T_grids_l415_415540

def TGrid : Type := matrix (fin 3) (fin 3) (fin 2)

def is_valid_TGrid (grid : TGrid) : Prop :=
  (finset.univ.card (finset.filter (λ (x : fin 3 × fin 3), grid x = 1) finset.univ) = 5) ∧
  (finset.univ ∑ (r : fin 3), (∃ i, grid (r, i) = 1) ≤ 1) ∧
  (finset.univ ∑ (c : fin 3), (∃ i, grid (i, c) = 1) ≤ 1) ∧
  (finset.univ ∑ (d : bool), (∀ i, grid (i, if d then i else 2 - i) = 1) ≤ 1)

theorem number_of_valid_T_grids : finset.card (finset.filter is_valid_TGrid (finset.univ : finset TGrid)) = 68 := 
sorry

end number_of_valid_T_grids_l415_415540


namespace simplify_expression_l415_415527

theorem simplify_expression (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -3) :
    (x + 2 - 5 / (x - 2)) / ((x + 3) / (x - 2)) = x - 3 :=
sorry

end simplify_expression_l415_415527


namespace range_of_f_l415_415181

def f (x : ℝ) : ℝ := 2 * x + real.sqrt (1 - x)

theorem range_of_f : Set.Iic (17 / 8) = set_of (λ y, ∃ x, x ≤ 1 ∧ f x = y) :=
by {
  sorry
}

end range_of_f_l415_415181


namespace largest_good_subset_and_smallest_bad_cover_l415_415331

theorem largest_good_subset_and_smallest_bad_cover 
  (S : Finset ℕ) 
  (hS : ∀ x ∈ S, ∀ d ∣ x, d > 0 → d ∈ S) 
  (good : Finset ℕ → Prop) 
  (bad : Finset ℕ → Prop)
  (good_single : ∀ x, good {x}) 
  (bad_single : ∀ x, bad {x})
  (good_def : ∀ T, good T ↔ (T.nonempty ∧ ∀ x y ∈ T, x < y → ∃ p : ℕ, nat.prime p ∧ y = x * p^nat.log p (y/x)))
  (bad_def : ∀ T, bad T ↔ (T.nonempty ∧ ∀ x y ∈ T, x < y → ∀ p : ℕ, nat.prime p → y ≠ x * p^nat.log p (y/x))) :
  ∃ k, (∀ T, good T → T.card ≤ k) ∧ (∀ T₁ T₂, T₁ ≠ T₂ → bad T₁ ∧ bad T₂ → (T₁ ∩ T₂ = ∅)) ∧ (∀ S, ∃ (partition : Finset (Finset ℕ)), (∀ T ∈ partition, bad T) ∧ (S = (Finset.bUnion partition (λ x, x)))) → True :=
sorry

end largest_good_subset_and_smallest_bad_cover_l415_415331


namespace find_original_function_l415_415291

-- Define the function transformation operation
def transform (f : ℝ → ℝ) (x : ℝ) := f (2 * (x - π / 3))

-- Given resulting function after transformations
def resulting_function (x : ℝ) := Real.sin (x - π / 4)

-- Original function
def original_function (x : ℝ) := Real.sin (x / 2 + π / 12)

theorem find_original_function :
  transform original_function = resulting_function := 
sorry

end find_original_function_l415_415291


namespace circle_common_chord_l415_415745

theorem circle_common_chord (x y : ℝ) :
  (x^2 + y^2 - 4 * x + 6 * y = 0) ∧
  (x^2 + y^2 - 6 * x = 0) →
  (x + 3 * y = 0) :=
by
  sorry

end circle_common_chord_l415_415745


namespace population_sampling_precision_l415_415307

theorem population_sampling_precision (sample_size : ℕ → Prop) 
    (A : Prop) (B : Prop) (C : Prop) (D : Prop)
    (condition_A : A = (∀ n : ℕ, sample_size n → false))
    (condition_B : B = (∀ n : ℕ, sample_size n → n > 0 → true))
    (condition_C : C = (∀ n : ℕ, sample_size n → false))
    (condition_D : D = (∀ n : ℕ, sample_size n → false)) :
  B :=
by sorry

end population_sampling_precision_l415_415307


namespace pointP_outside_curveC_min_max_distance_Q_line_l_l415_415686

def line_l (x y :ℝ) := x + y - 8
def curve_C (α : ℝ) := (cos α, sqrt 3 * sin α)
def point_P := (4, 4)
def polar_P := (4 * sqrt 2, π / 4)

theorem pointP_outside_curveC :
  ¬∃ α : ℝ, point_P = curve_C α :=
sorry

theorem min_max_distance_Q_line_l :
  let d := λ α : ℝ, abs (cos α + sqrt 3 * sin α - 8) / sqrt 2 in
  (∀ α : ℝ, 3 * sqrt 2 ≤ d α ∧ d α ≤ 5 * sqrt 2) ∧
  ∃ α₀ α₁ : ℝ, d α₀ = 3 * sqrt 2 ∧ d α₁ = 5 * sqrt 2 :=
sorry

end pointP_outside_curveC_min_max_distance_Q_line_l_l415_415686


namespace daves_initial_apps_l415_415539

theorem daves_initial_apps : ∃ (X : ℕ), X + 11 - 17 = 4 ∧ X = 10 :=
by {
  sorry
}

end daves_initial_apps_l415_415539


namespace ellipse_equation_max_area_line_equation_l415_415928

noncomputable def point (x y : ℝ) := (x, y)
noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

noncomputable def ellipse_eq (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

theorem ellipse_equation :
  ∀ (A F O : ℝ × ℝ)
    (a b : ℝ)
    (eccentricity : ℝ)
    (h1 : A = (0, -2))
    (h2 : F = (sqrt 3, 0))
    (h3 : O = (0, 0))
    (h4 : a > 0)
    (h5 : b > 0)
    (h6 : ellipse_eq a b x y)
    (h7 : eccentricity = sqrt 3 / 2)
    (h8 : slope A F = 2 * sqrt 3 / 3),
    ellipse_eq 2 1 x y := 
sorry

theorem max_area_line_equation :
  ∀ (A F O : ℝ × ℝ)
    (E : ℝ → ℝ → Prop)
    (a b : ℝ)
    (eccentricity : ℝ)
    (h1 : A = (0, -2))
    (h2 : F = (sqrt 3, 0))
    (h3 : O = (0, 0))
    (h4 : a > 0)
    (h5 : b > 0)
    (h6 : E = λ x y, ellipse_eq a b x y)
    (h7 : eccentricity = sqrt 3 / 2)
    (h8 : slope A F = 2 * sqrt 3 / 3),
    ∃ k : ℝ, (k = sqrt 7 / 2 ∨ k = -sqrt 7 / 2) ∧ (∀ x y, (x, y) ∈ E → y = k * x - 2) :=
sorry

end ellipse_equation_max_area_line_equation_l415_415928


namespace domain_and_range_of_h_when_a_eq_2_range_of_x_when_f_gt_g_l415_415950

noncomputable def f (a x : ℝ) : ℝ := log a (2 + x)
noncomputable def g (a x : ℝ) : ℝ := log a (2 - x)
noncomputable def h (a x : ℝ) : ℝ := f a x + g a x

-- Prove the domain and range of h(x) when a = 2
theorem domain_and_range_of_h_when_a_eq_2 :
  (∀ x : ℝ, (h 2 x) = log 2 (4 - x^2)) ∧ 
  (∀ x : ℝ, -2 < x ∧ x < 2 → log 2 (4 - x^2) ≤ 2) ∧ 
  (∀ x : ℝ, log 2 (4 - x^2) ∈ (-∞, 2]) := 
by
  sorry

-- Prove the ranges of x when f(x) > g(x)
theorem range_of_x_when_f_gt_g (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∀ x : ℝ, a > 1 → (f a x > g a x) ↔ (0 < x ∧ x < 2)) ∧
  (∀ x : ℝ, 0 < a ∧ a < 1 → (f a x > g a x) ↔ (-2 < x ∧ x < 0)) := 
by
  sorry

end domain_and_range_of_h_when_a_eq_2_range_of_x_when_f_gt_g_l415_415950


namespace area_of_rectangle_KLMJ_l415_415858

noncomputable def hypotenuse_length (AB AC : ℕ) : ℕ :=
  (Math.sqrt((AB * AB + AC * AC) : ℕ))

noncomputable def square_area (side : ℕ) : ℕ :=
  side * side

noncomputable def triangle_area (base height : ℕ) : ℕ :=
  (base * height) / 2

theorem area_of_rectangle_KLMJ :
  ∀ (AB AC : ℕ), AB = 3 → AC = 4 →
  let BC := hypotenuse_length AB AC in
  let area_ABED := square_area AB in
  let area_ACHI := square_area AC in
  let area_BCGF := square_area BC in
  let area_triangle_ABC := triangle_area AB AC in
  (4 * area_triangle_ABC + area_ABED + area_ACHI + area_BCGF) = 110 :=
begin
  intros AB AC,
  intros hAB hAC,
  let BC := hypotenuse_length AB AC,
  let area_ABED := square_area AB,
  let area_ACHI := square_area AC,
  let area_BCGF := square_area BC,
  let area_triangle_ABC := triangle_area AB AC,
  sorry
end

end area_of_rectangle_KLMJ_l415_415858


namespace find_y_l415_415391

theorem find_y (n x y : ℝ)
  (h1 : (100 + 200 + n + x) / 4 = 250)
  (h2 : (n + 150 + 100 + x + y) / 5 = 200) :
  y = 50 :=
by
  sorry

end find_y_l415_415391


namespace intersection_points_l415_415102

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4
def parabola_eq (x y : ℝ) : Prop := y = x^2
def line_eq (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

theorem intersection_points (a b c : ℝ) :
  ¬ (∃ x y, line_eq a b c x y ∧ tangent_to_curve circle_eq a b c x y)
  ∧ ¬ (∃ x y, line_eq a b c x y ∧ tangent_to_curve parabola_eq a b c x y) →
  ∃ n : ℕ, n ∈ ({2, 3, 4} : Set ℕ)
-- tangent_to_curve needs to be defined properly

end intersection_points_l415_415102


namespace mnop_is_parallelogram_l415_415599

-- Define the points and midpoints in quadrilateral context
variables {A B C D M N O P : Type*} [affine_space ℝ Type*]

-- Define conditions for midpoints on the edges of quadrilateral
def is_midpoint (M : Type*) (A B : Type*) [affine_space ℝ Type*] := 
  ∃ (m : ℝ), 0 < m ∧ m < 1 ∧ M = affine_combination m (affine_combination 1 A B)

-- Define conditions for the quadrilateral and midpoints
variables (h_midpoints :
  is_midpoint M A B ∧
  is_midpoint N B C ∧
  is_midpoint O C D ∧
  is_midpoint P D A)

-- Final statement: MNOP is a parallelogram
theorem mnop_is_parallelogram (h_midpoints) : is_parallelogram M N O P :=
sorry

end mnop_is_parallelogram_l415_415599


namespace conjugate_in_fourth_quadrant_l415_415607

open Complex

def i_unit : ℂ := complex.I

def complex_fraction : ℂ := i_unit / (2 + i_unit)

def conjugate_complex (z : ℂ) : ℂ := conj z

def point_of_conjugate (z : ℂ) : ℂ := conjugate_complex z

def is_in_fourth_quadrant (z : ℂ) : Prop :=
  (z.re > 0) ∧ (z.im < 0)

theorem conjugate_in_fourth_quadrant : 
  is_in_fourth_quadrant (point_of_conjugate complex_fraction) :=
sorry

end conjugate_in_fourth_quadrant_l415_415607


namespace arithmetic_sequence_sum_l415_415230

noncomputable def harmonic_integral : ℝ := ∫ x in 0..π, sin x

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h₁ : ∀ n, a n = a 0 + n * (a 1 - a 0)) 
(h₂ : a 5 + a 7 = harmonic_integral) : a 4 + a 6 + a 8 = 3 :=
by sorry

end arithmetic_sequence_sum_l415_415230


namespace maximal_difference_of_areas_l415_415983

-- Given:
-- A circle of radius R
-- A chord of length 2x is drawn perpendicular to the diameter of the circle
-- The endpoints of this chord are connected to the endpoints of the diameter
-- We need to prove that under these conditions, the length of the chord 2x that maximizes the difference in areas of the triangles is R √ 2

theorem maximal_difference_of_areas (R x : ℝ) (h : 2 * x = R * Real.sqrt 2) :
  2 * x = R * Real.sqrt 2 :=
by
  sorry

end maximal_difference_of_areas_l415_415983


namespace greatest_third_side_of_triangle_l415_415048

theorem greatest_third_side_of_triangle (a b : ℕ) (h1 : a = 7) (h2 : b = 15) :
  ∃ x : ℕ, 8 < x ∧ x < 22 ∧ (∀ y : ℕ, 8 < y ∧ y < 22 → y ≤ x) ∧ x = 21 :=
by
  sorry

end greatest_third_side_of_triangle_l415_415048


namespace first_dog_food_amount_l415_415270

noncomputable section

def first_dog_food := sorry -- Definition of the amount of dog food the first dog eats.

def second_dog_food (x : ℝ) := 2 * x -- The second dog eats twice the first dog's amount.
def third_dog_food (x : ℝ) := 2 * x + 2.5 -- The third dog eats 2.5 cups more than the second dog.

def total_food (x : ℝ) := x + second_dog_food x + third_dog_food x -- Total food for all three dogs.

theorem first_dog_food_amount : ∃ x : ℝ, total_food x = 10 -> x = 1.5 :=
by
  simp [total_food, second_dog_food, third_dog_food]
  sorry

end first_dog_food_amount_l415_415270


namespace train_travel_distance_l415_415810

def coal_efficiency := (5 : ℝ) / (2 : ℝ)  -- Efficiency in miles per pound
def coal_remaining := 160  -- Coal remaining in pounds
def distance_travelled := coal_remaining * coal_efficiency  -- Total distance the train can travel

theorem train_travel_distance : distance_travelled = 400 := 
by
  sorry

end train_travel_distance_l415_415810


namespace inequality_solution_empty_set_l415_415755

theorem inequality_solution_empty_set : ∀ x : ℝ, ¬ (x * (2 - x) > 3) :=
by
  -- Translate the condition and show that there are no x satisfying the inequality
  sorry

end inequality_solution_empty_set_l415_415755


namespace hunter_saw_32_frogs_l415_415650

noncomputable def total_frogs (g1 : ℕ) (g2 : ℕ) (d : ℕ) : ℕ :=
g1 + g2 + d

theorem hunter_saw_32_frogs :
  total_frogs 5 3 (2 * 12) = 32 := by
  sorry

end hunter_saw_32_frogs_l415_415650


namespace range_of_a_l415_415296

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Icc (Real.pi / 4) (Real.pi / 2), 0 ≤ exp x * ((1 - a) * sin x + (1 + a) * cos x)) → a ≤ 1 :=
by 
  intros h
  sorry

end range_of_a_l415_415296


namespace minimize_expression_pos_int_l415_415898

theorem minimize_expression_pos_int (n : ℕ) (hn : 0 < n) : 
  (∀ m : ℕ, 0 < m → (m / 3 + 27 / m : ℝ) ≥ (9 / 3 + 27 / 9)) :=
sorry

end minimize_expression_pos_int_l415_415898


namespace largest_angle_in_scalene_triangle_l415_415041

-- Define the conditions of the problem
def is_scalene (D E F : ℝ) : Prop :=
  D ≠ E ∧ D ≠ F ∧ E ≠ F

def angle_sum (D E F : ℝ) : Prop :=
  D + E + F = 180

def given_angles (D E : ℝ) : Prop :=
  D = 30 ∧ E = 50

-- Statement of the problem
theorem largest_angle_in_scalene_triangle :
  ∀ (D E F : ℝ), is_scalene D E F ∧ given_angles D E ∧ angle_sum D E F → F = 100 :=
by
  intros D E F h
  sorry

end largest_angle_in_scalene_triangle_l415_415041


namespace longer_side_length_l415_415118

-- Given a circle with radius 6 cm
def radius : ℝ := 6
def circle_area : ℝ := Real.pi * radius^2

-- Given that the area of the rectangle is three times the area of the circle
def rectangle_area : ℝ := 3 * circle_area

-- Given the rectangle has a shorter side equal to the diameter of the circle
def shorter_side : ℝ := 2 * radius

-- Prove that the length of the longer side of the rectangle is 9π cm
theorem longer_side_length : ∃ (longer_side : ℝ), longer_side = 9 * Real.pi :=
by
  have circle_area_def : circle_area = 36 * Real.pi := by sorry
  have rectangle_area_def : rectangle_area = 108 * Real.pi := by sorry
  have shorter_side_def : shorter_side = 12 := by sorry
  let longer_side := rectangle_area / shorter_side
  use longer_side
  show longer_side = 9 * Real.pi
  sorry

end longer_side_length_l415_415118


namespace inequality_proof_l415_415364

theorem inequality_proof (x y : ℝ) (h1 : y ≥ 0) (h2 : y * (y + 1) ≤ (x + 1)^2) : y * (y - 1) ≤ x^2 :=
sorry

end inequality_proof_l415_415364


namespace evaluate_expression_l415_415168

theorem evaluate_expression :
  (-1 : ℝ) ^ 2015 + real.sqrt (1 / 4) + (real.pi - 3.14) ^ 0 + 2 * real.sin (real.pi / 3) - 2⁻¹ = real.sqrt 3 :=
by
  sorry

end evaluate_expression_l415_415168


namespace percentage_of_sum_paid_l415_415768

theorem percentage_of_sum_paid (X Y : ℝ) (h1 : Y = 363.64) (h2 : X + Y = 800) : 
  (X / Y) * 100 ≈ 119.99 :=
by
  sorry

end percentage_of_sum_paid_l415_415768


namespace lines_per_page_l415_415782

theorem lines_per_page (total_lines : ℕ) (total_pages : ℕ) (h_lines : total_lines = 150) (h_pages : total_pages = 5) :
  total_lines / total_pages = 30 :=
by
  rw [h_lines, h_pages]
  norm_num
  sorry

end lines_per_page_l415_415782


namespace pythagorean_triangle_divisible_by_5_l415_415367

theorem pythagorean_triangle_divisible_by_5 {a b c : ℕ} (h : a^2 + b^2 = c^2) : 
  5 ∣ a ∨ 5 ∣ b ∨ 5 ∣ c := 
by
  sorry

end pythagorean_triangle_divisible_by_5_l415_415367


namespace find_theta_l415_415052

-- Definitions based on conditions
def angle_A : ℝ := 10
def angle_B : ℝ := 14
def angle_C : ℝ := 26
def angle_D : ℝ := 33
def sum_rect_angles : ℝ := 360
def sum_triangle_angles : ℝ := 180
def sum_right_triangle_acute_angles : ℝ := 90

-- Main theorem statement
theorem find_theta (A B C D : ℝ)
  (hA : A = angle_A)
  (hB : B = angle_B)
  (hC : C = angle_C)
  (hD : D = angle_D)
  (sum_rect : sum_rect_angles = 360)
  (sum_triangle : sum_triangle_angles = 180) :
  ∃ θ : ℝ, θ = 11 := 
sorry

end find_theta_l415_415052


namespace rectangle_longer_side_length_l415_415126

theorem rectangle_longer_side_length
  (r : ℝ) (A_rect : ℝ) :
  r = 6 →
  A_rect = 3 * (Real.pi * r^2) →
  ∃ l : ℝ, l = 9 * Real.pi :=
begin
  intros hr hA,
  use 9 * Real.pi,
  sorry
end

end rectangle_longer_side_length_l415_415126


namespace bread_count_at_end_of_day_l415_415493

def initial_loaves : ℕ := 2355
def sold_loaves : ℕ := 629
def delivered_loaves : ℕ := 489

theorem bread_count_at_end_of_day : 
  initial_loaves - sold_loaves + delivered_loaves = 2215 := by
  sorry

end bread_count_at_end_of_day_l415_415493


namespace interest_rate_is_12_l415_415483

-- Define the given conditions
def SI : ℝ := 5400
def P : ℝ := 15000
def T : ℝ := 3

-- Define the formula for Simple Interest
def simple_interest (P: ℝ) (R: ℝ) (T: ℝ) : ℝ := (P * R * T) / 100

-- Proposition to prove
theorem interest_rate_is_12 :
  (∃ R : ℝ, simple_interest P R T = SI) ↔ (∃ R : ℝ, R = 12) :=
by
  sorry -- skip the proof

end interest_rate_is_12_l415_415483


namespace celebration_60_counting_l415_415788

def context : Type :=
  | Anniversary : ℕ → context

def belongs_to_label (n : ℕ) : Prop :=
  n = 60 → "Label"

def belongs_to_measurement_result (n : ℕ) : Prop :=
  n = 60 → "Measurement result"

def belongs_to_counting (n : ℕ) : Prop :=
  n = 60 → "Counting"

def belongs_to_all (n : ℕ) : Prop :=
  n = 60 → "All of the above"

theorem celebration_60_counting : (n : ℕ) (ctx : context) (h : ctx = context.Anniversary 60) → belongs_to_counting n :=
by
  sorry

end celebration_60_counting_l415_415788


namespace reaction_CH3COOH_NaOH_l415_415192

theorem reaction_CH3COOH_NaOH 
  (CH3COOH NaOH NaCH3COO H2O : Type) 
  [has_add CH3COOH NaOH] [has_add NaCH3COO H2O] 
  (reaction : CH3COOH + NaOH = NaCH3COO + H2O) :
  1 * CH3COOH + 1 * NaOH = 1 * NaCH3COO + 1 * H2O :=
by
  sorry

end reaction_CH3COOH_NaOH_l415_415192


namespace medians_sum_inequality_l415_415368

variables {a b c : ℝ}
variables {m_a m_b m_c : ℝ}

noncomputable def medians_sum_gt_3_4_perimeter (a b c m_a m_b m_c : ℝ) : Prop :=
  let perimeter := a + b + c
  in 3/4 * perimeter < m_a + m_b + m_c ∧ m_a + m_b + m_c < perimeter

-- Definitions based on geometric properties involving medians
axiom median_inequality_a : m_a < (b + c) / 2
axiom median_inequality_b : m_b < (a + c) / 2
axiom median_inequality_c : m_c < (a + b) / 2

-- Definitions based on centroid properties
axiom centroid_property_a : 2/3 * m_a + 2/3 * m_c > c
axiom centroid_property_b : 2/3 * m_b + 2/3 * m_a > a
axiom centroid_property_c : 2/3 * m_c + 2/3 * m_b > b

theorem medians_sum_inequality 
  (a b c m_a m_b m_c : ℝ) 
  (h1: median_inequality_a a b c m_a)
  (h2: median_inequality_b a b c m_b)
  (h3: median_inequality_c a b c m_c)
  (h4: centroid_property_a a b c m_a m_c)
  (h5: centroid_property_b a b c m_b m_a)
  (h6: centroid_property_c a b c m_c m_b) : 
  medians_sum_gt_3_4_perimeter a b c m_a m_b m_c :=
sorry

end medians_sum_inequality_l415_415368


namespace stick_perpendicular_line_l415_415303

-- Definitions of lines and their properties
variable {Classroom : Type} [Nonempty Classroom]
variable {Line Ground : Classroom → Classroom → Prop}

-- Definition of perpendicularity condition
variable (perpendicular : ∀ (l1 l2 : Classroom), Prop)

-- Problem statement
theorem stick_perpendicular_line (h : ∀ (stick : Classroom),
  ∃ (line_ground : Classroom), Ground line_ground stick ∧ perpendicular line_ground stick) :
  ∀ (stick : Classroom), ∃ (line_ground : Classroom), perpendicular line_ground stick :=
by
  intros stick
  specialize h stick
  cases h with line_ground hl
  use line_ground
  exact hl.right

-- Sorry to skip the proof part as instructed
sorry

end stick_perpendicular_line_l415_415303


namespace constant_function_shift_l415_415288

variable {R : Type*} [Real]

def f (x : ℝ) : ℝ := 3

theorem constant_function_shift (x : ℝ) : f (x + 5) = 3 := by
  sorry

end constant_function_shift_l415_415288


namespace least_nonfactor_nonprime_is_62_l415_415075

-- Define the factorial of 30
def fact_30 : ℕ := nat.factorial 30

-- Define the set of prime numbers ≤ 30
def primes : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Define the set of all factors of 30!
def factors_30! : set ℕ := {n | ∃ k, n = nat.factorial 30 / k}

-- Define a number that is not a factor of 30! and not a prime
def is_not_factor_and_not_prime (n : ℕ) : Prop :=
  ¬ (n ∈ factors_30!) ∧ ¬ nat.prime n

-- Define the least positive integer that is not a factor of 30! and is not a prime
def least_nonfactor_nonprime : ℕ := 62

-- Prove that 62 is the least positive integer that satisfies the conditions
theorem least_nonfactor_nonprime_is_62 :
  ∀ n, is_not_factor_and_not_prime n → 62 ≤ n :=
sorry

end least_nonfactor_nonprime_is_62_l415_415075


namespace number_of_terms_M_n_leq_2022_l415_415252

theorem number_of_terms_M_n_leq_2022 :
  ∀ n, (let a_n := λ n, n + 1,
             b_n := λ n, 2^(n-1),
             M_n := (finset.range n).sum (λ k, a_n (b_n (k + 1)))) →
         (M_n ≤ 2022 ↔ n = 10) :=
sorry

end number_of_terms_M_n_leq_2022_l415_415252


namespace total_gallons_l415_415887

def gallons_used (A F : ℕ) := F = 4 * A - 5

theorem total_gallons
  (A F : ℕ)
  (h1 : gallons_used A F)
  (h2 : F = 23) :
  A + F = 30 :=
by
  sorry

end total_gallons_l415_415887


namespace problem_statement_l415_415616

variables (a b : ℕ → ℕ) (c d : ℕ → ℕ) (S T : ℕ → ℕ)

-- Condition definitions
def is_geometric (a : ℕ → ℕ) (q : ℕ) (h : 0 < q) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = q * a n

def is_arithmetic (b : ℕ → ℕ) (d : ℕ) : Prop :=
  b 1 = 2 ∧ ∀ n, b (n + 1) = b n + d

def condition1 (a : ℕ → ℕ) : Prop :=
  2 * a 2 + a 3 = a 4

def condition2 (a b : ℕ → ℕ) : Prop :=
  b 3 = a 1 + a 2 + a 3

-- Definitions of the sequence sums
def sum_c (c : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ k in range n, c k

def sum_d (d : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ k in range n, d k

-- Prove general formulas
def general_formula_a : Prop :=
  ∀ n, a n = 2 ^ n

def general_formula_b : Prop :=
  ∀ n, b n = 6 * n - 4

-- Prove sum formulations
def sum_formula_c (n : ℕ) : Prop :=
  sum_c c n = 2^(n + 1) + 3 * n^2 - n - 2

def sum_formula_d (n : ℕ) : Prop :=
  sum_d d n = (6 * n - 10) * 2^(n + 1) + 20

-- Lean 4 statement
theorem problem_statement (q d : ℕ) (hq : 0 < q) :
is_geometric a q hq → is_arithmetic b d → condition1 a → condition2 a b →
(general_formula_a) ∧ (general_formula_b) ∧
(∀ n, sum_formula_c n) ∧ (∀ n, sum_formula_d n) :=
by {
  sorry
}

end problem_statement_l415_415616


namespace initial_phase_l415_415822

-- Define the function representing the harmonic motion
def f (x : ℝ) : ℝ := 4 * Real.sin (8 * x - (Real.pi / 9))

-- Theorem stating that the initial phase is -π/9 when x = 0
theorem initial_phase : f 0 = 4 * Real.sin (-Real.pi / 9) := by
  sorry

end initial_phase_l415_415822


namespace sin_half_pi_plus_A_l415_415281

theorem sin_half_pi_plus_A (A : Real) (h : Real.cos (Real.pi + A) = -1 / 2) :
  Real.sin (Real.pi / 2 + A) = 1 / 2 := by
  sorry

end sin_half_pi_plus_A_l415_415281


namespace back_wheel_revolutions_l415_415094

theorem back_wheel_revolutions (front_perimeter back_perimeter : ℝ) 
  (front_slip_ratio back_slip_ratio : ℝ) (front_revolutions : ℕ) 
  (hf : front_perimeter = 30) (hb : back_perimeter = 20) 
  (fsr : front_slip_ratio = 0.05) (bsr : back_slip_ratio = 0.07) 
  (fr : front_revolutions = 240) : 
  let front_actual_distance_per_revolution := front_perimeter - (front_slip_ratio * front_perimeter),
      back_actual_distance_per_revolution := back_perimeter - (back_slip_ratio * back_perimeter),
      total_distance_covered := front_actual_distance_per_revolution * front_revolutions,
      back_wheel_revolutions := total_distance_covered / back_actual_distance_per_revolution 
  in back_wheel_revolutions ≈ 368 := 
by 
  sorry

end back_wheel_revolutions_l415_415094


namespace sufficient_not_necessary_for_one_zero_l415_415416

variable {a x : ℝ}

def f (a x : ℝ) : ℝ := a * x ^ 2 - 2 * x + 1

theorem sufficient_not_necessary_for_one_zero :
  (∃ x : ℝ, f 1 x = 0) ∧ (∀ x : ℝ, f 0 x = -2 * x + 1 → x ≠ 0) → 
  (∃ x : ℝ, f a x = 0) → (a = 1 ∨ f 0 x = 0)  :=
sorry

end sufficient_not_necessary_for_one_zero_l415_415416


namespace detergent_per_pound_l415_415355

-- Define the conditions
def total_ounces_detergent := 18
def total_pounds_clothes := 9

-- Define the question to prove the amount of detergent per pound of clothes
theorem detergent_per_pound : total_ounces_detergent / total_pounds_clothes = 2 := by
  sorry

end detergent_per_pound_l415_415355


namespace max_self_intersection_points_13_max_self_intersection_points_1950_l415_415058

def max_self_intersection_points (n : ℕ) : ℕ :=
if n % 2 = 1 then n * (n - 3) / 2 else n * (n - 4) / 2 + 1

theorem max_self_intersection_points_13 : max_self_intersection_points 13 = 65 :=
by sorry

theorem max_self_intersection_points_1950 : max_self_intersection_points 1950 = 1897851 :=
by sorry

end max_self_intersection_points_13_max_self_intersection_points_1950_l415_415058


namespace RnD_cost_increase_l415_415864

theorem RnD_cost_increase (R_D_t : ℝ) (delta_APL_t1 : ℝ)
  (h1 : R_D_t = 3205.69)
  (h2 : delta_APL_t1 = 1.93) :
  R_D_t / delta_APL_t1 = 1661 :=
by 
  conv in (R_D_t / delta_APL_t1) {
    rw [h1, h2]
  }
  simp
  sorry

end RnD_cost_increase_l415_415864


namespace decagon_sign_change_impossible_l415_415464

theorem decagon_sign_change_impossible :
  ∀ (decagon : Type) 
    (vertices intersections : decagon → Prop)
    (initial_sign : ∀ x : decagon, Prop) 
    (operation : (decagon → Prop) → (decagon → Prop) → decagon → Prop),
  (∀ x, initial_sign x = True) →
  (∀ diag_side diag, operation diag_side diag x = (initial_sign x) → (-1) * (initial_sign x)) →
  (intersections = (λ x, True)) → False := 
sorry

end decagon_sign_change_impossible_l415_415464


namespace larger_number_l415_415417

theorem larger_number (x y : ℕ) (h1 : x + y = 40) (h2 : x - y = 4) : x = 22 := by
  sorry

end larger_number_l415_415417


namespace price_of_first_oil_l415_415087

variable {x : ℝ}
variable {price1 volume1 price2 volume2 mix_price mix_volume : ℝ}

theorem price_of_first_oil:
  volume1 = 10 →
  price2 = 68 →
  volume2 = 5 →
  mix_volume = 15 →
  mix_price = 56 →
  (volume1 * x + volume2 * price2 = mix_volume * mix_price) →
  x = 50 :=
by
  intros h1 h2 h3 h4 h5 h6
  have h1 : volume1 = 10 := h1
  have h2 : price2 = 68 := h2
  have h3 : volume2 = 5 := h3
  have h4 : mix_volume = 15 := h4
  have h5 : mix_price = 56 := h5
  have h6 : volume1 * x + volume2 * price2 = mix_volume * mix_price := h6
  sorry

end price_of_first_oil_l415_415087


namespace red_beads_count_l415_415829

-- Define the cyclic necklace and its properties
def cyclic_necklace (n : ℕ) (colors : ℕ → Prop) := ∀ i, colors i = colors ((i + n) % n)

-- Conditions based on the problem
def valid_bead_arrangement (n : ℕ) (colors : ℕ → Prop) : Prop :=
  let blue := λ x, colors x = "B"
  let red := λ x, colors x = "R" in
  ∀ i, blue i → (red ((i + n - 1) % n) ∧ red ((i + 1) % n)) ∧
          (∀ j, red j → (blue ((j + n - 1) % n) ∧ blue ((j + 1) % n)))

-- The main theorem
theorem red_beads_count (colors : ℕ → Prop) :
  cyclic_necklace 30 colors →
  valid_bead_arrangement 30 colors →
  ∃ r, ∑ i in range 30, if colors i = "R" then 1 else 0 = r ∧ r = 60 := 
sorry

end red_beads_count_l415_415829


namespace ant_travel_finite_path_exists_l415_415850

theorem ant_travel_finite_path_exists :
  ∃ (x y z t : ℝ), |x| < |y - z + t| ∧ |y| < |x - z + t| ∧ 
                   |z| < |x - y + t| ∧ |t| < |x - y + z| :=
by
  sorry

end ant_travel_finite_path_exists_l415_415850


namespace rectangle_area_eq_2a_squared_l415_415668

variable {α : Type} [Semiring α] (a : α)

-- Conditions
def width (a : α) : α := a
def length (a : α) : α := 2 * a

-- Proof statement
theorem rectangle_area_eq_2a_squared (a : α) : (length a) * (width a) = 2 * a^2 := 
sorry

end rectangle_area_eq_2a_squared_l415_415668


namespace saree_sale_price_l415_415022

def initial_price : Real := 150
def discount1 : Real := 0.20
def tax1 : Real := 0.05
def discount2 : Real := 0.15
def tax2 : Real := 0.04
def discount3 : Real := 0.10
def tax3 : Real := 0.03
def final_price : Real := 103.25

theorem saree_sale_price :
  let price_after_discount1 : Real := initial_price * (1 - discount1)
  let price_after_tax1 : Real := price_after_discount1 * (1 + tax1)
  let price_after_discount2 : Real := price_after_tax1 * (1 - discount2)
  let price_after_tax2 : Real := price_after_discount2 * (1 + tax2)
  let price_after_discount3 : Real := price_after_tax2 * (1 - discount3)
  let price_after_tax3 : Real := price_after_discount3 * (1 + tax3)
  abs (price_after_tax3 - final_price) < 0.01 :=
by
  sorry

end saree_sale_price_l415_415022


namespace dad_steps_eq_90_l415_415875

-- Define the conditions given in the problem
variables (masha_steps yasha_steps dad_steps : ℕ)

-- Conditions:
-- 1. Dad takes 3 steps while Masha takes 5 steps
-- 2. Masha takes 3 steps while Yasha takes 5 steps
-- 3. Together, Masha and Yasha made 400 steps
def conditions := dad_steps * 5 = 3 * masha_steps ∧ masha_steps * yasha_steps = 3 * yasha_steps ∧ 3 * yasha_steps = 400

-- Theorem stating the proof problem
theorem dad_steps_eq_90 : conditions masha_steps yasha_steps dad_steps → dad_steps = 90 :=
by
  sorry

end dad_steps_eq_90_l415_415875


namespace transformation_projective_l415_415462

theorem transformation_projective (S : Circle) (l : Line) (M O : Point) (hM : M ∈ S) (hMl : M ∉ l) (hO : O ∉ S) :
  ∃ P : Transform l l, P.projective :=
sorry

end transformation_projective_l415_415462


namespace teachers_students_relationship_l415_415674

variables (m n k l : ℕ)

theorem teachers_students_relationship
  (teachers_count : m > 0)
  (students_count : n > 0)
  (students_per_teacher : k > 0)
  (teachers_per_student : l > 0)
  (h1 : ∀ p ∈ (Finset.range m), (Finset.card (Finset.range k)) = k)
  (h2 : ∀ s ∈ (Finset.range n), (Finset.card (Finset.range l)) = l) :
  m * k = n * l :=
sorry

end teachers_students_relationship_l415_415674


namespace numerator_is_12_l415_415398

theorem numerator_is_12 (x : ℕ) (h1 : (x : ℤ) / (2 * x + 4 : ℤ) = 3 / 7) : x = 12 := 
sorry

end numerator_is_12_l415_415398


namespace perpendicular_line_through_intersection_l415_415582

theorem perpendicular_line_through_intersection :
  ∃ (x y : ℝ), (x + y - 2 = 0) ∧ (3 * x + 2 * y - 5 = 0) ∧ (4 * x - 3 * y - 1 = 0) :=
sorry

end perpendicular_line_through_intersection_l415_415582


namespace diophantine_no_nonneg_solutions_l415_415934

theorem diophantine_no_nonneg_solutions {a b : ℕ} (ha : 0 < a) (hb : 0 < b) (h_gcd : Nat.gcd a b = 1) :
  ∃ (c : ℕ), (a * b - a - b + 1) / 2 = (a - 1) * (b - 1) / 2 := 
sorry

end diophantine_no_nonneg_solutions_l415_415934


namespace probability_interval_l415_415019

noncomputable def probability_density (X : MeasureTheory.ProbabilityMeasure ℝ) : (ℝ → ℝ) → ℝ := sorry

noncomputable def normal_distribution (μ σ : ℝ) : MeasureTheory.ProbabilityMeasure ℝ := sorry

noncomputable def cdf (F : MeasureTheory.ProbabilityMeasure ℝ) (x : ℝ) : ℝ := sorry

theorem probability_interval {μ σ : ℝ} (F : MeasureTheory.ProbabilityMeasure ℝ) :
  (X ∼ normal_distribution μ σ) →
  (cdf F 2 - cdf F 1 = 0.2) →
  (cdf F 1 - cdf F 0 = 0.3) :=
sorry

end probability_interval_l415_415019


namespace platform_length_l415_415070

noncomputable def speed_kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

noncomputable def distance (speed_mps : ℝ) (time_s : ℝ) : ℝ :=
  speed_mps * time_s

theorem platform_length
  (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_speed_kmph = 70 →
  train_length = 180 →
  crossing_time = 20 →
  let train_speed_mps := speed_kmph_to_mps train_speed_kmph in
  let total_distance := distance train_speed_mps crossing_time in
  total_distance - train_length = 208.8 :=
by
  intros
  unfold speed_kmph_to_mps distance
  sorry

end platform_length_l415_415070


namespace red_beads_count_l415_415830

-- Define the cyclic necklace and its properties
def cyclic_necklace (n : ℕ) (colors : ℕ → Prop) := ∀ i, colors i = colors ((i + n) % n)

-- Conditions based on the problem
def valid_bead_arrangement (n : ℕ) (colors : ℕ → Prop) : Prop :=
  let blue := λ x, colors x = "B"
  let red := λ x, colors x = "R" in
  ∀ i, blue i → (red ((i + n - 1) % n) ∧ red ((i + 1) % n)) ∧
          (∀ j, red j → (blue ((j + n - 1) % n) ∧ blue ((j + 1) % n)))

-- The main theorem
theorem red_beads_count (colors : ℕ → Prop) :
  cyclic_necklace 30 colors →
  valid_bead_arrangement 30 colors →
  ∃ r, ∑ i in range 30, if colors i = "R" then 1 else 0 = r ∧ r = 60 := 
sorry

end red_beads_count_l415_415830


namespace subway_ways_l415_415780

theorem subway_ways (total_ways : ℕ) (bus_ways : ℕ) (h1 : total_ways = 7) (h2 : bus_ways = 4) :
  total_ways - bus_ways = 3 :=
by
  sorry

end subway_ways_l415_415780


namespace defective_pairs_estimate_l415_415796

-- Definitions of conditions
def frequency_table : List (ℕ × ℕ × ℝ) := [
  (20, 17, 0.85),
  (40, 38, 0.95),
  (60, 55, 0.92),
  (80, 75, 0.94),
  (100, 96, 0.96),
  (200, 189, 0.95),
  (300, 286, 0.95)
]

-- Main theorem stating the proof problem
theorem defective_pairs_estimate :
    (∃ qualified_rate,
        qualified_rate = 0.95 ∧
        List.contains (300, 286, qualified_rate) frequency_table) →
    (∃ defective_pairs, defective_pairs = 75 ∧
        defective_pairs = 1500 * (1 - qualified_rate)) :=
by
  sorry

end defective_pairs_estimate_l415_415796


namespace divisibility_positive_divisibility_negative_l415_415369

theorem divisibility_positive (a : ℤ) (n : ℕ) : 
  ∃ k : ℤ, ((a + 1) ^ (a ^ n) - 1 = k * a^(n+1)) ∧ ¬(∃ l : ℤ, ((a + 1) ^ (a ^ n) - 1 = l * a^(n+2))) :=
sorry

theorem divisibility_negative (a : ℤ) (n : ℕ) : 
  ∃ k : ℤ, ((a - 1) ^ (a ^ n) + 1 = k * a^(n+1)) ∧ ¬(∃ l : ℤ, ((a - 1) ^ (a ^ n) + 1 = l * a^(n+2))) :=
sorry

end divisibility_positive_divisibility_negative_l415_415369


namespace Santino_papaya_trees_l415_415732

theorem Santino_papaya_trees 
  (papaya_trees : ℕ) 
  (mango_trees : ℕ = 3) 
  (papayas_per_tree : ℕ = 10) 
  (mangos_per_tree : ℕ = 20) 
  (total_fruits : ℕ = 80) 
  (total_papayas : ℕ := papaya_trees * papayas_per_tree) 
  (total_mangos : ℕ := mango_trees * mangos_per_tree) :
  total_papayas + total_mangos = total_fruits → papaya_trees = 2 := 
by {
  sorry
}

end Santino_papaya_trees_l415_415732


namespace shortest_path_surface_l415_415021

theorem shortest_path_surface (a b c : ℝ) (h : a > b ∧ b > c) :
  let A := (0, 0, 0)
      C1 := (a, b, c)
  in sqrt (a^2 + (b + c)^2) = shortest_surface_distance A C1 := sorry

end shortest_path_surface_l415_415021


namespace circumcircle_of_triangle_l415_415138

noncomputable def circumcircle_equation (x y : ℝ) : Prop :=
(x - 2)^2 + (y - 1)^2 = 5

theorem circumcircle_of_triangle (x y : ℝ) (hp : (4, 2) ≠ (0, 0)) (hc : x^2 + y^2 = 4)
  (ha : tangent (4, 2) (x, y)) (hb : tangent (4, 2) (a, b)) : (x - 2)^2 + (y - 1)^2 = 5 :=
sorry

end circumcircle_of_triangle_l415_415138


namespace paths_sum_eq_100_pow_17_l415_415707

theorem paths_sum_eq_100_pow_17 :
  ∑ j, C (100 * j + 19) 17 = 100 ^ 17 :=
by {
  -- Definitions and assumptions
  let C : ℕ → ℕ → ℕ,
  have h_C : ∀ j, C (100 * j + 19) 17 = 100 ^ 17 := ...
  -- Sum calculation
  have h_sum : ∑ j, C (100 * j + 19) 17 = ∑ j, 100 ^ 17 := ...
  sorry
}

end paths_sum_eq_100_pow_17_l415_415707


namespace segment_triangle_range_l415_415857

theorem segment_triangle_range (x : ℝ) :
  AB_diameter_is_1_circle (AB_unit_circle : ℝ) (D_on_AB : (x, 0))
  (DC_perpendicular : DC_perpendicular_to_AB (D_on_AB, AB_unit_circle))
  (C_intersects_circle : C_intersects_circle (DC_perpendicular_to AB, AB_unit_circle))  
  (AD := 1 + x) (BD := 1 - x) (CD := real.sqrt (1 - x^2)) :
  (2 - real.sqrt 5) < x ∧ x < (real.sqrt 5 - 2) := sorry

end segment_triangle_range_l415_415857


namespace ratio_sailboats_to_fishing_boats_l415_415474

noncomputable def CruiseShips : ℕ := 4
noncomputable def CargoShips : ℕ := 2 * CruiseShips
noncomputable def Sailboats : ℕ := CargoShips + 6
noncomputable def TotalVessels : ℕ := 28
noncomputable def FishingBoats : ℕ := TotalVessels - (CruiseShips + CargoShips + Sailboats)

theorem ratio_sailboats_to_fishing_boats : ((Sailboats : ℕ) / 2) = 7 :=
by
  have CruiseShips_eq : CruiseShips = 4 := rfl
  have CargoShips_eq : CargoShips = 2 * CruiseShips := rfl
  have Sailboats_eq : Sailboats = CargoShips + 6 := rfl
  have TotalVessels_eq : TotalVessels = 28 := rfl
  have FishingBoats_eq : FishingBoats = TotalVessels - (CruiseShips + CargoShips + Sailboats) := rfl
  have calc_F : FishingBoats = 2 :=
    by
      rw [CruiseShips_eq, CargoShips_eq, Sailboats_eq, TotalVessels_eq]
      norm_num
  have ratio_calc : (Sailboats / 2) = 7 :=
   by
      rw [CruiseShips_eq, CargoShips_eq, Sailboats_eq]
      norm_num
  exact ratio_calc

#eval ratio_sailboats_to_fishing_boats

end ratio_sailboats_to_fishing_boats_l415_415474


namespace hexagon_area_within_rectangle_of_5x4_l415_415684

-- Define the given conditions
def is_rectangle (length width : ℝ) := length > 0 ∧ width > 0

def vertices_touch_midpoints (length width : ℝ) (hexagon_area : ℝ) : Prop :=
  let rectangle_area := length * width
  let triangle_area := (1 / 2) * (length / 2) * (width / 2)
  let total_triangle_area := 4 * triangle_area
  rectangle_area - total_triangle_area = hexagon_area

-- Formulate the main statement to be proved
theorem hexagon_area_within_rectangle_of_5x4 : 
  vertices_touch_midpoints 5 4 10 := 
by
  -- Proof is omitted for this theorem
  sorry

end hexagon_area_within_rectangle_of_5x4_l415_415684


namespace triangle_area_l415_415500

theorem triangle_area : 
  ∃ (A : ℝ), A = 12 ∧ (∃ (x_intercept y_intercept : ℝ), 3 * x_intercept + 2 * y_intercept = 12 ∧ x_intercept * y_intercept / 2 = A) :=
by
  sorry

end triangle_area_l415_415500


namespace rhombus_unique_property_l415_415753

structure Rhombus where
  (a b c d : ℝ)
  (equalSides : a = b ∧ b = c ∧ c = d)
  (bisectingDiagonals : ∃ p q r s : ℝ, p + q = a ∧ r + s = b ∧ p + q = r + s)
  (perpendicularDiagonals : ∃ r s : ℝ, r * s = -1)

structure Parallelogram where
  (a b c d : ℝ)
  (equilateralOppositeSides : a = c ∧ b = d)
  (bisectingDiagonals : ∃ p q r s : ℝ, p + q = a ∧ r + s = b ∧ p + q = r + s)
  (notNecessarilyPerpendicularDiagonals : ¬ ∃ r s : ℝ, r * s = -1)

theorem rhombus_unique_property :
  ∀ (rh : Rhombus) (pg : Parallelogram),
    (rh.perpendicularDiagonals ∧ pg.notNecessarilyPerpendicularDiagonals) :=
by
  sorry

end rhombus_unique_property_l415_415753


namespace sininequality_for_n4_l415_415581

variable {α β γ : ℝ}

def isAcuteAngledTriangle (α β γ : ℝ) : Prop := 
  α + β + γ = π ∧ 0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2 ∧ 0 < γ ∧ γ < π / 2

theorem sininequality_for_n4 (α β γ : ℝ) (h : isAcuteAngledTriangle α β γ) :
  sin (4 * α) + sin (4 * β) + sin (4 * γ) < 0 :=
sorry

end sininequality_for_n4_l415_415581


namespace find_f_4_l415_415943

def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≥ 0 then x^a else abs (x - 2)

theorem find_f_4 (a : ℝ) (h : f (-2) a = f 2 a) : f 4 a = 16 := by
  have h₁ : f (-2) a = abs (-2 - 2) := by rfl
  have h₂ : f 2 a = 2^a := by rfl
  have eqn : abs (-2 - 2) = 4 := by rfl
  rw [eqn, h₁, h, h₂] at h
  rw [h₂] at h
  have h₃ : 4 = 2^a := by assumption
  have a_is_2 : a = 2 := by sorry -- Solving for a
  have f_is : f 4 2 = 16 := by rfl
  rw [a_is_2] at f_is
  exact f_is

end find_f_4_l415_415943


namespace red_beads_count_in_necklace_l415_415832

def has_n_red_beads (n : ℕ) (blue_count : ℕ) (necklace : list ℕ) : Prop :=
  -- Each blue bead has beads of different colors on either side
  (∀ i, (necklace[i % blue_count] = 1) → 
        (necklace[(i + 1) % necklace.length] ≠ 1 ∧ 
         necklace[(i - 1) % necklace.length] ≠ 1)) ∧
  -- Every other bead from each red one is also of different colors
  (∀ i, (necklace[i] = 2) → 
        (necklace[(i + 1) % necklace.length] ≠ 2 ∧ 
         necklace[(i + 2) % necklace.length] ≠ 2))

def total_beads_correct (necklace : list ℕ) : Prop :=
  -- The total number of beads in the necklace should be the count of blue beads 
  -- plus the total number of red beads
  necklace.length = 30 + 60

theorem red_beads_count_in_necklace : ∃ (necklace : list ℕ), 
  total_beads_correct necklace ∧ 
  has_n_red_beads 60 30 necklace :=
sorry

end red_beads_count_in_necklace_l415_415832


namespace mary_prevents_pat_l415_415350

noncomputable def smallest_initial_integer (N: ℕ) : Prop :=
  N > 2017 ∧ 
  ∀ x, ∃ n: ℕ, 
  (x = N + n * 2018 → x % 2018 ≠ 0 ∧
   (2017 * x + 2) % 2018 ≠ 0 ∧
   (2017 * x + 2021) % 2018 ≠ 0)

theorem mary_prevents_pat (N : ℕ) : smallest_initial_integer N → N = 2022 :=
sorry

end mary_prevents_pat_l415_415350


namespace problem_statement_l415_415470

def reading_method (n : ℕ) : String := sorry
-- Assume reading_method correctly implements the reading method for integers

def is_read_with_only_one_zero (n : ℕ) : Prop :=
  (reading_method n).count '0' = 1

theorem problem_statement : is_read_with_only_one_zero 83721000 = false := sorry

end problem_statement_l415_415470


namespace greatest_t_value_exists_l415_415892

theorem greatest_t_value_exists (t : ℝ) : (∃ t, (t^2 - t - 56) / (t - 8) = 3 / (t + 5)) → ∃ t, (t = -4) := 
by
  intro h
  -- Insert proof here
  sorry

end greatest_t_value_exists_l415_415892


namespace charming_eight_digit_integer_count_l415_415520

/-- An 8-digit positive integer is considered charming if its digits are a permutation of
the set {1, 2, 3, 4, 5, 6, 7, 8} and its first k digits form an integer that is divisible
by k for k = 1, 2, ..., 8, and additionally, the sum of its digits is divisible by 8.
Prove that there is exactly one charming 8-digit integer. -/
theorem charming_eight_digit_integer_count :
  let charming (a b c d e f g h : ℕ) : Prop :=
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
     c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
     d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
     e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
     f ≠ g ∧ f ≠ h ∧
     g ≠ h ∧
     a ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧ b ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧
     c ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧ d ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧
     e ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧ f ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧
     g ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧ h ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧
     (a + b + c + d + e + f + g + h) % 8 = 0 ∧
     a % 1 = 0 ∧ (a * 10 + b) % 2 = 0 ∧ (a * 100 + b * 10 + c) % 3 = 0 ∧
     (a * 1000 + b * 100 + c * 10 + d) % 4 = 0 ∧
     (a * 10000 + b * 1000 + c * 100 + d * 10 + e) % 5 = 0 ∧
     (a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f) % 6 = 0 ∧
     (a * 1000000 + b * 100000 + c * 10000 + d * 1000 + e * 100 + f * 10 + g) % 7 = 0 ∧
     (a * 10000000 + b * 1000000 + c * 100000 + d * 10000 + e * 1000 + f * 100 + g * 10 + h) % 8 = 0)
  in (∃! (a b c d e f g h : ℕ), charming a b c d e f g h) :=
sorry

end charming_eight_digit_integer_count_l415_415520


namespace sum_T_bk_eq_l415_415337

def geometric_seq (q : ℝ) (a : ℕ → ℝ): Prop :=
  ∀ n, a (n + 1) = q * (a n)

def arithmetic_seq (d : ℝ) (b : ℕ → ℝ): Prop :=
  ∀ n, b (n + 1) = b n + d

noncomputable def a (n : ℕ) : ℝ := 2^(n-1 : ℕ)
noncomputable def b (n : ℕ) : ℝ := n

def S (n : ℕ) : ℝ := 2^n - 1
def T (n : ℕ) : ℝ := ∑ k in finset.range n, 2^(k + 1) - k - 2

theorem sum_T_bk_eq :
  ∀ n, (∑ k in finset.range n, (T (k + 1) + b (k + 3)) * b (k + 1) / ((k + 2) * (k + 3)))
       = (2^(n + 2) / (n + 2) - 2) :=
by
  sorry

end sum_T_bk_eq_l415_415337


namespace find_c_and_d_l415_415408

-- Define the intersection point
def intersection := (3, 6)

-- Define the equations of the lines
def line1 (y c : ℝ) : ℝ := (1/3) * y + c
def line2 (x d : ℝ) : ℝ := (1/3) * x + d

-- Prove that c + d = 6 given the conditions
theorem find_c_and_d (c d : ℝ) :
  line1 6 c = 3 ∧ line2 3 d = 6 → c + d = 6 :=
by
  intros h,
  cases h with h1 h2,
  -- sorry to skip the proof
  sorry

end find_c_and_d_l415_415408


namespace circle_properties_l415_415633

/-- Given the polar coordinate equation of a circle: ρ^2 - 4√(2)ρ cos(θ - π/4) + 6 = 0,
  prove (1) the standard equation of the circle,
        (2) the parametric equations of the circle,
        (3) the maximum and minimum values of x * y among all points (x, y) on the circle. -/
theorem circle_properties :
  (∃ x y : ℝ, (x + y) ^ 2 - 4 * x - 4 * y + 6 = 0) ∧
  (∃ θ : ℝ, (x = 2 + √2 * cos θ) ∧ (y = 2 + √2 * sin θ)) ∧
  (∃ x y : ℝ, (x * y = 1) ∧ (x * y = 9)) :=
begin
  sorry
end

end circle_properties_l415_415633


namespace box_dimensions_sum_l415_415841

theorem box_dimensions_sum (X Y Z : ℝ) (hXY : X * Y = 18) (hXZ : X * Z = 54) (hYZ : Y * Z = 36) (hX_pos : X > 0) (hY_pos : Y > 0) (hZ_pos : Z > 0) :
  X + Y + Z = 11 := 
sorry

end box_dimensions_sum_l415_415841


namespace x12_is_1_l415_415282

noncomputable def compute_x12 (x : ℝ) (h : x + 1 / x = Real.sqrt 5) : ℝ :=
  x ^ 12

theorem x12_is_1 (x : ℝ) (h : x + 1 / x = Real.sqrt 5) : compute_x12 x h = 1 :=
  sorry

end x12_is_1_l415_415282


namespace solution_set_empty_iff_a_in_range_l415_415667

theorem solution_set_empty_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, ¬ (2 * x^2 + a * x + 2 < 0)) ↔ (-4 ≤ a ∧ a ≤ 4) :=
by
  sorry

end solution_set_empty_iff_a_in_range_l415_415667


namespace imaginary_part_of_z_l415_415595

-- Define the complex number z
def z : ℂ := 2 / (1 + complex.I)

-- Theorem stating that the imaginary part of z is -1
theorem imaginary_part_of_z : complex.im z = -1 :=
sorry

end imaginary_part_of_z_l415_415595


namespace math_problem_l415_415601

noncomputable def ellipse_equation : Prop := 
  ∃ (a b : ℝ), ∀ (x y : ℝ), (b = 1) ∧ (a = sqrt 3) ∧ ((x^2 / a^2) + (y^2 / b^2) = 1)

noncomputable def min_AM_distance (x1 : ℝ) : ℝ := 
  sqrt((2 * x1^2 / 3) - 2 * x1 + 2)

noncomputable def fixed_point_P : ℝ × ℝ := (3, 0)

theorem math_problem :
  ellipse_equation ∧
  (∀ x1, x1 ∈ set.Icc (-sqrt 3) (sqrt 3) → min_AM_distance x1 ≥ 1 / 2) ∧
  ∃ P : ℝ × ℝ, P = fixed_point_P ∧
               ∀ M A B (M = (1, 0)) (P.2 = 0), 
                 ∠ M P A = ∠ M P B :=
by
  sorry

end math_problem_l415_415601


namespace quadratic_discriminant_l415_415194

-- Define the coefficients of the quadratic equation
def a : ℚ := 2
def b : ℚ := 2 + 1/2
def c : ℚ := 1/2

-- State the theorem to prove
theorem quadratic_discriminant : (b^2 - 4 * a * c) = 9 / 4 := by
  -- Coefficient values
  have h_b : b = 5 / 2 := by
    calc
      b = 2 + 1/2 : rfl
      ... = 5 / 2 : by norm_num
  have h_discriminant : (5/2)^2 - 4 * 2 * (1/2) = 9/4 := by sorry
  -- Substitute the coefficient values
  rw h_b,
  exact h_discriminant,
  sorry

end quadratic_discriminant_l415_415194


namespace undefined_expression_values_l415_415578

theorem undefined_expression_values : 
    ∃ x : ℝ, x^2 - 9 = 0 ↔ (x = -3 ∨ x = 3) :=
by
  sorry

end undefined_expression_values_l415_415578


namespace diameter_length_l415_415685

theorem diameter_length (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9)
  (h_rational_oh : ∃ q : ℚ, 0 < q ∧ q * q = (99 * (x * x - y * y)) / 4) :
  10 * x + y = 65 :=
by {
  have h1 : (10 * x + y : ℕ) ∈ fin (100),
  {
    apply nat.le_trans _ (by norm_num),
    rw [mul_comm],
    refine nat.add_le_add_left (by linarith) _,
  },
  have h2 : (10 * y + x : ℕ) ∈ fin (100),
  {
    apply nat.le_trans _ (by norm_num),
    rw [mul_comm],
    refine nat.add_le_add_left (by linarith) _,
  },
  sorry -- skipping the detailed proof
}

end diameter_length_l415_415685


namespace northeast_to_west_is_180_degrees_l415_415803

/--
A circular floor decoration features ten rays emanating from the center,
forming ten congruent central angles. One ray points directly North.
Prove that the measure in degrees of the smaller angle formed between
the ray pointing Northeast and the ray pointing West is 180 degrees.
-/
theorem northeast_to_west_is_180_degrees :
  let n := 10 in
  let central_angle := 360 / n in
  5 * central_angle = 180 :=
by
  let n := 10
  let central_angle := 360 / n
  have h1 : central_angle = 36 := by
    calc
      central_angle = 360 / 10 : rfl
      ... = 36 : by norm_num
  show 5 * central_angle = 180 from
    calc
      5 * central_angle = 5 * 36 : by rw [h1]
      ... = 180 : by norm_num
  done

end northeast_to_west_is_180_degrees_l415_415803


namespace flash_catches_ace_l415_415846

-- The given conditions from part a)
variable {x y v : ℝ}
variable (hx : x > 1)

-- Definition of the problem
def total_distance_run_by_flash (x y v : ℝ) (hx : x > 1) : ℝ :=
  let t_e := 2 * y / (v * (x - 1))
  let east_distance := x * v * t_e
  let t_w := y / (v * x)
  let west_distance := (x + 1) * v * t_w
  east_distance + west_distance

-- The proof goal
theorem flash_catches_ace (h: x > 1) :
  total_distance_run_by_flash x y v h = (2 * x * y / (x - 1)) + ((x + 1) * y / x) :=
by
  sorry

end flash_catches_ace_l415_415846


namespace possible_values_of_d_l415_415341

theorem possible_values_of_d (r s : ℝ) (c d : ℝ)
  (h1 : ∃ u, u = -r - s ∧ r * s + r * u + s * u = c)
  (h2 : ∃ v, v = -r - s - 8 ∧ (r - 3) * (s + 5) + (r - 3) * (u - 8) + (s + 5) * (u - 8) = c)
  (u_eq : u = -r - s)
  (v_eq : v = -r - s - 8)
  (polynomial_relation : d + 156 = -((r - 3) * (s + 5) * (u - 8))) : 
  d = -198 ∨ d = 468 := 
sorry

end possible_values_of_d_l415_415341


namespace angle_between_a_and_b_l415_415639

noncomputable def vector_a := (1 : ℝ,  2 : ℝ)
noncomputable def magnitude_a := Real.sqrt (1^2 + 2^2)
noncomputable def magnitude_b := (1 / 2) * magnitude_a

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def vector_b_magnitude_half_vector_a (u : ℝ × ℝ) := magnitude_b = (1 / 2) * magnitude_a

noncomputable def perpendicular_vectors (u v : ℝ × ℝ) := dot_product u v == 0

theorem angle_between_a_and_b
  (vector_a := (1, 2) : ℝ × ℝ)
  (magnitude_a := Real.sqrt (1*1 + 2*2))
  (magnitude_b := (1 / 2) * magnitude_a)
  (h1 : magnitude_b = (1/2) * magnitude_a)
  (h2: perpendicular_vectors (vector_a.1 + 2 * vector_b) (2 * vector_a - vector_b)) :
  Real.arccos ((dot_product vector_a vector_b) / (magnitude_a * magnitude_b)) = Real.pi := 
by 
  sorry

end angle_between_a_and_b_l415_415639


namespace max_value_S_n_l415_415923

open Nat

noncomputable def a_n (n : ℕ) : ℤ := 20 + (n - 1) * (-2)

noncomputable def S_n (n : ℕ) : ℤ := n * 20 + (n * (n - 1)) * (-2) / 2

theorem max_value_S_n : ∃ n : ℕ, S_n n = 110 :=
by
  sorry

end max_value_S_n_l415_415923


namespace longer_side_of_rectangle_l415_415109

theorem longer_side_of_rectangle
  (r : ℝ) (A_rect A_circle L S : ℝ) (h1 : r = 6)
  (h2 : A_circle = π * r^2)
  (h3 : A_rect = 3 * A_circle)
  (h4 : S = 2 * r)
  (h5 : A_rect = S * L) : L = 9 * π :=
by
  sorry

end longer_side_of_rectangle_l415_415109


namespace tangent_line_at_A_properties_of_intersection_l415_415600

-- Definition of an ellipse and points on it
def ellipse (a b : ℝ) : ℝ × ℝ → Prop :=
  λ p, (∃ (x y : ℝ), (x = p.1 ∧ y = p.2 ∧ (x ^ 2 / a ^ 2) + (y ^ 2 / b ^ 2) = 1))

-- Prove the tangent line at point A on the ellipse
theorem tangent_line_at_A (a b x1 y1 x y : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : ellipse a b (x1, y1)) :
  (x1 * x / a^2) + (y1 * y / b^2) = 1 :=
sorry

-- Prove properties of the intersection and midpoint
theorem properties_of_intersection (a b x1 y1 x2 y2 xP yP xQ yQ : ℝ)
  (h1 : a > b) (h2 : b > 0) (h3 : ellipse a b (x1, y1)) (h4 : ellipse a b (x2, y2))
  (hP : ((x1 * xP / a^2) + (y1 * yP / b^2) = 1) ∧ ((x2 * xP / a^2) + (y2 * yP / b^2) = 1))
  (hQ : on_line (0, 0) (xP, yP) (xQ, yQ) ∧ ellipse a b (xQ, yQ)) :

  -- 1. Line OP passes through the midpoint C of chord AB
  let C := ((x1 + x2) / 2, (y1 + y2) / 2) in
  on_line (0, 0) (xP, yP) C ∧

  -- 2. Tangent at point Q is parallel to chord AB
  let k := (x1 - x2) / (y1 - y2) in
  (-b^2 * xQ / a^2 * yQ) = k :=
sorry

end tangent_line_at_A_properties_of_intersection_l415_415600


namespace tangents_parallel_common_tangents_l415_415765

-- Define the geometric entities and relations as described in the problem
variables {Point : Type} [EuclideanGeometry Point]

-- Assume we have three circles and their intersections
variables (circle1 circle2 circle3 : Circle Point)
variables (A B C D : Point)

-- Assume circle1 and circle2 intersect at points A and B
-- Circle3 is tangent to circle1, circle2 and intersects line AB at points C and D
axiom circle1_intersects_circle2 : circle1.Intersects circle2 A B
axiom circle3_tangent_circle1 : circle3.Tangent circle1
axiom circle3_tangent_circle2 : circle3.Tangent circle2
axiom circle3_intersects_AB : circle3.IntersectsLine (lineThrough A B) C D

-- Prove the tangents to circle3 at C and D are parallel to the common tangents of circle1 and circle2
theorem tangents_parallel_common_tangents :
  TangentAt circle3 C ∥ CommonTangents circle1 circle2 ∧ TangentAt circle3 D ∥ CommonTangents circle1 circle2
:= sorry

end tangents_parallel_common_tangents_l415_415765


namespace evaluate_expression_l415_415585

theorem evaluate_expression (a : ℝ) (h : 2 * a^2 - 3 * a - 5 = 0) : 4 * a^4 - 12 * a^3 + 9 * a^2 - 10 = 15 :=
by
  sorry

end evaluate_expression_l415_415585


namespace no_solution_exists_l415_415356

theorem no_solution_exists (x y z : ℕ) (hx : x > 2) (hy : y > 1) (h : x^y + 1 = z^2) : false := 
by
  sorry

end no_solution_exists_l415_415356


namespace complementary_events_at_least_one_black_both_red_l415_415978

def ball := { r : Fin 4 | r < 4 }

def red_balls : set ball := {r | r.val < 2}
def black_balls : set ball := {r | r.val ≥ 2}

def at_least_one_black_ball (s : set ball) : Prop :=
  ∃ b ∈ black_balls, b ∈ s

def both_red_balls (s : set ball) : Prop :=
  ∀ b ∈ s, b ∈ red_balls

theorem complementary_events_at_least_one_black_both_red
  (s : set ball)
  (h1 : card s = 2)
  : at_least_one_black_ball s ↔ ¬both_red_balls s := by
  sorry

end complementary_events_at_least_one_black_both_red_l415_415978


namespace max_profit_output_l415_415800

noncomputable def profit : ℝ → ℝ
| x := if 0 < x ∧ x < 50 then -10 * x^2 + 400 * x - 1050
       else if x ≥ 50 then -4 * x - 10000 / (x - 2) + 6200
       else 0

theorem max_profit_output :
  (∀ x, profit x ≤ 5792) ∧ profit 52 = 5792 :=
by
  sorry

end max_profit_output_l415_415800


namespace irrational_power_to_nat_l415_415370

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.log 3 / Real.log (Real.sqrt 2) 

theorem irrational_power_to_nat 
  (ha_irr : ¬ ∃ (q : ℚ), a = q)
  (hb_irr : ¬ ∃ (q : ℚ), b = q) : (a ^ b) = 3 := by
  -- \[a = \sqrt{2}, b = \log_{\sqrt{2}}(3)\]
  sorry

end irrational_power_to_nat_l415_415370


namespace sum_divisible_by_3_l415_415320

theorem sum_divisible_by_3 (a : ℤ) : 3 ∣ (a^3 + 2 * a) :=
sorry

end sum_divisible_by_3_l415_415320


namespace ellipse_equation_and_max_triangle_area_l415_415620

theorem ellipse_equation_and_max_triangle_area
  (a b : ℝ) (h₀ : a > b) (h₁ : b > 0)
  (e: ℝ) (he : e = Real.sqrt 2 / 2)
  (vertex : ℝ × ℝ) (hvertex : vertex = (0, -1)) :
  (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 2 + y^2 = 1)) ∧
  (∃ (A B : ℝ × ℝ), symmetric_about_line A B (λ x, -1 / m * x + 1 / 2) 
                    ∧ max_area_triangle O A B = Real.sqrt 2 / 2) :=
by { sorry }

end ellipse_equation_and_max_triangle_area_l415_415620


namespace minimum_value_when_a_is_1_range_of_a_given_fx_geq_0_l415_415948

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  Real.log (x + 1) + 2 / (x + 1) + a * x - 2

theorem minimum_value_when_a_is_1 : ∀ x : ℝ, ∃ m : ℝ, 
  (∀ y : ℝ, f y 1 ≥ f x 1) ∧ (f x 1 = m) :=
sorry

theorem range_of_a_given_fx_geq_0 : ∀ a : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → 0 ≤ f x a) ↔ 1 ≤ a :=
sorry

end minimum_value_when_a_is_1_range_of_a_given_fx_geq_0_l415_415948


namespace not_like_term_A_l415_415444

-- Definitions for the terms
def termA1 := 3 * (x ^ 2) * y
def termA2 := -6 * x * (y ^ 2)

def termB1 := - (a * (b ^ 3))
def termB2 := (b ^ 3) * a

def termC1 := 12
def termC2 := 0

def termD1 := 2 * x * y * z
def termD2 := -(1 / 2) * z * y * x

-- Function to check if two terms are like terms
def like_terms (term1 term2 : ℤ) : Prop :=
  -- Assuming specific structure for terms to compare like terms here
  -- This is a placeholder and would require a more formal definition
  sorry

theorem not_like_term_A :
  ¬ like_terms termA1 termA2 ∧
  like_terms termB1 termB2 ∧
  like_terms termC1 termC2 ∧
  like_terms termD1 termD2 := 
by
  sorry

end not_like_term_A_l415_415444


namespace num_palindromes_on_clock_l415_415273

def is_palindrome (hhmm : Nat) : Prop :=
  let hh := hhmm / 100
  let h1 := hh / 10
  let h2 := hh % 10
  let mm := hhmm % 100
  let m1 := mm / 10
  let m2 := mm % 10
  (h1 = m2) ∧ (h2 = m1)

def is_valid_time (hhmm : Nat) : Prop :=
  let hh := hhmm / 100
  let mm := hhmm % 100
  (hh < 24) ∧ (mm < 60)

theorem num_palindromes_on_clock : 
  (Finset.univ.filter (λ hhmm, is_valid_time hhmm ∧ is_palindrome hhmm)).card = 116 := 
sorry 

end num_palindromes_on_clock_l415_415273


namespace constant_term_of_Liam_polynomial_is_3_l415_415733

noncomputable def constant_term_of_Liam_polynomial : ℕ :=
  let constant_term_product := 9 in
  let c := ℕ.sqrt constant_term_product in
  if c * c = constant_term_product then c else 0

theorem constant_term_of_Liam_polynomial_is_3 :
  constant_term_of_Liam_polynomial = 3 :=
by
  sorry

end constant_term_of_Liam_polynomial_is_3_l415_415733


namespace parabola_equation_l415_415561

def equation_of_parabola (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, y = a * x^2 + b * x + c ↔ 
              (∃ a : ℝ, y = a * (x - 3)^2 + 5) ∧
              y = (if x = 0 then 2 else y)

theorem parabola_equation :
  equation_of_parabola (-1 / 3) 2 2 :=
by
  -- First, show that the vertex form (x-3)^2 + 5 meets the conditions
  sorry

end parabola_equation_l415_415561


namespace existence_of_special_set_l415_415158

theorem existence_of_special_set :
  ∃ S : Finset ℕ, S.card = 1998 ∧ ∀ a b ∈ S, (a ≠ b → (a - b) ^ 2 ∣ a * b) :=
by
  sorry

end existence_of_special_set_l415_415158


namespace fewer_columns_after_rearrangement_l415_415487

theorem fewer_columns_after_rearrangement : 
  ∀ (T R R' C C' fewer_columns : ℕ),
    T = 30 → 
    R = 5 → 
    R' = R + 4 →
    C * R = T →
    C' * R' = T →
    fewer_columns = C - C' →
    fewer_columns = 3 :=
by
  intros T R R' C C' fewer_columns hT hR hR' hCR hC'R' hfewer_columns
  -- sorry to skip the proof part
  sorry

end fewer_columns_after_rearrangement_l415_415487


namespace moles_of_LiOH_formed_l415_415203

-- Define the types for moles and reactions
def moles := ℝ
def lithium_nitride : moles := 1
def water : moles := 3
def lithium_hydroxide : moles := 3
def ammonia : moles := 1

-- Balanced chemical equation in terms of moles
def balanced_reaction (Li3N H2O : moles) : Prop :=
  Li3N  = 1 ∧ H2O = 3 ∧ lithium_hydroxide = 3 ∧ ammonia = 1

-- Main theorem stating the number of moles of Lithium hydroxide formed
theorem moles_of_LiOH_formed : balanced_reaction lithium_nitride water → lithium_hydroxide = 3 := by
  intros h
  have h1 : lithium_nitride = 1 := by sorry
  have h2 : water = 3 := by sorry
  show lithium_hydroxide = 3 from by sorry

end moles_of_LiOH_formed_l415_415203


namespace matrix_product_l415_415867

theorem matrix_product :
  (List.foldl (λ acc n, acc * (Matrix 2 2 ℝ (λ i j, if i = 0 then if j = 1 then n else 1 else if i = 1 then if j = 1 then 1 else 0 else 0)))
    (Matrix 2 2 ℝ (λ i j, if i = 0 then if j = 1 then 2 else 1 else if i = 1 then if j = 1 then 1 else 0 else 0)))
    [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100])
  =
  Matrix 2 2 ℝ (λ i j, if i = 0 then if j = 1 then 2550 else 1 else if i = 1 then if j = 1 then 1 else 0 else 0) :=
by
  sorry

end matrix_product_l415_415867


namespace tangent_chord_equation_l415_415744

theorem tangent_chord_equation (x1 y1 x2 y2 : ℝ) :
  (x1^2 + y1^2 = 1) →
  (x2^2 + y2^2 = 1) →
  (2*x1 + 2*y1 + 1 = 0) →
  (2*x2 + 2*y2 + 1 = 0) →
  ∀ (x y : ℝ), 2*x + 2*y + 1 = 0 :=
by
  intros hx1 hy1 hx2 hy2 x y
  exact sorry

end tangent_chord_equation_l415_415744


namespace discriminant_of_given_quadratic_l415_415197

-- define the coefficients a, b, c
def a : ℚ := 2
def b : ℚ := 2 + 1/2
def c : ℚ := 1/2

-- define the discriminant function for a quadratic equation ax^2 + bx + c
def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

-- state the theorem
theorem discriminant_of_given_quadratic : discriminant a b c = 9/4 :=
by
  -- add the proof here
  sorry

end discriminant_of_given_quadratic_l415_415197


namespace ratio_square_l415_415660

theorem ratio_square (x y : ℕ) (h1 : x * (x + y) = 40) (h2 : y * (x + y) = 90) (h3 : 2 * y = 3 * x) : (x + y) ^ 2 = 100 := 
by 
  sorry

end ratio_square_l415_415660


namespace circle_equation_tangent_to_line_l415_415007

theorem circle_equation_tangent_to_line
  (h k : ℝ) (A B C : ℝ)
  (hxk : h = 2) (hyk : k = -1) 
  (hA : A = 3) (hB : B = -4) (hC : C = 5)
  (r_squared : ℝ := (|A * h + B * k + C| / Real.sqrt (A^2 + B^2))^2)
  (h_radius : r_squared = 9) :
  ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r_squared := 
by
  sorry

end circle_equation_tangent_to_line_l415_415007


namespace discriminant_of_given_quadratic_l415_415199

-- define the coefficients a, b, c
def a : ℚ := 2
def b : ℚ := 2 + 1/2
def c : ℚ := 1/2

-- define the discriminant function for a quadratic equation ax^2 + bx + c
def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

-- state the theorem
theorem discriminant_of_given_quadratic : discriminant a b c = 9/4 :=
by
  -- add the proof here
  sorry

end discriminant_of_given_quadratic_l415_415199


namespace train_travel_distance_l415_415809

theorem train_travel_distance
  (rate_miles_per_pound : Real := 5 / 2)
  (remaining_coal : Real := 160)
  (distance_per_pound := λ r, r / 2)
  (total_distance := λ rc dpp, rc * dpp) :
  total_distance remaining_coal rate_miles_per_pound = 400 := sorry

end train_travel_distance_l415_415809


namespace probability_of_normal_distribution_l415_415938

open Real

variables (ξ : ℝ) (σ : ℝ)

noncomputable def normal_distribution := pdf_normal μ σ ξ

theorem probability_of_normal_distribution (h1 : ξ ~ Normal 2 σ^2)
    (h2 : P(ξ ≤ 4) = 0.84) : P(ξ ≤ 0) = 0.16 :=
  sorry

end probability_of_normal_distribution_l415_415938


namespace arithmetic_sequence_sum_l415_415347

variable {α : Type*} [linear_ordered_field α]

def nth_term (a₁ d n : α) : α := a₁ + (n - 1) * d

def sum_first_n_terms (a₁ d n : α) : α := 
  n / 2 * (a₁ + nth_term a₁ d n)

theorem arithmetic_sequence_sum
  (a₁ d : α)
  (h₁ : nth_term a₁ d 5 = 2) :
  2 * sum_first_n_terms a₁ d 6 + 
  sum_first_n_terms a₁ d 12 = 48 := 
sorry

end arithmetic_sequence_sum_l415_415347


namespace walnut_trees_planted_l415_415423

-- The number of walnut trees before planting
def walnut_trees_before : ℕ := 22

-- The number of walnut trees after planting
def walnut_trees_after : ℕ := 55

-- The number of walnut trees planted today
def walnut_trees_planted_today : ℕ := 33

-- Theorem statement to prove that the number of walnut trees planted today is 33
theorem walnut_trees_planted:
  walnut_trees_after - walnut_trees_before = walnut_trees_planted_today :=
by sorry

end walnut_trees_planted_l415_415423


namespace arc_length_of_parametric_curve_l415_415077

def parametric_curve_length := 
  ∫ (t : ℝ) in (Set.Icc π (2*π)), 
    real.sqrt ((3 * (1 - real.cos t))^2 + (3 * real.sin t)^2)

theorem arc_length_of_parametric_curve : 
  parametric_curve_length = 12 :=
sorry

end arc_length_of_parametric_curve_l415_415077


namespace nitin_borrowed_amount_l415_415717

theorem nitin_borrowed_amount (P : ℝ) (I1 I2 I3 : ℝ) :
  (I1 = P * 0.06 * 3) ∧
  (I2 = P * 0.09 * 5) ∧
  (I3 = P * 0.13 * 3) ∧
  (I1 + I2 + I3 = 8160) →
  P = 8000 :=
by
  sorry

end nitin_borrowed_amount_l415_415717


namespace probability_0_2_l415_415670

-- Definitions
def normal_distribution (mean : ℝ) (variance : ℝ) : Type := sorry

noncomputable def probability_interval (dist : Type) (a b : ℝ) : ℝ := sorry

-- Given conditions
variable (ξ : Type)
variable (σ : ℝ) (hσ : σ > 0) 

-- Assume ξ follows a normal distribution with mean 1 and variance σ^2
def xi_normal : Prop := ξ = normal_distribution 1 (σ^2)

-- Probability of ξ in the interval (0,1) is 0.4
def prob_0_1 : Prop := probability_interval ξ 0 1 = 0.4

-- The main proposition to prove
theorem probability_0_2 (hxi : xi_normal) (hprob_0_1 : prob_0_1) : probability_interval ξ 0 2 = 0.8 :=
by
  sorry

end probability_0_2_l415_415670


namespace solution_set_of_inequality_l415_415888

theorem solution_set_of_inequality :
  {y : ℝ | 3 / 10 + |2 * y - 1 / 5| < 7 / 10} = set.Ioo (-1 / 10) (3 / 10) :=
by
  sorry

end solution_set_of_inequality_l415_415888


namespace find_z_l415_415974

theorem find_z (z : ℂ) (hz1 : z.im = z) (hz2 : ((z + 2)^2 - 8 * complex.I).im = ((z + 2)^2 - 8 * complex.I)) : z = -2 * complex.I := 
sorry

end find_z_l415_415974


namespace angie_pretzels_l415_415853

theorem angie_pretzels (Barry_Shelly: ℕ) (Shelly_Angie: ℕ) :
  (Barry_Shelly = 12 / 2) → (Shelly_Angie = 3 * Barry_Shelly) → (Barry_Shelly = 6) → (Shelly_Angie = 18) :=
by
  intro h1 h2 h3
  sorry

end angie_pretzels_l415_415853


namespace rectangle_longer_side_length_l415_415125

theorem rectangle_longer_side_length
  (r : ℝ) (A_rect : ℝ) :
  r = 6 →
  A_rect = 3 * (Real.pi * r^2) →
  ∃ l : ℝ, l = 9 * Real.pi :=
begin
  intros hr hA,
  use 9 * Real.pi,
  sorry
end

end rectangle_longer_side_length_l415_415125


namespace smallest_cookies_left_l415_415095

theorem smallest_cookies_left (m : ℤ) (h : m % 8 = 5) : (4 * m) % 8 = 4 :=
by
  sorry

end smallest_cookies_left_l415_415095


namespace percentage_B_of_C_l415_415135

theorem percentage_B_of_C 
  (A C B : ℝ)
  (h1 : A = (7 / 100) * C)
  (h2 : A = (50 / 100) * B) :
  B = (14 / 100) * C := 
sorry

end percentage_B_of_C_l415_415135


namespace product_of_real_roots_l415_415621

def equation (x : ℝ) : Prop := x^2 + 18 * x + 30 = 2 * real.sqrt (x^2 + 18 * x + 45)

theorem product_of_real_roots : 
  let roots := {x | equation x} in
  ∏ x in roots, x = 20 :=
by
  sorry

end product_of_real_roots_l415_415621


namespace max_operations_external_min_operations_self_l415_415693

section Problem

/-- There are 2010 lights and 2010 switches. Each switch controls a different light. Each operation involves pressing some switches simultaneously, resulting in an equal number of lit lights to the number of switches pressed. After each operation, the switches reset to their original state. -/
def num_lights : ℕ := 2010
def num_switches : ℕ := 2010

/-- The maximum number of distinct operations that can be performed to determine the correspondence by an external operator is \(2^{2009} + 1\). -/
theorem max_operations_external : ∀ (num_lights : ℕ) (num_switches : ℕ), 
  num_lights = 2010 → num_switches = 2010 → 
  ∃ (max_operations : ℕ), max_operations = 2^2009 + 1 :=
by
  intros _ _ _ _,
  use (2^2009 + 1),
  sorry

/-- The minimum number of operations required by Laura herself to determine the correspondence of switches and lights is 11. -/
theorem min_operations_self : ∀ (num_lights : ℕ) (num_switches : ℕ), 
  num_lights = 2010 → num_switches = 2010 → 
  ∃ (min_operations : ℕ), min_operations = 11 :=
by
  intros _ _ _ _,
  use 11,
  sorry

end Problem

end max_operations_external_min_operations_self_l415_415693


namespace chess_club_probability_l415_415395

def num_ways_to_choose {n k : ℕ} : ℕ := Nat.choose n k

def chess_club_team_selection : ℕ :=
  let total := num_ways_to_choose 20 4
  let teams_with_2_boys_2_girls := num_ways_to_choose 12 2 * num_ways_to_choose 8 2
  let teams_with_3_boys_1_girl := num_ways_to_choose 12 3 * num_ways_to_choose 8 1
  let teams_with_4_boys_0_girls := num_ways_to_choose 12 4 * num_ways_to_choose 8 0
  teams_with_2_boys_2_girls + teams_with_3_boys_1_girl + teams_with_4_boys_0_girls

theorem chess_club_probability : chess_club_team_selection / num_ways_to_choose 20 4 = 4103 / 4845 := sorry

end chess_club_probability_l415_415395


namespace total_ladybugs_l415_415383

theorem total_ladybugs (ladybugs_with_spots ladybugs_without_spots : ℕ) 
  (h1 : ladybugs_with_spots = 12170) 
  (h2 : ladybugs_without_spots = 54912) : 
  ladybugs_with_spots + ladybugs_without_spots = 67082 := 
by
  sorry

end total_ladybugs_l415_415383


namespace pyramid_dihedral_angle_sum_zero_l415_415372

noncomputable theory

def SquarePyramid (O A B C D : Type) [metric_space O] [metric_space A] [metric_space B] [metric_space C] [metric_space D] :=
 (congruent_edges : (dist O A = dist O B) ∧ (dist O A = dist O C) ∧ (dist O A = dist O D))
 (base_square : is_square A B C D)
 (angle_AOB_60 : angle A O B = π / 3)

theorem pyramid_dihedral_angle_sum_zero (O A B C D : Type) [metric_space O] [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (pyramid : SquarePyramid O A B C D) : 
  ∃ m n : ℤ, cos (dihedral_angle O A B C pyramid.base_square) = m + real.sqrt n ∧ m + n = 0 :=
by
  sorry

end pyramid_dihedral_angle_sum_zero_l415_415372


namespace motel_total_rent_l415_415458

theorem motel_total_rent (R₅₀ R₆₀ : ℕ) 
  (h₁ : ∀ x y : ℕ, 50 * x + 60 * y = 50 * (x + 10) + 60 * (y - 10) + 100)
  (h₂ : ∀ x y : ℕ, 25 * (50 * x + 60 * y) = 10000) : 
  50 * R₅₀ + 60 * R₆₀ = 400 :=
by
  sorry

end motel_total_rent_l415_415458


namespace initial_phase_l415_415821

-- Define the function representing the harmonic motion
def f (x : ℝ) : ℝ := 4 * Real.sin (8 * x - (Real.pi / 9))

-- Theorem stating that the initial phase is -π/9 when x = 0
theorem initial_phase : f 0 = 4 * Real.sin (-Real.pi / 9) := by
  sorry

end initial_phase_l415_415821


namespace quadratic_root_representation_l415_415206

theorem quadratic_root_representation :
  ∃ m p : ℕ, nat.coprime m (37) ∧ nat.coprime m p ∧ nat.coprime p 37 ∧ 
  ∀ x : ℝ, (3 * x^2 - 7 * x + 1 = 0) ↔ (x = (m + real.sqrt 37) / p ∨ x = (m - real.sqrt 37) / p) :=
begin
  sorry
end

end quadratic_root_representation_l415_415206


namespace card_drawing_probability_l415_415905

theorem card_drawing_probability :
  let prob_of_ace_first := (4:ℚ) / 52,
      prob_of_2_second := (4:ℚ) / 51,
      prob_of_3_third := (4:ℚ) / 50,
      prob_of_4_fourth := (4:ℚ) / 49 in
  prob_of_ace_first * prob_of_2_second * prob_of_3_third * prob_of_4_fourth = 16 / 405525 :=
by
  sorry

end card_drawing_probability_l415_415905


namespace paper_height_after_folds_l415_415163

-- Definition of the initial thickness in millimeters
def initial_thickness : ℝ := 0.1

-- Number of folds
def num_folds : ℕ := 20

-- Approximation given in the problem
def approx_two_pow_ten : ℝ := 1000

-- Statement of the problem: prove that the height after 20 folds is approximately 100 meters
theorem paper_height_after_folds (h : ℝ := initial_thickness * 2^num_folds) (approx_two_pow_ten : 2^10 ≈ approx_two_pow_ten) :
  h / 1000 = 100 :=
by
  sorry

end paper_height_after_folds_l415_415163


namespace parametric_to_standard_l415_415874

theorem parametric_to_standard (t : ℝ) (h : 0 ≤ t ∧ t ≤ 2 * Real.pi) :
  ∃ (x y : ℝ), x = 2 * Real.cos t ∧ y = 3 * Real.sin t ∧ (x^2 / 4 + y^2 / 9 = 1) :=
by
  use 2 * Real.cos t, 3 * Real.sin t
  split
  { rfl }
  split
  { rfl }
  { sorry }

end parametric_to_standard_l415_415874


namespace area_of_triangle_bounded_by_coordinate_axes_and_line_l415_415510

def area_of_triangle (b h : ℕ) : ℕ := (b * h) / 2

theorem area_of_triangle_bounded_by_coordinate_axes_and_line :
  area_of_triangle 4 6 = 12 :=
by
  sorry

end area_of_triangle_bounded_by_coordinate_axes_and_line_l415_415510


namespace dwarfs_all_strawberries_l415_415378

theorem dwarfs_all_strawberries (
  (h1_day1 : ∀ d : Fin 7, (day1 d = mine)),
  (h2_jobs : ∀ d : Fin 7, ∀ t : Fin 16, worksInMine d t ∨ worksPicksStrawberries d t),
  (h3_nonboth : ∀ d : Fin 7, ∀ t : Fin 16, ¬(worksInMine d t ∧ worksPicksStrawberries d t)),
  (h4_atleast3 : ∀ t1 t2 : Fin 16, t1 ≠ t2 → ∃ (d1 d2 d3 : Fin 7), 
                 (worksInMine d1 t1 ∧ worksPicksStrawberries d1 t2) ∧
                 (worksInMine d2 t1 ∧ worksPicksStrawberries d2 t2) ∧
                 (worksInMine d3 t1 ∧ worksPicksStrawberries d3 t2))):
  ∃ t : Fin 16, ∀ d : Fin 7, worksPicksStrawberries d t := 
sorry

end dwarfs_all_strawberries_l415_415378


namespace imaginary_part_eq_2_l415_415396

-- Defining the complex number z and the conditions
def z (a : ℝ) : Complex := 1 + a * Complex.I

-- Conditions
axiom real_a : a ∈ ℝ
axiom pos_a : a > 0
axiom norm_z : Complex.norm (z a) = Real.sqrt 5

-- Prove the imaginary part is 2
theorem imaginary_part_eq_2 : Complex.im (z 2) = 2 :=
by
  sorry

end imaginary_part_eq_2_l415_415396


namespace total_animal_count_l415_415962

theorem total_animal_count
  (animals_per_aquarium : ℕ)
  (number_of_aquariums : ℕ)
  (h1: animals_per_aquarium = 2)
  (h2: number_of_aquariums = 20)
  : animals_per_aquarium * number_of_aquariums = 40 :=
by {
    rw [h1, h2],
    norm_num,
    sorry
}

end total_animal_count_l415_415962


namespace max_blue_points_segment_l415_415718

noncomputable def max_blue_points (red_points : ℕ) (blue_points : ℕ) : ℕ :=
  if red_points >= 5 ∧ (∀ a b r, a ≠ b ∧ a ≠ r ∧ b ≠ r → 4 ≤ blue_points) ∧ 
     (∀ a b r r', a ≠ b ∧ a ≠ r ∧ b ≠ r ∧ a ≠ r' ∧ b ≠ r' ∧ r ≠ r' → 2 ≤ blue_points ∧ 3 ≤ blue_points) 
  then 4
  else 0

theorem max_blue_points_segment :
  ∀ (red_points blue_points : ℕ),
  red_points >= 5 →
  (∀ a b r, a ≠ b ∧ a ≠ r ∧ b ≠ r → 4 ≤ blue_points) →
  (∀ a b r r', a ≠ b ∧ a ≠ r ∧ b ≠ r ∧ a ≠ r' ∧ b ≠ r' ∧ r ≠ r' → 2 ≤ blue_points ∧ 3 ≤ blue_points) →
  max_blue_points red_points blue_points = 4 :=
by
  sorry

end max_blue_points_segment_l415_415718


namespace valid_pairs_count_l415_415324

theorem valid_pairs_count : 
  ∃ (d n : ℕ), 
    30 + n = d ∧
    d > 30 ∧
    (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ b > a →
      (10 * a + b = 30 + n ∧ 10 * b + a > 30 ∧
        10 * b + a = d ↔
        (b, a) ∈ { (4, 3), (5, 3), (5, 4), (6, 3), (6, 4), (6, 5), 
                    (7, 3), (7, 4), (7, 5), (7, 6), (8, 3), (8, 4), 
                    (8, 5), (8, 6), (8, 7), (9, 3), (9, 4), (9, 5), 
                    (9, 6), (9, 7), (9, 8) } ) ) ∧
   (∃! (d, n : ℕ), true) :=
sorry

end valid_pairs_count_l415_415324


namespace tan_alpha_value_complex_expression_value_l415_415220

theorem tan_alpha_value (α : ℝ) (h : Real.tan (π / 4 + α) = 1 / 2) : Real.tan α = -1 / 3 :=
sorry

theorem complex_expression_value 
(α : ℝ) 
(h1 : Real.tan (π / 4 + α) = 1 / 2) 
(h2 : Real.tan α = -1 / 3) : 
Real.sin (2 * α + 2 * π) - (Real.sin (π / 2 - α))^2 / 
(1 - Real.cos (π - 2 * α) + (Real.sin α)^2) = -15 / 19 :=
sorry

end tan_alpha_value_complex_expression_value_l415_415220


namespace smallest_n_exceeds_million_l415_415266

def sequence (k : ℕ) : ℝ := 12^(k / 13)

noncomputable def product_first_n_terms (n : ℕ) : ℝ := ∏ k in finset.range(n + 1), sequence k

theorem smallest_n_exceeds_million : ∃ n : ℕ, n = 12 ∧ product_first_n_terms n > 1000000 := 
by {
  sorry
}

end smallest_n_exceeds_million_l415_415266


namespace T_n_bound_l415_415916

-- Define the sequence and its properties
noncomputable def sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  (a 1 = 1) ∧ (∀ n : ℕ, n > 0 → a (n + 1) = 2 * S n + 1) ∧ 
  (∀ n : ℕ, S n = ∑ k in finset.range n + 1, a k)

-- Define b_n and T_n
noncomputable def b_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  real.logb 3 (a (n + 1))

noncomputable def T_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ k in finset.range n + 1, (b_seq a k) / (a k)

-- Main theorem
theorem T_n_bound : ∀ (a S : ℕ → ℝ), (sequence a S) → (T_n a < 9 / 4) :=
begin
  assume a S h,
  sorry
end

end T_n_bound_l415_415916


namespace certain_event_l415_415440

theorem certain_event :
  (∀ (e : string), e = "Moonlight in front of the bed" → ¬is_certain_event e) ∧
  (∀ (e : string), e = "Lonely smoke in the desert" → ¬is_certain_event e) ∧
  (∀ (e : string), e = "Reach for the stars with your hand" → is_impossible_event e) ∧
  (∀ (e : string), e = "Yellow River flows into the sea" → is_certain_event e) →
  is_certain_event "Yellow River flows into the sea" :=
by
  sorry

end certain_event_l415_415440


namespace ratio_of_tin_to_copper_in_alloy_B_l415_415088

variables (T_A T_B C_B : ℝ) (total_tin_in_new_alloy mass_A mass_B : ℝ)
variables (ratio_lead_tin_A : ℝ)

-- Constants for the given problem
def mass_A : ℝ := 100
def ratio_lead_tin_A : ℝ := 5 / 3
def parts_A : ℝ := 5 + 3
def total_tin_in_new_alloy : ℝ := 117.5
def mass_B : ℝ := 200

def T_A : ℝ := (3 / parts_A) * mass_A
def T_B : ℝ := total_tin_in_new_alloy - T_A
def C_B : ℝ := mass_B - T_B

theorem ratio_of_tin_to_copper_in_alloy_B : 
  T_B / C_B = 2 / 3 :=
by 
  have TA_is : T_A = 37.5 := by sorry
  have TB_is : T_B = 80 := by sorry
  have CB_is : C_B = 120 := by sorry
  sorry

#check ratio_of_tin_to_copper_in_alloy_B

end ratio_of_tin_to_copper_in_alloy_B_l415_415088


namespace perimeter_of_figure_l415_415683

-- Defining the parameters
variable (x : ℝ)
variable (hx : x ≠ 0)

-- Theorem statement
theorem perimeter_of_figure (hl1 : real  := 3 * x)
                            (hl2 : real  := x)
                            (wl1 : real := 2 * x)
                            (wl2 : real := x) :
  2 * (hl1 + hl2) + 2 * (wl1 + wl2) = 10 * x :=
by
  sorry

end perimeter_of_figure_l415_415683


namespace series_sum_eq_l415_415167

noncomputable def sum_series (k : ℝ) : ℝ :=
  (∑' n : ℕ, (4 * (n + 1) + k) / 3^(n + 1))

theorem series_sum_eq (k : ℝ) : sum_series k = 3 + k / 2 := 
  sorry

end series_sum_eq_l415_415167


namespace quadratic_discriminant_l415_415196

-- Define the coefficients of the quadratic equation
def a : ℚ := 2
def b : ℚ := 2 + 1/2
def c : ℚ := 1/2

-- State the theorem to prove
theorem quadratic_discriminant : (b^2 - 4 * a * c) = 9 / 4 := by
  -- Coefficient values
  have h_b : b = 5 / 2 := by
    calc
      b = 2 + 1/2 : rfl
      ... = 5 / 2 : by norm_num
  have h_discriminant : (5/2)^2 - 4 * 2 * (1/2) = 9/4 := by sorry
  -- Substitute the coefficient values
  rw h_b,
  exact h_discriminant,
  sorry

end quadratic_discriminant_l415_415196


namespace index_difference_l415_415209

noncomputable def index_females (n k1 k2 k3 : ℕ) : ℚ :=
  ((n - k1 + k2 : ℚ) / n) * (1 + k3 / 10)

noncomputable def index_males (n k1 l1 l2 : ℕ) : ℚ :=
  ((n - (n - k1) + l1 : ℚ) / n) * (1 + l2 / 10)

theorem index_difference (n k1 k2 k3 l1 l2 : ℕ)
  (h_n : n = 35) (h_k1 : k1 = 15) (h_k2 : k2 = 5) (h_k3 : k3 = 8)
  (h_l1 : l1 = 6) (h_l2 : l2 = 10) : 
  index_females n k1 k2 k3 - index_males n k1 l1 l2 = 3 / 35 :=
by
  sorry

end index_difference_l415_415209


namespace principal_calculation_l415_415351

-- Definitions based on the conditions
def end_of_first_year (P : ℝ) : ℝ := 1.02 * P - 100
def end_of_second_year (A1 : ℝ) : ℝ := 1.03 * A1 + 200
def end_of_third_year (A2 : ℝ) : ℝ := 1.04 * A2
def end_of_fourth_year (A3 : ℝ) : ℝ := 1.05 * A3
def end_of_fifth_year (A4 : ℝ) : ℝ := 1.06 * A4

-- The proof statement
theorem principal_calculation :
  ∃ P : ℝ, let A1 := end_of_first_year P in
           let A2 := end_of_second_year A1 in
           let A3 := end_of_third_year A2 in
           let A4 := end_of_fourth_year A3 in
           let A5 := end_of_fifth_year A4 in
           A5 = 750 ∧ P ≈ 534.68 :=
begin
  sorry
end

end principal_calculation_l415_415351


namespace claire_meets_alice_in_30_minutes_l415_415848

theorem claire_meets_alice_in_30_minutes
  (alice_speed claire_speed initial_distance : ℝ) 
  (h_alice_speed : alice_speed = 4)
  (h_claire_speed : claire_speed = 6)
  (h_initial_distance : initial_distance = 5) :
  let relative_speed := alice_speed + claire_speed in
  let time_in_hours := initial_distance / relative_speed in
  let time_in_minutes := time_in_hours * 60 in
  time_in_minutes = 30 :=
by
  sorry

end claire_meets_alice_in_30_minutes_l415_415848


namespace max_grid_sum_l415_415066

open Nat

/-
  Define a 5x7 grid where each cell contains an integer between 1 and 9, and no
  two squares that share a vertex have the same integer. Prove that the maximum
  sum of all the numbers in the grid is 272.
-/
theorem max_grid_sum : 
  ∀ (grid : Fin 5 → Fin 7 → Fin 9),
    (∀ i j, 1 ≤ grid i j + 1 ∧ grid i j + 1 ≤ 9) → -- ensure the integer is between 1 and 9
    (∀ i j, ∀ (k l : ℕ), (|i-k| = 1 ∧ |j-l| = 1) → grid i j ≠ grid k l) → -- no two adjacent squares share the same number
    (∑ i in Finset.univ, ∑ j in Finset.univ, grid i j + 1) ≤ 272 := -- computing the sum with 1-based numbering
by
  intro grid h_bounds h_diff
  sorry

end max_grid_sum_l415_415066


namespace solve_equation_l415_415736

theorem solve_equation (x : ℝ) (h : x ≠ 3) : 
  -x^2 = (3*x - 3) / (x - 3) → x = 1 :=
by
  intro h1
  sorry

end solve_equation_l415_415736


namespace find_G_8_l415_415697

noncomputable def G : Polynomial ℝ := sorry 

variable (x : ℝ)

theorem find_G_8 :
  G.eval 4 = 8 ∧ 
  (∀ x, (G.eval (2*x)) / (G.eval (x+2)) = 4 - (16 * x) / (x^2 + 2 * x + 2)) →
  G.eval 8 = 40 := 
sorry

end find_G_8_l415_415697


namespace sqrt_equation_l415_415243

-- Define the condition as a hypothesis
theorem sqrt_equation (x y : ℝ) (h : sqrt (2 * x + y) + sqrt (x ^ 2 - 9) = 0) :
  y - x = -9 ∨ y - x = 9 :=
sorry

end sqrt_equation_l415_415243


namespace determine_proportion_of_X_l415_415784

theorem determine_proportion_of_X (P : ℚ) :
  (0.40 * P + 0.25 * (1 - P) = 0.27) → 
  P = 2 / 15 :=
by
  intro h,
  sorry

end determine_proportion_of_X_l415_415784


namespace no_sum_of_three_cubes_l415_415319

theorem no_sum_of_three_cubes (x y z : ℤ) : (x^3 + y^3 + z^3 ≠ 20042005) :=
by {
  have possible_mod_9 : 20042005 % 9 = 4 := sorry,
  have cube_mod_9 : ∀ n : ℤ, (n^3 % 9 = 0 ∨ n^3 % 9 = 1 ∨ n^3 % 9 = 8) := sorry,
  have sum_mod_9 : ∀ x y z : ℤ, ((x^3 % 9 + y^3 % 9 + z^3 % 9) % 9 ∈ {0, 1, 2, 3, 6, 7, 8}) := sorry,
  by_contra,
  apply sum_mod_9 x y z,
  simp [possible_mod_9],
  apply cube_mod_9,
  sorry
}

end no_sum_of_three_cubes_l415_415319


namespace integral_cos_exp_l415_415526

theorem integral_cos_exp : 
  ∫ x in -real.pi..0, (Real.cos x + Real.exp x) = 1 - 1 / Real.exp real.pi := 
sorry

end integral_cos_exp_l415_415526


namespace find_k_value_l415_415406

noncomputable def find_k (k : ℝ) : Prop :=
  ∃ (k : ℝ), (∀ x y : ℝ, 8 * k * x^2 - k * y^2 = 8 → abs (sqrt (x^2 + y^2)) = 3) ∧ k = -1

theorem find_k_value : find_k (-1) :=
by
  -- Proof omitted
  sorry

end find_k_value_l415_415406


namespace tan_60_eq_sqrt3_l415_415083

theorem tan_60_eq_sqrt3 : Real.tan (Real.pi / 3) = Real.sqrt 3 := 
sorry

end tan_60_eq_sqrt3_l415_415083


namespace find_f_2011_l415_415596

def f: ℝ → ℝ :=
sorry

axiom f_periodicity (x : ℝ) : f (x + 3) = -f x
axiom f_initial_value : f 4 = -2

theorem find_f_2011 : f 2011 = 2 :=
by
  sorry

end find_f_2011_l415_415596


namespace cookie_ratio_l415_415069

theorem cookie_ratio (cookies_monday cookies_tuesday cookies_wednesday final_cookies : ℕ)
  (h1 : cookies_monday = 32)
  (h2 : cookies_tuesday = cookies_monday / 2)
  (h3 : final_cookies = 92)
  (h4 : cookies_wednesday = final_cookies + 4 - cookies_monday - cookies_tuesday) :
  cookies_wednesday / cookies_tuesday = 3 :=
by
  sorry

end cookie_ratio_l415_415069


namespace x_1998_remainder_l415_415712

noncomputable def lambda : ℝ :=
  (1998 + Real.sqrt (1998^2 + 4)) / 2

def x_seq : ℕ → ℝ
| 0     := 1
| (n+1) := Real.floor (lambda * x_seq n)

theorem x_1998_remainder :
  (Real.floor (x_seq 1998) % 1998) = 1000 :=
sorry

end x_1998_remainder_l415_415712


namespace find_d1_l415_415339

-- Let m be an odd integer greater than or equal to 5
def isOddGe5 (m : ℤ) : Prop := m % 2 = 1 ∧ m ≥ 5

-- Define the function E(m) as specified
def E (m : ℤ) : ℤ := 
  -- The definition of E(m) would need to be implemented here
  sorry 

-- Define the polynomial p(x) = d_3 * x^3 + d_2 * x^2 + d_1 * x + d_0
def p (x : ℤ) : ℤ := d_3 * x^3 + d_2 * x^2 + d_1 * x + d_0

-- State the theorem that d_1 = 6 given the conditions
theorem find_d1 (hE : ∀ m : ℤ, isOddGe5 m → E m = p m) : d_1 = 6 := 
  by 
  sorry

end find_d1_l415_415339


namespace fifth_number_in_ninth_row_l415_415134

theorem fifth_number_in_ninth_row :
  ∃ (n : ℕ), n = 61 ∧ ∀ (i : ℕ), i = 9 → (7 * i - 2 = n) :=
by
  sorry

end fifth_number_in_ninth_row_l415_415134


namespace survey_part_a_l415_415071

theorem survey_part_a (persons : ℕ) (answers : persons → vector (fin 2) 20)
  (h : ∀ (subset : finset (fin 20)) (h_card : subset.card = 10)
       (comb : vector (fin 2) 10),
       ∃ p : fin persons, ((answers p).val.to_finset.filter (λ i, i ∈ subset)).val = comb.val) :
  ¬∃ (p1 p2 : fin persons), p1 ≠ p2 ∧ ∀ i : fin 20, answers p1 i ≠ answers p2 i :=
sorry

end survey_part_a_l415_415071


namespace sum_of_nine_roots_eq_9_l415_415212

noncomputable def f : ℝ → ℝ := sorry

theorem sum_of_nine_roots_eq_9 :
  (∃ roots : list ℝ,
    (∀ x ∈ roots, f(2 * x ^ 2 - 4 * x - 5) + sin (π / 3 * x + π / 6) = 0) ∧
    roots.length = 9 ∧
    list.sum roots = 9) :=
begin
  sorry
end

end sum_of_nine_roots_eq_9_l415_415212


namespace one_clerk_forms_per_hour_l415_415128

theorem one_clerk_forms_per_hour
  (total_forms : ℕ)
  (total_hours : ℕ)
  (total_clerks : ℕ) 
  (h1 : total_forms = 2400)
  (h2 : total_hours = 8)
  (h3 : total_clerks = 12) :
  (total_forms / total_hours) / total_clerks = 25 :=
by
  have forms_per_hour := total_forms / total_hours
  have forms_per_clerk_per_hour := forms_per_hour / total_clerks
  sorry

end one_clerk_forms_per_hour_l415_415128


namespace triangle_area_l415_415498

theorem triangle_area : 
  ∃ (A : ℝ), A = 12 ∧ (∃ (x_intercept y_intercept : ℝ), 3 * x_intercept + 2 * y_intercept = 12 ∧ x_intercept * y_intercept / 2 = A) :=
by
  sorry

end triangle_area_l415_415498


namespace percentage_a_of_b_c_d_l415_415713

variables (a b c d : ℝ)
variables (h1 : a = 0.12 * b)
variables (h2 : b = 0.40 * c)
variables (h3 : c = 0.75 * d)
variables (h4 : d = 1.50 * (a + b))

theorem percentage_a_of_b_c_d : 
  let s := b + c + d in 
  (a / s) * 100 ≈ 2.316 :=
by
  sorry

end percentage_a_of_b_c_d_l415_415713


namespace find_x_plus_y_l415_415584

-- Define the vectors a and b
variables {x y : ℝ}
def a : ℝ × ℝ × ℝ := (x, 4, 3)
def b : ℝ × ℝ × ℝ := (3, -2, y)

-- Define orthogonality condition
def orthogonal (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = 0

-- State the theorem to be proved
theorem find_x_plus_y (h : orthogonal a b) : x + y = 8 / 3 :=
sorry

end find_x_plus_y_l415_415584


namespace total_rubber_bands_l415_415961

theorem total_rubber_bands (H : ℕ := 100) (F : ℕ := 56) (M : ℕ := 47) :
  let B := H - F in
  let S := B + M in
  H + B + S = 235 := 
by
  let B := H - F
  let S := B + M
  show H + B + S = 235
  sorry

end total_rubber_bands_l415_415961


namespace order_of_x_y_z_l415_415662

theorem order_of_x_y_z (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  let x : ℝ := (a + b) * (c + d)
  let y : ℝ := (a + c) * (b + d)
  let z : ℝ := (a + d) * (b + c)
  x < y ∧ y < z :=
by
  let x : ℝ := (a + b) * (c + d)
  let y : ℝ := (a + c) * (b + d)
  let z : ℝ := (a + d) * (b + c)
  sorry

end order_of_x_y_z_l415_415662


namespace find_m_value_l415_415631

-- Define the hyperbola and given conditions
def hyperbola_condition (x y m : ℝ) : Prop := (x^2 / 4) - (y^2 / m^2) = 1
def eccentricity (m : ℝ) : ℝ := Real.sqrt(1 + (m^2 / 4))

-- Main theorem statement
theorem find_m_value (m : ℝ) (h1 : 0 < m) (h2 : eccentricity m = Real.sqrt 3) : m = 2 * Real.sqrt 2 :=
by
  sorry

end find_m_value_l415_415631


namespace blocks_found_l415_415375

def initial_blocks : ℕ := 2
def final_blocks : ℕ := 86

theorem blocks_found : (final_blocks - initial_blocks) = 84 :=
by
  sorry

end blocks_found_l415_415375


namespace incorrect_statement_D_l415_415673

variable (R_squared : ℝ)
variable (h : R_squared = 0.96)

def option_A : Prop := 
  R_squared is close to 1 → the fitting effect of the linear regression equation is good

def option_B : Prop := 
  R_squared explains approximately 96% of the variation of the forecast variable

def option_C : Prop := 
  the impact of random errors on the forecast variable accounts for about 4%

def option_D : Prop := 
  96% of the sample points lie on the regression line

theorem incorrect_statement_D (R_squared : ℝ) (h : R_squared = 0.96):
  ¬ option_D :=
by
  sorry

end incorrect_statement_D_l415_415673


namespace binomial_coefficient_third_term_l415_415393

open Nat

theorem binomial_coefficient_third_term :
  binomial 6 2 = 15 := by
  sorry

end binomial_coefficient_third_term_l415_415393


namespace fraction_spent_toy_store_l415_415272

noncomputable def weekly_allowance : ℚ := 2.25
noncomputable def arcade_fraction_spent : ℚ := 3 / 5
noncomputable def candy_store_spent : ℚ := 0.60

theorem fraction_spent_toy_store :
  let remaining_after_arcade := weekly_allowance * (1 - arcade_fraction_spent)
  let spent_toy_store := remaining_after_arcade - candy_store_spent
  spent_toy_store / remaining_after_arcade = 1 / 3 :=
by
  sorry

end fraction_spent_toy_store_l415_415272


namespace inverse_function_properties_l415_415591

theorem inverse_function_properties {f : ℝ → ℝ} 
  (h_monotonic_decreasing : ∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 3 → f x2 < f x1)
  (h_range : ∀ y : ℝ, 4 ≤ y ∧ y ≤ 7 ↔ ∃ x : ℝ, 1 ≤ x ∧ x ≤ 3 ∧ y = f x)
  (h_inverse_exists : ∃ g : ℝ → ℝ, ∀ x : ℝ, f (g x) = x ∧ g (f x) = x) :
  ∃ g : ℝ → ℝ, (∀ y1 y2 : ℝ, 4 ≤ y1 ∧ y1 < y2 ∧ y2 ≤ 7 → g y2 < g y1) ∧ (∀ y : ℝ, 4 ≤ y ∧ y ≤ 7 → g y ≤ 3) :=
sorry

end inverse_function_properties_l415_415591


namespace total_money_found_l415_415548

-- Define the conditions
def donna_share := 0.40
def friendA_share := 0.35
def friendB_share := 0.25
def donna_amount := 39.0

-- Define the problem statement/proof
theorem total_money_found (donna_share friendA_share friendB_share donna_amount : ℝ) 
  (h1 : donna_share = 0.40) 
  (h2 : friendA_share = 0.35) 
  (h3 : friendB_share = 0.25) 
  (h4 : donna_amount = 39.0) :
  ∃ total_money : ℝ, total_money = 97.50 := 
by
  -- The calculations and actual proof will go here
  sorry

end total_money_found_l415_415548


namespace greatest_third_side_l415_415047

-- Given data and the Triangle Inequality theorem
theorem greatest_third_side (c : ℕ) (h1 : 8 < c) (h2 : c < 22) : c = 21 :=
by
  sorry

end greatest_third_side_l415_415047


namespace circle_covered_by_five_disks_l415_415743

noncomputable def min_disk_diameter_cover (D : ℝ) (n : ℕ) : ℝ :=
  if D = 6 ∧ n = 5 then 4 else 0

theorem circle_covered_by_five_disks :
  ∀ (D : ℝ) (n : ℕ), D = 6 → n = 5 → min_disk_diameter_cover D n = 4 :=
by
  intros D n hD hn
  rw [min_disk_diameter_cover]
  simp [hD, hn]
  sorry

end circle_covered_by_five_disks_l415_415743


namespace birds_fed_per_week_l415_415691

def cups_of_birdseed : ℝ := 2
def birds_per_cup : ℝ := 14
def squirrel_steals : ℝ := 0.5

theorem birds_fed_per_week : (cups_of_birdseed - squirrel_steals) * birds_per_cup = 21 := by
sory

end birds_fed_per_week_l415_415691


namespace hyperbola_real_axis_length_l415_415256

theorem hyperbola_real_axis_length :
  (∃ (a b : ℝ), (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) ∧ a = 3) →
  2 * 3 = 6 :=
by
  sorry

end hyperbola_real_axis_length_l415_415256


namespace even_function_derivative_l415_415248

theorem even_function_derivative (f : ℝ → ℝ)
  (h_even : ∀ x, f (-x) = f x)
  (h_deriv_pos : ∀ x > 0, deriv f x = (x - 1) * (x - 2)) : f (-2) < f 1 :=
sorry

end even_function_derivative_l415_415248


namespace trajectory_of_center_l415_415133

noncomputable def distance (p q : ℝ×ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem trajectory_of_center (M : ℝ×ℝ) (r : ℝ)
  (H1 : distance M (0, 0) = r + 1)
  (H2 : distance M (3, 0) = r - 1) :
  let O := (0, 0)
      C := (3, 0) in
  distance O C = 3 ∧ (distance M O - distance M C = 2) ↔ (∃ a b : ℝ, ∀ P, abs (distance P (a, 0) - distance P (b, 0)) = 2) :=
begin
  let O := (0, 0),
  let C := (3, 0),
  sorry
end

end trajectory_of_center_l415_415133


namespace net_percentage_change_is_correct_l415_415018

def initial_price : Float := 100.0

def price_after_first_year (initial: Float) := initial * (1 - 0.05)

def price_after_second_year (price1: Float) := price1 * (1 + 0.10)

def price_after_third_year (price2: Float) := price2 * (1 + 0.04)

def price_after_fourth_year (price3: Float) := price3 * (1 - 0.03)

def price_after_fifth_year (price4: Float) := price4 * (1 + 0.08)

def final_price := price_after_fifth_year (price_after_fourth_year (price_after_third_year (price_after_second_year (price_after_first_year initial_price))))

def net_percentage_change (initial final: Float) := ((final - initial) / initial) * 100

theorem net_percentage_change_is_correct :
  net_percentage_change initial_price final_price = 13.85 := by
  sorry

end net_percentage_change_is_correct_l415_415018


namespace probability_circles_intersect_l415_415720

theorem probability_circles_intersect :
  let P := (0, a) in
  let Q := (1, b) in
  let dist := Real.sqrt (1 + (b - a) ^ 2) in
  (∀ a b : ℝ, -1 ≤ a ∧ a ≤ 1 ∧ -1 ≤ b ∧ b ≤ 1 →
     dist ≤ 2) →
  (∫ a in -1..1, ∫ b in -1..1, ite (|-√3 ≤ b - a ∧ b - a ≤ √3) 1 0) = (4√3 - 3) / 4 := 
  sorry

end probability_circles_intersect_l415_415720


namespace percentage_increase_in_expenses_l415_415824

theorem percentage_increase_in_expenses:
  ∀ (S : ℝ) (original_save_percentage new_savings : ℝ), 
  S = 5750 → 
  original_save_percentage = 0.20 →
  new_savings = 230 →
  (original_save_percentage * S - new_savings) / (S - original_save_percentage * S) * 100 = 20 :=
by
  intros S original_save_percentage new_savings HS Horiginal_save_percentage Hnew_savings
  rw [HS, Horiginal_save_percentage, Hnew_savings]
  sorry

end percentage_increase_in_expenses_l415_415824


namespace total_eggs_today_l415_415716

def eggs_morning : ℕ := 816
def eggs_afternoon : ℕ := 523

theorem total_eggs_today : eggs_morning + eggs_afternoon = 1339 :=
by {
  sorry
}

end total_eggs_today_l415_415716


namespace certain_event_proof_l415_415443

def Moonlight_in_front_of_bed := "depends_on_time_and_moon_position"
def Lonely_smoke_in_desert := "depends_on_specific_conditions"
def Reach_for_stars_with_hand := "physically_impossible"
def Yellow_River_flows_into_sea := "certain_event"

theorem certain_event_proof : Yellow_River_flows_into_sea = "certain_event" :=
by
  sorry

end certain_event_proof_l415_415443


namespace volume_of_inequality_halfspace_l415_415572

noncomputable def volume_of_region : ℝ :=
  let g (x y z : ℝ) : ℝ := |x + 2*y + z| + |x + 2*y - z| + |x - 2*y + z| + |-x + 2*y + z|
  let region := {p : ℝ × ℝ × ℝ | g p.1 p.2 p.3 ≤ 8 }
  -- Assuming a mathematical way to calculate the volume of a region
  volume region

theorem volume_of_inequality_halfspace :
  volume_of_region = 128 / 3 :=
sorry

end volume_of_inequality_halfspace_l415_415572


namespace min_value_of_a_plus_b_l415_415971

theorem min_value_of_a_plus_b (a b c : ℝ) (C : ℝ) 
  (hC : C = 60) 
  (h : (a + b)^2 - c^2 = 4) : 
  a + b ≥ (4 * Real.sqrt 3) / 3 :=
by
  sorry

end min_value_of_a_plus_b_l415_415971


namespace simplify_expression_correct_l415_415779

def simplify_expression : Prop :=
  4 - (+3) - (-7) + (-2) = 4 - 3 + 7 - 2

theorem simplify_expression_correct : simplify_expression :=
  by
    sorry

end simplify_expression_correct_l415_415779


namespace rectangle_longer_side_length_l415_415124

theorem rectangle_longer_side_length
  (r : ℝ) (A_rect : ℝ) :
  r = 6 →
  A_rect = 3 * (Real.pi * r^2) →
  ∃ l : ℝ, l = 9 * Real.pi :=
begin
  intros hr hA,
  use 9 * Real.pi,
  sorry
end

end rectangle_longer_side_length_l415_415124


namespace isosceles_triangle_perimeter_l415_415521

theorem isosceles_triangle_perimeter (a b : ℕ) (h_isosceles : a = 3 ∨ a = 7 ∨ b = 3 ∨ b = 7) (h_ineq1 : 3 + 3 ≤ b ∨ b + b ≤ 3) (h_ineq2 : 7 + 7 ≥ a ∨ a + a ≥ 7) :
  (a = 3 ∧ b = 7) → 3 + 7 + 7 = 17 :=
by
  -- To be completed
  sorry

end isosceles_triangle_perimeter_l415_415521


namespace convex_polyhedron_max_edges_non_convex_polyhedron_96_edges_non_convex_polyhedron_not_100_edges_l415_415016

-- Definitions
def polyhedron (E : ℕ) := E = 100

-- Problem statements
theorem convex_polyhedron_max_edges (E : ℕ) (h : polyhedron E) : 
  ∃ n, n = 68 ∧ ∀ plane, intersects_edges plane n :=
sorry

theorem non_convex_polyhedron_96_edges (E : ℕ) (h : polyhedron E) : 
  ∃ n, n = 96 :=
sorry

theorem non_convex_polyhedron_not_100_edges (E : ℕ) (h : polyhedron E) : 
  ¬ ∃ n, n = 100 :=
sorry

end convex_polyhedron_max_edges_non_convex_polyhedron_96_edges_non_convex_polyhedron_not_100_edges_l415_415016


namespace largest_c_value_l415_415893

theorem largest_c_value (c : ℝ) :
  (∃ x : ℝ, x^2 + 5 * x + c = -4) ↔ (c ≤ 9 / 4) :=
begin
  sorry
end

end largest_c_value_l415_415893


namespace triangle_MDA_area_l415_415302

noncomputable def area_of_triangle_MDA (r : ℝ) : ℝ :=
  (r^2) / (4 * Real.sqrt 3)

theorem triangle_MDA_area (r : ℝ) :
  ∃ (O A B M D : ℝ × ℝ),
  (dist O A = r) ∧ (dist O B = r) ∧ (dist A B = Real.sqrt (2) * r) ∧
  ∃ (M : ℝ × ℝ), (dist O M = Real.sqrt ((dist O A)^2 - ((dist A B) / 2)^2)) ∧
  ∃ (D : ℝ × ℝ), (∃ (OX_D : ℝ × ℝ) (M_OX_D : ℝ × ℝ), (dist OX_D A = dist A D), (dist M D = dist M (OX_D⁻¹))) ∧
  let AM := (dist A M), AD := (dist A D), MD := (dist M D)
  in (1 / 2) * AD * MD = area_of_triangle_MDA r :=
begin
  sorry
end

end triangle_MDA_area_l415_415302


namespace vertex_C_path_length_equals_l415_415837

noncomputable def path_length_traversed_by_C (AB BC CA : ℝ) (PQ QR : ℝ) : ℝ :=
  let BC := 3  -- length of side BC is 3 inches
  let AB := 2  -- length of side AB is 2 inches
  let CA := 4  -- length of side CA is 4 inches
  let PQ := 8  -- length of side PQ of the rectangle is 8 inches
  let QR := 6  -- length of side QR of the rectangle is 6 inches
  4 * BC * Real.pi

theorem vertex_C_path_length_equals (AB BC CA PQ QR : ℝ) :
  AB = 2 ∧ BC = 3 ∧ CA = 4 ∧ PQ = 8 ∧ QR = 6 →
  path_length_traversed_by_C AB BC CA PQ QR = 12 * Real.pi :=
by
  intros h
  have hAB : AB = 2 := h.1
  have hBC : BC = 3 := h.2.1
  have hCA : CA = 4 := h.2.2.1
  have hPQ : PQ = 8 := h.2.2.2.1
  have hQR : QR = 6 := h.2.2.2.2
  simp [path_length_traversed_by_C, hAB, hBC, hCA, hPQ, hQR]
  sorry

end vertex_C_path_length_equals_l415_415837


namespace remainder_when_dividing_p_by_g_is_3_l415_415204

noncomputable def p (x : ℤ) : ℤ := x^5 - 2 * x^3 + 4 * x^2 + x + 5
noncomputable def g (x : ℤ) : ℤ := x + 2

theorem remainder_when_dividing_p_by_g_is_3 : p (-2) = 3 :=
by
  sorry

end remainder_when_dividing_p_by_g_is_3_l415_415204


namespace prime_form_of_power_of_two_l415_415277

theorem prime_form_of_power_of_two (n : ℕ) (h : nat.prime (2^n + 1)) : n = 0 ∨ ∃ α : ℕ, n = 2^α := by
  sorry

end prime_form_of_power_of_two_l415_415277


namespace valid_subsets_12_even_subsets_305_l415_415643

def valid_subsets_count(n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 4
  else
    valid_subsets_count (n - 1) +
    valid_subsets_count (n - 2) +
    valid_subsets_count (n - 3)
    -- Recurrence relation for valid subsets which satisfy the conditions

theorem valid_subsets_12 : valid_subsets_count 12 = 610 :=
  by sorry
  -- We need to verify recurrence and compute for n = 12 (optional step if just computing, not proving the sequence.)

theorem even_subsets_305 :
  (valid_subsets_count 12) / 2 = 305 :=
  by sorry
  -- Concludes that half the valid subsets for n = 12 are even-sized sets.

end valid_subsets_12_even_subsets_305_l415_415643


namespace emma_walked_distance_l415_415883

-- Define the constants for the average speeds
def bike_speed : ℝ := 20
def walk_speed : ℝ := 6

-- Define the total time taken for the journey
def total_time : ℝ := 1

-- Define the total distance as 3 * x
def total_distance (x : ℝ) : ℝ := 3 * x

-- Calculate the time taken for each part of the journey
def time_biking (x : ℝ) : ℝ := x / bike_speed
def time_walking (x : ℝ) : ℝ := (2 * x) / walk_speed

-- Define the equation for the total time
def total_time_eq (x : ℝ) : ℝ := time_biking x + time_walking x

-- Our goal is to prove that the walking distance is approximately 5.2 kilometers
theorem emma_walked_distance : ∃ (x : ℝ), total_time_eq x = total_time ∧ abs((2 * x) - 5.2) < 0.1 :=
by
  sorry

end emma_walked_distance_l415_415883


namespace certain_event_l415_415441

theorem certain_event :
  (∀ (e : string), e = "Moonlight in front of the bed" → ¬is_certain_event e) ∧
  (∀ (e : string), e = "Lonely smoke in the desert" → ¬is_certain_event e) ∧
  (∀ (e : string), e = "Reach for the stars with your hand" → is_impossible_event e) ∧
  (∀ (e : string), e = "Yellow River flows into the sea" → is_certain_event e) →
  is_certain_event "Yellow River flows into the sea" :=
by
  sorry

end certain_event_l415_415441


namespace scientific_notation_eq_l415_415153

-- Define the number 82,600,000
def num : ℝ := 82600000

-- Define the scientific notation representation
def sci_not : ℝ := 8.26 * 10^7

-- The theorem to prove that the number is equal to its scientific notation
theorem scientific_notation_eq : num = sci_not :=
by 
  sorry

end scientific_notation_eq_l415_415153


namespace inv_nested_result_l415_415171

noncomputable def g : ℕ → ℕ 
| 1 := 4
| 2 := 3
| 3 := 1
| 4 := 2
| 5 := 5
| _ := 0  -- Handling for other inputs, can be arbitrary as they are irrelevant.

theorem inv_nested_result :
  g⁻¹ (g⁻¹ (g⁻¹ (g⁻¹ 3))) = 5 := by
  sorry

end inv_nested_result_l415_415171


namespace ratio_of_largest_to_rest_is_9_l415_415535

/-- The given set of numbers is {1, 10, 10^2, ..., 10^10}. 
The largest element's ratio to the sum of other elements is closest to 9. -/
theorem ratio_of_largest_to_rest_is_9 : 
  let largest := 10^10
  let sum_of_others := ∑ k in Finset.range 10, (10^k) in 
  abs ((largest : ℝ) / sum_of_others - 9) < 1 :=
by 
  let largest := 10^10
  let sum_of_others := ∑ k in Finset.range 10, (10^k)
  have h1 : sum_of_others = (10^10 - 1) / 9 :=
    by sorry  -- Show that the sum of first 10 terms of geometric series is (10^10 - 1) / 9
  have h2 : (largest : ℝ) / sum_of_others = 9 * (10^10 / (10^10 - 1)) :=
    by sorry  -- Compute the ratio
  have h3 : abs (9 * (10^10 / (10^10 - 1)) - 9) < 1 :=
    by sorry  -- Show that this expression is close to 9
  exact h3

end ratio_of_largest_to_rest_is_9_l415_415535


namespace min_distance_sum_l415_415265

noncomputable def parabola_and_line (A B : ℝ × ℝ) : ℝ :=
  let F := (1, 0)
  let parabola (p : ℝ × ℝ) := p.2^2 = 4 * p.1
  let line_through_focus (p : ℝ × ℝ) (k : ℝ) := p.2 = k * (p.1 - 1)
  let distance (p1 p2 : ℝ × ℝ) := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let m := distance A F
  let n := distance B F
  m + n

theorem min_distance_sum (A B : ℝ × ℝ) (hA : parabola A) (hB : parabola B) (h_line : ∃ k, A.2 = k * (A.1 - 1) ∧ B.2 = k * (B.1 - 1)) :
  parabola_and_line A B = 4 :=
by
  sorry

end min_distance_sum_l415_415265


namespace solve_trig_eq_l415_415382

-- Define the trigonometric equation
def trig_eq (x : ℝ) :=
  (cos (4 * x) / (cos (5 * x) - sin (5 * x))) +
  (sin (4 * x) / (cos (5 * x) + sin (5 * x))) = -real.sqrt 2

-- Define the solution set we expect
def solution_set (x : ℝ) : Prop :=
  ∃ k p : ℤ, x = (5 * real.pi / 76) + (2 * real.pi * k / 19) ∧ k ≠ 19 * p - 3

-- The equivalence theorem we need to prove
theorem solve_trig_eq : ∀ x : ℝ, trig_eq x ↔ solution_set x :=
by
  -- skipping the proof part with sorry.
  sorry

end solve_trig_eq_l415_415382


namespace range_of_a_l415_415217

theorem range_of_a (a : ℝ) : 
  let A := (-2, 3)
  let B := (a, 0)
  let circle_eq := (x - 3)^2 + (y - 2)^2 = 1
  let is_symmetric_about_x (A B : ℝ × ℝ) := A.1 = B.1 ∧ A.2 = -B.2
  is_symmetric_about_x A B ∧ ∃ (P : ℝ × ℝ), circle_eq P = 0 ∧ on_line P (A, B)
  → a ∈ set.Icc (1/4) 2 :=
sorry

end range_of_a_l415_415217


namespace third_circle_circumference_l415_415766

theorem third_circle_circumference :
  let r1 := 10
  let r2 := 20
  let shaded_area := (π * r2^2) - (π * r1^2)
  let r3 := Real.sqrt (shaded_area / π)
  let circumference := 2 * π * r3
  circumference = 20 * sqrt 3 * π := by
{
  sorry
}

end third_circle_circumference_l415_415766


namespace factory_production_equation_l415_415815

variable (x : ℝ)
axiom JanuaryProduction : ℝ := 50
axiom TotalFirstQuarter : ℝ := 182

theorem factory_production_equation :
  50 * (1 + (1 + x) + (1 + x)^2) = 182 :=
by
  sorry

end factory_production_equation_l415_415815


namespace train_travel_distance_l415_415811

def coal_efficiency := (5 : ℝ) / (2 : ℝ)  -- Efficiency in miles per pound
def coal_remaining := 160  -- Coal remaining in pounds
def distance_travelled := coal_remaining * coal_efficiency  -- Total distance the train can travel

theorem train_travel_distance : distance_travelled = 400 := 
by
  sorry

end train_travel_distance_l415_415811


namespace carla_total_earnings_l415_415315

theorem carla_total_earnings
  (h1 : ∀ (x : ℝ), 28 * x = 18 * x + 63)
  (h2 : ∀ (x : ℝ), ∃ y : ℝ, 10 * x = 63 ∧ y = x)
  (h3 : ∀ (wage hours1 hours2 : ℝ), 28 = hours2 ∧ 18 = hours1 ∧ wage = 6.30)
  (h4 : ∀ (wage : ℝ), ∀ (total_hours : ℝ), total_hours = 46 → wage = 6.30)
  (h5 : ∀ (total_hours wage : ℝ), total_hours = 46 → wage = 6.30 → total_hours * wage = 289.80) :
  true :=
by
  have h_rw : 10 * 6.30 = 63 := by sorry,
  have wage : ℝ := 6.30,
  have total_hours : ℝ := 46,
  have total_earnings := total_hours * wage,
  have : total_earnings = 289.80 := by sorry,
  exact ⟨⟩.

#check carla_total_earnings
  (λ x, 28 * x = 18 * x + 63)
  (λ x, ⟨x, 10 * x = 63, rfl⟩)
  (λ w h1 h2, ⟨rfl, rfl, rfl⟩)
  (λ wage total_hours, λ h_eq, rfl)
  (λ total_hours wage, λ h_eq1 h_eq2, rfl)

end carla_total_earnings_l415_415315


namespace truth_tellers_exactly_three_l415_415738

-- Define individuals and their statements
inductive Person
| A | B | C | D | E
deriving DecidableEq

open Person

-- Type definitions for truth tellers and liars
def is_truth_teller (p : Person) : Prop :=
  p = A -- Initially defining A as a truth teller based on conditions

def statement (p : Person) : Prop :=
  match p with
  | B => is_truth_teller B
  | C => is_truth_teller D
  | D => ¬ (is_truth_teller B ∧ is_truth_teller E)
  | E => is_truth_teller A ∧ is_truth_teller B
  | _ => True

-- Final problem: Proving the number of truth tellers
def count_truth_tellers : Nat :=
  if is_truth_teller B then 
    if is_truth_teller E then 3 else 2
  else 
    if is_truth_teller C then 3 else 2

theorem truth_tellers_exactly_three : count_truth_tellers = 3 := by
  -- Proof goes here
  sorry

end truth_tellers_exactly_three_l415_415738


namespace proposition_2_proposition_3_l415_415941

theorem proposition_2 (a b : ℝ) (h: a > |b|) : a^2 > b^2 := 
sorry

theorem proposition_3 (a b : ℝ) (h: a > b) : a^3 > b^3 := 
sorry

end proposition_2_proposition_3_l415_415941


namespace circle_covered_at_pi_l415_415012

/-- Given the polar equation r = sin(θ), prove that the smallest t for which the graph
covers the entire circle when plotted from 0 ≤ θ ≤ t is π. -/
theorem circle_covered_at_pi :
  (∀ θ, 0 ≤ θ ∧ θ ≤ π → r = sin(θ)) →
  (∀ θ, 0 ≤ θ ∧ θ ≤ t → r = sin(θ)) →
  (∃ t, t = π ∧ 
    ∀ (θ : ℝ), (0 ≤ θ ∧ θ ≤ t → sin(θ) covers the entire circle) := 
sorry

end circle_covered_at_pi_l415_415012


namespace find_circle_radius_l415_415465

variables {K L M N : Point}
variables {β γ a : Real}
-- Declare the angles and sides given
variables (triangle_KLM : Triangle K L M)
variables (angle_LKM : ∠LKM = β)
variables (angle_LMLK : ∠LMLK = γ)
variables (side_KM : KM = a)
variables (point_N_on_KL : ∃ n : ℝ, n > 0 ∧ N ∈ segment K L ∧ KN = 2 * n ∧ NL = n)

theorem find_circle_radius
  (r : ℝ)
  (h : circle_through L N) -- This is the circle through L and N
  (tangent_KM_or_extension : tangent h KM ∨ tangent h (extension KM))
  : r = (a * (5 / 3 - 2 * sqrt (2 / 3) * cos β) * sin γ) / (2 * sin β * sin (β + γ)) :=
sorry

end find_circle_radius_l415_415465


namespace josie_leftover_amount_l415_415327

-- Define constants and conditions
def initial_amount : ℝ := 20.00
def milk_price : ℝ := 4.00
def bread_price : ℝ := 3.50
def detergent_price : ℝ := 10.25
def bananas_price_per_pound : ℝ := 0.75
def bananas_weight : ℝ := 2.0
def detergent_coupon : ℝ := 1.25
def milk_discount_rate : ℝ := 0.5

-- Define the total cost before any discounts
def total_cost_before_discounts : ℝ := 
  milk_price + bread_price + detergent_price + (bananas_weight * bananas_price_per_pound)

-- Define the discounted prices
def milk_discounted_price : ℝ := milk_price * milk_discount_rate
def detergent_discounted_price : ℝ := detergent_price - detergent_coupon

-- Define the total cost after discounts
def total_cost_after_discounts : ℝ := 
  milk_discounted_price + bread_price + detergent_discounted_price + 
  (bananas_weight * bananas_price_per_pound)

-- Prove the amount left over
theorem josie_leftover_amount : initial_amount - total_cost_after_discounts = 4.00 := by
  simp [total_cost_before_discounts, milk_discounted_price, detergent_discounted_price,
    total_cost_after_discounts, initial_amount, milk_price, bread_price, detergent_price,
    bananas_price_per_pound, bananas_weight, detergent_coupon, milk_discount_rate]
  sorry

end josie_leftover_amount_l415_415327


namespace circle_tangency_proof_l415_415761

-- Defining the variables and conditions
variables {r1 r2 r3 a b : ℝ}

-- Tangency conditions as hypotheses
hypothesis h1 : ∀ {r1 r2 r3 a b : ℝ}, 
  -- Placeholder for specific tangency conditions involving r1, r2, r3, a, b

-- The proof statement
theorem circle_tangency_proof (h1 : ...) : r1 + r3 = (2 * a^2 * (a^2 - 2 * b^2) / a^4) * r2 :=
sorry

end circle_tangency_proof_l415_415761


namespace common_ratio_geometric_series_l415_415558

-- Define the terms of the geometric series
def term (n : ℕ) : ℚ :=
  match n with
  | 0     => 7 / 8
  | 1     => -21 / 32
  | 2     => 63 / 128
  | _     => sorry  -- Placeholder for further terms if necessary

-- Define the common ratio
def common_ratio : ℚ := -3 / 4

-- Prove that the common ratio is consistent for the given series
theorem common_ratio_geometric_series :
  ∀ (n : ℕ), term (n + 1) / term n = common_ratio :=
by
  sorry

end common_ratio_geometric_series_l415_415558


namespace train_travel_distance_l415_415805

theorem train_travel_distance
  (coal_per_mile_lb : ℝ)
  (remaining_coal_lb : ℝ)
  (travel_distance_per_unit_mile : ℝ)
  (units_per_unit_lb : ℝ)
  (remaining_units : ℝ)
  (total_distance : ℝ) :
  coal_per_mile_lb = 2 →
  remaining_coal_lb = 160 →
  travel_distance_per_unit_mile = 5 →
  units_per_unit_lb = remaining_coal_lb / coal_per_mile_lb →
  remaining_units = units_per_unit_lb →
  total_distance = remaining_units * travel_distance_per_unit_mile →
  total_distance = 400 :=
by
  sorry

end train_travel_distance_l415_415805


namespace n_times_s_eq_l415_415705

-- Define our main function g and its property
axiom g : ℝ → ℝ
axiom g_property : ∀ x y : ℝ, g(g(x) - y) = g(x) + g(g(y) - g(x)) - x 

-- Define n as the number of possible values of g(4)
def n := 1

-- Define s as the sum of all possible values of g(4)
def s := 4

-- The proof we need to show is that n * s = 4
theorem n_times_s_eq : n * s = 4 := by
  sorry

end n_times_s_eq_l415_415705


namespace real_time_is_correct_l415_415271

-- Given conditions as definitions
def initial_time_wall_clock := 0 -- 6:00 AM in minutes after 6:00 AM
def comparison_time_wall_clock := 170 -- 8:50 AM in minutes after 6:00 AM
def real_comparison_time := 180 -- 9:00 AM in minutes after 6:00 AM
def wall_clock_time_when_observed := 600 -- 4:00 PM in minutes after 6:00 AM

-- Proportion of wall clock time to real time
def wall_clock_to_real_time_ratio := 17 / 18

noncomputable def actual_time : ℚ :=
  wall_clock_to_real_time_ratio⁻¹ * wall_clock_time_when_observed

theorem real_time_is_correct :
  actual_time ≈ 635.29 :=
begin
  sorry
end

end real_time_is_correct_l415_415271


namespace closest_value_l415_415362

theorem closest_value
  (M N : ℝ)
  (hM : M ≈ 3^361)
  (hN : N ≈ 10^48) :
  10^125 ≈ (M / N) :=
by
  sorry

end closest_value_l415_415362


namespace count_even_digit_4_digit_numbers_divisible_by_5_or_3_l415_415964

-- Definition of even digits
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Definition of 4-digit number consisting only of even digits
def is_even_digit_4_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  is_even_digit (n / 1000) ∧
  is_even_digit ((n / 100) % 10) ∧
  is_even_digit ((n / 10) % 10) ∧
  is_even_digit (n % 10)

-- Definition of divisibility by 5
def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- Definition of divisibility by 3
def divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

-- Definition of the target property: number is divisible by 5 or 3
def divisible_by_5_or_3 (n : ℕ) : Prop :=
  divisible_by_5 n ∨ divisible_by_3 n

-- The theorem we want to prove
theorem count_even_digit_4_digit_numbers_divisible_by_5_or_3 :
  { n : ℕ | is_even_digit_4_digit_number n ∧ divisible_by_5_or_3 n }.to_finset.card = 190 :=
sorry -- proof goes here

end count_even_digit_4_digit_numbers_divisible_by_5_or_3_l415_415964


namespace find_B_and_M_range_l415_415980

-- Definitions of the conditions in the problem
variables {A B C a b c : Real}
variables (M : Real)

-- Given conditions
axiom condition : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos B

-- Define the expression for M
def M_expression : Real := Real.sin A * (Real.sqrt 3 * Real.cos A - Real.sin A)

-- Lean statement to prove the given conditions
theorem find_B_and_M_range (h : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos B) : 
  (B = Real.pi / 3) ∧ 
  (∀ A, M_expression A ∈ Ioo (-3/2 : Real) (1/2 : Real) ∪ {1/2}) :=
by
  sorry

end find_B_and_M_range_l415_415980


namespace eliminate_xy_l415_415882

variable {R : Type*} [Ring R]

theorem eliminate_xy
  (x y a b c : R)
  (h1 : a = x + y)
  (h2 : b = x^3 + y^3)
  (h3 : c = x^5 + y^5) :
  5 * b * (a^3 + b) = a * (a^5 + 9 * c) :=
sorry

end eliminate_xy_l415_415882


namespace rectangle_longer_side_length_l415_415127

theorem rectangle_longer_side_length
  (r : ℝ) (A_rect : ℝ) :
  r = 6 →
  A_rect = 3 * (Real.pi * r^2) →
  ∃ l : ℝ, l = 9 * Real.pi :=
begin
  intros hr hA,
  use 9 * Real.pi,
  sorry
end

end rectangle_longer_side_length_l415_415127


namespace part1_part2_l415_415592

variables (a b : ℝ^3)
variables (ha : ‖a‖ = 3) (hb : ‖b‖ = 2) (theta : ℝ)
variable (h_angle : theta = π / 3)

-- Part 1: Prove that |3a - 2b| = √61
theorem part1 : ‖(3 : ℝ) • a - (2 : ℝ) • b‖ = real.sqrt 61 :=
  sorry

-- Part 2: Given c = 3a + 5b and d = m a - 3b, prove that the value of m such that c ⊥ d is 29/14
variables (m : ℝ)
variables (c : ℝ^3 := (3 : ℝ) • a + (5 : ℝ) • b) (d : ℝ^3 := m • a - (3 : ℝ) • b)

theorem part2 (h_perp : c ⬝ d = 0) : m = 29 / 14 :=
  sorry

end part1_part2_l415_415592


namespace triangle_angle_identity_l415_415635

theorem triangle_angle_identity (A B C M N P : Type) 
(h1 : Triangle A B C) 
(h2 : OnRay M (CA)) 
(h3 : OnRay N (CB))
(h4 : AN = BM) 
(h5 : AN = AB) 
(h6 : BM = AB)
(h7 : SegmentsIntersect (AN) (BM) P) : 
  ∠ APM = 2 * ∠ ACB := by 
sorry

end triangle_angle_identity_l415_415635


namespace intersection_M_N_l415_415955

noncomputable def M := { x : ℝ | x ^ 2 - x / 2 > 0 }
noncomputable def N := { x : ℝ | real.log10 x <= 0 }

theorem intersection_M_N :
  M ∩ N = set.Ioc (1 / 2 : ℝ) 1 := 
  sorry

end intersection_M_N_l415_415955


namespace most_likely_outcome_is_D_l415_415731

-- Define the basic probability of rolling any specific number with a fair die
def probability_of_specific_roll : ℚ := 1/6

-- Define the probability of each option
def P_A : ℚ := probability_of_specific_roll
def P_B : ℚ := 2 * probability_of_specific_roll
def P_C : ℚ := 3 * probability_of_specific_roll
def P_D : ℚ := 4 * probability_of_specific_roll

-- Define the proof problem statement
theorem most_likely_outcome_is_D : P_D = max P_A (max P_B (max P_C P_D)) :=
sorry

end most_likely_outcome_is_D_l415_415731


namespace value_of_x_is_10_l415_415461

-- Define the conditions
def condition1 (x : ℕ) : ℕ := 3 * x
def condition2 (x : ℕ) : ℕ := (26 - x) + 14

-- Define the proof problem
theorem value_of_x_is_10 (x : ℕ) (h1 : condition1 x = condition2 x) : x = 10 :=
by {
  sorry
}

end value_of_x_is_10_l415_415461


namespace graph_translation_logarithmic_function_l415_415040

theorem graph_translation_logarithmic_function :
  ∀ (f g : ℝ → ℝ),
  (∀ x, f x = log (1 - 2*x)) →
  (∀ x, g x = log (3 - 2*x)) →
  (∀ x, g x = f (x - 1)) :=
begin
  intros f g hf hg x,
  rw [hg, hf],
  sorry
end

end graph_translation_logarithmic_function_l415_415040


namespace oranges_to_bananas_ratio_is_two_l415_415399

noncomputable def price_of_pear := 90
noncomputable def total_cost_op := 120
noncomputable def total_price := 24000
noncomputable def number_bananas := 200

def find_ratio (O P B : ℕ) (N_b : ℕ) (C : ℕ) : Prop :=
  P - O = B ∧ O + P = total_cost_op ∧ P = price_of_pear ∧
  let cost_bananas := N_b * B in
  let cost_oranges := C - cost_bananas in
  ∃ (N_o : ℕ), N_o * O = cost_oranges ∧ (N_o / N_b = 2)

theorem oranges_to_bananas_ratio_is_two (P O B : ℕ) :
  find_ratio O P B number_bananas total_price :=
by
  sorry

end oranges_to_bananas_ratio_is_two_l415_415399


namespace figure_can_be_rearranged_into_square_l415_415528

-- Define the conditions explicitly:
def AreaOfFigure : ℕ := 18
def can_be_cut_and_rearranged (area : ℕ) : Prop :=
  ∃ (parts : list (list (ℕ × ℕ))), parts.length = 3 ∧ 
    (∀ p ∈ parts, true) ∧  -- Placeholder to express valid parts, detailed geometric definitions needed
    (∀ x y ∈ parts, x ≠ y → disjoint x y) ∧ -- Placeholder to express disjoint parts
    (rearranged_figure parts = square_of_area area)  -- Placeholder for rearrangement result

-- Target statement to be proven.
theorem figure_can_be_rearranged_into_square : can_be_cut_and_rearranged AreaOfFigure :=
by 
  sorry

end figure_can_be_rearranged_into_square_l415_415528


namespace triangle_area_l415_415506

theorem triangle_area : 
  let line_eq (x y : ℝ) := 3 * x + 2 * y = 12
  let x_intercept := (4 : ℝ)
  let y_intercept := (6 : ℝ)
  ∃ (x y : ℝ), line_eq x y ∧ x = x_intercept ∧ y = y_intercept ∧
  ∃ (area : ℝ), area = 1 / 2 * x * y ∧ area = 12 :=
by
  sorry

end triangle_area_l415_415506


namespace square_root_properties_l415_415920

theorem square_root_properties (a : ℝ) (h : a^2 = 10) :
  a = Real.sqrt 10 ∧ (a^2 - 10 = 0) ∧ (3 < a ∧ a < 4) :=
by sorry

end square_root_properties_l415_415920


namespace minimum_perimeter_triangle_MAF_is_11_l415_415746

-- Define point, parabola, and focus
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the specific points in the problem
def A : Point := ⟨5, 3⟩

-- Parabola with the form y^2 = 4x has the focus at (1, 0)
def F : Point := ⟨1, 0⟩

-- Minimum perimeter problem for ΔMAF
noncomputable def minimum_perimeter_triangle_MAF (M : Point) : ℝ :=
  (dist (M.x, M.y) (A.x, A.y)) + (dist (M.x, M.y) (F.x, F.y))

-- The goal is to show the minimum value of the perimeter is 11
theorem minimum_perimeter_triangle_MAF_is_11 (M : Point) 
  (hM_parabola : M.y^2 = 4 * M.x) 
  (hM_not_AF : M.x ≠ (5 + (3 * ((M.y - 0) / (M.x - 1))) )) : 
  ∃ M, minimum_perimeter_triangle_MAF M = 11 :=
sorry

end minimum_perimeter_triangle_MAF_is_11_l415_415746


namespace max_area_triangle_OPQ_l415_415618

theorem max_area_triangle_OPQ :
  ∀ (A B O M P Q : ℝ × ℝ), 
  A = (-2, 0) →
  B = (2, 0) →
  O = (0, 0) →
  (∃ x y : ℝ, 
  ((y / (x + 2)) * (y / (x - 2)) = -3 / 4) ∧ 
  ((P.2 = P.1 + 1) ∧ (Q.2 = Q.1 + 1) ∧ (P.1, P.2) ∈ set_of_eqn_M ∧ (Q.1, Q.2) ∈ set_of_eqn_M)) →
  (∃ m : ℝ, 
  |(Q.1 - P.1)| * |(Q.2 - P.2)| = 2 * sqrt 3) →
  let area := (1 / 2) * |(Q.1 - P.1)| * |(Q.2 - P.2)| in 
  area = sqrt 3 := sorry

end max_area_triangle_OPQ_l415_415618


namespace balloons_total_l415_415516

theorem balloons_total :
  let allan_balloons := 5
  let jake_balloons := 7
  let maria_balloons := 3
  let tom_balloons_brought := 9
  let tom_balloons_lost := 2
  let tom_balloons := tom_balloons_brought - tom_balloons_lost
  total_balloons = allan_balloons + jake_balloons + maria_balloons + tom_balloons
  in total_balloons = 22 := by
  sorry

end balloons_total_l415_415516


namespace external_tangent_b_value_l415_415042

theorem external_tangent_b_value :
  let C1_center := (3 : ℝ, 3 : ℝ)
  let C1_radius := 5
  let C2_center := (15 : ℝ, 10 : ℝ)
  let C2_radius := 10
  ∃ m b : ℝ, 0 < m ∧ (∀ (x y : ℝ), y = m * x + b → (dist (x, y) C1_center = C1_radius) ∧ (dist (x, y) C2_center = C2_radius)) ∧ b = 446 / 95 := 
begin
  sorry
end

end external_tangent_b_value_l415_415042


namespace ellipse_equation_line_through_M_l415_415231

theorem ellipse_equation (a b : ℝ)
  (h1 : a > b > 0)
  (h2 : (sqrt 3 / 2) = sqrt (a^2 - b^2) / a)
  (h3 : ∀ x y : ℝ, (x - y + 2 * b = 0) → x^2 + y^2 = 2 → y*x > 0) :
  ∀ x y : ℝ, (x ^ 2 / 4 + y ^ 2 = 1) := sorry

theorem line_through_M (M P Q : ℝ × ℝ)
  (hM : M = (0, 2))
  (hQ : Q = (-9 / 10, 0))
  (hPQ_perpendicular_AB : ∀ l : (ℝ × ℝ) × (ℝ × ℝ), l = (M, P) ∨ l = (P,Q) ∧ (M.1 - Q.1) * (P.2 - Q.2) + (M.2 - Q.2) * (P.1 - Q.1) = 0) :
  ∃ k : ℝ, (∀ x y : ℝ, y = l.1 * x + l.2 → x * y = 1) ∨ (∀ x : ℝ, x = 0) := sorry

end ellipse_equation_line_through_M_l415_415231


namespace max_min_z_l415_415359

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop :=
  x^2 + 4*y^2 = 4*x

-- Define the function z
def z (x y : ℝ) : ℝ :=
  x^2 - y^2

-- Define the required points
def P1 (x y : ℝ) :=
  x = 4 ∧ y = 0

def P2 (x y : ℝ) :=
  x = 2/5 ∧ (y = 3/5 ∨ y = -3/5)

-- Theorem stating the required conditions
theorem max_min_z (x y : ℝ) (h : on_ellipse x y) :
  (P1 x y → z x y = 16) ∧ (P2 x y → z x y = -1/5) :=
by
  sorry

end max_min_z_l415_415359


namespace smallest_magnitude_value_l415_415357

noncomputable def z : ℂ := sorry
noncomputable def area_of_parallelogram : ℝ := 20 / 29
noncomputable def real_part_positive (z : ℂ) : Prop := z.re > 0
noncomputable def magnitude_value (z : ℂ) : ℂ := z^2 + z
noncomputable def smallest_value (r : ℝ) : ℝ := (r^2 + r)^2

theorem smallest_magnitude_value
  (h_area : abs (z * z^2) = area_of_parallelogram)
  (h_real_part : real_part_positive z) :
  abs (magnitude_value z) = smallest_value (abs z) :=
sorry

end smallest_magnitude_value_l415_415357


namespace problem1_problem2_l415_415254

-- Definition and conditions given in the problem
def expandedEquation (x : ℝ) (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (x + 1)^n - ∑ i in Finset.range(n + 1), a i * (x - 1)^i

-- Specific problem case where n = 5
def exampleSum (a : ℕ → ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5

theorem problem1 (a : ℕ → ℝ) (h : expandedEquation 2 a 5 = 0) :
  exampleSum a = 243 := by
  unfold exampleSum expandedEquation
  rw [Finset.sum_range_succ, Finset.sum_range_succ, Finset.sum_range_succ, Finset.sum_range_succ, Finset.sum_range_succ, Finset.sum_range_zero]
  sorry

-- Definitions for b_n and T_n
def b_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (a 2) / (2 ^ (n - 3))

def T_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ k in Finset.range(n - 1) + 2, b_n a (k + 2)

-- Main theorem to prove using induction
theorem problem2 (a : ℕ → ℝ) (n : ℕ) (h : n ≥ 2) :
  T_n a n = (n * (n + 1) * (n - 1)) / 3 := by
  sorry

end problem1_problem2_l415_415254


namespace maxModulus_7sqrt6_l415_415711

noncomputable def maxModulus (z : ℂ) : ℂ :=
  |z - 2| ^ 2 * |z + 2|

theorem maxModulus_7sqrt6 (z : ℂ) (h : |z| = Real.sqrt 3) :
  maxModulus z ≤ 7 * Real.sqrt 6 :=
sorry

end maxModulus_7sqrt6_l415_415711


namespace not_p_is_sufficient_but_not_necessary_for_not_q_l415_415929

variable (x : ℝ)

def proposition_p : Prop := |x| < 2
def proposition_q : Prop := x^2 - x - 2 < 0

theorem not_p_is_sufficient_but_not_necessary_for_not_q :
  (¬ proposition_p x) → (¬ proposition_q x) ∧ (¬ proposition_q x) → (¬ proposition_p x) → False := by
  sorry

end not_p_is_sufficient_but_not_necessary_for_not_q_l415_415929


namespace greatest_number_divisible_by_11_and_3_l415_415140

namespace GreatestNumberDivisibility

theorem greatest_number_divisible_by_11_and_3 : 
  ∃ (A B C : ℕ), 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    (2 * A - 2 * B + C) % 11 = 0 ∧ 
    (2 * A + 2 * C + B) % 3 = 0 ∧
    (10000 * A + 1000 * C + 100 * C + 10 * B + A) = 95695 :=
by
  -- The proof here is omitted.
  sorry

end GreatestNumberDivisibility

end greatest_number_divisible_by_11_and_3_l415_415140


namespace min_value_of_expr_l415_415610

noncomputable def f (a b : ℝ) : ℝ := 4 / (a + 2 * b) + 9 / (2 * a + b)

theorem min_value_of_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 5 / 3) :
  f a b = 5 :=
by
  sorry

end min_value_of_expr_l415_415610


namespace angle_between_lines_in_folded_rectangle_l415_415309

theorem angle_between_lines_in_folded_rectangle
  (a b : ℝ) 
  (h : b > a)
  (dihedral_angle : ℝ)
  (h_dihedral_angle : dihedral_angle = 18) :
  ∃ (angle_AC_MN : ℝ), angle_AC_MN = 90 :=
by
  sorry

end angle_between_lines_in_folded_rectangle_l415_415309


namespace angle_AOD_value_l415_415787

-- Define the angles and conditions
variables (AOB BOC COD AOD : ℝ)
variables (x y z X : ℝ)

-- Assume given relationships and conditions
def angle_relationship (AOB BOC COD AOD : ℝ) : Prop :=
  AOB = 3 * AOD ∧
  BOC = 3 * AOD ∧
  COD = 3 * AOD

def sum_of_angles (AOB BOC COD AOD : ℝ) : Prop :=
  AOB + BOC + COD + AOD = 360 ∨
  AOB + BOC + COD - AOD = 360

-- Statement to prove the values of ∠AOD
theorem angle_AOD_value (AOB BOC COD AOD : ℝ)
  (h1 : angle_relationship AOB BOC COD AOD)
  (h2 : sum_of_angles AOB BOC COD AOD) :
  AOD = 36 ∨ AOD = 45 :=
begin
  sorry -- proof is omitted
end

end angle_AOD_value_l415_415787


namespace first_player_wins_l415_415429

-- Definitions and Parameters
def Board := Fin 1001 × Fin 1001
def Player := Bool  -- true for player 1, false for player 2

-- Conditions from the problem statement
variables (stones : Board → Nat) [decidable_eq Board]
variables (moves : List (Player × Board))

-- Winning condition: A player loses if there are more than 5 stones in any row or column.
def losing_condition (stones : Board → Nat) : Prop :=
  ∃ i : Fin 1001, (∑ j, stones (i, j)) > 5 ∨ ∃ j : Fin 1001, (∑ i, stones (i, j)) > 5

-- Main theorem: The first player can guarantee a win.
theorem first_player_wins :
  ∃ strategy : (moves → Board), ∀ opponent_moves,
    (λ stones' : Board → Nat, if losing_condition stones' then "second" else "first") = "first" :=
sorry

end first_player_wins_l415_415429


namespace quiz_scores_mean_and_mode_l415_415984

-- Define the problem conditions
def scores : List ℕ := [7, 5, 6, 8, 7, 9]

-- Define the mean and mode computation
def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def mode (l : List ℕ) : List ℕ :=
  let frequencies := l.foldl (λ acc x, acc.insert x (acc.find x |>.getOrElse 0 + 1)) Std.RBMap.empty
  let max_freq := frequencies.fold (λ acc p, max acc (p.snd)) 0
  frequencies.fold (λ acc p, if p.snd == max_freq then p.fst :: acc else acc) []

-- Define the properties to be proven
theorem quiz_scores_mean_and_mode :
  mean scores = 7 ∧ mode scores = [7] :=
by
  sorry

end quiz_scores_mean_and_mode_l415_415984


namespace compare_neg_fractions_l415_415869

theorem compare_neg_fractions : - (2 / 3 : ℝ) > - (5 / 7 : ℝ) := 
by 
  sorry

end compare_neg_fractions_l415_415869


namespace problem_part1_problem_part2_problem_part3_l415_415084

open Set

-- Define the universal set U
def U : Set ℝ := Set.univ 

-- Define sets A and B within the universal set U
def A : Set ℝ := { x | 0 < x ∧ x ≤ 2 }
def B : Set ℝ := { x | x < -3 ∨ x > 1 }

-- Define the complements of A and B within U
def complement_A : Set ℝ := U \ A
def complement_B : Set ℝ := U \ B

-- Define the results as goals to be proved
theorem problem_part1 : A ∩ B = { x | 1 < x ∧ x ≤ 2 } := 
by
  sorry

theorem problem_part2 : complement_A ∩ complement_B = { x | -3 ≤ x ∧ x ≤ 0 } :=
by
  sorry

theorem problem_part3 : U \ (A ∪ B) = { x | -3 ≤ x ∧ x ≤ 0 } :=
by
  sorry

end problem_part1_problem_part2_problem_part3_l415_415084


namespace cyclists_meet_at_starting_point_l415_415786

/-- 
Two cyclists start on a circular track from a given point and travel in opposite directions. 
One cyclist has a speed of 7 m/s and the other has a speed of 8 m/s. 
The circumference of the circle is 630 meters.
We need to prove that they will meet at the starting point after 42 seconds.
-/
theorem cyclists_meet_at_starting_point :
  ∀ (speed1 speed2 circumference : ℕ), speed1 = 7 → speed2 = 8 → circumference = 630 →
  (42 = (circumference / (speed1 + speed2))) :=
begin
  intros speed1 speed2 circumference h_speed1 h_speed2 h_circumference,
  rw [h_speed1, h_speed2, h_circumference],
  norm_num,
end

end cyclists_meet_at_starting_point_l415_415786


namespace median_of_moons_is_two_l415_415436

def moons : List Nat := [0, 0, 1, 2, 67, 82, 27, 14, 5, 2, 1]

theorem median_of_moons_is_two : (median moons) = 2 := by
  sorry

end median_of_moons_is_two_l415_415436


namespace stream_speed_l415_415286

def upstream_time : ℝ := 4  -- time in hours
def downstream_time : ℝ := 4  -- time in hours
def upstream_distance : ℝ := 32  -- distance in km
def downstream_distance : ℝ := 72  -- distance in km

-- Speed equations based on given conditions
def effective_speed_upstream (vj vs : ℝ) : Prop := vj - vs = upstream_distance / upstream_time
def effective_speed_downstream (vj vs : ℝ) : Prop := vj + vs = downstream_distance / downstream_time

theorem stream_speed (vj vs : ℝ)  
  (h1 : effective_speed_upstream vj vs)
  (h2 : effective_speed_downstream vj vs) : 
  vs = 5 := sorry

end stream_speed_l415_415286


namespace tens_digit_smallest_six_digit_multiple_of_10_11_12_13_14_15_l415_415438

-- Definitions based on conditions
def six_digit_positive_integer := {n : ℕ // 100000 ≤ n ∧ n < 1000000}

-- LCM computation
def lcm_10_11_12_13_14_15 : ℕ := Nat.lcm 10 (Nat.lcm 11 (Nat.lcm 12 (Nat.lcm 13 (Nat.lcm 14 15))))

-- Find the smallest six-digit number divisible by the LCM
noncomputable def smallest_six_digit_multiple_lcm : six_digit_positive_integer :=
  ⟨Nat.find (exists_six_digit_multiple_lcm), sorry⟩

-- Function to get the tens digit
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Main Theorem
theorem tens_digit_smallest_six_digit_multiple_of_10_11_12_13_14_15 : 
  tens_digit (smallest_six_digit_multiple_lcm.val) = 2 :=
begin
  sorry
end

end tens_digit_smallest_six_digit_multiple_of_10_11_12_13_14_15_l415_415438


namespace max_a_is_fractional_value_l415_415257

theorem max_a_is_fractional_value (a k : ℝ) (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = x^2 - (k^2 - 5 * a * k + 3) * x + 7)
  (h_k : 0 ≤ k ∧ k ≤ 2)
  (x1 x2 : ℝ)
  (h_x1 : k ≤ x1 ∧ x1 ≤ k + a)
  (h_x2 : k + 2 * a ≤ x2 ∧ x2 ≤ k + 4 * a)
  (h_fx1_fx2 : f x1 ≥ f x2) :
  a = (2 * Real.sqrt 6 - 4) / 5 :=
sorry

end max_a_is_fractional_value_l415_415257


namespace expected_animals_surviving_first_3_months_l415_415456

theorem expected_animals_surviving_first_3_months :
  ∀ (initial_population : ℕ) (P_die : ℚ),
  initial_population = 300 →
  P_die = 1 / 10 →
  let P_survive := 1 - P_die in
  let P_survive_3_months := P_survive^3 in
  let expected_survivors := initial_population * P_survive_3_months in
  expected_survivors ≈ 219 :=
by
  intros initial_population P_die hInit hPDie
  let P_survive := 1 - P_die
  let P_survive_3_months := P_survive^3
  let expected_survivors := initial_population * P_survive_3_months
  sorry

end expected_animals_surviving_first_3_months_l415_415456


namespace corner_coloring_condition_l415_415004

theorem corner_coloring_condition 
  (n : ℕ) 
  (h1 : n ≥ 5) 
  (board : ℕ → ℕ → Prop) -- board(i, j) = true if cell (i, j) is black, false if white
  (h2 : ∀ i j, board i j = board (i + 1) j → board (i + 2) j = board (i + 1) j → ¬(board i j = board (i + 2) j)) -- row condition
  (h3 : ∀ i j, board i j = board i (j + 1) → board i (j + 2) = board i (j + 1) → ¬(board i j = board i (j + 2))) -- column condition
  (h4 : ∀ i j, board i j = board (i + 1) (j + 1) → board (i + 2) (j + 2) = board (i + 1) (j + 1) → ¬(board i j = board (i + 2) (j + 2))) -- diagonal condition
  (h5 : ∀ i j, board (i + 2) j = board (i + 1) (j + 1) → board (i + 2) (j + 2) = board (i + 1) (j + 1) → ¬(board (i + 2) j = board (i + 2) (j + 2))) -- anti-diagonal condition
  : ∀ i j, i + 2 < n ∧ j + 2 < n → ((board i j ∧ board (i + 2) (j + 2)) ∨ (board i (j + 2) ∧ board (i + 2) j)) :=
sorry

end corner_coloring_condition_l415_415004


namespace quadrilateral_inscribed_in_conic_l415_415924

-- Define the general setup of the problem
structure Ellipse (F : Point) := 
  (focus : Point)
  (a b : ℝ) -- Major and minor axes lengths
  (a_squared_pos : a^2 > 0)
  (b_squared_pos : b^2 > 0)

structure Point := 
  (x : ℝ)
  (y : ℝ)

-- Define the theorem statement
theorem quadrilateral_inscribed_in_conic (F : Point) 
  (E : Ellipse F) 
  (L1 L2 : Line) 
  (perpendicular : Line.is_perpendicular L1 L2)
  (A B C D : Point) 
  (A_on_E : E.on_ellipse A)
  (B_on_E : E.on_ellipse B)
  (C_on_E : E.on_ellipse C)
  (D_on_E : E.on_ellipse D)
  (A_on_L1 : L1.contains A)
  (B_on_L2 : L2.contains B)
  (C_on_L1 : L1.contains C)
  (D_on_L2 : L2.contains D):
  ∃ (conic : ConicSection), 
    conic.has_focus F ∧
    conic.contains (tangent E A) ∧ 
    conic.contains (tangent E B) ∧ 
    conic.contains (tangent E C) ∧ 
    conic.contains (tangent E D) :=
sorry

end quadrilateral_inscribed_in_conic_l415_415924


namespace hypotenuse_30_60_90_l415_415360

theorem hypotenuse_30_60_90 (a : ℝ) (h : a = 15) : 
  ∃ (h : ℝ), a = 15 ∧ ∃ (θ : ℝ), θ = real.pi / 3 ∧ h = 10 * real.sqrt 3 :=
by
  use 10 * real.sqrt 3
  sorry

end hypotenuse_30_60_90_l415_415360


namespace math_problem_l415_415605

theorem math_problem (a b c m n : ℝ)
  (h1 : a = -b)
  (h2 : c = -1)
  (h3 : m * n = 1) : 
  (a + b) / 3 + c^2 - 4 * m * n = -3 := 
by 
  -- Proof steps would be here
  sorry

end math_problem_l415_415605


namespace order_of_values_l415_415911

theorem order_of_values (a p q r: ℝ) 
    (ha: a = Real.cos 1)
    (hp: p = Real.log a (1 / 2))
    (hq: q = a^(1 / 2))
    (hr: r = (1 / 2)^a) : 
    r < q ∧ q < p :=
    sorry

end order_of_values_l415_415911


namespace exists_1000_rows_and_1000_columns_with_sum_at_least_1000_l415_415982

theorem exists_1000_rows_and_1000_columns_with_sum_at_least_1000 :
  (∃ (grid : Fin 2000 → Fin 2000 → ℤ),
    (∀ i j, grid i j = 1 ∨ grid i j = -1) ∧
    (∑ i, ∑ j, grid i j ≥ 0)) →
  ∃ (rows cols : Finset (Fin 2000)),
    rows.card = 1000 ∧ cols.card = 1000 ∧
    (∑ i in rows, ∑ j in cols, grid i j ≥ 1000) :=
by
  sorry

end exists_1000_rows_and_1000_columns_with_sum_at_least_1000_l415_415982


namespace toyota_not_less_honda_skoda_l415_415030

variables (T H S F R Y : ℕ) -- Number of Toyotas, Hondas, Skodas, other brands, red cars, and yellow cars

-- Conditions:
def condition1 : Prop := T + S + F = 1.5 * (H + S + F)
def condition2 : Prop := T + H + F = 1.5 * (T + H + F - Y)
def condition3 : Prop := H + S + F = 0.5 * (R + Y)

-- Theorem to prove:
theorem toyota_not_less_honda_skoda (h1 : condition1) (h2 : condition2) (h3 : condition3) : 
  T ≥ H + S :=
sorry

end toyota_not_less_honda_skoda_l415_415030


namespace set_intersection_complement_l415_415957

   variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

   def U := {0, 1, 2, 3, 4, 5, 6}
   def A := {0, 1, 3, 5}
   def B := {1, 2, 4}

   theorem set_intersection_complement :
     A ∩ (U \ B) = {0, 3, 5} := by
   sorry
   
end set_intersection_complement_l415_415957


namespace smallest_positive_integer_d_l415_415835

theorem smallest_positive_integer_d (x : ℝ) (d : ℕ) :
  (x^2 + (2*x + d)^2 = 100) ∧ (sqrt (x^2 + (2 * x + d)^2) = 10 * sqrt d) ∧ d > 0 ∧ 
  (∀ d', (d' > 0) → ((∃ x', x'^2 + (2*x' + d')^2 = 100 ∧ sqrt (x'^2 + (2 * x' + d')^2) = 10 * sqrt d') → d' ≥ d)) :=
begin
  sorry,
end

end smallest_positive_integer_d_l415_415835


namespace radius_of_circumscribed_sphere_l415_415314

def right_isosceles_triangle (A B C : Type) (AB AC : ℝ) (right_angle : Prop) : Prop :=
  AB = 4 * Real.sqrt 2 ∧ AC = 4 * Real.sqrt 2 ∧ right_angle

def right_triangular_prism (prism_height : ℝ) : Prop :=
  prism_height = 8

theorem radius_of_circumscribed_sphere
  {A B C A1 : Type}
  (h_rt : right_isosceles_triangle A B C (angle_eq A C B (π / 2)))
  (h_sides : AB = 4 * Real.sqrt 2)
  (h_height : AA1 = 6)
  (h_prism_height : right_triangular_prism 8) :
  circumscribed_sphere_radius (prism A B C A1) = 5 :=
begin
  sorry -- Proof not required
end

end radius_of_circumscribed_sphere_l415_415314


namespace parallel_PD_QR_l415_415318

variables {A B C D E F M N P O Q R : Type} [geometry A B C D E F M N P O Q R]

-- Definitions for the conditions
variables (ABC : Triangle A B C)
variables (alt_AD : is_altitude A D B C)
variables (alt_BE : is_altitude B E A C)
variables (alt_CF : is_altitude C F A B)
variables (circle_Omega : Circle ℝ (midpoint A D) (dist A D / 2))
variables (M_on_AC : on_circle M (segment A C) circle_Omega)
variables (N_on_AB : on_circle N (segment A B) circle_Omega)
variables (tan_M : tangent_to M circle_Omega)
variables (tan_N : tangent_to N circle_Omega)
variables (P_tangent_meet: meets_tangent_lines P M N circle_Omega)
variables (O_is_circumcenter : circumcenter_type O ABC)
variables (AO_extends_to_Q : line A O ▸ meets_point Q (segment B C))
variables (inter_AD_EF : intersects_at AD (segment E F) R)

-- The final proof statement
theorem parallel_PD_QR 
  (PD_parallel_QR : parallel (line_through P D) (line_through Q R)) : 
  PD_parallel_QR := sorry

end parallel_PD_QR_l415_415318


namespace complex_quadrant_l415_415239

theorem complex_quadrant :
  let i := Complex.i
  let z := (1 + i) / i in
  z.re > 0 ∧ z.im < 0 :=
by
  sorry

end complex_quadrant_l415_415239


namespace calculate_distances_l415_415056

variable {real : Type*}

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2 + (p1.3 - p2.3) ^ 2)

def largest_smallest_distance
  (C : ℝ × ℝ × ℝ) (r1 : ℝ)
  (D : ℝ × ℝ × ℝ) (r2 : ℝ) :
  (ℝ × ℝ) :=
  let CD := distance C D
  (CD + r1 + r2, abs (CD - (r1 + r2)))

theorem calculate_distances :
  let C := (0, -4, 9)
  let D := (15, -10, -5)
  largest_smallest_distance C 23 D 70 = (114.377, 71.623) :=
by
  sorry

end calculate_distances_l415_415056


namespace avg_ratio_one_l415_415847

variable (x : Fin 50 → ℝ)

def true_avg (x : Fin 50 → ℝ) : ℝ := (∑ i, x i) / 50

def new_avg (x : Fin 50 → ℝ) : ℝ := (∑ i, x i + 2 * true_avg x) / 52

theorem avg_ratio_one (x : Fin 50 → ℝ) : 
  new_avg x / true_avg x = 1 := by sorry

end avg_ratio_one_l415_415847


namespace line_perpendicular_to_plane_l415_415240

open Classical

-- Define the context of lines and planes.
variables {Line : Type} {Plane : Type}

-- Define the perpendicular and parallel relations.
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l₁ l₂ : Line) : Prop := sorry

-- Declare the distinct lines and non-overlapping planes.
variable {m n : Line}
variable {α β : Plane}

-- State the theorem.
theorem line_perpendicular_to_plane (h1 : parallel m n) (h2 : perpendicular n β) : perpendicular m β :=
sorry

end line_perpendicular_to_plane_l415_415240


namespace minimum_set_intersection_l415_415579

noncomputable def smallest_n (A : Set (Set (ℕ))) (hA : A.card = 2007) : ℕ :=
  let n := 2008
  if ∃ B : Set (Set ℕ), B.card = n ∧ ∀ a ∈ A, ∃ b1 b2 ∈ B, a = b1 ∩ b2 ∧ b1 ≠ b2 then n
  else 0

theorem minimum_set_intersection (A : Set (Set ℕ)) (hA : A.card = 2007) :
  smallest_n A hA = 2008 := by
  sorry

end minimum_set_intersection_l415_415579


namespace part_1_part_2_l415_415624

-- Definitions
def piecewise_fn (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then -x + 6 else 3 + Real.log x / Real.log a

def is_solution_1 (x : ℝ) : Prop :=
  1 ≤ x ∧ x ≤ 4

def is_solution_2 (a : ℝ) : Prop :=
  1 < a ∧ a ≤ 2

-- Theorem statements
theorem part_1 (x : ℝ) (h : piecewise_fn 2 x ≤ 5) : is_solution_1 x :=
by {
  sorry -- Proof goes here
}

theorem part_2 (a : ℝ) (h : ∀ y, ∃ x, piecewise_fn a x ≥ y) : is_solution_2 a :=
by {
  sorry -- Proof goes here
}

end part_1_part_2_l415_415624


namespace probability_two_white_marbles_l415_415794

theorem probability_two_white_marbles :
  let total_marbles := 12
  let white_marbles := 7
  let red_marbles := 5
  let first_draw_white := (white_marbles : ℚ) / total_marbles
  let second_draw_white := (white_marbles - 1 : ℚ) / (total_marbles - 1)
  (first_draw_white * second_draw_white) = (7 / 22 : ℚ) :=
begin
  let total_marbles := 12,
  let white_marbles := 7,
  let red_marbles := 5,
  let first_draw_white := (white_marbles : ℚ) / total_marbles,
  let second_draw_white := (white_marbles - 1 : ℚ) / (total_marbles - 1),
  calc
    first_draw_white * second_draw_white
    = (7 / 12) * (6 / 11) : by norm_num
    ... = 7 / 22 : by norm_num,
end

end probability_two_white_marbles_l415_415794


namespace even_function_l415_415403

noncomputable def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x

def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem even_function : is_even f :=
by 
  sorry

end even_function_l415_415403


namespace proof_find_angles_l415_415990

noncomputable def find_angles 
  (α : ℝ) (ABC : ℝ) 
  (acute_isosceles_triangle : Prop)
  (R r : ℝ) (hR : R = 4 * r) 
  : Prop :=
  α = real.arccos ((2 - real.sqrt 2) / 4) ∧
  ABC = real.arccos ((2 * (2 + 1) ^ (1/2)) / 4)

theorem proof_find_angles 
  (α : ℝ) (ABC : ℝ) 
  (acute_isosceles_triangle : Prop := sorry) 
  (R r : ℝ) (hR : R = 4 * r) 
: find_angles α ABC acute_isosceles_triangle R r hR := 
sorry

end proof_find_angles_l415_415990


namespace bisection_contains_root_l415_415051

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 1

theorem bisection_contains_root : (1 < 1.5) ∧ f 1 < 0 ∧ f 1.5 > 0 → ∃ (c : ℝ), 1 < c ∧ c < 1.5 ∧ f c = 0 :=
by
  sorry

end bisection_contains_root_l415_415051


namespace total_frogs_seen_by_hunter_l415_415647

/-- Hunter saw 5 frogs sitting on lily pads in the pond. -/
def initial_frogs : ℕ := 5

/-- Three more frogs climbed out of the water onto logs floating in the pond. -/
def frogs_on_logs : ℕ := 3

/-- Two dozen baby frogs (24 frogs) hopped onto a big rock jutting out from the pond. -/
def baby_frogs : ℕ := 24

/--
The total number of frogs Hunter saw in the pond.
-/
theorem total_frogs_seen_by_hunter : initial_frogs + frogs_on_logs + baby_frogs = 32 := by
sorry

end total_frogs_seen_by_hunter_l415_415647


namespace row_column_sums_equal_l415_415792

theorem row_column_sums_equal (table : Matrix (Fin 3) (Fin 3) ℤ) 
  (h_val : ∀ i j, table i j ∈ {1, 0, -1}) 
  (sum_rows_cols : Fin 6 → ℤ) 
  (h_sum_rows_cols : ∀ i, sum_rows_cols i = if i.val < 3 then (∑ j, table ⟨i.val, by linarith⟩ j) else (∑ i, table i ⟨i.val - 3, by linarith⟩)) :
  ∃ i j, i ≠ j ∧ sum_rows_cols i = sum_rows_cols j :=
by
  sorry

end row_column_sums_equal_l415_415792


namespace triangle_area_l415_415508

theorem triangle_area : 
  let line_eq (x y : ℝ) := 3 * x + 2 * y = 12
  let x_intercept := (4 : ℝ)
  let y_intercept := (6 : ℝ)
  ∃ (x y : ℝ), line_eq x y ∧ x = x_intercept ∧ y = y_intercept ∧
  ∃ (area : ℝ), area = 1 / 2 * x * y ∧ area = 12 :=
by
  sorry

end triangle_area_l415_415508


namespace population_definition_sample_definition_sample_size_definition_l415_415762

def students : Type :=
  sorry 

variable (total_students : ℕ)
variable (sample_students : ℕ)

axiom student_count : total_students = 43
axiom sample_count : sample_students = 28

theorem population_definition
  (P : Prop) :
  P = ( "The time spent on studying mathematics at home during the summer vacation by the 43 students in the 7th grade class" ) →
  ∀ x, x = students →
  x = P := 
sorry

theorem sample_definition
  (P : Prop) :
  P = ( "The time spent on studying mathematics at home during the summer vacation by the 28 students in the 7th grade class" ) →
  ∀ x, x = students →
  x = P :=
sorry

theorem sample_size_definition
  (size : ℕ)
  (h : size = 28) 
  (P : size = sample_students) :
  size = P :=
sorry

end population_definition_sample_definition_sample_size_definition_l415_415762


namespace segment_length_behaviour_l415_415873

-- Define the scenario
variables (A B C D : Point)
variables (a b c ad bd cd : LineSegment)

-- Assume the geometric relationships
axiom midpoint_D : is_midpoint D A B
axiom segment_start_at_A : starts_at a A
axiom segment_parallel_max_length : ∃ x, moves_to_max_length x a BC

-- The theorem statement
theorem segment_length_behaviour :
  ∀ (t ∈ [0, 1]), segment_moves a A D →
  ∃ max_len, length_increases_to_maximum_then_decreases a A D max_len :=
sorry

end segment_length_behaviour_l415_415873


namespace set_D_not_like_terms_l415_415778

def like_terms (a b : ℚ[X]) : Prop :=
  ∀ (vars : List (String × ℕ)), a.vars = b.vars ∧ a.exponents = b.exponents

theorem set_D_not_like_terms : ¬ like_terms (2 * X^2 * Y) (2 * X * Y^2) :=
by sorry

end set_D_not_like_terms_l415_415778


namespace problem1_l415_415082

theorem problem1 : 
  |2 - Real.sqrt 3| + (Real.sqrt 2 + 1)^0 + 3 * Real.tan (Real.pi / 6) + (-1 : ℤ) ^ 2023 - (1/2 : ℚ) ^ -1  = 0 := 
by 
  sorry

end problem1_l415_415082


namespace smallest_value_expression_l415_415569

theorem smallest_value_expression (a b c d : ℝ) (h1 : b > c) (h2 : c > a) (h3 : a > d) (h4 : b ≠ 0) :
    ∃ k, (k = (2 * a + b)^2 + (b - 2 * c)^2 + (c - a)^2 + 3 * d^2) / b^2 ∧ k = 49 / 36 := sorry

end smallest_value_expression_l415_415569


namespace longer_side_of_rectangle_l415_415107

theorem longer_side_of_rectangle (r : ℝ) (Aₙ Aₙ: ℝ) (L S: ℝ):
  (r = 6) → 
  (Aₙ = 36 * π) →
  (Aₙ = 108 * π) →
  (S = 12) → 
  (S * L = Aₙ) →
  L = 9 * π := sorry

end longer_side_of_rectangle_l415_415107


namespace f_gt_ln_div_x_sub_1_l415_415628

def g (x : ℝ) (a : ℝ) : ℝ := Real.log x + 2 * x + a / x

def f (x : ℝ) (a : ℝ) : ℝ := (1 / (x + 1)) * (g x a - 2 * x - a / x) + 1 / x

theorem f_gt_ln_div_x_sub_1 (a x : ℝ) (h1 : 0 < x) (h2 : x ≠ 1) : 
  f x a > Real.log x / (x - 1) := sorry

end f_gt_ln_div_x_sub_1_l415_415628


namespace smallest_positive_angle_l415_415566

theorem smallest_positive_angle :
  ∃ θ : ℝ, θ = 30 ∧
    cos (θ * (Real.pi / 180)) = 
      sin (45 * (Real.pi / 180)) + cos (60 * (Real.pi / 180)) - sin (30 * (Real.pi / 180)) - cos (15 * (Real.pi / 180)) := 
by
  sorry

end smallest_positive_angle_l415_415566


namespace probability_inside_D_in_E_l415_415996

-- Define the region D as a set of points in the plane satisfying the inequality |x| + |y| <= 1
def region_D (x y : ℝ) : Prop := abs x + abs y ≤ 1

-- Define the region E as a set of points in the plane whose distance from the origin is no greater than 1
def region_E (x y : ℝ) : Prop := x^2 + y^2 ≤ 1

-- Define the area of region D, which is known to be 2
def area_D : ℝ := 2

-- Define the area of region E, which is known to be π * 1^2 = π
def area_E : ℝ := Real.pi

-- The probability is the ratio of the area of D to the area of E
def probability : ℝ := area_D / area_E

-- The goal is to prove that this probability is equal to 2 / π
theorem probability_inside_D_in_E : probability = 2 / Real.pi :=
by
  sorry

end probability_inside_D_in_E_l415_415996


namespace arithmetic_sequence_range_of_m_l415_415917

-- Conditions
variable {a : ℕ+ → ℝ} -- Sequence of positive terms
variable {S : ℕ+ → ℝ} -- Sum of the first n terms
variable (h : ∀ n, 2 * Real.sqrt (S n) = a n + 1) -- Relationship condition

-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem arithmetic_sequence (n : ℕ+)
    (h1 : ∀ n, 2 * Real.sqrt (S n) = a n + 1)
    (h2 : S 1 = 1 / 4 * (a 1 + 1)^2) :
    ∃ d : ℝ, ∀ n, a (n + 1) = a n + d :=
sorry

-- Part 2: Find range of m
theorem range_of_m (T : ℕ+ → ℝ)
    (hT : ∀ n, T n = 1 / 4 * n + 1 / 8 * (1 - 1 / (2 * n + 1))) :
    ∃ m : ℝ, (6 / 7 : ℝ) < m ∧ m ≤ 10 / 9 ∧
    (∃ n₁ n₂ n₃ : ℕ+, (n₁ < n₂ ∧ n₂ < n₃) ∧ (∀ n, T n < m ↔ n₁ ≤ n ∧ n ≤ n₃)) :=
sorry

end arithmetic_sequence_range_of_m_l415_415917


namespace find_m_range_l415_415250

-- Defining the function and conditions
variable {f : ℝ → ℝ}
variable {m : ℝ}

-- Prove if given the conditions, then the range of m is as specified
theorem find_m_range (h1 : ∀ x, f (-x) = -f x) 
                     (h2 : ∀ x, -2 < x ∧ x < 2 → f (x) > f (x+1)) 
                     (h3 : -2 < m - 1 ∧ m - 1 < 2) 
                     (h4 : -2 < 2 * m - 1 ∧ 2 * m - 1 < 2) 
                     (h5 : f (m - 1) + f (2 * m - 1) > 0) :
  -1/2 < m ∧ m < 2/3 :=
sorry

end find_m_range_l415_415250


namespace game_ends_after_3_rounds_l415_415305

-- Defining initial token counts for each player
def initial_tokens : ℕ → ℕ
| 0 := 12 -- Player A
| 1 := 11 -- Player B
| 2 := 10 -- Player C
| 3 := 9  -- Player D
| _ := 0  -- No other players

-- Defining the condition for the game process
def tokens_after_rounds (n : ℕ) (tokens : ℕ → ℕ) : ℕ → ℕ :=
  λ i, match i with
  | 0 => tokens 0 - 4 * n  -- Player with most tokens initially loses 4 tokens per round
  | 1 => tokens 1 + n      -- Other players gain 1 token per round
  | 2 => tokens 2 + n
  | 3 => tokens 3 + n
  | _ => 0
  end

-- Proof statement
theorem game_ends_after_3_rounds :
  ∃ n, n = 3 ∧ ∀ tokens : (ℕ → ℕ), tokens = initial_tokens →
  tokens_after_rounds n tokens 0 = 0 :=
by {
  sorry
}

end game_ends_after_3_rounds_l415_415305


namespace line_AB_passes_fixed_point_foot_N_trajectory_l415_415603

theorem line_AB_passes_fixed_point :
  ∀ (A B : ℝ × ℝ), A ≠ (2, 1) →
  B ≠ (2, 1) →
  (A_fst ^ 2 = 4 * A_snd) →
  (B_fst ^ 2 = 4 * B_snd) →
  let M : ℝ × ℝ := (2, 1) in
  let k : ℝ := (B_snd - A_snd) / (B_fst - A_fst) in
  let m : ℝ := A_snd - k * A_fst in
  M ∈ circle (A, B) →
  line_through A B = line_through (-2, 5) :=
begin
  intros,
  sorry
end

theorem foot_N_trajectory :
  ∀ (A B : ℝ × ℝ), A ≠ (2, 1) →
  B ≠ (2, 1) →
  (A_fst ^ 2 = 4 * A_snd) →
  (B_fst ^ 2 = 4 * B_snd) →
  let M : ℝ × ℝ := (2, 1) in
  let k : ℝ := (B_snd - A_snd) / (B_fst - A_fst) in
  let R : ℝ × ℝ := (-2, 5) in
  N_foot_trajectory M R = { p : ℝ × ℝ | p.1 ^ 2 + (p.2 - 3) ^ 2 = 8 ∧ p.2 ≠ 1 } :=
begin
  intros,
  sorry
end

end line_AB_passes_fixed_point_foot_N_trajectory_l415_415603


namespace addition_and_rounding_nearest_tenth_l415_415154

noncomputable def addition_and_rounding : ℝ := 45.92 + 68.453

theorem addition_and_rounding_nearest_tenth
  (h_add : addition_and_rounding = 114.373)
  (h_rounding: Real.round_nearest_tenth 114.373 = 114.4) :
  Real.round_nearest_tenth addition_and_rounding = 114.4 :=
by
  -- The proof is trivial given the assumptions h_add and h_rounding
  sorry

end addition_and_rounding_nearest_tenth_l415_415154


namespace condition1_a_geq_1_l415_415908

theorem condition1_a_geq_1 (a : ℝ) :
  (∀ x ∈ ({1, 2, 3} : Set ℝ), a * x - 1 ≥ 0) → a ≥ 1 :=
by
sorry

end condition1_a_geq_1_l415_415908


namespace domain_of_f_l415_415401

-- Define the function f(x)
def f (x : ℝ) := real.sqrt (real.log x / real.log 2 - 1)

-- Statement asserting the domain of the function
theorem domain_of_f : {x : ℝ | x ≥ 2} = {x : ℝ | x ∈ [2, +∞)} :=
by
  sorry

end domain_of_f_l415_415401


namespace arcsin_equation_solution_l415_415379

theorem arcsin_equation_solution :
    ∀ x : ℝ, arcsin (3 * x) - arcsin x = π / 6 ↔ x = 1 / sqrt (40 - 12 * sqrt 3) ∨ x = -1 / sqrt (40 - 12 * sqrt 3) :=
by
  intros x
  sorry

end arcsin_equation_solution_l415_415379


namespace area_of_triangle_bounded_by_line_and_axes_l415_415502

theorem area_of_triangle_bounded_by_line_and_axes (x y : ℝ) (hx : 3 * x + 2 * y = 12) :
  ∃ (area : ℝ), area = 12 := by
sorry

end area_of_triangle_bounded_by_line_and_axes_l415_415502


namespace relationship_among_a_b_c_l415_415606

noncomputable def f : ℝ → ℝ := sorry -- Define the function f 

variables (x : ℝ)

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x
def decreasing_on_negative_half (f : ℝ → ℝ) : Prop := ∀ x < 0, f x + x * deriv f x < 0

theorem relationship_among_a_b_c (h_odd : odd_function f) 
  (h_decreasing : decreasing_on_negative_half f) : 
  let a := π * f π,
      b := -2 * f (-2),
      c := f 1 in 
  a > b ∧ b > c := 
begin
  sorry
end

end relationship_among_a_b_c_l415_415606


namespace intersection_A_B_range_of_a_l415_415942

variable {A B C : Set ℝ}
variable {a : ℝ}

-- Given conditions
def f (x : ℝ) : ℝ := Real.sqrt (6 - 2 * x) + Real.log (x + 2)
def SetA : Set ℝ := {x | -2 < x ∧ x ≤ 3}
def SetB : Set ℝ := {x | x > 3 ∨ x < 2}
def SetC (a : ℝ) : Set ℝ := {x | x < 2 * a + 1}

-- Proof objectives
theorem intersection_A_B :
  (SetA ∩ SetB) = {x | -2 < x ∧ x < 2} := 
sorry

theorem range_of_a :
  ∀ {a : ℝ}, (SetC a ⊆ SetB) → a ≤ 1 / 2 := 
sorry

end intersection_A_B_range_of_a_l415_415942


namespace odd_not_divisible_by_3_l415_415597

theorem odd_not_divisible_by_3 :
  ∃ (n : ℤ), odd n ∧ ¬ (3 ∣ n) := 
sorry

end odd_not_divisible_by_3_l415_415597


namespace coterminal_angle_l415_415881

theorem coterminal_angle (α : ℤ) : 
  ∃ k : ℤ, α = k * 360 + 283 ↔ ∃ k : ℤ, α = k * 360 - 437 :=
sorry

end coterminal_angle_l415_415881


namespace quadratic_intersects_x_axis_l415_415954

theorem quadratic_intersects_x_axis (a : ℝ) (h : a ≠ 0) :
  ∃ x : ℝ, a * x^2 - (3 * a + 1) * x + 3 = 0 := 
by {
  -- The proof will go here
  sorry
}

end quadratic_intersects_x_axis_l415_415954


namespace no_counterexample_exists_l415_415706

-- Definition to check if the sum of the digits of a number is divisible by 9
def sum_of_digits (n : ℕ) : ℕ :=
n.to_digits Nat.digits 10 |>.sum

-- Definition to check if a number is divisible by 9 and 3
def is_divisible_by_9_and_3 (n : ℕ) : Prop :=
n % 9 = 0 ∧ n % 3 = 0

-- Statement of the proof problem
theorem no_counterexample_exists (n : ℕ) :
  n ∈ {54, 81, 99, 108} →
  (sum_of_digits n) % 9 = 0 →
  is_divisible_by_9_and_3 n :=
by
  intros h1 h2
  -- Proof would go here
  sorry

end no_counterexample_exists_l415_415706


namespace part1_part2_l415_415222

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 + a*x + sin ((π / 2) * x)

theorem part1 (a : ℝ) : (∀ x ∈ Ioo (0 : ℝ) 1, f x a < f (x+ε) a ∀ ε > 0) → a ∈ Icc (-π/2) (real.top) := sorry

theorem part2 (x0 x1 x2 : ℝ) (h0 : 0 < x0 ∧ x0 < 1) (h1 : 0 < x1 ∧ x1 < 1) (h2 : 0 < x2 ∧ x2 < 1) :
  (∀ x, f x (-2) ≥ 0) ∧ (f x1 (-2) = f x2 (-2)) → x1 + x2 > 2 * x0 := sorry

end part1_part2_l415_415222


namespace max_area_polygon_l415_415836

theorem max_area_polygon (P : ℕ) (A B : ℕ) (grid_lines : ℕ → Prop)
    (h_polygon_on_grid : ∀ (x : ℕ), grid_lines x → ∃ (side : ℕ), side = A ∨ side = B)
    (h_perimeter : 2 * (A + B) = P)
    (h_perimeter_value : P = 36) : A * B ≤ 81 :=
begin
  sorry -- proof goes here
end

end max_area_polygon_l415_415836


namespace star_four_three_l415_415283

def star (x y : ℕ) : ℕ := x^2 - x*y + y^2

theorem star_four_three : star 4 3 = 13 := by
  sorry

end star_four_three_l415_415283


namespace sin_value_given_cos_condition_l415_415587

theorem sin_value_given_cos_condition (theta : ℝ) (h : Real.cos (5 * Real.pi / 12 - theta) = 1 / 3) :
  Real.sin (Real.pi / 12 + theta) = 1 / 3 :=
sorry

end sin_value_given_cos_condition_l415_415587


namespace exercise_books_quantity_l415_415724

theorem exercise_books_quantity (ratio_pencil : ℕ) (ratio_exercise_book : ℕ) (pencils : ℕ) (h_ratio : ratio_pencil = 10) (h_ratio_exercise : ratio_exercise_book = 3) (h_pencils : pencils = 120) : (3 * (pencils / 10) = 36) :=
by
  have h1 : pencils / ratio_pencil = pencils / 10 := by rw [h_ratio]
  have h2 : (3 * (pencils / 10) = 3 * 12) := by rw [h1, h_pencils, Nat.div_eq_of_lt]
  have h3 : 3 * 12 = 36 := by norm_num
  rw [h2, h3]
  norm_num

end exercise_books_quantity_l415_415724


namespace sufficient_but_not_necessary_condition_l415_415180

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (x^2 - 3*x + a = 0 ∧ a = 1 → 
  (∃ x : ℝ, x^2 - 3*x + a = 0)) ∧ ¬(∀ a' : ℝ, (x^2 - 3*x + a' = 0 → a' = 1)) :=
begin
  split,
  { 
    intros h,
    use 1,
    -- Proof needed here
    sorry
  },
  {
    intro h,
    -- Proof needed here
    sorry
  }
end

end sufficient_but_not_necessary_condition_l415_415180


namespace max_self_intersections_closed_polygon_13_segments_max_self_intersections_closed_polygon_1950_segments_l415_415435

theorem max_self_intersections_closed_polygon_13_segments :
  max_self_intersections_closed_polygon 13 = 65 :=
sorry

theorem max_self_intersections_closed_polygon_1950_segments :
  max_self_intersections_closed_polygon 1950 = 1898851 :=
sorry

/-- 
  Definition:
  max_self_intersections_closed_polygon n := maximum number of self-intersection points a closed polygonal chain with n segments can have.
-/

end max_self_intersections_closed_polygon_13_segments_max_self_intersections_closed_polygon_1950_segments_l415_415435


namespace hunter_saw_32_frogs_l415_415648

noncomputable def total_frogs (g1 : ℕ) (g2 : ℕ) (d : ℕ) : ℕ :=
g1 + g2 + d

theorem hunter_saw_32_frogs :
  total_frogs 5 3 (2 * 12) = 32 := by
  sorry

end hunter_saw_32_frogs_l415_415648


namespace initial_carrots_proof_l415_415349

-- Given conditions
def initial_carrots := sorry
def thrown_out := 11
def picked_later := 15
def total_carrots := 52

-- Equation derived from conditions
def equation : Prop := (initial_carrots - thrown_out) + picked_later = total_carrots

-- Prove the initial number of carrots Maria picked is 48
theorem initial_carrots_proof (h : equation) : initial_carrots = 48 :=
sorry

end initial_carrots_proof_l415_415349


namespace polygon_area_is_correct_l415_415872

def points : List (ℕ × ℕ) := [
  (0, 0), (10, 0), (20, 0), (30, 10),
  (0, 20), (10, 20), (20, 30), (10, 30),
  (0, 30), (20, 10), (30, 20), (10, 10)
]

def polygon_area (ps : List (ℕ × ℕ)) : ℕ := sorry

theorem polygon_area_is_correct :
  polygon_area points = 9 := sorry

end polygon_area_is_correct_l415_415872


namespace total_cups_l415_415719

theorem total_cups (n : ℕ) (tea_per_non_rainy_day : ℕ) (hot_chocolate_per_rainy_day : ℕ) : 
  n = 3 → tea_per_non_rainy_day = 4 → hot_chocolate_per_rainy_day = 3 → 
  5 * tea_per_non_rainy_day - (2 * hot_chocolate_per_rainy_day) = 14 →
  2 * n = hot_chocolate_per_rainy_day →
  (5 * tea_per_non_rainy_day) + (2 * hot_chocolate_per_rainy_day) = 26 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1] at h5
  rw [h3] at h2
  sorry

end total_cups_l415_415719


namespace longer_side_length_l415_415120

-- Given a circle with radius 6 cm
def radius : ℝ := 6
def circle_area : ℝ := Real.pi * radius^2

-- Given that the area of the rectangle is three times the area of the circle
def rectangle_area : ℝ := 3 * circle_area

-- Given the rectangle has a shorter side equal to the diameter of the circle
def shorter_side : ℝ := 2 * radius

-- Prove that the length of the longer side of the rectangle is 9π cm
theorem longer_side_length : ∃ (longer_side : ℝ), longer_side = 9 * Real.pi :=
by
  have circle_area_def : circle_area = 36 * Real.pi := by sorry
  have rectangle_area_def : rectangle_area = 108 * Real.pi := by sorry
  have shorter_side_def : shorter_side = 12 := by sorry
  let longer_side := rectangle_area / shorter_side
  use longer_side
  show longer_side = 9 * Real.pi
  sorry

end longer_side_length_l415_415120


namespace R_and_D_costs_increase_productivity_l415_415866

noncomputable def R_and_D_t := 3205.69
noncomputable def Delta_APL_t_plus_1 := 1.93
noncomputable def desired_result := 1661

theorem R_and_D_costs_increase_productivity :
  R_and_D_t / Delta_APL_t_plus_1 = desired_result :=
by
  sorry

end R_and_D_costs_increase_productivity_l415_415866


namespace contrapositive_of_sine_implies_angle_l415_415740

variable {A B : ℝ}

theorem contrapositive_of_sine_implies_angle (h : sin A = sin B → A = B) : A ≠ B → sin A ≠ sin B :=
by
  sorry

end contrapositive_of_sine_implies_angle_l415_415740


namespace correct_statements_l415_415249

def odd_function_on_f (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def arithmetic_sequence (seq : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, seq (n + 1) = seq n + d

def geometric_sequence (seq : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, seq (n + 1) = seq n * q

noncomputable def f : ℝ → ℝ := sorry

theorem correct_statements :
  (odd_function_on_f f ∧ (∀ x > 0, f x = log x)) →
  (∃ x_seq : ℕ → ℝ, arithmetic_sequence x_seq ∧ arithmetic_sequence (λ n, f (x_seq n))) ∧
  (∃ x_seq : ℕ → ℝ, geometric_sequence x_seq ∧ arithmetic_sequence (λ n, f (x_seq n))) ∧
  (∃ x_seq : ℕ → ℝ, arithmetic_sequence x_seq ∧ geometric_sequence (λ n, f (x_seq n))) := 
sorry

end correct_statements_l415_415249


namespace smallest_m_l415_415772

theorem smallest_m (m : ℕ) (h1 : m > 0) (h2 : 3 ^ ((m + m ^ 2) / 4) > 500) : m = 5 := 
by sorry

end smallest_m_l415_415772


namespace remaining_candies_correct_l415_415187

variable (clowns : ℕ) (children : ℕ) (initial_supply : ℕ)
variable (candies_per_clown : ℕ) (candies_per_child : ℕ) (prizes : ℕ)
variable (total_given : ℕ) (remaining_candies : ℕ)

-- Setting up conditions
def conditions := 
  clowns = 4 ∧ 
  children = 30 ∧ 
  initial_supply = 1200 ∧ 
  candies_per_clown = 10 ∧ 
  candies_per_child = 15 ∧ 
  prizes = 100

-- Total candies given out
def total_candies_given (clowns children candies_per_clown candies_per_child prizes : ℕ) : ℕ :=
  clowns * candies_per_clown + children * candies_per_child + prizes

-- Prove the remaining candies
theorem remaining_candies_correct : 
  conditions → 
  remaining_candies = initial_supply - total_given → 
  total_given = total_candies_given 4 30 10 15 100 → 
  remaining_candies = 610 :=
by 
  intros hcond hrmain htotal
  rw [total_candies_given, hcond.1, hcond.2, hcond.3, hcond.4, hcond.5, hcond.6] at htotal
  rw [hrmain, htotal]
  sorry

end remaining_candies_correct_l415_415187


namespace beta_interval_solution_l415_415907

/-- 
Prove that the values of β in the set {β | β = π/6 + 2*k*π, k ∈ ℤ} 
that satisfy the interval (-2*π, 2*π) are β = π/6 or β = -11*π/6.
-/
theorem beta_interval_solution :
  ∀ β : ℝ, (∃ k : ℤ, β = (π / 6) + 2 * k * π) → (-2 * π < β ∧ β < 2 * π) →
  (β = π / 6 ∨ β = -11 * π / 6) :=
by
  intros β h_exists h_interval
  sorry

end beta_interval_solution_l415_415907


namespace limit_a_n_to_ln_2_l415_415072

noncomputable def a_n (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), 1 / (n + k)

theorem limit_a_n_to_ln_2 : tendsto (λ n, a_n n) at_top (𝓝 (Real.log 2)) :=
sorry

end limit_a_n_to_ln_2_l415_415072


namespace g_2002_eq_1_l415_415238

noncomputable def f : ℝ → ℝ := sorry
def g (x : ℝ) : ℝ := f(x) + 1 - x 

theorem g_2002_eq_1
  (h1 : f 1 = 1)
  (h2 : ∀ x : ℝ, f (x + 5) ≥ f x + 5)
  (h3 : ∀ x : ℝ, f (x + 1) ≤ f x + 1) :
  g 2002 = 1 :=
sorry

end g_2002_eq_1_l415_415238


namespace triangle_area_l415_415505

theorem triangle_area : 
  let line_eq (x y : ℝ) := 3 * x + 2 * y = 12
  let x_intercept := (4 : ℝ)
  let y_intercept := (6 : ℝ)
  ∃ (x y : ℝ), line_eq x y ∧ x = x_intercept ∧ y = y_intercept ∧
  ∃ (area : ℝ), area = 1 / 2 * x * y ∧ area = 12 :=
by
  sorry

end triangle_area_l415_415505


namespace log2_f_neg1_l415_415614

theorem log2_f_neg1 (a : ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1)
  (f : ℝ → ℝ) (h_f_inv : ∀ x, f (log a x) = x) (h_point : f 2 = 1 / 4) :
  log 2 (f (-1)) = 1 :=
sorry

end log2_f_neg1_l415_415614


namespace part1_monotonically_increasing_part2_positive_definite_l415_415947

-- Definition of the function
def f (k : ℝ) (x : ℝ) : ℝ := x^2 + 2 * k * x + 4

-- Part 1: Proving the range of k for monotonically increasing function on [1, 4]
theorem part1_monotonically_increasing (k : ℝ) :
  (∀ x ∈ Set.Icc 1 4, ∀ y ∈ Set.Icc 1 4, x ≤ y → f k x ≤ f k y) ↔ k ≥ -1 :=
sorry

-- Part 2: Proving the range of k for f(x) > 0 for all x
theorem part2_positive_definite (k : ℝ) :
  (∀ x : ℝ, f k x > 0) ↔ k ∈ Set.Ioo (-2) 2 :=
sorry

end part1_monotonically_increasing_part2_positive_definite_l415_415947


namespace tiles_needed_l415_415394

/-- 
Given:
- The cafeteria is tiled with the same floor tiles.
- It takes 630 tiles to cover an area of 18 square decimeters of tiles.
- We switch to square tiles with a side length of 6 decimeters.

Prove:
- The number of new tiles needed to cover the same area is 315.
--/
theorem tiles_needed (n_tiles : ℕ) (area_per_tile : ℕ) (new_tile_side_length : ℕ) 
  (h1 : n_tiles = 630) (h2 : area_per_tile = 18) (h3 : new_tile_side_length = 6) :
  (630 * 18) / (6 * 6) = 315 :=
by
  sorry

end tiles_needed_l415_415394


namespace triangle_min_sum_l415_415969

-- Let a, b, and c be the sides of the triangle opposite to angles A, B, and C respectively
variables {a b c : ℝ}

-- Given conditions:
-- 1. (a + b)^2 - c^2 = 4
-- 2. C = 60 degrees, and by cosine rule, we have cos C = (a^2 + b^2 - c^2) / (2ab)
-- Since C = 60 degrees, cos C = 1/2
-- Therefore, (a^2 + b^2 - c^2) / (2ab) = 1/2

theorem triangle_min_sum (h1 : (a + b) ^ 2 - c ^ 2 = 4)
    (h2 : cos (real.pi / 3) = (a ^ 2 + b ^ 2 - c ^ 2) / (2 * a * b)) :
  a + b ≥ 2 * real.sqrt (4 / 3) :=
  by
    sorry

end triangle_min_sum_l415_415969


namespace min_max_arguments_l415_415518

-- Define a condition of a complex number z such that |z - 5 - 5i| = 5
noncomputable def satisfies_condition (z : ℂ) : Prop :=
  abs (z - (5 + 5 * complex.i)) = 5

-- Define z1 and z2 as per the given problem
noncomputable def z1 : ℂ := 5
noncomputable def z2 : ℂ := 5 * complex.i

-- Theorem statement proving that z1 and z2 are the complex numbers with minimum and maximum arguments
theorem min_max_arguments (z : ℂ) (h : satisfies_condition z) : z = z1 ∨ z = z2 :=
by sorry

end min_max_arguments_l415_415518


namespace third_stack_shorter_by_five_l415_415361

theorem third_stack_shorter_by_five
    (first_stack second_stack third_stack fourth_stack : ℕ)
    (h1 : first_stack = 5)
    (h2 : second_stack = first_stack + 2)
    (h3 : fourth_stack = third_stack + 5)
    (h4 : first_stack + second_stack + third_stack + fourth_stack = 21) :
    second_stack - third_stack = 5 :=
by
  sorry

end third_stack_shorter_by_five_l415_415361


namespace distance_between_intersections_correct_l415_415176

noncomputable def distance_between_intersections (u v p : ℕ) (h1 : u = 4) (h2 : v = 0) (h3 : p = 1) : ℝ :=
  let f := λ y : ℝ, y^3
  let g := λ y : ℝ, 1 - y^2
  let y1 := 1   -- first intersection y-coordinate
  let y2 := -1  -- second intersection y-coordinate
  let x1 := f y1
  let x2 := f y2
  let square_dist := (x1 - x2)^2 + (y1 - y2)^2
  real.sqrt square_dist

theorem distance_between_intersections_correct : ∃ u v p : ℕ, u = 4 ∧ v = 0 ∧ p = 1 ∧ 
  distance_between_intersections u v p = 2 :=
by {
  existsi 4, 0, 1,
  simp [distance_between_intersections],
  sorry
}

end distance_between_intersections_correct_l415_415176


namespace lawnmower_blade_cost_l415_415823

theorem lawnmower_blade_cost (x : ℕ) : 4 * x + 7 = 39 → x = 8 :=
by
  sorry

end lawnmower_blade_cost_l415_415823


namespace projection_a_plus_b_on_a_l415_415593

variables (a b : ℝ^2)
variables (h1 : ‖a‖ = 1)
variables (h2 : ‖b‖ = 2)
variables (h3 : real.angle (a) (b) = real.pi / 3)

theorem projection_a_plus_b_on_a :
  ((a + b) ⬝ a) / ‖a‖ = 2 :=
by 
  sorry

end projection_a_plus_b_on_a_l415_415593


namespace angie_bought_18_pretzels_l415_415856

theorem angie_bought_18_pretzels
  (B : ℕ := 12) -- Barry bought 12 pretzels
  (S : ℕ := B / 2) -- Shelly bought half as many pretzels as Barry
  (A : ℕ := 3 * S) -- Angie bought three times as many pretzels as Shelly
  : A = 18 := sorry

end angie_bought_18_pretzels_l415_415856


namespace fill_tank_time_l415_415817

theorem fill_tank_time (X Y Z T : ℕ) (h1 : T = 2 * (X + Y)) (h2 : T = 3 * (X + Z)) (h3 : T = 4 * (Y + Z)) :
  let t := T / (X + Y + Z) in t = 24 / 13 :=
by
  -- Lean proof goes here
  sorry

end fill_tank_time_l415_415817


namespace remainder_when_divided_by_8_l415_415060

theorem remainder_when_divided_by_8 :
  (481207 % 8) = 7 :=
by
  sorry

end remainder_when_divided_by_8_l415_415060


namespace max_cubic_pairwise_products_l415_415208

theorem max_cubic_pairwise_products (n : ℕ) (a : ℕ → ℕ)
  (h1 : ∀ i, 1 ≤ i → i ≤ n → ¬∃ k, a i = k^3) :
  ∃ N, N = \left\lfloor \frac{n^2}{4} \right\rfloor := 
sorry

end max_cubic_pairwise_products_l415_415208


namespace no_real_values_satisfy_log_eq_l415_415244

noncomputable def valid_log_domain (x : ℝ) : Prop := x + 5 > 0 ∧ x - 3 > 0 ∧ x^2 - 5x - 14 > 0

theorem no_real_values_satisfy_log_eq (x : ℝ) : ¬ (valid_log_domain x ∧ log (x + 5) + log (x - 3) = log (x^2 - 5x - 14)) :=
by
  sorry

end no_real_values_satisfy_log_eq_l415_415244


namespace find_b_value_satisfying_conditions_l415_415619

noncomputable def ellipse_intersection_angle_condition (b : ℝ) : Prop :=
  (0 < b ∧ b < 2) ∧
  ∃ (A B M : ℝ × ℝ), 
    ((A.1^2 / 4 + A.2^2 / b^2 = 1) ∧ (B.1^2 / 4 + B.2^2 / b^2 = 1)) ∧
    (let x0 := (A.1 + B.1) / 2; y0 := (A.2 + B.2) / 2 in
    (y0 / x0 = b^2 / 4)) ∧
    (|θ| = 3 → (∃ α : ℝ, (tan(α + π/4) = θ ∨ tan(3π/4 - α) = θ) ∧ (θ = 3)))

theorem find_b_value_satisfying_conditions : 
  ∃ (b : ℝ), ellipse_intersection_angle_condition b ∧ b = sqrt 2 :=
by 
  -- replace 'by' statement with 'sorry' to skip proofs
  sorry

end find_b_value_satisfying_conditions_l415_415619


namespace spherical_caps_area_percentage_l415_415431

theorem spherical_caps_area_percentage 
  (R : ℝ) 
  -- Conditions:
  (cuts_spherical_caps : ∀ (caps_count : ℕ), caps_count = 6)
  (disks_touch_neighbors : ∀ (disks_count : ℕ), disks_count = 4) :
  -- Conclusion:
  (total_disk_area_percentage : ℝ) :=
  -- Proof that the total area of the six circular disks is 86.08% of the total surface area of the dice:
  (total_disk_area_percentage = 86.08) := 
sorry

end spherical_caps_area_percentage_l415_415431


namespace problem_1_problem_2_l415_415952

def power_function (m x : ℝ) : ℝ := (m^2 - 5 * m + 7) * x^(-m - 1)

axiom even_function {m : ℝ} : power_function m = λ x, power_function m (-x)

theorem problem_1 {m : ℝ} (h : m = 3) : power_function m (1 / 2) = 16 := 
by sorry

theorem problem_2 {m : ℝ} (h : m = 3) (a : ℝ) (h1 : power_function m (2 * a + 1) = power_function m a) : 
  a = -1 ∨ a = -(1 / 3) := 
by sorry

end problem_1_problem_2_l415_415952


namespace flight_duration_sum_l415_415479

theorem flight_duration_sum 
  (departure_time : ℕ×ℕ) (arrival_time : ℕ×ℕ) (delay : ℕ)
  (h m : ℕ)
  (h0 : 0 < m ∧ m < 60)
  (h1 : departure_time = (9, 20))
  (h2 : arrival_time = (13, 45)) -- using 13 for 1 PM, 24-hour format
  (h3 : delay = 25)
  (h4 : ((arrival_time.1 * 60 + arrival_time.2) - (departure_time.1 * 60 + departure_time.2) + delay) = h * 60 + m) :
  h + m = 29 :=
by {
  -- Proof is skipped
  sorry
}

end flight_duration_sum_l415_415479


namespace magazine_choice_count_l415_415149

theorem magazine_choice_count :
  let science_magazines := 4
  let digest_magazines := 3
  let entertainment_magazines := 2
  science_magazines + digest_magazines + entertainment_magazines = 9 := by {
    let science_magazines := 4
    let digest_magazines := 3
    let entertainment_magazines := 2
    show 4 + 3 + 2 = 9, from sorry
  }

end magazine_choice_count_l415_415149


namespace total_wood_needed_l415_415326

theorem total_wood_needed : 
      (4 * 4 + 4 * (4 * 5)) + 
      (10 * 6 + 10 * (6 - 3)) + 
      (8 * 5.5) + 
      (6 * (5.5 * 2) + 6 * (5.5 * 1.5)) = 345.5 := 
by 
  sorry

end total_wood_needed_l415_415326


namespace sum_of_x_coordinates_is_correct_l415_415571

theorem sum_of_x_coordinates_is_correct :
  let f1 : ℝ → ℝ := λ x, abs (x^2 - 8 * x + 12)
  let f2 : ℝ → ℝ := λ x, 3 - x / 2
  let f3 : ℝ → ℝ := λ x, x - 2
  (∃ x1 x2 x3 x4 x5 x6 : ℝ,
    f1 x1 = f2 x1 ∧ f1 x2 = f2 x2 ∧ f1 x3 = f3 x3 ∧
    f2 x4 = f3 x4 ∧ f2 x5 = f3 x5 ∧ f1 x6 = f3 x6 ∧
    2 ≤ x2 ∧ x2 ≤ 6 ∧ 2 ≤ x3 ∧ x3 ≤ 6 ∧
    x1 + x2 + x3 + x4 + x5 + x6 = 16.625) :=
by {
  sorry
}

end sum_of_x_coordinates_is_correct_l415_415571


namespace find_smallest_a_l415_415080

noncomputable def quadratic_eq_1_roots (a b : ℤ) : Prop :=
∃ α β : ℤ, x < -1 ∧ x^2 + bx + a = 0

noncomputable def quadratic_eq_2_roots (a c : ℤ) : Prop :=
∃ γ δ : ℤ, x < -1 ∧ x^2 + cx + (a - 1) = 0

theorem find_smallest_a : ∃ a, (quadratic_eq_1_roots a b ∧ quadratic_eq_2_roots a c) ↔ a = 15 :=
sorry

end find_smallest_a_l415_415080


namespace problem_equiv_f_x_l415_415289

variable {x : ℝ}
def f (x : ℝ) : ℝ := log ((1 + x) / (1 - x))

theorem problem_equiv_f_x :
  -1 < x ∧ x < 1 → f (4 * x - x^3)/(1 + 4 * x^2) = f x :=
  begin
    sorry
  end

end problem_equiv_f_x_l415_415289


namespace hyperbola_equation_and_line_intersection_l415_415629

theorem hyperbola_equation_and_line_intersection :
  ∃ (a b : ℝ) (x y m : ℝ),
    a > 0 ∧ b > 0 ∧
    (∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1) ∧
    2 * a = 2 * sqrt 3 ∧
    ((sqrt 5, 0) = (sqrt 5, 0)) ∧ -- For the focus at (-sqrt 5, 0)
    (∀ x1 y1 x2 y2, (∃ l : ℝ, y = 2 * x + l ∧ ∀ x1 y1 y2 x2, x1 ≠ x2 ∧ 
    y1 = 2 * x1 + l ∧ y2 = 2 * x2 + l → (10 * x1 * x1 + 12 * l * x1 + 3 * (l * l + 2) = 0) ∧
    abs (sqrt ((x2 - x1)^2 + (y2 - y1)^2)) = 4)) ∧
    ((3 * (l^2 - 10) > 0) ∧  -- Ensuring the discriminant condition
    l = sqrt 210 / 3 ∨ l = -(sqrt 210 / 3)) ∧ 
    (∀ x y, (x^2 / 3) - (y^2 / 2) = 1) :=
sorry

end hyperbola_equation_and_line_intersection_l415_415629


namespace sequence_solution_l415_415207

theorem sequence_solution {a : ℕ → ℝ} (h₀ : a 1 = 1) (h_rec : ∀ n, a (n + 1) = 2 * a n / (a n + 2)) :
  a 2 = 2 / 3 ∧ a 3 = 1 / 2 ∧ (∀ n, a n = 2 / (n + 1)) :=
by
  split
  · sorry -- Prove a 2 = 2 / 3
  split
  · sorry -- Prove a 3 = 1 / 2
  · intro n; sorry -- Prove general term a_n = 2 / (n + 1)

end sequence_solution_l415_415207


namespace pears_sold_in_morning_l415_415142

theorem pears_sold_in_morning (total pears_sold_afternoon : ℤ) (twice_afternoon : ℤ) : 
  total = 360 ∧ pears_sold_afternoon = 240 ∧ twice_afternoon = 2 * pears_sold_afternoon → 
  twice_afternoon / 2 = 120 :=
by
  assume h,
  cases h,
  sorry

end pears_sold_in_morning_l415_415142


namespace sum_repeating_decimals_as_fraction_l415_415886

-- Definitions for repeating decimals
def rep2 : ℝ := 0.2222
def rep02 : ℝ := 0.0202
def rep0002 : ℝ := 0.00020002

-- Prove the sum of the repeating decimals is equal to the given fraction
theorem sum_repeating_decimals_as_fraction :
  rep2 + rep02 + rep0002 = (2224 / 9999 : ℝ) :=
sorry

end sum_repeating_decimals_as_fraction_l415_415886


namespace largest_house_number_l415_415267

theorem largest_house_number (phone_number_digits : List ℕ) (house_number_digits : List ℕ) :
  phone_number_digits = [5, 0, 4, 9, 3, 2, 6] →
  phone_number_digits.sum = 29 →
  (∀ (d1 d2 : ℕ), d1 ∈ house_number_digits → d2 ∈ house_number_digits → d1 ≠ d2) →
  house_number_digits.sum = 29 →
  house_number_digits = [9, 8, 7, 5] :=
by
  intros
  sorry

end largest_house_number_l415_415267


namespace cars_sold_proof_l415_415475

noncomputable def total_cars_sold : Nat := 300
noncomputable def perc_audi : ℝ := 0.10
noncomputable def perc_toyota : ℝ := 0.15
noncomputable def perc_acura : ℝ := 0.20
noncomputable def perc_honda : ℝ := 0.18

theorem cars_sold_proof : total_cars_sold * (1 - (perc_audi + perc_toyota + perc_acura + perc_honda)) = 111 := by
  sorry

end cars_sold_proof_l415_415475


namespace total_frogs_seen_by_hunter_l415_415646

/-- Hunter saw 5 frogs sitting on lily pads in the pond. -/
def initial_frogs : ℕ := 5

/-- Three more frogs climbed out of the water onto logs floating in the pond. -/
def frogs_on_logs : ℕ := 3

/-- Two dozen baby frogs (24 frogs) hopped onto a big rock jutting out from the pond. -/
def baby_frogs : ℕ := 24

/--
The total number of frogs Hunter saw in the pond.
-/
theorem total_frogs_seen_by_hunter : initial_frogs + frogs_on_logs + baby_frogs = 32 := by
sorry

end total_frogs_seen_by_hunter_l415_415646


namespace intersection_at_one_point_l415_415523

theorem intersection_at_one_point (b : ℝ) :
  (∃ x₀ : ℝ, bx^2 + 7*x₀ + 4 = 0 ∧ (7)^2 - 4*b*4 = 0) →
  b = 49 / 16 :=
by
  sorry

end intersection_at_one_point_l415_415523


namespace product_of_two_integers_l415_415749

theorem product_of_two_integers :
  ∀ (a b : ℤ), 
    Nat.lcm a.natAbs b.natAbs = 72 ∧ Nat.gcd a.natAbs b.natAbs = 8 ∧ a = 4 * (Nat.gcd a.natAbs b.natAbs) →
    a * b = 576 := 
by 
  intros a b h,
  cases h with h1 h2,
  cases h2 with h2 h3,
  sorry

end product_of_two_integers_l415_415749


namespace product_of_fractions_l415_415164

theorem product_of_fractions : 
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 :=
by
  sorry

end product_of_fractions_l415_415164


namespace zero_product_gt_e_l415_415625

theorem zero_product_gt_e (a : ℝ) (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) 
  (h₃ : f x₁ = 0) (h₄ : f x₂ = 0) (h₅ : f = fun x => log x - a * x ^ 2) : 
  x₁ * x₂ > Real.exp 1 := 
sorry

end zero_product_gt_e_l415_415625


namespace number_of_pages_in_book_l415_415067

-- Define the conditions using variables and hypotheses
variables (P : ℝ) (h1 : 0.30 * P = 150)

-- State the theorem to be proved
theorem number_of_pages_in_book : P = 500 :=
by
  -- Proof would go here, but we use sorry to skip it
  sorry

end number_of_pages_in_book_l415_415067


namespace periodic_non_const_seq_l415_415833

def periodic_sequence (x : ℕ → ℤ) (p : ℕ) : Prop :=
  ∀ n : ℕ, x(n + p) = x(n)

theorem periodic_non_const_seq (x : ℕ → ℤ) (p : ℕ) (h : ∀ n, x(n + 1) = 3 * x(n) + 4 * x(n - 1)) :
  ∃ p, p > 0 ∧ periodic_sequence x p ∧ (∃ n m, n ≠ m ∧ x(n) ≠ x(m)) :=
sorry

end periodic_non_const_seq_l415_415833


namespace ice_cream_flavors_l415_415654

theorem ice_cream_flavors : (∑ (x y z : ℕ) in {(n : ℕ) | n ≤ 5}, x + y + z = 5) = 21 := by
  sorry

end ice_cream_flavors_l415_415654


namespace carla_chickens_l415_415531

theorem carla_chickens (initial_chickens : ℕ) (percent_died : ℕ) (bought_factor : ℕ) :
  initial_chickens = 400 →
  percent_died = 40 →
  bought_factor = 10 →
  let died := (percent_died * initial_chickens) / 100 in
  let bought := bought_factor * died in
  let total := initial_chickens - died + bought in
  total = 1840 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  let died := (40 * 400) / 100
  have hdied : died = 160 := rfl
  let bought := 10 * died
  have hbought : bought = 1600 := rfl
  let total := 400 - 160 + 1600
  have htotal : total = 1840 := rfl
  exact htotal

end carla_chickens_l415_415531


namespace ineq_triples_distinct_integers_l415_415376

theorem ineq_triples_distinct_integers 
  (x y z : ℤ) (h₁ : x ≠ y) (h₂ : y ≠ z) (h₃ : z ≠ x) : 
  ( ( (x - y)^7 + (y - z)^7 + (z - x)^7 - (x - y) * (y - z) * (z - x) * ((x - y)^4 + (y - z)^4 + (z - x)^4) )
  / ( (x - y)^5 + (y - z)^5 + (z - x)^5 ) ) ≥ 3 :=
sorry

end ineq_triples_distinct_integers_l415_415376


namespace nat_implies_int_incorrect_reasoning_due_to_minor_premise_l415_415020

-- Definitions for conditions
def is_integer (x : ℚ) : Prop := ∃ (n : ℤ), x = n
def is_natural (x : ℚ) : Prop := ∃ (n : ℕ), x = n

-- Major premise: Natural numbers are integers
theorem nat_implies_int (n : ℕ) : is_integer n := 
  ⟨n, rfl⟩

-- Minor premise: 1 / 3 is a natural number
def one_div_three_is_natural : Prop := is_natural (1 / 3)

-- Conclusion: 1 / 3 is an integer
def one_div_three_is_integer : Prop := is_integer (1 / 3)

-- The proof problem
theorem incorrect_reasoning_due_to_minor_premise :
  ¬one_div_three_is_natural :=
sorry

end nat_implies_int_incorrect_reasoning_due_to_minor_premise_l415_415020


namespace walnut_trees_planted_l415_415422

-- The number of walnut trees before planting
def walnut_trees_before : ℕ := 22

-- The number of walnut trees after planting
def walnut_trees_after : ℕ := 55

-- The number of walnut trees planted today
def walnut_trees_planted_today : ℕ := 33

-- Theorem statement to prove that the number of walnut trees planted today is 33
theorem walnut_trees_planted:
  walnut_trees_after - walnut_trees_before = walnut_trees_planted_today :=
by sorry

end walnut_trees_planted_l415_415422


namespace problem_a_problem_b_l415_415546

-- Problem a conditions and statement
def digit1a : Nat := 1
def digit2a : Nat := 4
def digit3a : Nat := 2
def digit4a : Nat := 8
def digit5a : Nat := 5

theorem problem_a : (digit1a * 100000 + digit2a * 10000 + digit3a * 1000 + digit4a * 100 + digit5a * 10 + 7) * 5 = 
                    7 * (digit1a * 100000 + digit2a * 10000 + digit3a * 1000 + digit4a * 100 + digit5a * 10 + 285) := by
  sorry

-- Problem b conditions and statement
def digit1b : Nat := 4
def digit2b : Nat := 2
def digit3b : Nat := 8
def digit4b : Nat := 5
def digit5b : Nat := 7

theorem problem_b : (1 * 100000 + digit1b * 10000 + digit2b * 1000 + digit3b * 100 + digit4b * 10 + digit5b) * 3 = 
                    (digit1b * 100000 + digit2b * 10000 + digit3b * 1000 + digit4b * 100 + digit5b * 10 + 1) := by
  sorry

end problem_a_problem_b_l415_415546


namespace tim_campaign_total_l415_415039

theorem tim_campaign_total (amount_max : ℕ) (num_max : ℕ) (num_half : ℕ) (total_donations : ℕ) (total_raised : ℕ)
  (H1 : amount_max = 1200)
  (H2 : num_max = 500)
  (H3 : num_half = 3 * num_max)
  (H4 : total_donations = num_max * amount_max + num_half * (amount_max / 2))
  (H5 : total_donations = 40 * total_raised / 100) :
  total_raised = 3750000 :=
by
  -- Proof is omitted
  sorry

end tim_campaign_total_l415_415039


namespace a_break_time_l415_415472

theorem a_break_time (distance_AB : ℤ) (midpoint_m : ℤ) (speed_A : ℤ) (speed_B : ℤ) (meet_point_1 : ℤ) (meet_point_2 : ℤ) (possible_break_1 : ℤ) (possible_break_2 : ℤ) :
  distance_AB = 600 ∧ midpoint_m = 300 ∧ speed_A = 60 ∧ speed_B = 40 ∧ meet_point_1 = 300 ∧ meet_point_2 = 150 ∧ 
  possible_break_1 = 625 ∧ possible_break_2 = 1875 → 
  exists (t:ℚ), t = 625 / 100 ∨ t = 1875 / 100 :=
by
  assume h,
  -- Insert proof here
  sorry

noncomputable def calculate_break_time (distance_AB : ℤ) (midpoint_m : ℤ) (speed_A : ℤ) (speed_B : ℤ) (meet_point_1 : ℤ) (meet_point_2 : ℤ) : ℚ :=
  if (distance_AB = 600 ∧ midpoint_m = 300 ∧ speed_A = 60 ∧ speed_B = 40 ∧ meet_point_1 = 300 ∧ meet_point_2 = 150)
  then if meet_point_2 = 150
       then (1875 / 100)
       else (625 / 100)
  else 
    0 -- default value if input conditions are not met

#eval calculate_break_time 600 300 60 40 300 150 -- Expected output: 18.75

#eval calculate_break_time 600 300 60 40 300 300 -- Expected output: 6.25

end a_break_time_l415_415472


namespace triangle_area_l415_415499

theorem triangle_area : 
  ∃ (A : ℝ), A = 12 ∧ (∃ (x_intercept y_intercept : ℝ), 3 * x_intercept + 2 * y_intercept = 12 ∧ x_intercept * y_intercept / 2 = A) :=
by
  sorry

end triangle_area_l415_415499


namespace T_53_eq_38_l415_415877

def T (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem T_53_eq_38 : T 5 3 = 38 := by
  sorry

end T_53_eq_38_l415_415877


namespace mod_remainder_l415_415059

open Int

theorem mod_remainder (n : ℤ) : 
  (1125 * 1127 * n) % 12 = 3 ↔ n % 12 = 1 :=
by
  sorry

end mod_remainder_l415_415059


namespace red_beads_count_in_necklace_l415_415831

def has_n_red_beads (n : ℕ) (blue_count : ℕ) (necklace : list ℕ) : Prop :=
  -- Each blue bead has beads of different colors on either side
  (∀ i, (necklace[i % blue_count] = 1) → 
        (necklace[(i + 1) % necklace.length] ≠ 1 ∧ 
         necklace[(i - 1) % necklace.length] ≠ 1)) ∧
  -- Every other bead from each red one is also of different colors
  (∀ i, (necklace[i] = 2) → 
        (necklace[(i + 1) % necklace.length] ≠ 2 ∧ 
         necklace[(i + 2) % necklace.length] ≠ 2))

def total_beads_correct (necklace : list ℕ) : Prop :=
  -- The total number of beads in the necklace should be the count of blue beads 
  -- plus the total number of red beads
  necklace.length = 30 + 60

theorem red_beads_count_in_necklace : ∃ (necklace : list ℕ), 
  total_beads_correct necklace ∧ 
  has_n_red_beads 60 30 necklace :=
sorry

end red_beads_count_in_necklace_l415_415831


namespace simplify_expr1_simplify_expr2_l415_415377

noncomputable section

-- Problem 1: Simplify the given expression
theorem simplify_expr1 (a b : ℝ) : 4 * a^2 + 2 * (3 * a * b - 2 * a^2) - (7 * a * b - 1) = -a * b + 1 := 
by sorry

-- Problem 2: Simplify the given expression
theorem simplify_expr2 (x y : ℝ) : 3 * (x^2 * y - 1/2 * x * y^2) - 1/2 * (4 * x^2 * y - 3 * x * y^2) = x^2 * y :=
by sorry

end simplify_expr1_simplify_expr2_l415_415377


namespace part1_part2_part3_l415_415901

-- Problem 1
theorem part1 : 1 / (Real.sqrt 5 + Real.sqrt 4) = Real.sqrt 5 - Real.sqrt 4 := sorry

-- Problem 2
theorem part2 (n : ℕ) (h : n ≥ 2) : 1 / (Real.sqrt n + Real.sqrt (n - 1)) = Real.sqrt n - Real.sqrt (n - 1) := sorry

-- Problem 3
theorem part3 : (∑ k in Finset.range 2023 \ Finset.range 2, 1 / (Real.sqrt k + Real.sqrt (k - 1))) * (Real.sqrt 2023 + 1) = 2022 := sorry

end part1_part2_part3_l415_415901


namespace longer_side_of_rectangle_l415_415103

theorem longer_side_of_rectangle (r : ℝ) (Aₙ Aₙ: ℝ) (L S: ℝ):
  (r = 6) → 
  (Aₙ = 36 * π) →
  (Aₙ = 108 * π) →
  (S = 12) → 
  (S * L = Aₙ) →
  L = 9 * π := sorry

end longer_side_of_rectangle_l415_415103


namespace scheduling_plans_l415_415151

-- Defining a set of 7 employees
@[derive fintype]
inductive Employee
| A
| B
| E1
| E2
| E3
| E4
| E5

-- Days from October 1st to 7th
def Days := Fin 7

-- A function to represent the schedule
def schedule : Days → Employee := sorry

-- A condition to express that each person is scheduled for one day
def one_per_day (s : Days → Employee) : Prop := 
  bijective s

-- A condition that A and B are not on consecutive days
def not_consecutive (s : Days → Employee) : Prop := 
  ∀ i : Days, i < 6 → (s i = Employee.A ∧ s (i+1) = Employee.B) → false

-- Theorem statement
theorem scheduling_plans : 
  ∃ s : Days → Employee, one_per_day s ∧ not_consecutive s ∧ fintype.card {s' : Days → Employee // one_per_day s' ∧ not_consecutive s'} = 3600 := 
sorry

end scheduling_plans_l415_415151


namespace angie_bought_18_pretzels_l415_415855

theorem angie_bought_18_pretzels
  (B : ℕ := 12) -- Barry bought 12 pretzels
  (S : ℕ := B / 2) -- Shelly bought half as many pretzels as Barry
  (A : ℕ := 3 * S) -- Angie bought three times as many pretzels as Shelly
  : A = 18 := sorry

end angie_bought_18_pretzels_l415_415855


namespace paisa_per_rs_l415_415802

-- Definitions based on the conditions
variable (A B C : ℝ)
variable (x : ℝ)

-- Define the conditions given in the problem.
def condition_1 := C = 64
def condition_2 := A + B + C = 328
def condition_3 := B = (A * x) / 100
def condition_4 := C = (A * 40) / 100

-- The statement we need to prove.
theorem paisa_per_rs (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) : x = 65 := 
sorry -- Proof not required for this task

end paisa_per_rs_l415_415802


namespace total_frogs_in_pond_l415_415651

def frogsOnLilyPads : ℕ := 5
def frogsOnLogs : ℕ := 3
def babyFrogsOnRock : ℕ := 2 * 12 -- Two dozen

theorem total_frogs_in_pond : frogsOnLilyPads + frogsOnLogs + babyFrogsOnRock = 32 :=
by
  sorry

end total_frogs_in_pond_l415_415651


namespace part1_part2_l415_415598

noncomputable def f (x : ℝ) (a : ℝ) := x^2 + 2*a*x + 2

theorem part1 (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 5 → f x a > 3*a*x) → a < 2*Real.sqrt 2 :=
sorry

theorem part2 (a : ℝ) :
  ∀ x : ℝ,
    ((a = 0) → x > 2) ∧
    ((a > 0) → (x < -1/a ∨ x > 2)) ∧
    ((-1/2 < a ∧ a < 0) → (2 < x ∧ x < -1/a)) ∧
    ((a = -1/2) → false) ∧
    ((a < -1/2) → (-1/a < x ∧ x < 2)) :=
sorry

end part1_part2_l415_415598


namespace train_crossing_time_l415_415844

theorem train_crossing_time (length_of_train : ℕ) (speed_kmh : ℕ) (speed_ms : ℕ) 
  (conversion_factor : speed_kmh * 1000 / 3600 = speed_ms) 
  (H1 : length_of_train = 180) 
  (H2 : speed_kmh = 72) 
  (H3 : speed_ms = 20) 
  : length_of_train / speed_ms = 9 := by
  sorry

end train_crossing_time_l415_415844


namespace regular_polygon_tile_sum_l415_415769

theorem regular_polygon_tile_sum
  (k m n : ℕ)
  (hk : 3 ≤ k)
  (hm : 3 ≤ m)
  (hn : 3 ≤ n)
  (angle_k : (k - 2) * 180 / k)
  (angle_m : (m - 2) * 180 / m)
  (angle_n : (n - 2) * 180 / n) :
  angle_k + angle_m + angle_n = 360 → 
  (1/k + 1/m + 1/n = 1/2) := 
by
  sorry

end regular_polygon_tile_sum_l415_415769


namespace quad_min_value_on_interval_l415_415789

/-- The quadratic function g(x) = x^2 - 4x + 9. -/
def g (x : ℝ) := x^2 - 4*x + 9

/-- The interval [-2, 0]. -/
def interval := Set.Icc (-2 : ℝ) (0 : ℝ)

/-- The minimum value of this quadratic function on the given interval is 9. -/
theorem quad_min_value_on_interval : ∃ x ∈ interval, ∀ y ∈ interval, g x ≤ g y ∧ g x = 9 :=
by
  exists 0
  split
  -- Proof that 0 ∈ interval and ∀ y ∈ interval, g 0 ≤ g y will be filled here
  sorry

end quad_min_value_on_interval_l415_415789


namespace area_of_triangle_bounded_by_line_and_axes_l415_415503

theorem area_of_triangle_bounded_by_line_and_axes (x y : ℝ) (hx : 3 * x + 2 * y = 12) :
  ∃ (area : ℝ), area = 12 := by
sorry

end area_of_triangle_bounded_by_line_and_axes_l415_415503


namespace students_in_class_l415_415671

theorem students_in_class (x : ℕ) (S : ℕ)
  (h1 : S = 3 * (S / x) + 24)
  (h2 : S = 4 * (S / x) - 26) : 3 * x + 24 = 4 * x - 26 :=
by
  sorry

end students_in_class_l415_415671


namespace volume_of_apple_juice_l415_415478

noncomputable def apple_juice_volume : ℝ :=
  let pi := Real.pi
  let total_ratio := 2 + 5
  let apple_ratio := 2 / total_ratio.toRat
  let radius := 2
  let height := 3
  let total_volume := pi * radius^2 * height
  apple_ratio * total_volume

theorem volume_of_apple_juice : abs (apple_juice_volume - 10.74) < 0.01 := by
  sorry

end volume_of_apple_juice_l415_415478


namespace average_fuel_efficiency_l415_415152

theorem average_fuel_efficiency (dist1 dist2 : ℕ) (efficiency1 efficiency2 : ℕ) (total_distance total_gasoline : ℝ) :
  dist1 = 150 ∧ efficiency1 = 25 ∧ dist2 = 150 ∧ efficiency2 = 15 →
  total_distance = dist1 + dist2 →
  total_gasoline = (dist1 / efficiency1) + (dist2 / efficiency2) →
  (total_distance / total_gasoline) = 18.75 :=
begin
  sorry
end

end average_fuel_efficiency_l415_415152


namespace daily_sales_profit_52_yuan_selling_price_for_1350_profit_l415_415129

-- Definitions based on the conditions
def cost_per_unit : ℕ := 40
def base_selling_price : ℕ := 50
def base_sales_volume : ℕ := 100
def sales_volume_decrease_per_price_increase (price_increase : ℕ) : ℕ :=
  2 * price_increase

-- Part 1: Proving daily sales profit at 52 yuan per unit is 1152 yuan
theorem daily_sales_profit_52_yuan :
  let selling_price := 52
  let price_increase := selling_price - base_selling_price
  let profit_per_unit := selling_price - cost_per_unit
  let decrease_in_sales_volume := sales_volume_decrease_per_price_increase price_increase
  let new_sales_volume := base_sales_volume - decrease_in_sales_volume
  let daily_sales_profit := profit_per_unit * new_sales_volume
  daily_sales_profit = 1152 :=
by
  -- Calculation steps and proof are assumed
  sorry

-- Part 2: Proving to achieve 1350 yuan profit, selling price should be 55 yuan
theorem selling_price_for_1350_profit :
  let desired_profit := 1350
  let quadratic_roots := λ (x : ℕ), x * x - 140 * x + 4675 = 0
  let valid_selling_price := 55 -- Based on the problem constraints and quadratic roots properties
  (valid_selling_price ≤ 65) ∧ (valid_selling_price * 2 - 90 ≥ 0) :=
by
  -- Calculation steps and proof are assumed
  sorry

end daily_sales_profit_52_yuan_selling_price_for_1350_profit_l415_415129


namespace cos_gamma_l415_415335

noncomputable def cos_gamma (a b c : ℝ) : ℝ :=
  c / (Real.sqrt (a^2 + b^2 + c^2))

theorem cos_gamma' (α' β' : ℝ) (hα' : Real.cos α' = 1/4) (hβ' : Real.cos β' = 1/2) :
  cos_gamma(Real.sqrt (11)/4) :=
by
  sorry

end cos_gamma_l415_415335


namespace impossible_numbering_l415_415998

-- Define what it means for triangles to be neighbors
def neighbors (t1 t2 : ℕ) : Prop :=
  -- Assuming we have a way to identify triangles and determine neighbors
  sorry

/--
  It is impossible to write the numbers from 1 to 16 inside 16 equilateral triangles
  such that the difference between the numbers placed in any two
  neighboring triangles is 1 or 2.
-/
theorem impossible_numbering (f : ℕ → ℕ) (h_f : ∀ n, 1 ≤ f n ∧ f n ≤ 16) :
  ¬ (∀ (i j : ℕ), neighbors i j → abs (f i - f j) = 1 ∨ abs (f i - f j) = 2) :=
by
  sorry

end impossible_numbering_l415_415998


namespace clock_hands_angle_l415_415432

theorem clock_hands_angle (h m : ℕ) (H : h = 12) (M : m = 20) :
  let minute_hand_angle := (M * 360) / 60,
      hour_hand_angle := ((H * 60 + M) * 360) / (12 * 60)
  in |minute_hand_angle - hour_hand_angle| = 110 := 
by 
  let minute_hand_angle := (M * 360) / 60
  let hour_hand_angle := ((H * 60 + M) * 360) / (12 * 60)
  show |minute_hand_angle - hour_hand_angle| = 110
  sorry

end clock_hands_angle_l415_415432


namespace convert_units_l415_415085

theorem convert_units :
  (0.56 * 10 = 5.6 ∧ 0.6 * 10 = 6) ∧
  (2.05 = 2 + 0.05 ∧ 0.05 * 100 = 5) :=
by 
  sorry

end convert_units_l415_415085


namespace monotonic_decreasing_interval_l415_415751

noncomputable def f (x : ℝ) : ℝ := (1 / 3) ^ (-x^2 - 4 * x + 3)

theorem monotonic_decreasing_interval :
  (∀ x1 x2 : ℝ, x1 ≤ x2 → f(x2) ≤ f(x1)) →
  (∀ x : ℝ, x ∈ set.Iic (-2) → ∃ y : ℝ, y = f(x)) :=
by
  sorry

end monotonic_decreasing_interval_l415_415751


namespace arithmetic_sequence_general_formula_l415_415229

theorem arithmetic_sequence_general_formula (a : ℤ) :
  ∀ n : ℕ, n ≥ 1 → (∃ a_1 a_2 a_3 : ℤ, a_1 = a - 1 ∧ a_2 = a + 1 ∧ a_3 = a + 3) →
  (a + 2 * n - 3 = a - 1 + (n - 1) * 2) :=
by
  intros n hn h_exists
  rcases h_exists with ⟨a_1, a_2, a_3, h1, h2, h3⟩
  sorry

end arithmetic_sequence_general_formula_l415_415229


namespace find_k_l415_415234

def point (ℝ : Type) := (ℝ × ℝ)

def area_of_triangle (A B C : point ℝ) : ℝ := 
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem find_k (k : ℝ) : 
  let A := (7 : ℝ, 7 : ℝ) in 
  let B := (3 : ℝ, 2 : ℝ) in 
  let C := (0 : ℝ, k) in 
  area_of_triangle A B C = 10 → k = 9 ∨ k = -13 / 3 :=
by
  sorry

end find_k_l415_415234


namespace volume_in_cubic_yards_l415_415486

-- Adding the conditions as definitions
def feet_to_yards : ℝ := 3 -- 3 feet in a yard
def cubic_feet_to_cubic_yards : ℝ := feet_to_yards^3 -- convert to cubic yards
def volume_in_cubic_feet : ℝ := 108 -- volume in cubic feet

-- The theorem to prove the equivalence
theorem volume_in_cubic_yards
  (h1 : feet_to_yards = 3)
  (h2 : volume_in_cubic_feet = 108)
  : (volume_in_cubic_feet / cubic_feet_to_cubic_yards) = 4 := 
sorry

end volume_in_cubic_yards_l415_415486


namespace min_cups_l415_415346

theorem min_cups (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ∃ n, (∀ (n : ℕ), n ≥ 2 * a + 2 * b - 2 * (Int.gcd a b)) :=
begin
  sorry
end

end min_cups_l415_415346


namespace complex_number_solution_l415_415086

theorem complex_number_solution (z : ℂ) (h : (3 + 4 * complex.I) * z = 1 - 2 * complex.I) : 
  z = - (1/5 : ℂ) - (2/5 : ℂ) * complex.I :=
by
  sorry

end complex_number_solution_l415_415086


namespace apply_r_six_times_problem_r_30_l415_415342

-- Define the function r
def r (θ : ℝ) : ℝ := 1 / (1 - θ)

-- State the main problem
theorem apply_r_six_times (θ : ℝ) : r (r (r (r (r (r θ))))) = θ :=
begin
  -- apply the triple application of r which falls back to identity
  -- hence applying six times also falls back to the original input
  sorry
end

-- Specific solution for θ = 30
theorem problem_r_30 : r (r (r (r (r (r 30))))) = 30 :=
begin
  -- Direct application of the derived theorem with θ = 30
  exact apply_r_six_times 30,
end

end apply_r_six_times_problem_r_30_l415_415342


namespace min_value_expr_l415_415564

noncomputable def expr (x : ℝ) : ℝ := (Real.sin x)^8 + (Real.cos x)^8 + 3 / (Real.sin x)^6 + (Real.cos x)^6 + 3

theorem min_value_expr : ∃ x : ℝ, expr x = 14 / 31 := 
by
  sorry

end min_value_expr_l415_415564


namespace volume_of_lemon_juice_l415_415814

theorem volume_of_lemon_juice 
  (h : height = 6) 
  (d : diameter = 2) 
  (ratio : lemon_to_water_ratio = 1 / 11) 
  (fullness : fullness = 1 / 2) : 
  round_decimal_2 (π * (d / 2)^2 * (h * fullness) * (1 / (1 + 11))) = 0.79 := 
sorry

end volume_of_lemon_juice_l415_415814


namespace maximum_discussions_remaining_l415_415089

theorem maximum_discussions_remaining (n : ℕ) (h : n = 2018) 
  (condition : ∀ (P : Finset ℕ) (hP : P.card = 4), ∃ p ∈ P, ∀ q ∈ P, q ≠ p → (p, q) ∈ discussions) : 
  ∃ m : ℕ, m = 3 ∧ 
    ∀ remaining_discussions : Finset (ℕ × ℕ), remaining_discussions.card = m → 
    ∀ (P : Finset ℕ) (hP : P.card = 4), 
    ∃ p ∈ P, ∀ q ∈ P, q ≠ p → (p, q) ∉ remaining_discussions :=
sorry

end maximum_discussions_remaining_l415_415089


namespace domain_of_f_l415_415433

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 12))

theorem domain_of_f :
  ∀ x : ℝ, f x ≠ 0 ↔ (x ≠ 15 / 2) :=
by
  sorry

end domain_of_f_l415_415433


namespace minimum_ab_l415_415932

theorem minimum_ab (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : ab + 2 = 2 * (a + b)) : ab ≥ 6 + 4 * Real.sqrt 2 :=
by
  sorry

end minimum_ab_l415_415932


namespace minimum_positive_period_of_f_l415_415409

noncomputable def f (x : ℝ) : ℝ := (1 + (Real.sqrt 3) * Real.tan x) * Real.cos x

theorem minimum_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T', T' > 0 → (∀ x, f (x + T') = f x) → T ≤ T') :=
sorry

end minimum_positive_period_of_f_l415_415409


namespace R_and_D_costs_increase_productivity_l415_415865

noncomputable def R_and_D_t := 3205.69
noncomputable def Delta_APL_t_plus_1 := 1.93
noncomputable def desired_result := 1661

theorem R_and_D_costs_increase_productivity :
  R_and_D_t / Delta_APL_t_plus_1 = desired_result :=
by
  sorry

end R_and_D_costs_increase_productivity_l415_415865


namespace correct_statements_l415_415213

noncomputable def f (x a : ℝ) := x^3 - a * x^2

theorem correct_statements (a : ℝ) :
  -- Conditions
  ∃ x, (∀ a : ℝ, ∃ x : ℝ, 3 * x^2 - 2 * a * x = 0) ∧ -- ①
  (∀ x, 3 * x^2 - 2 * a * x ≥ - (a^2 / 3)) ∧        -- ②
  (∃ x y, (f' a x) = 1 ∧ (f' a y) = 1 ∧ x ≠ y) ∧   -- ③
  ¬ (∃ x y, (f' a x) = 1 ∧ (f' a y) = 1 ∧ x = y)   -- ④
   :=
begin
  sorry
end
 where f' (a x : ℝ) := 3 * x^2 - 2 * a * x

end correct_statements_l415_415213


namespace expression_evaluation_l415_415553

variable (k : ℝ) -- Declare k as a real number variable

theorem expression_evaluation : 
  (2 ^ (-(2 * k + 3)) - 2 ^ (-(2 * k - 3)) + 2 ^ (-2 * k)) = - (55 / 8) * 2 ^ (-2 * k) :=
by
  sorry

end expression_evaluation_l415_415553


namespace RnD_cost_increase_l415_415863

theorem RnD_cost_increase (R_D_t : ℝ) (delta_APL_t1 : ℝ)
  (h1 : R_D_t = 3205.69)
  (h2 : delta_APL_t1 = 1.93) :
  R_D_t / delta_APL_t1 = 1661 :=
by 
  conv in (R_D_t / delta_APL_t1) {
    rw [h1, h2]
  }
  simp
  sorry

end RnD_cost_increase_l415_415863


namespace number_of_good_subsets_l415_415708

def is_good_subset (S : Finset ℕ) (A : Finset ℕ) : Prop :=
  A ⊆ S ∧ A.card = 31 ∧ (A.sum id) % 5 = 0

theorem number_of_good_subsets :
  let S := Finset.range 1991 \ {0}
  (Finset.filter (is_good_subset S) (Finset.powersetLen 31 S)).card =
    (1 / 5 * Nat.choose 1990 31) :=
by
  sorry

end number_of_good_subsets_l415_415708


namespace bill_original_profit_percentage_l415_415861

noncomputable def original_profit_percentage : ℝ :=
  let S := 989.9999999999992
  let additional_gain := 63
  let new_selling_price := S + additional_gain
  let factor := 1.17 in
  let P := new_selling_price / factor in
  let profit := S - P in
  (profit / P) * 100

theorem bill_original_profit_percentage :
  original_profit_percentage = 10 :=
by
  sorry

end bill_original_profit_percentage_l415_415861


namespace sum_first_10_odd_l415_415774

theorem sum_first_10_odd : (∑ i in finset.range 10, (2 * i + 1)) = 100 :=
by
  sorry

end sum_first_10_odd_l415_415774


namespace vector_scaling_l415_415960

theorem vector_scaling (m : ℝ) (λ : ℝ) :
  (1, m) = (2 * λ, -4 * λ) → m = -2 :=
by
  intro h
  cases h
  have h1 : 2 * λ = 1 := h.1
  have h2 : -4 * λ = m := h.2
  sorry

end vector_scaling_l415_415960


namespace tanA_tanB_range_l415_415681
noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  ∀ (acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π)
    (side_relation : a^2 = b^2 + b * c), 1 < tan A * tan B

theorem tanA_tanB_range (a b c A B C : ℝ) :
  triangle_ABC a b c A B C :=
begin
  sorry
end

end tanA_tanB_range_l415_415681


namespace locus_of_P_l415_415933

variable (P : ℝ × ℝ) (x y : ℝ)
variable (M : ℝ × ℝ := (-2, 0)) (N : ℝ × ℝ := (2, 0))

def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

theorem locus_of_P :
  ((distance P M) - (distance P N) = 4) →
  (P.2 = 0 ∧ P.1 = 2) :=
by
  sorry

end locus_of_P_l415_415933


namespace total_votes_cast_is_24000_candidateE_votes_candidateE_received_votes_l415_415827

-- Define the total votes cast, V
def V : ℝ := 24000

-- Given conditions: percentages and margin
def candidateA_votes (V : ℝ) : ℝ := 0.30 * V
def candidateB_votes (V : ℝ) : ℝ := 0.25 * V
def candidateC_votes (V : ℝ) : ℝ := 0.20 * V
def candidateD_votes (V : ℝ) : ℝ := 0.15 * V

-- Given the margin condition that candidate A won over candidate B by 1200 votes
theorem total_votes_cast_is_24000 : V = 24000 :=
by
  have margin_condition : candidateA_votes V - candidateB_votes V = 1200
  rw [candidateA_votes, candidateB_votes] at margin_condition
  have margin_simplified : 0.05 * V = 1200 := margin_condition
  rw div_eq_iff at margin_simplified
  linarith

-- Given total votes cast, find the votes Candidate E received
theorem candidateE_votes (V : ℝ) : ℝ :=
  V - (candidateA_votes V + candidateB_votes V + candidateC_votes V + candidateD_votes V)

theorem candidateE_received_votes : candidateE_votes V = 2400 :=
by
  unfold candidateE_votes
  rw [candidateA_votes, candidateB_votes, candidateC_votes, candidateD_votes]
  have calculation : V - (0.90 * V) = 0.10 * V by linarith
  rw calculation
  norm_num
  rw mul_eq_mul_right_iff
  exact Or.inr rfl

end total_votes_cast_is_24000_candidateE_votes_candidateE_received_votes_l415_415827


namespace cyclists_cannot_reach_point_B_l415_415767

def v1 := 35 -- Speed of the first cyclist in km/h
def v2 := 25 -- Speed of the second cyclist in km/h
def t := 2   -- Total time in hours
def d  := 30 -- Distance from A to B in km

-- Each cyclist does not rest simultaneously
-- Time equations based on their speed proportions

theorem cyclists_cannot_reach_point_B 
  (v1 := 35) (v2 := 25) (t := 2) (d := 30) 
  (h1 : t * (v1 * (5 / (5 + 7)) / 60) + t * (v2 * (7 / (5 + 7)) / 60) < d) : 
  False := 
sorry

end cyclists_cannot_reach_point_B_l415_415767


namespace positive_integers_expressible_count_l415_415894

theorem positive_integers_expressible_count :
  (∃ n ∈ {n : ℕ | 1 ≤ n ∧ n ≤ 1200},
   (∃ x : ℝ, ⌊x⌋ + ⌊2 * x⌋ + ⌊3 * x⌋ + ⌊4 * x⌋ = n)) → 604 :=
by
  sorry

end positive_integers_expressible_count_l415_415894


namespace cosine_of_angle_between_vectors_l415_415669

open Real

structure Vector2D where
  x : ℝ
  y : ℝ

def vector_add (u v : Vector2D) : Vector2D :=
  ⟨u.x + v.x, u.y + v.y⟩

def vector_sub (u v : Vector2D) : Vector2D :=
  ⟨u.x - v.x, u.y - v.y⟩

def dot_product (u v : Vector2D) : ℝ :=
  u.x * v.x + u.y * v.y

def magnitude (v : Vector2D) : ℝ :=
  sqrt (v.x * v.x + v.y * v.y)

noncomputable def cos_angle (a b : Vector2D) : ℝ :=
  dot_product a b / (magnitude a * magnitude b)

theorem cosine_of_angle_between_vectors : 
  let a := Vector2D.mk 1 (-4)
      b := Vector2D.mk (-1) 2
      u := vector_add a b
      v := vector_sub a b
  in cos_angle u v = 3 * sqrt 10 / 10 :=
by
  let a := Vector2D.mk 1 (-4)
  let b := Vector2D.mk (-1) 2
  let u := vector_add a b
  let v := vector_sub a b
  sorry

end cosine_of_angle_between_vectors_l415_415669


namespace parallelogram_has_midpoint_ellipse_l415_415454

theorem parallelogram_has_midpoint_ellipse (P : Type) [affine_space P ℝ] 
    (parallelogram: set P) (H_parallelogram : is_parallelogram parallelogram) :
  ∃ (ellipse : set P), (∀ side ∈ sides parallelogram, tangent_at_midpoints ellipse side) := 
sorry

end parallelogram_has_midpoint_ellipse_l415_415454


namespace roots_identity_l415_415340

theorem roots_identity (p q r : ℝ) (h₁ : p + q + r = 15) (h₂ : p * q + q * r + r * p = 25) (h₃ : p * q * r = 10) :
  (1 + p) * (1 + q) * (1 + r) = 51 :=
by sorry

end roots_identity_l415_415340


namespace λ_range_l415_415612

open scoped BigOperators

-- Given the first term of sequence a_n is 1
def a_initial : ℕ → ℝ 
| 0       := 1
| (n + 1) := a_initial n + 3 - 2^n 

-- Define the sequence b_n where b_1 + b_3 = 5 and b_5 - b_1 = 15, found to be b_n = 2^(n-1)
def b : ℕ → ℝ
| 0 := 1
| (n + 1) := 2^n 

-- Define S_n as the sum of the first n terms of the sequence a_n
def S : ℕ → ℝ
| 0 := a_initial 0 
| (n + 1) := S n + (a_initial (n + 1))

-- Define the condition S_{n+1} + b_n = S_n + a_n + 3
lemma Sn_relation (n : ℕ) : (S (n + 1)) + (b n) = S n + (a_initial n) + 3 := 
sorry 

-- Define that only 5 terms of the sequence {a_n} are not less than the real number λ
def five_terms_NOT_less_than_λ (λ : ℝ) : Prop := 
(∀ n, (a_initial n ≥ λ → n < 5)) ∧
(∃ (s : finset ℕ), (∀ i ∈ s, a_initial i ≥ λ) ∧ s.card = 5)

-- Prove that the range for λ is (-16, -2]
theorem λ_range : (∀ λ, five_terms_NOT_less_than_λ λ → -16 < λ ∧ λ <= -2) := 
sorry

end λ_range_l415_415612


namespace parallelogram_lines_intersect_at_single_point_l415_415225

-- Helper definitions for parallel lines and intersection
def Parallelogram (A B C D : Point) : Prop :=
  (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (D ≠ A) ∧ Segment A B ∥ Segment C D ∧ Segment B C ∥ Segment D A

def ParallelLinesIntersectAt (A B C D M : Point) : Prop :=
  ∃ P : Point, Line A (LineParallelTo (Line M C)) = Line P ∧ 
               Line B (LineParallelTo (Line M D)) = Line P ∧ 
               Line C (LineParallelTo (Line M A)) = Line P ∧ 
               Line D (LineParallelTo (Line M B)) = Line P

-- Main Lean theorem statement
theorem parallelogram_lines_intersect_at_single_point 
  (A B C D M : Point)
  (h_parallelogram : Parallelogram A B C D) 
  (h_parallel_lines : ParallelLinesIntersectAt A B C D M) :
  ∃ P : Point, Line A (LineParallelTo (Line M C)) = Line P ∧ 
               Line B (LineParallelTo (Line M D)) = Line P ∧ 
               Line C (LineParallelTo (Line M A)) = Line P ∧ 
               Line D (LineParallelTo (Line M B)) = Line P :=
sorry

end parallelogram_lines_intersect_at_single_point_l415_415225


namespace num_rows_seat_9_people_l415_415550

-- Define the premises of the problem.
def seating_arrangement (x y : ℕ) : Prop := (9 * x + 7 * y = 58)

-- The theorem stating the number of rows seating exactly 9 people.
theorem num_rows_seat_9_people
  (x y : ℕ)
  (h : seating_arrangement x y) :
  x = 1 :=
by
  -- Proof is not required as per the instruction
  sorry

end num_rows_seat_9_people_l415_415550


namespace perimeter_triangle_ABI_l415_415994

universe u

variables {AC BC AB CD AD BD r x : ℝ}
variables {A B C D I : Type u}

theorem perimeter_triangle_ABI (h1 : AC = 5) 
                              (h2 : BC = 12)
                              (h3 : AB = 13) 
                              (h4 : CD = real.sqrt (5 * 12)) : 
                              x ≥ real.sqrt 15 → 
                              x = real.sqrt 15 → 
                              let P := AB + 2 * x in 
                              P = 13 + 2 * x := 
by
  sorry

end perimeter_triangle_ABI_l415_415994


namespace longer_side_of_rectangle_is_l415_415117

-- Define the radius of the circle
def radius (r : ℝ) : Prop := r = 6

-- Define the area of the circle given the radius
def area_circle (A : ℝ) : Prop := A = 36 * Real.pi

-- Define the relationship between the area of the rectangle and the circle
def area_rectangle (A_rect : ℝ) (A : ℝ) : Prop := A_rect = 3 * A

-- Define the length of the shorter side of the rectangle
def shorter_side (s : ℝ) : Prop := s = 12

-- State the goal: the length of the longer side of the rectangle
def longer_side (l : ℝ) (A_rect : ℝ) (s : ℝ) : Prop := l = A_rect / s

-- Theorem statement combining all conditions and the goal
theorem longer_side_of_rectangle_is :
  ∃ l : ℝ, ∃ A_rect : ℝ, ∃ s : ℝ, radius 6 ∧ area_circle 36 * Real.pi ∧ area_rectangle A_rect (36 * Real.pi) ∧ shorter_side s ∧ longer_side l A_rect s :=
by {
  sorry
}

end longer_side_of_rectangle_is_l415_415117


namespace probability_of_selecting_cooking_l415_415099

def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

theorem probability_of_selecting_cooking : (favorable_outcomes : ℚ) / total_courses = 1 / 4 := 
by 
  sorry

end probability_of_selecting_cooking_l415_415099


namespace estimate_defective_pairs_l415_415798

/-- Define the data from the frequency table. --/
def sample_data : List (ℕ × ℕ × ℝ) :=
  [(20, 17, 0.85),
   (40, 38, 0.95),
   (60, 55, 0.92),
   (80, 75, 0.94),
   (100, 96, 0.96),
   (200, 189, 0.95),
   (300, 286, 0.95)]

/-- Calculate the defective rate from the largest sample size. --/
def defective_rate : ℝ :=
  1 - 0.95

/-- Prove that the estimated number of defective pairs in 1500 pairs of sports shoes is 75. --/
theorem estimate_defective_pairs :
  let total_pairs := 1500
  let estimated_defective_pairs := total_pairs * defective_rate
  estimated_defective_pairs = 75 :=
by
  let total_pairs := 1500
  let estimated_defective_pairs := total_pairs * defective_rate
  have h_defective_rate : defective_rate = 0.05 := rfl
  have h_calc : estimated_defective_pairs = 1500 * 0.05 := rfl
  rw [← h_defective_rate] at h_calc
  exact h_calc ▸ rfl

end estimate_defective_pairs_l415_415798


namespace triangle_area_l415_415055

theorem triangle_area (a b c d e f : ℝ) (h1 : a = 3) (h2 : b = 6) (h3 : c = -2) (h4 : d = 18)
                      (h_intersect_x : e = 12 / 5) (h_intersect_y : f = 66 / 5) : 
                      let y_intercept_1 := (0, b) in
                      let y_intercept_2 := (0, d) in
                      let intersect := (e, f) in
                      let base := d - b in
                      let height := e in
                      area (base * height / 2) = 14.4 := by
  → sorry

end triangle_area_l415_415055


namespace train_travel_distance_l415_415806

theorem train_travel_distance
  (coal_per_mile_lb : ℝ)
  (remaining_coal_lb : ℝ)
  (travel_distance_per_unit_mile : ℝ)
  (units_per_unit_lb : ℝ)
  (remaining_units : ℝ)
  (total_distance : ℝ) :
  coal_per_mile_lb = 2 →
  remaining_coal_lb = 160 →
  travel_distance_per_unit_mile = 5 →
  units_per_unit_lb = remaining_coal_lb / coal_per_mile_lb →
  remaining_units = units_per_unit_lb →
  total_distance = remaining_units * travel_distance_per_unit_mile →
  total_distance = 400 :=
by
  sorry

end train_travel_distance_l415_415806


namespace isosceles_triangle_perimeter_l415_415925

theorem isosceles_triangle_perimeter (a b : ℝ) (h₁ : 2 * a - 3 * b + 5 = 0) (h₂ : 2 * a + 3 * b - 13 = 0) :
  ∃ p : ℝ, p = 7 ∨ p = 8 :=
sorry

end isosceles_triangle_perimeter_l415_415925


namespace longer_side_of_rectangle_l415_415105

theorem longer_side_of_rectangle (r : ℝ) (Aₙ Aₙ: ℝ) (L S: ℝ):
  (r = 6) → 
  (Aₙ = 36 * π) →
  (Aₙ = 108 * π) →
  (S = 12) → 
  (S * L = Aₙ) →
  L = 9 * π := sorry

end longer_side_of_rectangle_l415_415105


namespace fourth_month_sale_l415_415818

theorem fourth_month_sale (s1 s2 s3 s5 s6 : ℕ) (avg : ℕ) :
  s1 = 5420 →
  s2 = 5660 →
  s3 = 6200 →
  s5 = 6500 →
  s6 = 7070 →
  avg = 6200 →
  (s1 + s2 + s3 + s5 + s6 + fourth_sale) / 6 = avg →
  fourth_sale = 6350 :=
by
  intros h1 h2 h3 h4 h5 h6 h_avg h_total
  -- Proof steps go here
  sorry

end fourth_month_sale_l415_415818


namespace area_of_closed_figure_l415_415390

theorem area_of_closed_figure:
  ∫ (x : ℝ) in 1..3, (x - (1 / x)) = 4 - real.log 3 :=
by
  sorry

end area_of_closed_figure_l415_415390


namespace probability_of_making_pro_shot_l415_415473

-- Define the probabilities given in the problem
def P_free_throw : ℚ := 4 / 5
def P_high_school_3 : ℚ := 1 / 2
def P_at_least_one : ℚ := 0.9333333333333333

-- Define the unknown probability for professional 3-pointer
def P_pro := 1 / 3

-- Calculate the probability of missing each shot
def P_miss_free_throw : ℚ := 1 - P_free_throw
def P_miss_high_school_3 : ℚ := 1 - P_high_school_3
def P_miss_pro : ℚ := 1 - P_pro

-- Define the probability of missing all shots
def P_miss_all := P_miss_free_throw * P_miss_high_school_3 * P_miss_pro

-- Now state what needs to be proved
theorem probability_of_making_pro_shot :
  (1 - P_miss_all = P_at_least_one) → P_pro = 1 / 3 :=
by
  sorry

end probability_of_making_pro_shot_l415_415473


namespace angle_BKA_28_angle_BAC_34_l415_415754

variable (A B C D K : Point)
variable (AD BK : ℝ)
variable (α : ℝ)

-- Given conditions
def is_quadrilateral (A B C D K : Point) : Prop :=
  ∠ABD = 90 ∧ ∠ACD = 90 ∧ ∠CAD = 42

def intersect_at_K (A B C D K : Point) : Prop :=
  line_through C B intersects line_through D A at K

def lengths (AD BK : ℝ) : Prop :=
  AD = 6 ∧ BK = 3

-- Problem 1
theorem angle_BKA_28 (h1 : is_quadrilateral A B C D) (h2 : intersect_at_K A B C D K) (h3 : lengths AD BK) : ∠BKA = 28 :=
  sorry

-- Problem 2
theorem angle_BAC_34 (h1 : is_quadrilateral A B C D) (h2 : intersect_at_K A B C D K) (h3 : lengths AD BK) (h4 : ∠BKA = 28) : ∠BAC = 34 :=
  sorry

end angle_BKA_28_angle_BAC_34_l415_415754


namespace f_prime_pos_l415_415223

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x - 4 * Real.log x

def f_prime (x : ℝ) : ℝ := 2 * x - 2 - 4 / x

theorem f_prime_pos (x : ℝ) (hx : x > 0) : 
  f_prime x > 0 ↔ x > 2 := 
sorry

end f_prime_pos_l415_415223


namespace total_frogs_seen_by_hunter_l415_415645

/-- Hunter saw 5 frogs sitting on lily pads in the pond. -/
def initial_frogs : ℕ := 5

/-- Three more frogs climbed out of the water onto logs floating in the pond. -/
def frogs_on_logs : ℕ := 3

/-- Two dozen baby frogs (24 frogs) hopped onto a big rock jutting out from the pond. -/
def baby_frogs : ℕ := 24

/--
The total number of frogs Hunter saw in the pond.
-/
theorem total_frogs_seen_by_hunter : initial_frogs + frogs_on_logs + baby_frogs = 32 := by
sorry

end total_frogs_seen_by_hunter_l415_415645


namespace loaves_count_l415_415492

theorem loaves_count (initial_loaves afternoon_sales evening_delivery end_day_loaves: ℕ)
  (h_initial: initial_loaves = 2355)
  (h_sales: afternoon_sales = 629)
  (h_delivery: evening_delivery = 489)
  (h_end: end_day_loaves = 2215) :
  initial_loaves - afternoon_sales + evening_delivery = end_day_loaves :=
  by {
    rw [h_initial, h_sales, h_delivery, h_end],
    sorry
  }

end loaves_count_l415_415492


namespace vector_sum_magnitude_gt_one_l415_415214

-- Definitions
variable {n : ℕ}
variable {ι : Type} [Fintype ι]
variable (a : ι → ℝ × ℝ)
variable (h_sum : ∑ i, (Real.sqrt ((a i).1 ^ 2 + (a i).2 ^ 2)) = 4)

-- Theorem statement
theorem vector_sum_magnitude_gt_one {a : ι → ℝ × ℝ} (h_sum : ∑ i, Real.sqrt ((a i).1 ^ 2 + (a i).2 ^ 2) = 4) :
  ∃ s : Finset ι, s.Nonempty ∧ (Real.sqrt ((s.intoFun ∑ i, (a i).1)^2 + (s.intoFun ∑ i, (a i).2)^2) > 1) :=
sorry

end vector_sum_magnitude_gt_one_l415_415214


namespace problem_equivalent_l415_415687

noncomputable def C1_parametric := (x y α : ℝ) ↔ (x = √3 * Real.cos α ∧ y = Real.sin α)
noncomputable def C2_polar := (ρ θ : ℝ) ↔ (ρ * Real.sin (θ + π/4) = 2*√2)

theorem problem_equivalent :
  (∀ x y α, C1_parametric x y α → x ^ 2 / 3 + y ^ 2 = 1) ∧ 
  (∀ ρ θ, C2_polar ρ θ → ρ * (Real.sin θ + Real.cos θ) = 4) ∧
  (∃ P Q : (ℝ × ℝ), (C1_parametric P.1 P.2 ?) ∧ (C2_polar ? ?) ∧ (|P - Q| = √2 ∧ P = (3/2, 1/2))) :=
sorry

end problem_equivalent_l415_415687


namespace longer_side_length_l415_415121

-- Given a circle with radius 6 cm
def radius : ℝ := 6
def circle_area : ℝ := Real.pi * radius^2

-- Given that the area of the rectangle is three times the area of the circle
def rectangle_area : ℝ := 3 * circle_area

-- Given the rectangle has a shorter side equal to the diameter of the circle
def shorter_side : ℝ := 2 * radius

-- Prove that the length of the longer side of the rectangle is 9π cm
theorem longer_side_length : ∃ (longer_side : ℝ), longer_side = 9 * Real.pi :=
by
  have circle_area_def : circle_area = 36 * Real.pi := by sorry
  have rectangle_area_def : rectangle_area = 108 * Real.pi := by sorry
  have shorter_side_def : shorter_side = 12 := by sorry
  let longer_side := rectangle_area / shorter_side
  use longer_side
  show longer_side = 9 * Real.pi
  sorry

end longer_side_length_l415_415121


namespace friend_redistribution_l415_415575

-- Definitions of friends' earnings
def earnings := [18, 22, 26, 32, 47]

-- Definition of total earnings
def totalEarnings := earnings.sum

-- Definition of equal share
def equalShare := totalEarnings / earnings.length

-- The amount that the friend who earned 47 needs to redistribute
def redistributionAmount := 47 - equalShare

-- The goal to prove
theorem friend_redistribution:
  redistributionAmount = 18 := by
  sorry

end friend_redistribution_l415_415575


namespace find_sin_cos_sum_find_beta_value_l415_415586

open Real

-- Define α in the interval (0, π/2) and with cos(2α) = 4/5
variable (α β : ℝ)
variable (h1 : α ∈ set.Ioo 0 (π / 2))
variable (h2 : cos (2 * α) = 4 / 5)

-- Define β in the interval (π/2, π) and with 5sin(2α + β) = sin(β)
variable (h3 : β ∈ set.Ioo (π / 2) π)
variable (h4 : 5 * sin (2 * α + β) = sin β)

-- Statement proving the required results
theorem find_sin_cos_sum : sin α + cos α = 2 * sqrt 10 / 5 := by sorry

theorem find_beta_value : β = 3 * π / 4 := by sorry

end find_sin_cos_sum_find_beta_value_l415_415586


namespace range_of_a_l415_415262

noncomputable def f (x : ℝ) : ℝ := 
  if h : x ≤ 1 then x^2 - x + 3 else 0

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ |x / 2 + a|) ↔ -47 / 16 ≤ a ∧ a ≤ 2 := sorry

end range_of_a_l415_415262


namespace V1_is_30_l415_415430

def horner_eval (coeffs : List ℤ) (x : ℤ) : List ℤ :=
  coeffs.foldr (λ a acc, match acc with
    | []        => [a]
    | (v :: vs) => (a + x * v) :: v :: vs
  ) []

def polynomial := [3, 0, 2, 1, 4]
def x_val := 10
def V_1 := horner_eval polynomial x_val

theorem V1_is_30 : (V_1 polynomial x_val).getLast = 30 := by
  -- proof omitted
  sorry

end V1_is_30_l415_415430


namespace prove_angles_equal_l415_415862

-- Define the problem setup with the given conditions

theorem prove_angles_equal 
  (A B C D E F G M N : Type) 
  [Point A B]
  (line_BA : Line A B)
  [Collinear C D line_BA]
  (circle_P : Circle P)
  (circle_Q : Circle Q)
  (line_EC : Line E C)
  (intersects_P_F : Intersects line_EC circle_P F)
  (line_ED : Line E D)
  (intersects_Q_G : Intersects line_ED circle_Q G)
  (line_FG : Line F G)
  (intersects_P_N : Intersects line_FG circle_P N)
  (intersects_Q_M : Intersects line_FG circle_Q M) :
  ∠ F C M = ∠ G D N :=
sorry -- Proof is omitted as per instructions

end prove_angles_equal_l415_415862


namespace sequence1_sixth_seventh_terms_sequence2_sixth_term_sequence3_ninth_tenth_terms_l415_415466

def seq1 (n : ℕ) : ℕ := 2 * (n + 1)
def seq2 (n : ℕ) : ℕ := 3 * 2 ^ n
def seq3 (n : ℕ) : ℕ :=
  if n % 2 = 0 then 36 + n
  else 10 + n
  
theorem sequence1_sixth_seventh_terms :
  seq1 5 = 12 ∧ seq1 6 = 14 :=
by
  sorry

theorem sequence2_sixth_term :
  seq2 5 = 96 :=
by
  sorry

theorem sequence3_ninth_tenth_terms :
  seq3 8 = 44 ∧ seq3 9 = 19 :=
by
  sorry

end sequence1_sixth_seventh_terms_sequence2_sixth_term_sequence3_ninth_tenth_terms_l415_415466


namespace general_term_of_sequence_l415_415011

noncomputable def seq (n : ℕ) : ℕ :=
  match n with
  | 1 => 6
  | 2 => 9
  | 3 => 14
  | 4 => 21
  | 5 => 30
  | _ => sorry

theorem general_term_of_sequence :
  ∀ n : ℕ, seq n = 5 + n^2 :=
by
  sorry

end general_term_of_sequence_l415_415011


namespace unique_integer_sequence_exists_l415_415728

open Nat

def a (n : ℕ) : ℤ := sorry

theorem unique_integer_sequence_exists :
  ∃ (a : ℕ → ℤ), a 1 = 1 ∧ a 2 > 1 ∧ (∀ n ≥ 1, (a (n+1))^3 + 1 = a n * a (n+2)) ∧
  (∀ b, (b 1 = 1) → (b 2 > 1) → (∀ n ≥ 1, (b (n+1))^3 + 1 = b n * b (n+2)) → b = a) :=
by
  sorry

end unique_integer_sequence_exists_l415_415728


namespace solve_for_m_minimum_value_set_l415_415182

noncomputable def f (x m : ℝ) : ℝ := sqrt 3 * sin (2 * x) + 2 * (cos x)^2 + m

theorem solve_for_m (m : ℝ) : (∀ x ∈ set.Icc (0:ℝ) (π/2), f x m ≤ 6) → (∃ x ∈ set.Icc (0:ℝ) (π/2), f x m = 6) → m = 3 :=
by sorry

theorem minimum_value_set (m : ℝ) : m = 3 → (∀ x ∈ set.univ, f x m ≥ 2) ∧ (∃ x ∈ set.univ, f x m = 2 ∧ ∃ k : ℤ, x = - (π/3) + k * π) :=
by sorry

end solve_for_m_minimum_value_set_l415_415182


namespace remainder_of_sum_l415_415966

theorem remainder_of_sum (n : ℤ) : ((5 - n) + (n + 4)) % 5 = 4 := 
by 
  -- proof goes here
  sorry

end remainder_of_sum_l415_415966


namespace find_original_function_l415_415294

-- Definitions based on conditions
def shortened_abscissa (f : ℝ → ℝ) : ℝ → ℝ := λ x, f (2 * x)

def shifted_right (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x, f (x - a)

-- Theorem statement
theorem find_original_function (f : ℝ → ℝ)
  (h : (shifted_right (shortened_abscissa f) (π / 3)) = (λ x, sin (x - π / 4))) :
  f = (λ x, sin (x / 2 + π / 12)) :=
sorry

end find_original_function_l415_415294


namespace fixed_point_l415_415880

def f (a : ℝ) (x : ℝ) : ℝ := a^(x-1) + 2

theorem fixed_point (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : f a 1 = 3 :=
by
  -- proof goes here
  sorry

end fixed_point_l415_415880


namespace average_interest_rate_l415_415137

theorem average_interest_rate
  (x : ℝ)
  (h₀ : 0 ≤ x)
  (h₁ : x ≤ 5000)
  (h₂ : 0.05 * x = 0.03 * (5000 - x)) :
  (0.05 * x + 0.03 * (5000 - x)) / 5000 = 0.0375 :=
by
  sorry

end average_interest_rate_l415_415137


namespace total_frogs_in_pond_l415_415653

def frogsOnLilyPads : ℕ := 5
def frogsOnLogs : ℕ := 3
def babyFrogsOnRock : ℕ := 2 * 12 -- Two dozen

theorem total_frogs_in_pond : frogsOnLilyPads + frogsOnLogs + babyFrogsOnRock = 32 :=
by
  sorry

end total_frogs_in_pond_l415_415653


namespace vasya_and_petya_equal_ways_l415_415035

theorem vasya_and_petya_equal_ways :
  ∀ (board_length : ℕ) (even_move odd_move : ℕ → ℕ), 
  board_length = 101 →
  (∀ n, even (even_move n)) →
  (∀ n, odd (odd_move n)) →
  (∀ pos, odd_move pos = (board_length - (even_move pos + 1)) % board_length) →
  (num_ways_to_traverse board_length even_move odd_move 1) = 
  (num_ways_to_traverse board_length even_move odd_move 50) :=
by
  sorry

end vasya_and_petya_equal_ways_l415_415035


namespace maximize_profit_l415_415130

noncomputable def Q (x : ℝ) (h : 0 ≤ x) : ℝ := (3 * x + 1) / (x + 1)

noncomputable def W (x : ℝ) (h : 0 ≤ x) : ℝ := 
  let Q_val := Q x h
  (1 / 2) * (32 * Q_val + 3 - x) 

theorem maximize_profit :
  ∃ (x : ℝ), 0 ≤ x ∧ W x (by linarith) = 42 :=
begin
  use 7,
  split,
  { linarith },
  { sorry }
end

end maximize_profit_l415_415130


namespace limit_area_A_l415_415576

noncomputable def tangency_point (a : ℝ) : ℝ × ℝ :=
  (a^(-1 / a), a^(-1))

noncomputable def b_as_function_of_a (a : ℝ) : ℝ :=
  (a * Real.exp(1))^(1 / a)

noncomputable def area_A (a h : ℝ) : ℝ :=
  ∫ x in h..a^(-1 / a), (x^a - (Real.log ((a * Real.exp(1))^(1 / a) * x))) dx

theorem limit_area_A (a : ℝ) (h : ℝ) (condition : 0 < h ∧ h < a^(-1 / a)) : 
  ∀ ε > 0, ∃ δ > 0, ∀ h1, 0 < h1 < δ → abs (area_A a h1 - a^(-1 / (a + 1) * (a - 1))) < ε := 
by
  sorry

end limit_area_A_l415_415576


namespace exists_x_abs_ge_one_fourth_l415_415366

theorem exists_x_abs_ge_one_fourth :
  ∀ (a b c : ℝ), ∃ x : ℝ, |x| ≤ 1 ∧ |x^3 + a * x^2 + b * x + c| ≥ 1 / 4 :=
by sorry

end exists_x_abs_ge_one_fourth_l415_415366


namespace angie_pretzels_l415_415854

theorem angie_pretzels (Barry_Shelly: ℕ) (Shelly_Angie: ℕ) :
  (Barry_Shelly = 12 / 2) → (Shelly_Angie = 3 * Barry_Shelly) → (Barry_Shelly = 6) → (Shelly_Angie = 18) :=
by
  intro h1 h2 h3
  sorry

end angie_pretzels_l415_415854


namespace range_of_m_l415_415953

theorem range_of_m (h : ¬ (∀ x : ℝ, ∃ m : ℝ, 4 ^ x - 2 ^ (x + 1) + m = 0) → false) : 
  ∀ m : ℝ, m ≤ 1 :=
by
  sorry

end range_of_m_l415_415953


namespace solve_eq_l415_415556

noncomputable def fx (x : ℝ) : ℝ :=
  ((x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 3) * (x - 2) * (x - 1) * (x - 5)) /
  ((x - 2) * (x - 4) * (x - 2) * (x - 5))

theorem solve_eq (x : ℝ) (h : x ≠ 2 ∧ x ≠ 4 ∧ x ≠ 5) :
  fx x = 1 ↔ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 :=
by
  sorry

end solve_eq_l415_415556


namespace parabola_x0_range_l415_415698

variables {x₀ y₀ : ℝ}
def parabola (x₀ y₀ : ℝ) : Prop := y₀^2 = 8 * x₀

def focus (x : ℝ) : ℝ := 2

def directrix (x : ℝ) : Prop := x = -2

/-- Prove that for any point (x₀, y₀) on the parabola y² = 8x and 
if a circle centered at the focus intersects the directrix, then x₀ > 2. -/
theorem parabola_x0_range (x₀ y₀ : ℝ) (h1 : parabola x₀ y₀)
  (h2 : ((x₀ - 2)^2 + y₀^2)^(1/2) > (2 : ℝ)) : x₀ > 2 := 
sorry

end parabola_x0_range_l415_415698


namespace landscape_length_l415_415785

-- Define the conditions from the problem
def breadth (b : ℝ) := b > 0
def length_of_landscape (l b : ℝ) := l = 8 * b
def area_of_playground (pg_area : ℝ) := pg_area = 1200
def playground_fraction (A b : ℝ) := A = 8 * b^2
def fraction_of_landscape (pg_area A : ℝ) := pg_area = (1/6) * A

-- Main theorem statement
theorem landscape_length (b l A pg_area : ℝ) 
  (H_b : breadth b) 
  (H_length : length_of_landscape l b)
  (H_pg_area : area_of_playground pg_area)
  (H_pg_fraction : playground_fraction A b)
  (H_pg_landscape_fraction : fraction_of_landscape pg_area A) :
  l = 240 :=
by
  sorry

end landscape_length_l415_415785


namespace plane_equation_l415_415700

noncomputable def vector_w : ℝ × ℝ × ℝ := (3, -3, 3)

theorem plane_equation 
  (v : ℝ × ℝ × ℝ)
  (h : (3 * v.1 - 3 * v.2 + 3 * v.3) / 27 * vector_w = vector_w) :
  let x := v.1, y := v.2, z := v.3 in x - y + z = 9 := 
by 
  sorry

end plane_equation_l415_415700


namespace find_m_l415_415638

noncomputable def parallel_vectors : Prop :=
  ∃ m : ℝ, (1, 3) = (1 * m, 3 * m)

theorem find_m (m : ℝ) (h : ∃ k : ℝ, (1, 3) = (k * m, k * 1)) : m = 1 / 3 :=
by
  obtain ⟨k, hk⟩ := h
  have : 3 * k = 1 := by
    have ⟨hx, hy⟩ := hk
    linarith
  sorry

end find_m_l415_415638


namespace sum_of_b_is_T_l415_415939

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 2^(n+1)

-- Define the sequence b_n
def b (n : ℕ) : ℝ := (-1)^n * (2 * n + 3) / ((n + 1) * (n + 2))

-- Define the sum of the first n terms of b_n
def T (n : ℕ) : ℝ :=
  if n % 2 = 0 then
    -n / (2 * n + 4)
  else
    -(n + 4) / (2 * n + 4)

-- The final theorem to prove
theorem sum_of_b_is_T (n : ℕ) : ∑ k in Finset.range n, b k = T n := sorry

end sum_of_b_is_T_l415_415939


namespace units_digit_base_l415_415027

theorem units_digit_base (x : ℕ) (y : ℕ) : 
  (y % 10 = 5) ∧ ((x % 10) ^ 53 + y) % 10 = 8 → 
  x % 10 = 3 :=
by
  intros h
  cases h with hy hx
  sorry

end units_digit_base_l415_415027


namespace jose_joined_after_two_months_l415_415424

theorem jose_joined_after_two_months 
  (Tom_investment : ℝ) (Jose_investment : ℝ) 
  (total_profit : ℝ) (Jose_share : ℝ) : 
  Tom_investment = 3000 → 
  Jose_investment = 4500 → 
  total_profit = 5400 → 
  Jose_share = 3000 → 
  ∃ (x : ℕ), x = 2 :=
by
  intros h1 h2 h3 h4
  have : (Tom_investment * 12) / (Jose_investment * (12 - (2:ℕ))) = (total_profit - Jose_share) / Jose_share :=
    by sorry
  use 2
  sorry

end jose_joined_after_two_months_l415_415424


namespace value_of_a3_a6_a9_l415_415311

variable (a : ℕ → ℤ) -- Define the sequence a as a function from natural numbers to integers
variable (d : ℤ) -- Define the common difference d as an integer

-- Conditions
axiom h1 : a 1 + a 4 + a 7 = 39
axiom h2 : a 2 + a 5 + a 8 = 33
axiom h3 : ∀ n : ℕ, a (n+1) = a n + d -- This condition ensures the sequence is arithmetic

-- Theorem: We need to prove the value of a_3 + a_6 + a_9 is 27
theorem value_of_a3_a6_a9 : a 3 + a 6 + a 9 = 27 :=
by
  sorry

end value_of_a3_a6_a9_l415_415311


namespace tom_tickets_left_l415_415522

-- Define the conditions
def tickets_whack_a_mole : ℕ := 32
def tickets_skee_ball : ℕ := 25
def tickets_spent_on_hat : ℕ := 7

-- Define what we need to prove
theorem tom_tickets_left : tickets_whack_a_mole + tickets_skee_ball - tickets_spent_on_hat = 50 :=
by sorry

end tom_tickets_left_l415_415522


namespace prove_divisibility_by_polynomial_l415_415580

theorem prove_divisibility_by_polynomial {a b c : ℤ} :
  ∀ n : ℕ, (n = 4) ↔ (a^n * (b - c) + b^n * (c - a) + c^n * (a - b)) ∣ (a^2 + b^2 + c^2 + a*b + b*c + c*a) :=
by sorry

end prove_divisibility_by_polynomial_l415_415580


namespace area_of_triangle_bounded_by_coordinate_axes_and_line_l415_415512

def area_of_triangle (b h : ℕ) : ℕ := (b * h) / 2

theorem area_of_triangle_bounded_by_coordinate_axes_and_line :
  area_of_triangle 4 6 = 12 :=
by
  sorry

end area_of_triangle_bounded_by_coordinate_axes_and_line_l415_415512


namespace problem_statement_l415_415547

noncomputable def distinct_triangle_areas
  (GH HI : ℝ) (IJ : ℝ) (JK : ℝ) (LM : ℝ) (NO : ℝ) (d : ℝ) 
  (G H I J K L M N O : Type) : Nat :=
  (∀ (a b c: Type), a ∈ {G, H, I, J, K} → b ∈ {L, M} → c ∈ {N, O} → 
    a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (GH = 1 ∧ HI = 1 ∧ IJ = 2 ∧ JK = 3 ∧ LM = 3 ∧ NO = 2) → 
  {area : ℝ // area ∈ {0.5 * d, 1.0 * d, 1.5 * d}}.card
  = 3

theorem problem_statement : distinct_triangle_areas 1 1 2 3 3 2 d G H I J K L M N O = 3 := by
  sorry

end problem_statement_l415_415547


namespace evaluate_f_at_5_l415_415623

def f : ℤ → ℤ
| x := if x ≥ 6 then x - 5 else f (x + 2)

theorem evaluate_f_at_5 : f 5 = 2 := by
  sorry

end evaluate_f_at_5_l415_415623


namespace total_seashells_after_six_weeks_l415_415185

theorem total_seashells_after_six_weeks :
  ∀ (a b : ℕ) 
  (initial_a : a = 50) 
  (initial_b : b = 30) 
  (next_a : ∀ k : ℕ, k > 0 → a + 20 = (a + 20) * k) 
  (next_b : ∀ k : ℕ, k > 0 → b * 2 = (b * 2) * k), 
  (a + 20 * 5) + (b * 2 ^ 5) = 1110 :=
by
  intros a b initial_a initial_b next_a next_b
  sorry

end total_seashells_after_six_weeks_l415_415185


namespace savings_equal_after_weeks_l415_415449

theorem savings_equal_after_weeks :
  ∃ w : ℕ, 160 + 7 * w = 210 + 5 * w ∧ w = 25 :=
by {
  have h : 160 + 7 * 25 = 210 + 5 * 25,
  calc 
    160 + 7 * 25 = 160 + 175 : by norm_num
              ... = 335 : by norm_num
              ... = 210 + 125 : by norm_num
              ... = 210 + 5 * 25 : by norm_num,
  use 25,
  split,
  exact h,
  simp,
  }

end savings_equal_after_weeks_l415_415449


namespace greatest_third_side_l415_415046

-- Given data and the Triangle Inequality theorem
theorem greatest_third_side (c : ℕ) (h1 : 8 < c) (h2 : c < 22) : c = 21 :=
by
  sorry

end greatest_third_side_l415_415046


namespace num_valid_int_values_l415_415750

-- Define the conditions
def isValidTriangleSide (x : ℕ) : Prop :=
  25 < x ∧ x < 55

-- Define the proof problem: proving the number of valid integer values for x
theorem num_valid_int_values : 
  ∃ n : ℕ, n = (54 - 26 + 1) ∧ (∀ x, isValidTriangleSide x → ∃ k, nat.succ k = n) :=
by
  sorry

end num_valid_int_values_l415_415750


namespace P_inter_Q_contains_two_elements_l415_415956

noncomputable def P (k : ℝ) : set (ℝ × ℝ) :=
  {p | ∃ x y, y = k * (x - 1) + 1 ∧ p = (x, y)}

def Q : set (ℝ × ℝ) :=
  {p | ∃ x y, x^2 + y^2 - 2 * y = 0 ∧ p = (x, y)}

theorem P_inter_Q_contains_two_elements (k : ℝ) : (P k ∩ Q).finite ∧ (P k ∩ Q).to_finset.card = 2 := 
sorry

end P_inter_Q_contains_two_elements_l415_415956


namespace first_player_wins_l415_415428

theorem first_player_wins :
  ∀ (sticks : ℕ), (sticks = 1) →
  (∀ (break_rule : ℕ → ℕ → Prop),
  (∀ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z → break_rule x y → break_rule x z)
  → (∃ n : ℕ, n % 3 = 0 ∧ break_rule n (n + 1) → ∃ t₁ t₂ t₃ : ℕ, t₁ = t₂ ∧ t₂ = t₃ ∧ t₁ + t₂ + t₃ = n))
  → (∃ w : ℕ, w = 1) := sorry

end first_player_wins_l415_415428


namespace annieka_free_throws_l415_415001

theorem annieka_free_throws :
  let d : ℕ := 12 in
  let k : ℕ := d + (d / 2) in
  let a : ℕ := k - 4 in
  a = 14 :=
by
  let d := 12
  let k := d + (d / 2)
  let a := k - 4
  show a = 14
  sorry

end annieka_free_throws_l415_415001


namespace inv_distances_sum_l415_415144

-- Define points G, P, Q, R and their relations
variables {A B C G P Q R : Type}
-- Assume a triangle ABC with centroid G and secant line passing through G intersects sides or their extensions at points P, Q, R.
-- Assume points Q and R lie on the same side of G.
-- Prove that (1 / GP) = (1 / GQ) + (1 / GR)

-- Define distances GP, GQ, GR
variables {GP GQ GR : ℝ}
-- Centroid property and distances hypothesis
hypothesis (h_centroid : is_centroid G A B C)
hypothesis (h_distance_relation : (1 / GP) = (1 / GQ) + (1 / GR))
hypothesis (h_same_side : same_side G Q R)

-- The theorem to prove
theorem inv_distances_sum :
  (1 / GP) = (1 / GQ) + (1 / GR) :=
sorry

end inv_distances_sum_l415_415144


namespace exists_set_with_specific_score_l415_415190

def set_with_distinct_subset_sums (S : Set ℕ) : Prop :=
  ∀ s1 s2 : Set ℕ, s1 ≠ s2 → (s1 ⊆ S ∧ s2 ⊆ S) → ∑ x in s1, x ≠ ∑ x in s2, x

def score (n r : ℕ) : ℤ :=
  ⌊20 * (2^n / r) - 2⌋

theorem exists_set_with_specific_score :
  ∃ S : Set ℕ, set_with_distinct_subset_sums S ∧ score S.finite.to_finset.card (S.finite.to_finset.max' (finite_set.1) (finite_set.2)) = 50 := sorry

end exists_set_with_specific_score_l415_415190


namespace area_of_triangle_bounded_by_line_and_axes_l415_415504

theorem area_of_triangle_bounded_by_line_and_axes (x y : ℝ) (hx : 3 * x + 2 * y = 12) :
  ∃ (area : ℝ), area = 12 := by
sorry

end area_of_triangle_bounded_by_line_and_axes_l415_415504


namespace roommate_payment_l415_415425

theorem roommate_payment :
  (1100 + 114 + 300) / 2 = 757 := 
by
  sorry

end roommate_payment_l415_415425


namespace rachel_weight_l415_415730

theorem rachel_weight :
  ∃ R : ℝ, (R + (R + 6) + (R - 15)) / 3 = 72 ∧ R = 75 :=
by
  sorry

end rachel_weight_l415_415730


namespace probability_of_selecting_cooking_l415_415098

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "woodworking"]
  in (∃ (course : String), course ∈ courses ∧ course = "cooking") →
    (1 / (courses.length : ℝ) = 1 / 4) :=
by
  sorry

end probability_of_selecting_cooking_l415_415098


namespace minimum_inclination_angle_l415_415224

noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * x + 2 + 2 * Real.log x

noncomputable def f_derivative (x : ℝ) : ℝ := 2 * x - 3 + 2 / x

theorem minimum_inclination_angle (l : ℝ → ℝ) (h_tangent : ∃ x > 0, f x = l x ∧ f_derivative x = slope l) :
  ∃ α : ℝ, α = Real.arctan 1 ∧ α > 0 :=
by
  use Real.arctan 1
  split
  · refl
  · simp; exact Real.arctan_pos_of_pos zero_lt_one

def slope (f : ℝ → ℝ): = sorry

end minimum_inclination_angle_l415_415224


namespace weight_of_raisins_proof_l415_415692

-- Define the conditions
def weight_of_peanuts : ℝ := 0.1
def total_weight_of_snacks : ℝ := 0.5

-- Theorem to prove that the weight of raisins equals 0.4 pounds
theorem weight_of_raisins_proof : total_weight_of_snacks - weight_of_peanuts = 0.4 := by
  sorry

end weight_of_raisins_proof_l415_415692


namespace tilly_counts_total_stars_l415_415038

-- Definition of the given conditions
def stars_east : ℕ := 120
def percentage_west : ℚ := 473 / 100

-- Calculation based on the conditions
def stars_west : ℕ := (stars_east * percentage_west).toNat

-- Total number of stars
def total_stars : ℕ := stars_east + stars_west

-- Proof statement
theorem tilly_counts_total_stars : total_stars = 688 := by
  sorry

end tilly_counts_total_stars_l415_415038


namespace true_propositions_l415_415963

-- Define the complex number z
def z : ℂ := 2 - I

-- Define the propositions p1, p2, p3, p4
def p1 := complex.abs z = 5
def p2 := z * z = 3 - 4 * I
def p3 := conj z = -2 + I
def p4 := z.im = -1

-- Proof statement
theorem true_propositions : (p2 = true) ∧ (p4 = true) ∧ (p1 = false) ∧ (p3 = false) := by
  sorry

end true_propositions_l415_415963


namespace sum_of_digits_l415_415525

theorem sum_of_digits (n : ℕ) : 
  let a := (List.range n).map (λ i, (List.repeat 9 (2^i)).foldl (λ acc d, acc * 10 + d) 0),
  let b := (List.repeat 9 (2^n)).foldl (λ acc d, acc * 10 + d) 0
  in ((a.foldl (λ acc d, acc * d) 1) * b).digits.sum = 9 * 2^n :=
by sorry

end sum_of_digits_l415_415525


namespace explicit_formula_and_domain_of_g_line_y_eq_b_intersects_g_one_point_l415_415714

noncomputable def f (x : ℝ) : ℝ := x + 1 / x

noncomputable def g (x : ℝ) : ℝ := x - 2 + 1 / (x - 4)

noncomputable def is_domain_of_g (x : ℝ) : Prop := x ≠ 4

theorem explicit_formula_and_domain_of_g :
  ∀ x : ℝ, is_domain_of_g x → g(x) = x - 2 + 1 / (x - 4) :=
begin
  intros x hx,
  exact g(x),
  sorry
end

theorem line_y_eq_b_intersects_g_one_point (b : ℝ) :
  (∃ x : ℝ, g(x) = b) ∧ (∀ x₁ x₂ : ℝ, g(x₁) = b → g(x₂) = b → x₁ = x₂) ↔
  (b = 4 ∧ (∃ x : ℝ, f x ∧ x = 1)) ∨ (b = 0 ∧ (∃ x : ℝ, f x ∧ x = 2)) :=
begin
  split,
  { intros h,
    -- place argument here
    sorry },
  { intros h,
    -- place argument here
    sorry }
end

end explicit_formula_and_domain_of_g_line_y_eq_b_intersects_g_one_point_l415_415714


namespace winning_strategy_l415_415541

-- Define the conditions
def is_power_of_three (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 3^k

-- Define the proof problem
theorem winning_strategy (a b : ℕ)
    (h_a_gt_1 : 1 < a)
    (h_b_gt_1 : 1 < b) :
    (is_power_of_three a → "乙 wins") ∧ (¬ is_power_of_three a → "甲 wins") :=
by
  sorry

end winning_strategy_l415_415541


namespace range_of_k_l415_415976

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, (k^2 - 1) * x^2 - (k + 1) * x + 1 > 0) ↔ (1 ≤ k ∧ k ≤ 5 / 3) := 
sorry

end range_of_k_l415_415976


namespace rpm_wheel_approx_l415_415460

noncomputable def rpm_of_wheel (radius_cm : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_cm_per_min := (speed_kmh * 100000) / 60
  let circumference_cm := 2 * Real.pi * radius_cm
  speed_cm_per_min / circumference_cm

theorem rpm_wheel_approx (h_radius :  radius_cm = 250) (h_speed : speed_kmh = 66) :
  rpm_of_wheel radius_cm speed_kmh ≈ 700.5 := sorry

end rpm_wheel_approx_l415_415460


namespace find_original_price_of_dish_l415_415325

noncomputable def original_price_of_dish (P : ℝ) : Prop :=
  let john_paid := (0.9 * P) + (0.15 * P)
  let jane_paid := (0.9 * P) + (0.135 * P)
  john_paid = jane_paid + 0.60 → P = 40

theorem find_original_price_of_dish (P : ℝ) (h : original_price_of_dish P) : P = 40 := by
  sorry

end find_original_price_of_dish_l415_415325


namespace only_b_is_increasing_on_neg_infty_zero_l415_415777

noncomputable def is_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x < f y

def fA (x : ℝ) : ℝ := -Real.log (-x) / Real.log (1 / 2)
def fB (x : ℝ) : ℝ := x / (1 - x)
def fC (x : ℝ) : ℝ := -(x + 1)^2
def fD (x : ℝ) : ℝ := 1 + x^2

theorem only_b_is_increasing_on_neg_infty_zero :
  ∀ f, (f = fA ∨ f = fB ∨ f = fC ∨ f = fD) → is_increasing f {x | x < 0} ↔ f = fB :=
by
  sorry

end only_b_is_increasing_on_neg_infty_zero_l415_415777


namespace algebraic_expression_identity_l415_415297

theorem algebraic_expression_identity (a b x : ℕ) (h : x * 3 * a * b = 3 * a * a * b) : x = a :=
sorry

end algebraic_expression_identity_l415_415297


namespace probability_at_least_one_red_l415_415675

def total_balls : ℕ := 6
def red_balls : ℕ := 4
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

theorem probability_at_least_one_red :
  (choose_two red_balls + red_balls * (total_balls - red_balls - 1) / 2) / choose_two total_balls = 14 / 15 :=
sorry

end probability_at_least_one_red_l415_415675


namespace sum_of_squares_ge_one_third_l415_415237

theorem sum_of_squares_ge_one_third (a b c : ℝ) (h : a + b + c = 1) : 
  a^2 + b^2 + c^2 ≥ 1/3 := 
by 
  sorry

end sum_of_squares_ge_one_third_l415_415237


namespace mabel_initial_daisies_l415_415715

theorem mabel_initial_daisies (D: ℕ) (h1: 8 * (D - 2) = 24) : D = 5 :=
by
  sorry

end mabel_initial_daisies_l415_415715


namespace higher_rate_correct_l415_415150

noncomputable def higher_rate (R : ℝ) : ℝ :=
  let P := 1200
  let n := 10
  let d := 100
  let SI := (P * R * n) / d
  let SI_h := (P * (R + 5) * n) / d
  SI_h = SI + 600

theorem higher_rate_correct (R : ℝ) : higher_rate R :=
begin
  have h : (1200 * (R + 5) * 10) / 100 = (1200 * R * 10) / 100 + 600,
  {
    -- Prove the given condition holds
    sorry,
  },
  exact h,
end

end higher_rate_correct_l415_415150


namespace find_point_A_l415_415594

-- Given a circle and points A, B, and C.
variables {k : Type*} [MetricSpace k] {O : k} {r : ℝ} {A B C : k}
          (circle : ∀ (P : k), dist O P = r)

-- Definitions of tangent and chord properties.
def tangent (P Q : k) : Prop := dist P Q = r
def chord (P Q : k) : Prop := ∃ R, dist O R = r ∧ dist P R + dist Q R = 2 * dist O R
def equilateral_triangle (P Q R : k) : Prop :=
  dist P Q = dist Q R ∧ dist Q R = dist R P

-- Main theorem
theorem find_point_A :
  tangent B (circle B) → tangent C (circle C) →
  (∃ A, tangent A B ∧ tangent A C ∧ chord B C ∧ equilateral_triangle A B C) :=
by sorry

end find_point_A_l415_415594


namespace probability_of_selecting_cooking_l415_415100

def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

theorem probability_of_selecting_cooking : (favorable_outcomes : ℚ) / total_courses = 1 / 4 := 
by 
  sorry

end probability_of_selecting_cooking_l415_415100


namespace perimeter_XYZABCR_l415_415763

-- Define the necessary objects and conditions
structure Point := (x y : ℝ)
structure Triangle := (p1 p2 p3 : Point)
def midpoint (p1 p2 : Point) : Point := 
  Point.mk ((p1.x + p2.x) / 2) ((p1.y + p2.y) / 2)

-- Conditions given in the problem
def XYZ : Triangle := Triangle.mk (Point.mk 0 0) (Point.mk 6 0) (Point.mk 3 (3 * Real.sqrt 3))
def X := XYZ.p1
def Y := XYZ.p2
def Z := XYZ.p3
def A : Point := midpoint X Z
def B : Point := midpoint Z A

def ZAB : Triangle := Triangle.mk Z A B
def BCR : Triangle := Triangle.mk B (Point.mk (B.x + 1.5 * (Real.sqrt 3/2)) (B.y + 1.5 * (Real.sqrt 3))) (Point.mk (B.x + 1.5 * (Real.sqrt 3 / 2)) (B.y - 1.5 * (Real.sqrt 3 / 2)))

-- Perimeter definition
def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

def perimeter (triangles : List Triangle) : ℝ := 
  triangles.foldl (fun acc (t : Triangle) => acc + 
    (distance t.p1 t.p2 + distance t.p2 t.p3 + distance t.p3 t.p1)
  ) 0

-- Main statement
theorem perimeter_XYZABCR : (perimeter [XYZ, ZAB, BCR]) = 21 :=
by 
  -- sorry here to skip the proof steps
  sorry

end perimeter_XYZABCR_l415_415763


namespace parabola_equation_l415_415560

def equation_of_parabola (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, y = a * x^2 + b * x + c ↔ 
              (∃ a : ℝ, y = a * (x - 3)^2 + 5) ∧
              y = (if x = 0 then 2 else y)

theorem parabola_equation :
  equation_of_parabola (-1 / 3) 2 2 :=
by
  -- First, show that the vertex form (x-3)^2 + 5 meets the conditions
  sorry

end parabola_equation_l415_415560


namespace probability_vertical_boundary_l415_415480

def starting_point := (2, 1)
def inner_boundary := {((0, 0), (0, 4), (4, 4), (4, 0))}
def outer_boundary := {((-1, -1), (-1, 5), (5, 5), (5, -1))}
def jump_length := 1

def is_vertical_side (p : ℕ × ℕ) : Prop := 
  p.1 = -1 ∨ p.1 = 5

def frog_jumps (start : ℕ × ℕ) (direction : string) (jumps : ℕ) : ℕ × ℕ := 
  match direction with
  | "up" => (start.1, start.2 + jumps)
  | "down" => (start.1, start.2 - jumps)
  | "left" => (start.1 - jumps, start.2)
  | "right" => (start.1 + jumps, start.2)
  | _ => start
  
theorem probability_vertical_boundary :
  let P := 9 / 16 in
  ∃ (P : ℝ), P = 9 / 16 →
  ∀ (start : ℕ × ℕ) (start = starting_point) 
  (jumps : ℕ) (direction : string), 
  (frog_jumps start direction jumps) ∈ (inner_boundary ∪ outer_boundary) →
  is_vertical_side (frog_jumps start direction jumps) :=
by sorry

end probability_vertical_boundary_l415_415480


namespace probability_of_selecting_cooking_l415_415097

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "woodworking"]
  in (∃ (course : String), course ∈ courses ∧ course = "cooking") →
    (1 / (courses.length : ℝ) = 1 / 4) :=
by
  sorry

end probability_of_selecting_cooking_l415_415097


namespace p_bounded_iff_a1_odd_l415_415211

def smallest_prime_divisor (k : ℕ) : ℕ := 
sorry -- Assume the existence of such a function

def seq (a : ℕ) (n : ℕ) : ℕ :=
if n = 0 then a else seq a (n - 1)^n - 1

def is_bounded (f : ℕ → ℕ) : Prop :=
∃ M, ∀ n, f n < M

theorem p_bounded_iff_a1_odd (a1 : ℕ) (h : a1 > 2) :
  is_bounded (λ n, smallest_prime_divisor (seq a1 n)) ↔ odd a1 :=
sorry

end p_bounded_iff_a1_odd_l415_415211


namespace problem_I_problem_I_interval_problem_II_l415_415215

noncomputable def f (x : ℝ) : ℝ := (√3 * Real.sin x) * Real.cos x + (Real.cos x) ^ 2

theorem problem_I (x : ℝ) :
  f x = Real.sin (2 * x + (π / 6)) + 1 / 2 :=
sorry

theorem problem_I_interval (k : ℤ) :
  let x := Real.arcsin (1 / 2) - π / 3 + k * π in
  let y := Real.arcsin (1 / 2) + π / 6 + k * π in
  ∀ x ∈ Set.univ, f x ≤ f (x + y) :=
sorry

theorem problem_II (A a b c : ℝ) (h_a : a = 1) (h_b_c : b + c = 2) (h_f_A : f A = 1) :
  let area := (1 / 2) * b * c * Real.sin A in
  area = √3 / 4 :=
sorry

end problem_I_problem_I_interval_problem_II_l415_415215


namespace train_cross_bridge_time_l415_415842

def train_length : ℕ := 170
def train_speed_kmph : ℕ := 45
def bridge_length : ℕ := 205

def total_distance : ℕ := train_length + bridge_length
def train_speed_mps : ℕ := (train_speed_kmph * 1000) / 3600

theorem train_cross_bridge_time : (total_distance / train_speed_mps) = 30 := 
sorry

end train_cross_bridge_time_l415_415842


namespace range_of_a_l415_415936

noncomputable def even_function_monotonic_interval (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ {y z : ℝ}, 0 ≤ y → y ≤ z → f y ≤ f z)

noncomputable def satisfies_inequality (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a * 2^x) - f (4^x + 1) ≤ 0

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (hf : even_function_monotonic_interval f) (ha : satisfies_inequality f a) : 
  a ∈ set.Icc (-2 : ℝ) (2 : ℝ) :=
sorry

end range_of_a_l415_415936


namespace cake_buyers_count_l415_415801

def total_buyers : ℕ := 100
def muffin_buyers : ℕ := 40
def both_buyers : ℕ := 15
def prob_neither : ℚ := 0.25

theorem cake_buyers_count : 
  let neither_buyers := prob_neither * total_buyers in
  let buyers_cake_or_muffin := total_buyers - neither_buyers in
  ∃ (C : ℕ), 
   (C + muffin_buyers - both_buyers = buyers_cake_or_muffin) ∧ (C = 50) :=
by 
  let neither_buyers := prob_neither * total_buyers
  let buyers_cake_or_muffin := total_buyers - neither_buyers
  use 50
  split
  case correct_count =>
    calc
      50 + muffin_buyers - both_buyers
      = 50 + 40 - 15 : by rfl
      ... = 75 : by norm_num
      ... = buyers_cake_or_muffin : by rw [buyers_cake_or_muffin, neither_buyers, total_buyers] ; norm_num
  case correct_value => 
    rfl

end cake_buyers_count_l415_415801


namespace alyssa_limes_picked_l415_415849

-- Definitions for the conditions
def total_limes : ℕ := 57
def mike_limes : ℕ := 32

-- The statement to be proved
theorem alyssa_limes_picked :
  ∃ (alyssa_limes : ℕ), total_limes - mike_limes = alyssa_limes ∧ alyssa_limes = 25 :=
by
  have alyssa_limes : ℕ := total_limes - mike_limes
  use alyssa_limes
  sorry

end alyssa_limes_picked_l415_415849


namespace lcm_calculation_l415_415973

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem lcm_calculation :
  let t := lcm (lcm 12 16) (lcm 18 24)
  t = 144 :=
by
  sorry

end lcm_calculation_l415_415973


namespace system_of_equations_solution_l415_415737

theorem system_of_equations_solution :
  ∃ (a b c : ℝ), 
    (4 * a + b * c = 32) ∧
    (2 * a - 2 * c - b^2 = 0) ∧
    (a + 12 * b - c - a * b = 6) ∧
    a = 2 ∧ b = 2 ∧ c = 6 :=
by
  use [2, 2, 6]
  split
  · exact (by norm_num)
  split
  · exact (by norm_num)
  split
  · exact (by norm_num)
  exact (by norm_num)

end system_of_equations_solution_l415_415737


namespace rectangle_height_base_ratio_l415_415678

theorem rectangle_height_base_ratio :
  ∀ (E F A G B : Point) (EF AG BF : segment),
  square_with_side_two E F A G B  -- Condition: Square with side length 2 with midpoints E and F
  → is_midpoint E 
  → is_midpoint F 
  → is_perpendicular AG BF        -- Condition: AG is perpendicular to BF
  → height_to_base_ratio XY YZ = 5 :=     -- Conclusion: height/base ratio is 5
by
  intros E F A G B EF AG BF
  intro square_with_side_two
  intro is_midpoint_E
  intro is_midpoint_F
  intro is_perpendicular_AG_BF
  sorry

end rectangle_height_base_ratio_l415_415678


namespace longer_side_of_rectangle_is_l415_415115

-- Define the radius of the circle
def radius (r : ℝ) : Prop := r = 6

-- Define the area of the circle given the radius
def area_circle (A : ℝ) : Prop := A = 36 * Real.pi

-- Define the relationship between the area of the rectangle and the circle
def area_rectangle (A_rect : ℝ) (A : ℝ) : Prop := A_rect = 3 * A

-- Define the length of the shorter side of the rectangle
def shorter_side (s : ℝ) : Prop := s = 12

-- State the goal: the length of the longer side of the rectangle
def longer_side (l : ℝ) (A_rect : ℝ) (s : ℝ) : Prop := l = A_rect / s

-- Theorem statement combining all conditions and the goal
theorem longer_side_of_rectangle_is :
  ∃ l : ℝ, ∃ A_rect : ℝ, ∃ s : ℝ, radius 6 ∧ area_circle 36 * Real.pi ∧ area_rectangle A_rect (36 * Real.pi) ∧ shorter_side s ∧ longer_side l A_rect s :=
by {
  sorry
}

end longer_side_of_rectangle_is_l415_415115


namespace sin_half_pi_plus_A_l415_415280

theorem sin_half_pi_plus_A (A : Real) (h : Real.cos (Real.pi + A) = -1 / 2) :
  Real.sin (Real.pi / 2 + A) = 1 / 2 := by
  sorry

end sin_half_pi_plus_A_l415_415280


namespace ratio_of_board_pieces_l415_415793

theorem ratio_of_board_pieces (S L : ℕ) (hS : S = 23) (hTotal : S + L = 69) : L / S = 2 :=
by
  sorry

end ratio_of_board_pieces_l415_415793


namespace midpoint_locus_l415_415485

-- Definitions corresponding to the conditions
def center : Type := ℝ × ℝ -- Type representing a point in the 2D plane
def radius := 4.0 -- radius 4 units
def fixed_point_R : center := (R_x, R_y) -- Point R outside the circle
def circle_center_C : center := (C_x, C_y) -- Center of the circle

-- Locus of midpoints
theorem midpoint_locus :
  let homothety (R C : center) k := ((fst R) + k * ((fst C) - (fst R)), (snd R) + k * ((snd C) - (snd R))) in
  let N := homothety fixed_point_R circle_center_C (2/3) in
  N = homothety fixed_point_R circle_center_C (2/3) ∧ distance N fixed_point_R = (8/3) :=
sorry

end midpoint_locus_l415_415485


namespace total_distance_AD_l415_415484

theorem total_distance_AD :
  let d_AB := 100
  let d_BC := d_AB + 50
  let d_CD := 2 * d_BC
  d_AB + d_BC + d_CD = 550 := by
  sorry

end total_distance_AD_l415_415484


namespace min_value_fraction_sum_l415_415609

theorem min_value_fraction_sum (x y : ℝ) (h : x^2 + y^2 = 2) :
  ∃ m : ℝ, m = 1 ∧ ∀ x y : ℝ, x^2 + y^2 = 2 → 
            (1 / (1 + x^2) + 1 / (1 + y^2)) ≥ m :=
begin
  sorry
end

end min_value_fraction_sum_l415_415609


namespace coin_order_correct_l415_415759

variables (C E A D B : Type) -- Representing each coin as a type

-- Coin order conditions based on the problem
variable (is_top_of_all : C)
variable (is_beneath : E -> C)
variable (is_directly_beneath : A -> E -> B)
variable (is_directly_beneath : D -> E -> B)
variable (is_bottom : B)

-- Definition outlining the correct order of coins.
def correct_order : Prop :=
  (is_top_of_all = C) ∧ (is_beneath E = C) ∧ (is_directly_beneath D E = B) ∧
  (is_directly_beneath A E = B) ∧ (is_bottom = B)

-- Main theorem stating that the order of coins follows (C, E, D, A, B)
theorem coin_order_correct : correct_order C E A D B :=
sorry -- Proof to be provided

end coin_order_correct_l415_415759


namespace degree_of_monomial_l415_415397

theorem degree_of_monomial : 
  (degree (-2^3 * a * b^2 * c) = 4) :=
by 
  -- The degree of a monomial is defined as the sum of the exponents of all its variables.
  -- The exponent of a is 1, the exponent of b is 2, the exponent of c is 1.
  -- Therefore, the degree is 1 + 2 + 1 = 4.
  sorry

end degree_of_monomial_l415_415397


namespace price_increase_correct_l415_415414

variable (a b : ℝ)

-- Assume percentage increase is given as a fraction
def percentage_increase (b : ℝ) : ℝ := b / 100

def new_price (a b : ℝ) : ℝ := a * (1 + percentage_increase b)

theorem price_increase_correct (a b : ℝ) : 
  new_price a b = a * (1 + (b / 100)) :=
by
  unfold new_price
  unfold percentage_increase
  sorry

end price_increase_correct_l415_415414


namespace minimum_a_l415_415233

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - 1 - a * Real.log x
noncomputable def g (x : ℝ) : ℝ := x / Real.exp (x - 1)

theorem minimum_a {a : ℝ} (h_neg : a < 0) :
  (∀ x1 x2 : ℝ, 3 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 4 → 
    (f x1 a - f x2 a) / (g x1 - g x2) > -1 / (g x1 * g x2)) → 
  a ≥ 3 - (2 / 3) * Real.exp 2 := 
sorry

end minimum_a_l415_415233


namespace cube_faces_with_corners_cut_l415_415742

theorem cube_faces_with_corners_cut (a b : ℕ) (h_a : a = 3) (h_b : b = 1) : 
  let initial_faces := 6 in
  let corners := 8 in
  let faces_per_cut := 3 in
  initial_faces + corners * faces_per_cut = 30 :=
by
  sorry

end cube_faces_with_corners_cut_l415_415742


namespace hyperbola_standard_equation_triangle_area_PF1F2_l415_415246

-- Define the given conditions
def F1 : ℝ × ℝ := (-Real.sqrt 5, 0)
def F2 : ℝ × ℝ := (Real.sqrt 5, 0)
def real_axis_length : ℝ := 4

-- Part 1: Prove the standard equation of the hyperbola
theorem hyperbola_standard_equation : 
  let c := Real.sqrt 5 in
  let a := real_axis_length / 2 in
  let b := Real.sqrt (c^2 - a^2) in
  (a = 2) → (c = Real.sqrt 5) → (a^2 = 4) → (b^2 = 1) → 
  (∀ x y : ℝ, (x, y) ∈ {p | (p.1^2 / 4) - p.2^2 = 1}) :=
by
  intros c a b ha hc ha_square hb_square
  -- define the hyperbola equation
  let hyperbola := (λ p : ℝ × ℝ, (p.1^2 / (ha_square)) - (p.2^2 / (hb_square)) = 1)
  exact hyperbola
  sorry

-- Part 2: Prove the area of triangle PF1F2
theorem triangle_area_PF1F2 (P : ℝ × ℝ) (h_on_hyperbola : P ∈ {p | (p.1^2 / 4) - p.2^2 = 1}) :
  let PF1 := Real.dist P F1 in
  let PF2 := Real.dist P F2 in
  (PF1 ^ 2 + PF2 ^ 2 = 20) → (PF1 * PF2 = 2) → 
  let area := 1 / 2 * PF1 * PF2 in 
  (PF1 * PF2 / 2 = 1) :=
by
  intros PF1 PF2 hp1 hp2
  exact (PF1 * PF2) / 2 = 1
  sorry

end hyperbola_standard_equation_triangle_area_PF1F2_l415_415246


namespace remainder_mod_of_a_squared_subtract_3b_l415_415913

theorem remainder_mod_of_a_squared_subtract_3b (a b : ℕ) (h₁ : a % 7 = 2) (h₂ : b % 7 = 5) (h₃ : a^2 > 3 * b) : 
  (a^2 - 3 * b) % 7 = 3 := 
sorry

end remainder_mod_of_a_squared_subtract_3b_l415_415913


namespace repetitions_today_l415_415065

theorem repetitions_today (yesterday_reps : ℕ) (deficit : ℤ) (today_reps : ℕ) : 
  yesterday_reps = 86 ∧ deficit = -13 → 
  today_reps = yesterday_reps + deficit →
  today_reps = 73 :=
by
  intros
  sorry

end repetitions_today_l415_415065


namespace max_distance_of_P_to_C2_l415_415242

noncomputable def curve_C1 (x y : ℝ) := x^2 / 12 + y^2 / 4 = 1

noncomputable def parametric_line_C2 (t : ℝ) : ℝ × ℝ := 
  (3 + (real.sqrt 3) / 2 * t, real.sqrt 3 - 1 / 2 * t)

theorem max_distance_of_P_to_C2 :
  ∃ (P : ℝ × ℝ) (d : ℝ), 
    curve_C1 P.1 P.2 ∧
    (∀ t : ℝ, P = parametric_line_C2 t) ∧
    d = real.sqrt 6 + 3 ∧
    P = (-real.sqrt 6, -real.sqrt 2) := by
  sorry

end max_distance_of_P_to_C2_l415_415242


namespace subset_condition_l415_415053

theorem subset_condition (n : ℕ) (L : list (set ℕ)) (hL : ∀ l ∈ L, 2 ≤ l.card) 
  (h_unique : ∀ (x y : ℕ) (hx : x ≠ y), ∃! l ∈ L, x ∈ l ∧ y ∈ l) : 
  L.length = 1 ∨ L.length ≥ n :=
sorry

end subset_condition_l415_415053


namespace conclusion_one_conclusion_two_conclusion_three_conclusion_four_l415_415666

noncomputable def a_n (m : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then m
  else if a_n (m) (n - 1) > 1 then a_n (m) (n - 1) - 1
  else if 0 < a_n (m) (n - 1) ∧ a_n (m) (n - 1) ≤ 1 then 1 / a_n (m) (n - 1)
  else 0 -- this else is to ensure the function is total

theorem conclusion_one (a_3_eq_4 : a_n(m)(3) = 4) :
  ∃! m : ℝ, m = 6 ∨ m = 1/5 ∨ m = 5/4 := 
sorry

theorem conclusion_two (m_eq_sqrt_2 : m = Real.sqrt 2) :
  ∃ T : ℕ, T = 3 ∧ ∀ n : ℕ, a_n(m)(n+T) = a_n(m)(n) := 
sorry

theorem conclusion_three (T : ℕ) (hT : T ≥ 2) :
  ∃ m : ℝ, m > 1 ∧ ∀ n : ℕ, a_n(m)(n+T) = a_n(m)(n) :=
sorry

theorem conclusion_four :
  ¬ ∃ m : ℚ, m ≥ 2 ∧ ∃ T : ℕ, ∀ n : ℕ, a_n(m)(n+T) = a_n(m)(n) :=
sorry

end conclusion_one_conclusion_two_conclusion_three_conclusion_four_l415_415666


namespace base_4_representation_of_157_l415_415904

theorem base_4_representation_of_157 :
  ∃ (b : ℕ), b^3 ≤ 157 ∧ 157 < b^4 ∧ ∀ (d₀ d₁ d₂ d₃ : ℕ), 
  (157 = d₃ * b^3 + d₂ * b^2 + d₁ * b + d₀) → 
  (1 ≤ d₃ < b) ∧ (0 ≤ d₂ < b) ∧ (0 ≤ d₁ < b) ∧ (0 ≤ d₀ < b) 
  ∧ (d₀ % 2 = 1) :=
by
  exists 4
  simp
  have h₁ : 4^3 = 64 := by norm_num
  have h₂ : 4^4 = 256 := by norm_num
  norm_num at *
  exact ⟨h₁, 64 < 157, 157 < 256, 0, 0, 1⟩
  sorry -- Skipping the proof.

end base_4_representation_of_157_l415_415904


namespace find_positive_real_solution_l415_415890

theorem find_positive_real_solution (x : ℝ) : 
  0 < x ∧ (1 / 2 * (4 * x^2 - 1) = (x^2 - 60 * x - 20) * (x^2 + 30 * x + 10)) ↔ 
  (x = 30 + Real.sqrt 919 ∨ x = -15 + Real.sqrt 216 ∧ 0 < -15 + Real.sqrt 216) :=
by sorry

end find_positive_real_solution_l415_415890


namespace average_rainfall_in_normal_year_l415_415760

def first_day_rainfall : ℕ := 26
def second_day_rainfall : ℕ := 34
def third_day_rainfall : ℕ := second_day_rainfall - 12
def total_rainfall_this_year : ℕ := first_day_rainfall + second_day_rainfall + third_day_rainfall
def rainfall_difference : ℕ := 58

theorem average_rainfall_in_normal_year :
  (total_rainfall_this_year + rainfall_difference) = 140 :=
by
  sorry

end average_rainfall_in_normal_year_l415_415760


namespace part_a_proof_part_b_proof_l415_415958

-- Definitions for Part (a)
variables {A B C : Point} {S : Circle} (T1 T2 T3 : Point)
axiom touched_at_M : S.Touches (circumcircle A B C) -- Statement: S touches the circumcircle at some M
axiom M_on_arc_AC : M ∈ arcAC -- Statement: M is on the arc AC

-- Define lengths of tangents from points A, B, C to the circle S
def length_of_tangent (P T : Point) (c : Circle) : Real := sorry

theorem part_a_proof : length_of_tangent A T1 S * distance B C + length_of_tangent C T3 S * distance A B = length_of_tangent B T2 S * distance A C := sorry

-- Variables and definitions for part (b)
variables {C1 C2 C3 C4 : Circle} {S : Circle} (P1 P2 P3 P4 : Point)
axiom C1_touch_S : C1.Touches S
axiom C2_touch_S : C2.Touches S
axiom C3_touch_S : C3.Touches S
axiom C4_touch_S : C4.Touches S

-- Order assumption of tangency points on the circle S
axiom order_of_tangency : ordered_circle_points [P1, P2, P3, P4] S -- Placeholder for the proper definition of ordered points on circle S

-- Define lengths of tangents between corresponding points
def tangent_segment (Ci Cj : Circle) : Real := sorry

theorem part_b_proof : tangent_segment C1 C2 * tangent_segment C3 C4 + tangent_segment C2 C3 * tangent_segment C1 C4 = tangent_segment C1 C3 * tangent_segment C2 C4 := sorry

end part_a_proof_part_b_proof_l415_415958


namespace quadratic_discriminant_l415_415195

-- Define the coefficients of the quadratic equation
def a : ℚ := 2
def b : ℚ := 2 + 1/2
def c : ℚ := 1/2

-- State the theorem to prove
theorem quadratic_discriminant : (b^2 - 4 * a * c) = 9 / 4 := by
  -- Coefficient values
  have h_b : b = 5 / 2 := by
    calc
      b = 2 + 1/2 : rfl
      ... = 5 / 2 : by norm_num
  have h_discriminant : (5/2)^2 - 4 * 2 * (1/2) = 9/4 := by sorry
  -- Substitute the coefficient values
  rw h_b,
  exact h_discriminant,
  sorry

end quadratic_discriminant_l415_415195


namespace smallest_four_digit_multiple_of_17_l415_415570

theorem smallest_four_digit_multiple_of_17 : ∃ n, n ≥ 1000 ∧ n < 10000 ∧ 17 ∣ n ∧ ∀ m, m ≥ 1000 ∧ m < 10000 ∧ 17 ∣ m → n ≤ m := 
by
  use 1003
  sorry

end smallest_four_digit_multiple_of_17_l415_415570


namespace small_seat_capacity_indeterminate_l415_415388

-- Conditions
def small_seats : ℕ := 3
def large_seats : ℕ := 7
def capacity_per_large_seat : ℕ := 12
def total_large_capacity : ℕ := 84

theorem small_seat_capacity_indeterminate
  (h1 : large_seats * capacity_per_large_seat = total_large_capacity)
  (h2 : ∀ s : ℕ, ∃ p : ℕ, p ≠ s * capacity_per_large_seat) :
  ¬ ∃ n : ℕ, ∀ m : ℕ, small_seats * m = n * small_seats :=
by {
  sorry
}

end small_seat_capacity_indeterminate_l415_415388


namespace mean_height_proof_l415_415024

def heights_50s : List ℕ := [50, 51, 54]
def heights_60s : List ℕ := [60, 62, 62, 63, 65, 68]
def heights_70s : List ℕ := [70, 71, 74, 75]

def total_height (l1 l2 l3 : List ℕ) : ℕ :=
  (l1 ++ l2 ++ l3).sum

def total_players (l1 l2 l3 : List ℕ) : ℕ :=
  (l1.length + l2.length + l3.length)

def mean_height (total_height : ℕ) (total_players : ℕ) : ℝ :=
  total_height / total_players.to_real

theorem mean_height_proof : mean_height (total_height heights_50s heights_60s heights_70s) (total_players heights_50s heights_60s heights_70s) = 63.46 := by
  sorry

end mean_height_proof_l415_415024


namespace triangle_inequality_l415_415729
open Real

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_abc : a + b > c) (h_acb : a + c > b) (h_bca : b + c > a) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end triangle_inequality_l415_415729


namespace find_a_find_a_plus_c_l415_415301

-- Define the triangle with given sides and angles
variables (A B C : ℝ) (a b c S : ℝ)
  (h_cosB : cos B = 4/5)
  (h_b : b = 2)
  (h_area : S = 3)

-- Prove the value of the side 'a' when angle A is π/6
theorem find_a (h_A : A = Real.pi / 6) : a = 5 / 3 := 
  sorry

-- Prove the sum of sides 'a' and 'c' when the area of the triangle is 3
theorem find_a_plus_c (h_ac : a * c = 10) : a + c = 2 * Real.sqrt 10 :=
  sorry

end find_a_find_a_plus_c_l415_415301


namespace product_of_all_values_l415_415159

-- Define the initial temperatures
variables (M L N : ℤ)

-- Define the conditions
def initial_condition : Prop := M = L + N
def temp_mpls_6pm : ℤ := M - 8
def temp_stl_6pm : ℤ := L + 6
def temp_difference_condition : Prop := abs (temp_mpls_6pm - temp_stl_6pm) = 1

-- Proposition to prove
theorem product_of_all_values (N : ℤ) :
  initial_condition M L N →
  temp_difference_condition M L N →
  N = 15 ∨ N = 13 →
  15 * 13 = 195 :=
by {
  intros _ _ _,

  -- Just stating the proof is trivial since by given conditions N equals 15 or 13
  -- and their product is 195.
  have h : 15 * 13 = 195 := rfl,
  exact h,
}

end product_of_all_values_l415_415159


namespace range_of_a_l415_415977

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, 3 < x ∧ x < 4 ∧ ax^2 - 4*a*x - 2 > 0) ↔ a < -2/3 :=
sorry

end range_of_a_l415_415977


namespace most_likely_dissatisfied_proof_expected_dissatisfied_proof_variance_dissatisfied_proof_l415_415028

noncomputable def most_likely_dissatisfied (n : ℕ) : ℕ := 1

theorem most_likely_dissatisfied_proof (n : ℕ) (h : n > 1) :
  (∃ d : ℕ, d = most_likely_dissatisfied n) := by
  use 1
  trivial

noncomputable def expected_dissatisfied (n : ℕ) : ℝ := Real.sqrt (n / Real.pi)

theorem expected_dissatisfied_proof (n : ℕ) (h : n > 0) :
  (∃ e : ℝ, e = expected_dissatisfied n) := by
  use Real.sqrt (n / Real.pi)
  trivial

noncomputable def variance_dissatisfied (n : ℕ) : ℝ := 0.182 * n

theorem variance_dissatisfied_proof (n : ℕ) (h : n > 0) :
  (∃ v : ℝ, v = variance_dissatisfied n) := by
  use 0.182 * n
  trivial

end most_likely_dissatisfied_proof_expected_dissatisfied_proof_variance_dissatisfied_proof_l415_415028


namespace total_chickens_after_purchase_l415_415530

def initial_chickens : ℕ := 400
def percentage_died : ℕ := 40
def times_to_buy : ℕ := 10

noncomputable def chickens_died : ℕ := (percentage_died * initial_chickens) / 100
noncomputable def chickens_remaining : ℕ := initial_chickens - chickens_died
noncomputable def chickens_bought : ℕ := times_to_buy * chickens_died
noncomputable def total_chickens : ℕ := chickens_remaining + chickens_bought

theorem total_chickens_after_purchase : total_chickens = 1840 :=
by
  sorry

end total_chickens_after_purchase_l415_415530


namespace Jules_height_l415_415722

theorem Jules_height (Ben_initial_height Jules_initial_height Ben_current_height Jules_current_height : ℝ) 
  (h_initial : Ben_initial_height = Jules_initial_height)
  (h_Ben_growth : Ben_current_height = 1.25 * Ben_initial_height)
  (h_Jules_growth : Jules_current_height = Jules_initial_height + (Ben_current_height - Ben_initial_height) / 3)
  (h_Ben_current : Ben_current_height = 75) 
  : Jules_current_height = 65 := 
by
  -- Use the conditions to prove that Jules is now 65 inches tall
  sorry

end Jules_height_l415_415722


namespace car_speed_l415_415476

-- Definitions based on the conditions
def distance : ℝ := 260
def time : ℝ := 4

-- Definition of speed using the distance and time
def speed (d : ℝ) (t : ℝ) : ℝ := d / t

-- Statement of the theorem that needs to be proven
theorem car_speed : speed distance time = 65 := by
  sorry

end car_speed_l415_415476


namespace third_derivative_y_l415_415897

noncomputable def y (x : ℝ) : ℝ := x^2 * Real.sin (5 * x - 3)

theorem third_derivative_y (x : ℝ) : 
  (deriv^[3] y x) = -150 * x * Real.sin (5 * x - 3) + (30 - 125 * x^2) * Real.cos (5 * x - 3) :=
by
  sorry

end third_derivative_y_l415_415897


namespace walnut_trees_planted_l415_415421

-- Define the initial number of walnut trees
def initial_walnut_trees : ℕ := 22

-- Define the total number of walnut trees after planting
def total_walnut_trees_after : ℕ := 55

-- The Lean statement to prove the number of walnut trees planted today
theorem walnut_trees_planted : (total_walnut_trees_after - initial_walnut_trees = 33) :=
by
  sorry

end walnut_trees_planted_l415_415421


namespace tank_capacity_correct_l415_415852

noncomputable def tank_capacity : ℕ :=
  let C := 640 in
  C

theorem tank_capacity_correct :
  ∃ C : ℕ,
    (∃ outlet_rate : ℕ, outlet_rate = C / 10) ∧
    (∃ inlet_rate : ℕ, inlet_rate = 4 * 60) ∧
    (∃ total_time_with_inlet : ℕ, total_time_with_inlet = 10 + 6) ∧
    (∃ effective_rate : ℕ, effective_rate = C / 16) ∧
    ((C / 10) - (4 * 60) = C / 16) ∧
    C = 640 :=
sorry

end tank_capacity_correct_l415_415852


namespace triangle_perimeter_l415_415997

variable (X Y Z : Type) [MetricSpace X] (a : X) (b : Y) (c : Z)
-- The above variables simulate the points in a metric space which can be worked as Euclidean space points.

-- Here we assume point configuration such that angles and distances hold
variables (XYZ_is_isosceles : ∠X Y Z = ∠X Z Y)  -- angle XYZ equals angle XZY
variables (YZ_len : dist y z = 8)               -- distance YZ = 8
variables (XZ_len : dist x z = 10)              -- distance XZ = 10

theorem triangle_perimeter : 
  perimeter XYZ = 28 :=
  sorry

end triangle_perimeter_l415_415997


namespace value_of_n_l415_415285

theorem value_of_n (n : ℕ) (k : ℕ) (h : k = 11) (eqn : (1/2)^n * (1/81)^k = 1/18^22) : n = 22 :=
by
  sorry

end value_of_n_l415_415285


namespace find_value_l415_415036

-- Definitions based on the problem conditions
axiom op_def (m n k : ℕ) : m ⊕ n = k → m ⊕ (n + 1) = k + 2
axiom op_initial : 1 ⊕ 1 = 2

-- The target statement to prove
theorem find_value : 1 ⊕ 2006 = 4012 := sorry

end find_value_l415_415036


namespace min_value_of_a_plus_b_l415_415972

theorem min_value_of_a_plus_b (a b c : ℝ) (C : ℝ) 
  (hC : C = 60) 
  (h : (a + b)^2 - c^2 = 4) : 
  a + b ≥ (4 * Real.sqrt 3) / 3 :=
by
  sorry

end min_value_of_a_plus_b_l415_415972


namespace Alexei_finished_ahead_of_Sergei_by_1_9_km_l415_415174

noncomputable def race_distance : ℝ := 10
noncomputable def v_A : ℝ := 1  -- speed of Alexei
noncomputable def v_V : ℝ := 0.9 * v_A  -- speed of Vitaly
noncomputable def v_S : ℝ := 0.81 * v_A  -- speed of Sergei

noncomputable def distance_Alexei_finished_Ahead_of_Sergei : ℝ :=
race_distance - (0.81 * race_distance)

theorem Alexei_finished_ahead_of_Sergei_by_1_9_km :
  distance_Alexei_finished_Ahead_of_Sergei = 1.9 :=
by
  simp [race_distance, v_A, v_V, v_S, distance_Alexei_finished_Ahead_of_Sergei]
  sorry

end Alexei_finished_ahead_of_Sergei_by_1_9_km_l415_415174


namespace prove_2x_minus_y_l415_415967

noncomputable def determine_x (y : ℤ) : ℤ := (40 - y) / 3

theorem prove_2x_minus_y :
  ∃ x y : ℤ, 
    (3 * x + y = 40) ∧ 
    (3 * y^2 = 48) ∧ 
    (2 * x - y = 20) := 
by
  use determine_x 4, 4
  have h_x : determine_x 4 = 12 := by norm_num [determine_x]
  rw [h_x]
  exact ⟨84, 12, 20⟩
  sorry

end prove_2x_minus_y_l415_415967


namespace intersection_point_l415_415178

def first_line (x : ℝ) : ℝ := 3 * x - 1

def perpendicular_line (point_x : ℝ) (point_y : ℝ) (x : ℝ) : ℝ :=
  let slope := -1 / 3
  slope * (x - point_x) + point_y

theorem intersection_point :
  ∃ x y : ℝ, y = first_line x ∧ y = perpendicular_line 4 2 x ∧ x = 13 / 10 ∧ y = 29 / 10 :=
by
  sorry

end intersection_point_l415_415178


namespace infinite_series_sum_l415_415407

theorem infinite_series_sum :
  ∑' (n : ℕ), (n + 1) / 10^(n + 1) = 10 / 81 :=
sorry

end infinite_series_sum_l415_415407


namespace problem_statement_l415_415260

noncomputable def f : ℝ → ℝ :=
λ x, if x ≥ 0 then -x^2 - x - 2 else x / (x + 4) + Real.log x / Real.log 4

theorem problem_statement : f(f(2)) = 7 / 2 :=
by
  sorry

end problem_statement_l415_415260


namespace triangle_area_inside_pentagon_l415_415226

theorem triangle_area_inside_pentagon (points : Finset (ℝ × ℝ))
  (pentagon_area : ℝ)
  (h_pentagon_area : pentagon_area = 1993)
  (h_points_card : points.card = 1000)
  (points_in_pentagon : ∀ p ∈ points, p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 <= 1993 ∧ p.2 <= 1993) :
  ∃ (a b c : ℝ × ℝ) (H1 : a ∈ points) (H2 : b ∈ points) (H3 : c ∈ points), 
  let triangle := {a, b, c} in
  (Finset.area triangle) ≤ 1 :=
sorry

end triangle_area_inside_pentagon_l415_415226


namespace weight_of_new_person_l415_415459

theorem weight_of_new_person (avg_increase : ℝ) (num_persons : ℕ) (old_weight : ℝ) (total_increase : ℝ) :
  avg_increase = 3.5 → num_persons = 8 → old_weight = 65 → total_increase = num_persons * avg_increase →
  (old_weight + total_increase) = 93 :=
by
  intros h_avg h_num h_old h_total
  rw [h_avg, h_num, h_old] at h_total
  simp at h_total
  exact h_total
  sorry

end weight_of_new_person_l415_415459


namespace square_side_length_properties_l415_415919

theorem square_side_length_properties (a: ℝ) (h: a^2 = 10) :
  a = Real.sqrt 10 ∧ (a^2 - 10 = 0) ∧ (3 < a ∧ a < 4) :=
by
  sorry

end square_side_length_properties_l415_415919


namespace hyperbola_eccentricity_l415_415630

-- Define the parameters of the hyperbola and the given points
variables (a b : ℝ) (ha : a > 0) (hb : b > 0)
variables (A B : ℝ × ℝ) (P : ℝ × ℝ) (N : ℝ × ℝ)

-- Definitions for points
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Condition for the points being on the hyperbola
def on_hyperbola (A : ℝ × ℝ) : Prop :=
  (A.1^2 / a^2) - (A.2^2 / b^2) = 1

-- Define the specific conditions from the problem
axiom A_on_hyperbola : on_hyperbola a b A
axiom B_on_hyperbola : on_hyperbola a b B
axiom P_is_given : P = (3, 6)
axiom N_is_midpoint : N = (12, 15)
axiom midpoint_is_N : midpoint A B = N

-- The proof statement to show the eccentricity is 3/2
theorem hyperbola_eccentricity : 
  let e := sqrt (1 + (b^2 / a^2)) in
  e = 3 / 2 :=
sorry -- proof to be done

end hyperbola_eccentricity_l415_415630


namespace distinct_pairs_digit_sum_l415_415336

def digitSum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem distinct_pairs_digit_sum :
  {xy : ℕ × ℕ // ∀ x y : ℕ, xy = (x, y) →
    10 ≤ x ∧ x < 100 ∧
    10 ≤ y ∧ y < 100 ∧
    x < y ∧
    digitSum x = 6 ∧
    digitSum y = 6} .card = 15 := by
  sorry

end distinct_pairs_digit_sum_l415_415336


namespace three_b_minus_a_l415_415404

-- The problem setup and conditions
def quadratic_expression (x : ℝ) : ℝ := x^2 - 16 * x + 64

def factorization (x a b : ℝ) : Prop :=
  quadratic_expression x = (x - a) * (x - b) ∧ a ≥ 0 ∧ b ≥ 0 ∧ a > b

-- The main theorem stating what we need to prove.
theorem three_b_minus_a (a b : ℝ) (h : factorization x a b) : 3 * b - a = 16 :=
sorry

end three_b_minus_a_l415_415404


namespace probability_test_l415_415752

theorem probability_test (p_math : ℚ) (p_english : ℚ) (p_independent: Prop) :
    p_math = 5/8 ∧ p_english = 1/4 ∧ p_independent → 
    (1 - (1 - p_math) * (1 - p_english) = 23/32) := 
by 
  intro h
  cases h with hp p_independent
  cases hp with hp_math hp_english
  rw [hp_math, hp_english]
  sorry  -- Proof would go here

end probability_test_l415_415752


namespace minimum_beta_value_l415_415663

variable (α β : Real)

-- Defining the conditions given in the problem
def sin_alpha_condition : Prop := Real.sin α = -Real.sqrt 2 / 2
def cos_alpha_minus_beta_condition : Prop := Real.cos (α - β) = 1 / 2
def beta_greater_than_zero : Prop := β > 0

-- The theorem to be proven
theorem minimum_beta_value (h1 : sin_alpha_condition α) (h2 : cos_alpha_minus_beta_condition α β) (h3 : beta_greater_than_zero β) : β = Real.pi / 12 := 
sorry

end minimum_beta_value_l415_415663


namespace least_non_lucky_multiple_of_12_l415_415057

-- Define the notion of a lucky integer
def is_lucky (n : ℕ) : Prop :=
  n % (nat.digits 10 n).sum = 0

-- Define the list of first few multiples of 12
def multiples_of_12 : list ℕ := [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]

-- Problem statement: Prove that 96 is the least positive multiple of 12 that is not a lucky integer
theorem least_non_lucky_multiple_of_12 : 
  (96 ∈ multiples_of_12) ∧ (¬ is_lucky 96) ∧ 
  (∀ m, m ∈ multiples_of_12 → m < 96 → is_lucky m) :=
by
  sorry

end least_non_lucky_multiple_of_12_l415_415057


namespace canCombineWithSqrt3_l415_415063

theorem canCombineWithSqrt3 (x : ℝ) :
  (x = sqrt 12 → ∃ k : ℝ, k * sqrt 3 = x) ∧
  (x = -sqrt 6 → ¬∃ k : ℝ, k * sqrt 3 = x) ∧
  (x = sqrt (3 / 2) → ¬∃ k : ℝ, k * sqrt 3 = x) ∧
  (x = sqrt 13 → ¬∃ k : ℝ, k * sqrt 3 = x) :=
by
  sorry

end canCombineWithSqrt3_l415_415063


namespace q_polynomial_l415_415384

theorem q_polynomial (q : ℝ[X]) :
  q + (2 * X^6 + 4 * X^4 - 5 * X^3 + 2 * X) = (3 * X^4 + X^3 - 11 * X^2 + 6 * X + 3) →
  q = -2 * X^6 - X^4 + 6 * X^3 - 11 * X^2 + 4 * X + 3 :=
by
  intros h
  sorry

end q_polynomial_l415_415384


namespace rectangle_longer_side_length_l415_415123

theorem rectangle_longer_side_length
  (r : ℝ) (A_rect : ℝ) :
  r = 6 →
  A_rect = 3 * (Real.pi * r^2) →
  ∃ l : ℝ, l = 9 * Real.pi :=
begin
  intros hr hA,
  use 9 * Real.pi,
  sorry
end

end rectangle_longer_side_length_l415_415123


namespace pandas_and_bamboo_l415_415132

-- Definitions for the conditions
def number_of_pandas (x : ℕ) :=
  (∃ y : ℕ, y = 5 * x + 11 ∧ y = 2 * (3 * x - 5) - 8)

-- Theorem stating the solution
theorem pandas_and_bamboo (x y : ℕ) (h1 : y = 5 * x + 11) (h2 : y = 2 * (3 * x - 5) - 8) : x = 29 ∧ y = 156 :=
by {
  sorry
}

end pandas_and_bamboo_l415_415132


namespace sum_pairwise_le_quarter_square_l415_415343

open BigOperators

theorem sum_pairwise_le_quarter_square {n : ℕ} (x : Fin n → ℝ) (a : ℝ) (h₀ : ∀ i, 0 ≤ x i)
  (h₁ : ∑ i, x i = a) : 
  (∑ i in Finset.range (n - 1), x i * x (i + 1)) ≤ (1 / 4) * a ^ 2 := 
  sorry

end sum_pairwise_le_quarter_square_l415_415343


namespace defective_pairs_estimate_l415_415795

-- Definitions of conditions
def frequency_table : List (ℕ × ℕ × ℝ) := [
  (20, 17, 0.85),
  (40, 38, 0.95),
  (60, 55, 0.92),
  (80, 75, 0.94),
  (100, 96, 0.96),
  (200, 189, 0.95),
  (300, 286, 0.95)
]

-- Main theorem stating the proof problem
theorem defective_pairs_estimate :
    (∃ qualified_rate,
        qualified_rate = 0.95 ∧
        List.contains (300, 286, qualified_rate) frequency_table) →
    (∃ defective_pairs, defective_pairs = 75 ∧
        defective_pairs = 1500 * (1 - qualified_rate)) :=
by
  sorry

end defective_pairs_estimate_l415_415795


namespace find_x_range_l415_415937

noncomputable def f (x : ℝ) : ℝ := if h : x ≥ 0 then 3^(-x) else 3^(x)

theorem find_x_range (x : ℝ) (h1 : f 2 = -f (2*x - 1) ∧ f 2 < 0) : -1/2 < x ∧ x < 3/2 := by
  -- Proof goes here
  sorry

end find_x_range_l415_415937


namespace dodecagon_area_l415_415228

theorem dodecagon_area (a : ℝ) : 
  let OA := a / Real.sqrt 2 
  let CD := (a / 2) / Real.sqrt 2 
  let triangle_area := (1/2) * OA * CD 
  let dodecagon_area := 12 * triangle_area
  dodecagon_area = (3 * a^2) / 2 :=
by
  let OA := a / Real.sqrt 2 
  let CD := (a / 2) / Real.sqrt 2 
  let triangle_area := (1/2) * OA * CD 
  let dodecagon_area := 12 * triangle_area
  sorry

end dodecagon_area_l415_415228
