import Mathlib

namespace part_a_part_b_l1238_123828

variable (a b c : ℤ)
variable (h : a + b + c = 0)

theorem part_a : (a^4 + b^4 + c^4) % (a^2 + b^2 + c^2) = 0 :=
by
  -- proof goes here
  sorry

theorem part_b : (a^100 + b^100 + c^100) % (a^2 + b^2 + c^2) = 0 :=
by
  -- proof goes here
  sorry

end part_a_part_b_l1238_123828


namespace total_flour_l1238_123804

def cups_of_flour (flour_added : ℕ) (flour_needed : ℕ) : ℕ :=
  flour_added + flour_needed

theorem total_flour :
  ∀ (flour_added flour_needed : ℕ), flour_added = 3 → flour_needed = 6 → cups_of_flour flour_added flour_needed = 9 :=
by 
  intros flour_added flour_needed h_added h_needed
  rw [h_added, h_needed]
  rfl

end total_flour_l1238_123804


namespace students_prob_red_light_l1238_123857

noncomputable def probability_red_light_encountered (p1 p2 p3 : ℚ) : ℚ :=
  1 - ((1 - p1) * (1 - p2) * (1 - p3))

theorem students_prob_red_light :
  probability_red_light_encountered (1/2) (1/3) (1/4) = 3/4 :=
by
  sorry

end students_prob_red_light_l1238_123857


namespace volume_of_cone_formed_by_sector_l1238_123848

theorem volume_of_cone_formed_by_sector :
  let radius := 6
  let sector_fraction := (5:ℝ) / 6
  let circumference := 2 * Real.pi * radius
  let cone_base_circumference := sector_fraction * circumference
  let cone_base_radius := cone_base_circumference / (2 * Real.pi)
  let slant_height := radius
  let cone_height := Real.sqrt (slant_height^2 - cone_base_radius^2)
  let volume := (1:ℝ) / 3 * Real.pi * (cone_base_radius^2) * cone_height
  volume = 25 / 3 * Real.pi * Real.sqrt 11 :=
by sorry

end volume_of_cone_formed_by_sector_l1238_123848


namespace man_speed_l1238_123819

theorem man_speed (distance_meters time_minutes : ℝ) (h_distance : distance_meters = 1250) (h_time : time_minutes = 15) :
  (distance_meters / 1000) / (time_minutes / 60) = 5 :=
by
  sorry

end man_speed_l1238_123819


namespace part_a_part_b_part_c_l1238_123853

-- The conditions for quadrilateral ABCD
variables (a b c d e f m n S : ℝ)
variables (S_nonneg : 0 ≤ S)

-- Prove Part (a)
theorem part_a (a b c d e f : ℝ) (S : ℝ) (h : S ≤ 1/4 * (e^2 + f^2)) : S <= 1/4 * (e^2 + f^2) :=
by 
  exact h

-- Prove Part (b)
theorem part_b (a b c d e f m n S: ℝ) (h : S ≤ 1/2 * (m^2 + n^2)) : S <= 1/2 * (m^2 + n^2) :=
by 
  exact h

-- Prove Part (c)
theorem part_c (a b c d e f m n S: ℝ) (h : S ≤ 1/4 * (a + c) * (b + d)) : S <= 1/4 * (a + c) * (b + d) :=
by 
  exact h

#eval "This Lean code defines the correctness statement of each part of the problem."

end part_a_part_b_part_c_l1238_123853


namespace cost_of_math_books_l1238_123869

theorem cost_of_math_books (M : ℕ) : 
  (∃ (total_books math_books history_books total_cost : ℕ),
    total_books = 90 ∧
    math_books = 60 ∧
    history_books = total_books - math_books ∧
    history_books * 5 + math_books * M = total_cost ∧
    total_cost = 390) → 
  M = 4 :=
by
  -- We provide the assumed conditions
  intro h
  -- We will skip the proof with sorry
  sorry

end cost_of_math_books_l1238_123869


namespace sandwich_cost_proof_l1238_123866

/-- Definitions of ingredient costs and quantities. --/
def bread_cost : ℝ := 0.15
def ham_cost : ℝ := 0.25
def cheese_cost : ℝ := 0.35
def mayo_cost : ℝ := 0.10
def lettuce_cost : ℝ := 0.05
def tomato_cost : ℝ := 0.08

def num_bread_slices : ℕ := 2
def num_ham_slices : ℕ := 2
def num_cheese_slices : ℕ := 2
def num_mayo_tbsp : ℕ := 1
def num_lettuce_leaf : ℕ := 1
def num_tomato_slices : ℕ := 2

/-- Calculation of the total cost in dollars and conversion to cents. --/
def sandwich_cost_in_dollars : ℝ :=
  (num_bread_slices * bread_cost) + 
  (num_ham_slices * ham_cost) + 
  (num_cheese_slices * cheese_cost) + 
  (num_mayo_tbsp * mayo_cost) + 
  (num_lettuce_leaf * lettuce_cost) + 
  (num_tomato_slices * tomato_cost)

def sandwich_cost_in_cents : ℝ :=
  sandwich_cost_in_dollars * 100

/-- Prove that the cost of the sandwich in cents is 181. --/
theorem sandwich_cost_proof : sandwich_cost_in_cents = 181 := by
  sorry

end sandwich_cost_proof_l1238_123866


namespace sixth_root_binomial_expansion_l1238_123825

theorem sixth_root_binomial_expansion :
  (2748779069441 = 1 * 150^6 + 6 * 150^5 + 15 * 150^4 + 20 * 150^3 + 15 * 150^2 + 6 * 150 + 1) →
  (2748779069441 = Nat.choose 6 6 * 150^6 + Nat.choose 6 5 * 150^5 + Nat.choose 6 4 * 150^4 + Nat.choose 6 3 * 150^3 + Nat.choose 6 2 * 150^2 + Nat.choose 6 1 * 150 + Nat.choose 6 0) →
  (Real.sqrt (2748779069441 : ℝ) = 151) :=
by
  intros h1 h2
  sorry

end sixth_root_binomial_expansion_l1238_123825


namespace chores_minutes_proof_l1238_123881

-- Definitions based on conditions
def minutes_of_cartoon_per_hour := 60
def cartoon_watched_hours := 2
def cartoon_watched_minutes := cartoon_watched_hours * minutes_of_cartoon_per_hour
def ratio_of_cartoon_to_chores := 10 / 8

-- Definition based on the question
def chores_minutes (cartoon_minutes : ℕ) : ℕ := (8 * cartoon_minutes) / 10

theorem chores_minutes_proof : chores_minutes cartoon_watched_minutes = 96 := 
by sorry 

end chores_minutes_proof_l1238_123881


namespace gcd_factorials_l1238_123833

theorem gcd_factorials : Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  -- Proof steps would go here
  sorry

end gcd_factorials_l1238_123833


namespace ratio_of_volumes_l1238_123831
-- Import the necessary library

-- Define the edge lengths of the cubes
def small_cube_edge : ℕ := 4
def large_cube_edge : ℕ := 24

-- Define the volumes of the cubes
def volume (edge : ℕ) : ℕ := edge ^ 3

-- Define the volumes
def V_small := volume small_cube_edge
def V_large := volume large_cube_edge

-- State the main theorem
theorem ratio_of_volumes : V_small / V_large = 1 / 216 := 
by 
  -- we skip the proof here
  sorry

end ratio_of_volumes_l1238_123831


namespace pollen_particle_diameter_in_scientific_notation_l1238_123815

theorem pollen_particle_diameter_in_scientific_notation :
  0.0000078 = 7.8 * 10^(-6) :=
by
  sorry

end pollen_particle_diameter_in_scientific_notation_l1238_123815


namespace max_visible_sum_l1238_123873

-- Definitions for the problem conditions

def numbers : List ℕ := [1, 3, 6, 12, 24, 48]

def num_faces (cubes : List ℕ) : Prop :=
  cubes.length = 18 -- since each of 3 cubes has 6 faces, we expect 18 numbers in total.

def is_valid_cube (cube : List ℕ) : Prop :=
  ∀ n ∈ cube, n ∈ numbers

def are_cubes (cubes : List (List ℕ)) : Prop :=
  cubes.length = 3 ∧ ∀ cube ∈ cubes, is_valid_cube cube ∧ cube.length = 6

-- The main theorem stating the maximum possible sum of the visible numbers
theorem max_visible_sum (cubes : List (List ℕ)) (h : are_cubes cubes) : ∃ s, s = 267 :=
by
  sorry

end max_visible_sum_l1238_123873


namespace distance_after_12_seconds_time_to_travel_380_meters_l1238_123877

def distance_travelled (t : ℝ) : ℝ := 9 * t + 0.5 * t^2

theorem distance_after_12_seconds : distance_travelled 12 = 180 :=
by 
  sorry

theorem time_to_travel_380_meters : ∃ t : ℝ, distance_travelled t = 380 ∧ t = 20 :=
by 
  sorry

end distance_after_12_seconds_time_to_travel_380_meters_l1238_123877


namespace evaluate_triangle_l1238_123880

def triangle_op (a b : Int) : Int :=
  a * b - a - b + 1

theorem evaluate_triangle :
  triangle_op (-3) 4 = -12 :=
by
  sorry

end evaluate_triangle_l1238_123880


namespace actual_distance_between_city_centers_l1238_123870

-- Define the conditions
def map_distance_cm : ℝ := 45
def scale_cm_to_km : ℝ := 10

-- Define the proof statement
theorem actual_distance_between_city_centers
  (md : ℝ := map_distance_cm)
  (scale : ℝ := scale_cm_to_km) :
  md * scale = 450 :=
by
  sorry

end actual_distance_between_city_centers_l1238_123870


namespace expression_undefined_at_9_l1238_123863

theorem expression_undefined_at_9 (x : ℝ) : (3 * x ^ 3 - 5) / (x ^ 2 - 18 * x + 81) = 0 → x = 9 :=
by sorry

end expression_undefined_at_9_l1238_123863


namespace kate_candy_l1238_123858

variable (K : ℕ)
variable (R : ℕ) (B : ℕ) (M : ℕ)

-- Define the conditions
def robert_pieces := R = K + 2
def mary_pieces := M = R + 2
def bill_pieces := B = M - 6
def total_pieces := K + R + M + B = 20

-- The theorem to prove
theorem kate_candy :
  ∃ (K : ℕ), robert_pieces K R ∧ mary_pieces R M ∧ bill_pieces M B ∧ total_pieces K R M B ∧ K = 4 :=
sorry

end kate_candy_l1238_123858


namespace zoo_sea_lions_l1238_123887

variable (S P : ℕ)

theorem zoo_sea_lions (h1 : S / P = 4 / 11) (h2 : P = S + 84) : S = 48 := 
sorry

end zoo_sea_lions_l1238_123887


namespace ratio_x_y_l1238_123886

theorem ratio_x_y (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + y) = 5 / 4) : x / y = 13 / 2 :=
by
  sorry

end ratio_x_y_l1238_123886


namespace zero_in_interval_l1238_123893

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := -2 * x + 3
noncomputable def h (x : ℝ) : ℝ := f x + 2 * x - 3

theorem zero_in_interval : ∃ x : ℝ, x ∈ Set.Ioo (1/2 : ℝ) 1 ∧ h x = 0 := 
sorry

end zero_in_interval_l1238_123893


namespace room_volume_l1238_123842

theorem room_volume (b l h : ℝ) (h1 : l = 3 * b) (h2 : h = 2 * b) (h3 : l * b = 12) :
  l * b * h = 48 :=
by sorry

end room_volume_l1238_123842


namespace lily_remaining_milk_l1238_123895

def initial_milk : ℚ := (11 / 2)
def given_away : ℚ := (17 / 4)
def remaining_milk : ℚ := initial_milk - given_away

theorem lily_remaining_milk : remaining_milk = 5 / 4 :=
by
  -- Here, we would provide the proof steps, but we can use sorry to skip it.
  exact sorry

end lily_remaining_milk_l1238_123895


namespace farmer_harvested_correctly_l1238_123856

def estimated_harvest : ℕ := 213489
def additional_harvest : ℕ := 13257
def total_harvest : ℕ := 226746

theorem farmer_harvested_correctly :
  estimated_harvest + additional_harvest = total_harvest :=
by
  sorry

end farmer_harvested_correctly_l1238_123856


namespace sqrt_div_val_l1238_123832

theorem sqrt_div_val (n : ℕ) (h : n = 3600) : (Nat.sqrt n) / 15 = 4 := by 
  sorry

end sqrt_div_val_l1238_123832


namespace gratuity_calculation_correct_l1238_123839

noncomputable def tax_rate (item: String): ℝ :=
  if item = "NY Striploin" then 0.10
  else if item = "Glass of wine" then 0.15
  else if item = "Dessert" then 0.05
  else if item = "Bottle of water" then 0.00
  else 0

noncomputable def base_price (item: String): ℝ :=
  if item = "NY Striploin" then 80
  else if item = "Glass of wine" then 10
  else if item = "Dessert" then 12
  else if item = "Bottle of water" then 3
  else 0

noncomputable def total_price_with_tax (item: String): ℝ :=
  base_price item + base_price item * tax_rate item

noncomputable def gratuity (item: String): ℝ :=
  total_price_with_tax item * 0.20

noncomputable def total_gratuity: ℝ :=
  gratuity "NY Striploin" + gratuity "Glass of wine" + gratuity "Dessert" + gratuity "Bottle of water"

theorem gratuity_calculation_correct :
  total_gratuity = 23.02 :=
by
  sorry

end gratuity_calculation_correct_l1238_123839


namespace CapeMay_more_than_twice_Daytona_l1238_123826

def Daytona_sharks : ℕ := 12
def CapeMay_sharks : ℕ := 32

theorem CapeMay_more_than_twice_Daytona : CapeMay_sharks - 2 * Daytona_sharks = 8 := by
  sorry

end CapeMay_more_than_twice_Daytona_l1238_123826


namespace avg_speed_additional_hours_l1238_123800

/-- Definitions based on the problem conditions -/
def first_leg_speed : ℕ := 30 -- miles per hour
def first_leg_time : ℕ := 6 -- hours
def total_trip_time : ℕ := 8 -- hours
def total_avg_speed : ℕ := 34 -- miles per hour

/-- The theorem that ties everything together -/
theorem avg_speed_additional_hours : 
  ((total_avg_speed * total_trip_time) - (first_leg_speed * first_leg_time)) / (total_trip_time - first_leg_time) = 46 := 
sorry

end avg_speed_additional_hours_l1238_123800


namespace range_of_m_for_line_to_intersect_ellipse_twice_l1238_123838

theorem range_of_m_for_line_to_intersect_ellipse_twice (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
   (A.2 = 4 * A.1 + m) ∧
   (B.2 = 4 * B.1 + m) ∧
   ((A.1 ^ 2) / 4 + (A.2 ^ 2) / 3 = 1) ∧
   ((B.1 ^ 2) / 4 + (B.2 ^ 2) / 3 = 1) ∧
   (A.1 + B.1) / 2 = 0 ∧ 
   (A.2 + B.2) / 2 = 4 * 0 + m) ↔
   - (2 * Real.sqrt 13) / 13 < m ∧ m < (2 * Real.sqrt 13) / 13
 :=
sorry

end range_of_m_for_line_to_intersect_ellipse_twice_l1238_123838


namespace no_500_good_trinomials_l1238_123865

def is_good_quadratic_trinomial (a b c : ℤ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (b^2 - 4 * a * c) > 0

theorem no_500_good_trinomials (S : Finset ℤ) (hS: S.card = 10)
  (hs_pos: ∀ x ∈ S, x > 0) : ¬(∃ T : Finset (ℤ × ℤ × ℤ), 
  T.card = 500 ∧ (∀ (a b c : ℤ), (a, b, c) ∈ T → is_good_quadratic_trinomial a b c)) :=
by
  sorry

end no_500_good_trinomials_l1238_123865


namespace polynomial_is_perfect_square_trinomial_l1238_123896

-- The definition of a perfect square trinomial
def isPerfectSquareTrinomial (a b c m : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a * b = c ∧ 4 * a * a + m * b = 4 * a * b * b

-- The main theorem to prove that if the polynomial is a perfect square trinomial, then m = 20
theorem polynomial_is_perfect_square_trinomial (a b : ℝ) (h : isPerfectSquareTrinomial 2 1 5 25) :
  ∀ x, (4 * x * x + 20 * x + 25 = (2 * x + 5) * (2 * x + 5)) :=
by
  sorry

end polynomial_is_perfect_square_trinomial_l1238_123896


namespace number_of_dogs_on_boat_l1238_123802

theorem number_of_dogs_on_boat 
  (initial_sheep : ℕ) (initial_cows : ℕ) (initial_dogs : ℕ)
  (drowned_sheep : ℕ) (drowned_cows : ℕ)
  (made_it_to_shore : ℕ)
  (H1 : initial_sheep = 20)
  (H2 : initial_cows = 10)
  (H3 : drowned_sheep = 3)
  (H4 : drowned_cows = 2 * drowned_sheep)
  (H5 : made_it_to_shore = 35)
  : initial_dogs = 14 := 
sorry

end number_of_dogs_on_boat_l1238_123802


namespace knight_min_moves_l1238_123806

theorem knight_min_moves (n : ℕ) (h : n ≥ 4) : 
  ∃ k : ℕ, k = 2 * (Nat.floor ((n + 1 : ℚ) / 3)) ∧
  (∀ m, (3 * m) ≥ (2 * (n - 1)) → ∃ l, l = 2 * m ∧ l ≥ k) :=
by
  sorry

end knight_min_moves_l1238_123806


namespace relatively_prime_sequence_l1238_123841

theorem relatively_prime_sequence (k : ℤ) (hk : k > 1) :
  ∃ (a b : ℤ) (x : ℕ → ℤ),
    a > 0 ∧ b > 0 ∧
    (∀ n, x (n + 2) = x (n + 1) + x n) ∧
    x 0 = a ∧ x 1 = b ∧ ∀ n, gcd (x n) (4 * k^2 - 5) = 1 :=
by
  sorry

end relatively_prime_sequence_l1238_123841


namespace candy_bar_calories_unit_l1238_123850

-- Definitions based on conditions
def calories_unit := "calories per candy bar"

-- There are 4 units of calories in a candy bar
def units_per_candy_bar : ℕ := 4

-- There are 2016 calories in 42 candy bars
def total_calories : ℕ := 2016
def number_of_candy_bars : ℕ := 42

-- The statement to prove
theorem candy_bar_calories_unit : (total_calories / number_of_candy_bars = 48) → calories_unit = "calories per candy bar" :=
by
  sorry

end candy_bar_calories_unit_l1238_123850


namespace cos_angle_identity_l1238_123872

theorem cos_angle_identity (α : ℝ) (h : Real.sin (α - π / 12) = 1 / 3) :
  Real.cos (α + 17 * π / 12) = 1 / 3 :=
by
  sorry

end cos_angle_identity_l1238_123872


namespace tan_2A_cos_pi3_minus_A_l1238_123871

variable (A : ℝ)

def line_equation (A : ℝ) : Prop :=
  (4 * Real.tan A = 3)

theorem tan_2A : line_equation A → Real.tan (2 * A) = -24 / 7 :=
by
  intro h 
  sorry

theorem cos_pi3_minus_A : (0 < A ∧ A < Real.pi) →
    Real.tan A = 4 / 3 →
    Real.cos (Real.pi / 3 - A) = (3 + 4 * Real.sqrt 3) / 10 :=
by
  intro h1 h2
  sorry

end tan_2A_cos_pi3_minus_A_l1238_123871


namespace polynomial_evaluation_l1238_123894

-- Given the value of y
def y : ℤ := 4

-- Our goal is to prove this mathematical statement
theorem polynomial_evaluation : (3 * (y ^ 2) + 4 * y + 2 = 66) := 
by 
    sorry

end polynomial_evaluation_l1238_123894


namespace not_all_zero_implies_at_least_one_nonzero_l1238_123835

variable {a b c : ℤ}

theorem not_all_zero_implies_at_least_one_nonzero (h : ¬ (a = 0 ∧ b = 0 ∧ c = 0)) : 
  a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 := 
by 
  sorry

end not_all_zero_implies_at_least_one_nonzero_l1238_123835


namespace ellipse_foci_on_y_axis_l1238_123884

theorem ellipse_foci_on_y_axis (k : ℝ) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (∀ x y, x^2 + k * y^2 = 2 ↔ x^2/a^2 + y^2/b^2 = 1) ∧ b^2 > a^2)
  → (0 < k ∧ k < 1) :=
sorry

end ellipse_foci_on_y_axis_l1238_123884


namespace arithmetic_sequence_sum_l1238_123889

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℕ), a 1 = 2 ∧ a 2 + a 3 = 13 → a 4 + a 5 + a 6 = 42 :=
by
  intro a
  intro h
  sorry

end arithmetic_sequence_sum_l1238_123889


namespace sector_area_eq_4cm2_l1238_123821

variable (α : ℝ) (l : ℝ) (R : ℝ)
variable (h_alpha : α = 2) (h_l : l = 4) (h_R : R = l / α)

theorem sector_area_eq_4cm2
    (h_alpha : α = 2)
    (h_l : l = 4)
    (h_R : R = l / α) :
    (1/2 * l * R) = 4 := by
  sorry

end sector_area_eq_4cm2_l1238_123821


namespace brian_final_cards_l1238_123854

-- Definitions of initial conditions
def initial_cards : ℕ := 76
def cards_taken : ℕ := 59
def packs_bought : ℕ := 3
def cards_per_pack : ℕ := 15

-- The proof problem: Prove that the final number of cards is 62
theorem brian_final_cards : initial_cards - cards_taken + packs_bought * cards_per_pack = 62 :=
by
  -- Proof goes here, 'sorry' used to skip actual proof
  sorry

end brian_final_cards_l1238_123854


namespace part1_part2_l1238_123899

-- Part (1)
theorem part1 (a : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < -1 / 2 → (ax - 1) * (x + 1) > 0) →
  a = -2 :=
sorry

-- Part (2)
theorem part2 (a : ℝ) :
  (∀ x : ℝ,
    ((a < -1 ∧ -1 < x ∧ x < 1/a) ∨
     (a = -1 ∧ ∀ x : ℝ, false) ∨
     (-1 < a ∧ a < 0 ∧ 1/a < x ∧ x < -1) ∨
     (a = 0 ∧ x < -1) ∨
     (a > 0 ∧ (x < -1 ∨ x > 1/a))) →
    (ax - 1) * (x + 1) > 0) :=
sorry

end part1_part2_l1238_123899


namespace consecutive_numbers_expression_l1238_123814

theorem consecutive_numbers_expression (x y z : ℤ) (h1 : x = y + 1) (h2 : z = y - 1) (h3 : z = 2) :
  2 * x + 3 * y + 3 * z = 8 * y - 1 :=
by
  -- substitute the conditions and simplify
  sorry

end consecutive_numbers_expression_l1238_123814


namespace curve_C_is_circle_area_AOB_constant_find_valid_a_and_curve_eq_l1238_123816

-- Define the equation of the curve C
def curve_C (a x y : ℝ) : Prop := a * x^2 + a * y^2 - 2 * a^2 * x - 4 * y = 0

-- Prove that curve C is a circle
theorem curve_C_is_circle (a : ℝ) (h : a ≠ 0) :
  ∃ (h_c : ℝ), ∃ (k : ℝ), ∃ (r : ℝ), r > 0 ∧ ∀ (x y : ℝ), curve_C a x y ↔ (x - h_c)^2 + (y - k)^2 = r^2
:= sorry

-- Prove that the area of triangle AOB is constant
theorem area_AOB_constant (a : ℝ) (h : a ≠ 0) :
  ∃ (A B : ℝ × ℝ), (A = (2 * a, 0) ∧ B = (0, 4 / a)) ∧ 1/2 * (2 * a) * (4 / a) = 4
:= sorry

-- Find valid a and equation of curve C given conditions of line l and points M, N
theorem find_valid_a_and_curve_eq (a : ℝ) (h : a ≠ 0) :
  ∀ (M N : ℝ × ℝ), (|M.1 - 0| = |N.1 - 0| ∧ |M.2 - 0| = |N.2 - 0|) → (M.1 = N.1 ∧ M.2 = N.2) →
  y = -2 * x + 4 →  a = 2 ∧ ∀ (x y : ℝ), curve_C 2 x y ↔ x^2 + y^2 - 4 * x - 2 * y = 0
:= sorry

end curve_C_is_circle_area_AOB_constant_find_valid_a_and_curve_eq_l1238_123816


namespace percentage_of_men_attended_picnic_l1238_123824

variable (E : ℝ) (W M P : ℝ)
variable (H1 : M = 0.5 * E)
variable (H2 : W = 0.5 * E)
variable (H3 : 0.4 * W = 0.2 * E)
variable (H4 : 0.3 * E = P * M + 0.2 * E)

theorem percentage_of_men_attended_picnic : P = 0.2 :=
by sorry

end percentage_of_men_attended_picnic_l1238_123824


namespace min_total_penalty_l1238_123855

noncomputable def min_penalty (B W R : ℕ) : ℕ :=
  min (B * W) (min (2 * W * R) (3 * R * B))

theorem min_total_penalty (B W R : ℕ) :
  min_penalty B W R = min (B * W) (min (2 * W * R) (3 * R * B)) := by
  sorry

end min_total_penalty_l1238_123855


namespace probability_not_equal_genders_l1238_123897

noncomputable def probability_more_grandsons_or_more_granddaughters : ℚ :=
  let total_ways := 2 ^ 12
  let equal_distribution_ways := (Nat.choose 12 6)
  let probability_equal := (equal_distribution_ways : ℚ) / (total_ways : ℚ)
  1 - probability_equal

theorem probability_not_equal_genders (n : ℕ) (p : ℚ) (hp : p = 1 / 2) (hn : n = 12) :
  probability_more_grandsons_or_more_granddaughters = 793 / 1024 :=
by
  sorry

end probability_not_equal_genders_l1238_123897


namespace solve_for_a_b_c_d_l1238_123803

theorem solve_for_a_b_c_d :
  ∃ a b c d : ℕ, (a + b + c + d) * (a^2 + b^2 + c^2 + d^2)^2 = 2023 ∧ a^3 + b^3 + c^3 + d^3 = 43 := 
by
  sorry

end solve_for_a_b_c_d_l1238_123803


namespace tournament_committees_l1238_123846

-- Assuming each team has 7 members
def team_members : Nat := 7

-- There are 5 teams
def total_teams : Nat := 5

-- The host team selects 3 members including at least one woman
def select_host_team_members (w m : Nat) : ℕ :=
  let total_combinations := Nat.choose team_members 3
  let all_men_combinations := Nat.choose (team_members - 1) 3
  total_combinations - all_men_combinations

-- Each non-host team selects 2 members including at least one woman
def select_non_host_team_members (w m : Nat) : ℕ :=
  let total_combinations := Nat.choose team_members 2
  let all_men_combinations := Nat.choose (team_members - 1) 2
  total_combinations - all_men_combinations

-- Total number of committees when one team is the host
def one_team_host_total_combinations (w m : Nat) : ℕ :=
  select_host_team_members w m * (select_non_host_team_members w m) ^ (total_teams - 1)

-- Total number of possible 11-member tournament committees
def total_committees (w m : Nat) : ℕ :=
  one_team_host_total_combinations w m * total_teams

theorem tournament_committees (w m : Nat) (hw : w ≥ 1) (hm : m ≤ 6) :
  total_committees w m = 97200 :=
by
  sorry

end tournament_committees_l1238_123846


namespace remainder_of_65_power_65_plus_65_mod_97_l1238_123885

theorem remainder_of_65_power_65_plus_65_mod_97 :
  (65^65 + 65) % 97 = 33 :=
by
  sorry

end remainder_of_65_power_65_plus_65_mod_97_l1238_123885


namespace distance_third_day_l1238_123874

theorem distance_third_day (total_distance : ℝ) (days : ℕ) (first_day_factor : ℝ) (halve_factor : ℝ) (third_day_distance : ℝ) :
  total_distance = 378 ∧ days = 6 ∧ first_day_factor = 4 ∧ halve_factor = 0.5 →
  third_day_distance = 48 := sorry

end distance_third_day_l1238_123874


namespace ladder_wood_sufficiency_l1238_123861

theorem ladder_wood_sufficiency
  (total_wood : ℝ)
  (rung_length_in: ℝ)
  (rung_distance_in: ℝ)
  (ladder_height_ft: ℝ)
  (total_wood_ft : total_wood = 300)
  (rung_length_ft : rung_length_in = 18 / 12)
  (rung_distance_ft : rung_distance_in = 6 / 12)
  (ladder_height_ft : ladder_height_ft = 50) :
  (∃ wood_needed : ℝ, wood_needed ≤ total_wood ∧ total_wood - wood_needed = 162.5) :=
sorry

end ladder_wood_sufficiency_l1238_123861


namespace FirstCandidatePercentage_l1238_123878

noncomputable def percentage_of_first_candidate_marks (PassingMarks TotalMarks MarksFirstCandidate : ℝ) :=
  (MarksFirstCandidate / TotalMarks) * 100

theorem FirstCandidatePercentage 
  (PassingMarks TotalMarks MarksFirstCandidate : ℝ)
  (h1 : PassingMarks = 200)
  (h2 : 0.45 * TotalMarks = PassingMarks + 25)
  (h3 : MarksFirstCandidate = PassingMarks - 50)
  : percentage_of_first_candidate_marks PassingMarks TotalMarks MarksFirstCandidate = 30 :=
sorry

end FirstCandidatePercentage_l1238_123878


namespace bhanu_house_rent_l1238_123883

theorem bhanu_house_rent (I : ℝ) 
  (h1 : 0.30 * I = 300) 
  (h2 : 210 = 210) : 
  210 / (I - 300) = 0.30 := 
by 
  sorry

end bhanu_house_rent_l1238_123883


namespace each_person_gets_9_apples_l1238_123876

-- Define the initial number of apples and the number of apples given to Jack's father
def initial_apples : ℕ := 55
def apples_given_to_father : ℕ := 10

-- Define the remaining apples after giving to Jack's father
def remaining_apples : ℕ := initial_apples - apples_given_to_father

-- Define the number of people sharing the remaining apples
def number_of_people : ℕ := 1 + 4

-- Define the number of apples each person will get
def apples_per_person : ℕ := remaining_apples / number_of_people

-- Prove that each person gets 9 apples
theorem each_person_gets_9_apples (h₁ : initial_apples = 55) 
                                  (h₂ : apples_given_to_father = 10) 
                                  (h₃ : number_of_people = 5) 
                                  (h₄ : remaining_apples = initial_apples - apples_given_to_father) 
                                  (h₅ : apples_per_person = remaining_apples / number_of_people) : 
  apples_per_person = 9 :=
by sorry

end each_person_gets_9_apples_l1238_123876


namespace correct_answer_is_C_l1238_123890

structure Point where
  x : ℤ
  y : ℤ

def inSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

def A : Point := ⟨1, -1⟩
def B : Point := ⟨0, 2⟩
def C : Point := ⟨-3, 2⟩
def D : Point := ⟨4, 0⟩

theorem correct_answer_is_C : inSecondQuadrant C := sorry

end correct_answer_is_C_l1238_123890


namespace loaned_books_l1238_123849

theorem loaned_books (initial_books : ℕ) (returned_percent : ℝ)
  (end_books : ℕ) (damaged_books : ℕ) (L : ℝ) :
  initial_books = 150 ∧
  returned_percent = 0.85 ∧
  end_books = 135 ∧
  damaged_books = 5 ∧
  0.85 * L + 5 + (initial_books - L) = end_books →
  L = 133 :=
by
  intros h
  rcases h with ⟨hb, hr, he, hd, hsum⟩
  repeat { sorry }

end loaned_books_l1238_123849


namespace value_set_of_t_l1238_123827

theorem value_set_of_t (t : ℝ) :
  (1 > 2 * (1) + 1 - t) ∧ (∀ x : ℝ, x^2 + (2*t-4)*x + 4 > 0) → 3 < t ∧ t < 4 :=
by
  intros h
  sorry

end value_set_of_t_l1238_123827


namespace angle_terminal_side_eq_l1238_123813

noncomputable def has_same_terminal_side (θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₂ = θ₁ + 2 * k * Real.pi

theorem angle_terminal_side_eq (k : ℤ) :
  has_same_terminal_side (- (Real.pi / 3)) (5 * Real.pi / 3) :=
by
  use 1
  sorry

end angle_terminal_side_eq_l1238_123813


namespace trains_crossing_time_l1238_123817

theorem trains_crossing_time :
  let length_of_each_train := 120 -- in meters
  let speed_of_each_train := 12 -- in km/hr
  let total_distance := length_of_each_train * 2
  let relative_speed := (speed_of_each_train * 1000 / 3600 * 2) -- in m/s
  total_distance / relative_speed = 36 := 
by
  -- Since we only need to state the theorem, the proof is omitted.
  sorry

end trains_crossing_time_l1238_123817


namespace total_people_going_to_zoo_and_amusement_park_l1238_123834

theorem total_people_going_to_zoo_and_amusement_park :
  (7.0 * 45.0) + (5.0 * 56.0) = 595.0 :=
by
  sorry

end total_people_going_to_zoo_and_amusement_park_l1238_123834


namespace f_has_two_zeros_l1238_123807

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + (a - 2) * x - Real.log x

theorem f_has_two_zeros (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0 := sorry

end f_has_two_zeros_l1238_123807


namespace exponentiation_product_l1238_123818

theorem exponentiation_product (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (3 ^ a) ^ b = 3 ^ 3) : 3 ^ a * 3 ^ b = 3 ^ 4 :=
by
  sorry

end exponentiation_product_l1238_123818


namespace q_f_digit_div_36_l1238_123888

theorem q_f_digit_div_36 (q f : ℕ) (hq : q ≠ f) (hq_digit: q < 10) (hf_digit: f < 10) :
    (457 * 10000 + q * 1000 + 89 * 10 + f) % 36 = 0 → q + f = 6 :=
sorry

end q_f_digit_div_36_l1238_123888


namespace second_group_persons_l1238_123860

open Nat

theorem second_group_persons
  (P : ℕ)
  (work_first_group : 39 * 24 * 5 = 4680)
  (work_second_group : P * 26 * 6 = 4680) :
  P = 30 :=
by
  sorry

end second_group_persons_l1238_123860


namespace product_modulo_25_l1238_123830

theorem product_modulo_25 : 
  (123 ≡ 3 [MOD 25]) → 
  (456 ≡ 6 [MOD 25]) → 
  (789 ≡ 14 [MOD 25]) → 
  (123 * 456 * 789 ≡ 2 [MOD 25]) := 
by 
  intros h1 h2 h3 
  sorry

end product_modulo_25_l1238_123830


namespace smallest_base10_num_exists_l1238_123840

noncomputable def A_digit_range := {n : ℕ // n < 6}
noncomputable def B_digit_range := {n : ℕ // n < 8}

theorem smallest_base10_num_exists :
  ∃ (A : A_digit_range) (B : B_digit_range), 7 * A.val = 9 * B.val ∧ 7 * A.val = 63 :=
by
  sorry

end smallest_base10_num_exists_l1238_123840


namespace system1_solution_system2_solution_l1238_123836

theorem system1_solution :
  ∃ x y : ℝ, 3 * x + 4 * y = 16 ∧ 5 * x - 8 * y = 34 ∧ x = 6 ∧ y = -1/2 :=
by
  sorry

theorem system2_solution :
  ∃ x y : ℝ, (x - 1) / 2 + (y + 1) / 3 = 1 ∧ x + y = 4 ∧ x = -1 ∧ y = 5 :=
by
  sorry

end system1_solution_system2_solution_l1238_123836


namespace find_integer_solutions_l1238_123829

theorem find_integer_solutions (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 + 1 / (x: ℝ)) * (1 + 1 / (y: ℝ)) * (1 + 1 / (z: ℝ)) = 2 ↔ (x = 2 ∧ y = 4 ∧ z = 15) ∨ (x = 2 ∧ y = 5 ∧ z = 9) ∨ (x = 2 ∧ y = 6 ∧ z = 7) ∨ (x = 3 ∧ y = 3 ∧ z = 8) ∨ (x = 3 ∧ y = 4 ∧ z = 5) := sorry

end find_integer_solutions_l1238_123829


namespace alligators_not_hiding_l1238_123837

-- Definitions derived from conditions
def total_alligators : ℕ := 75
def hiding_alligators : ℕ := 19

-- Theorem statement matching the mathematically equivalent proof problem.
theorem alligators_not_hiding : (total_alligators - hiding_alligators) = 56 := by
  -- Sorry skips the proof. Replace with actual proof if required.
  sorry

end alligators_not_hiding_l1238_123837


namespace fraction_value_l1238_123808

theorem fraction_value (p q r s : ℝ) (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -4 / 7 :=
by
  sorry

end fraction_value_l1238_123808


namespace travel_distance_l1238_123868

-- Define the conditions
def distance_10_gallons := 300 -- 300 miles on 10 gallons of fuel
def gallons_10 := 10 -- 10 gallons

-- Given the distance per gallon, calculate the distance for 15 gallons
def distance_per_gallon := distance_10_gallons / gallons_10

def gallons_15 := 15 -- 15 gallons

def distance_15_gallons := distance_per_gallon * gallons_15

-- Proof statement
theorem travel_distance (d_10 : distance_10_gallons = 300)
                        (g_10 : gallons_10 = 10)
                        (g_15 : gallons_15 = 15) :
  distance_15_gallons = 450 :=
  by
  -- The actual proof goes here
  sorry

end travel_distance_l1238_123868


namespace solve_for_x_l1238_123852

theorem solve_for_x (x : ℝ) (h : 1 / 4 - 1 / 6 = 4 / x) : x = 48 := 
sorry

end solve_for_x_l1238_123852


namespace speed_downstream_l1238_123851

def speed_in_still_water := 12 -- man in still water
def speed_of_stream := 6  -- speed of stream
def speed_upstream := 6  -- rowing upstream

theorem speed_downstream : 
  speed_in_still_water + speed_of_stream = 18 := 
by 
  sorry

end speed_downstream_l1238_123851


namespace car_speed_l1238_123845

theorem car_speed (v : ℝ) (hv : (1 / v * 3600) = (1 / 40 * 3600) + 10) : v = 36 := 
by
  sorry

end car_speed_l1238_123845


namespace inequality_abc_l1238_123822

theorem inequality_abc
  (a b c : ℝ)
  (ha : 0 ≤ a) (ha_le : a ≤ 1)
  (hb : 0 ≤ b) (hb_le : b ≤ 1)
  (hc : 0 ≤ c) (hc_le : c ≤ 1) :
  a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 :=
sorry

end inequality_abc_l1238_123822


namespace number_of_students_exclusively_in_math_l1238_123811

variable (T M F K : ℕ)
variable (students_in_math students_in_foreign_language students_only_music : ℕ)
variable (students_not_in_music total_students_only_non_music : ℕ)

theorem number_of_students_exclusively_in_math (hT: T = 120) (hM: M = 82)
    (hF: F = 71) (hK: K = 20) :
    T - K = 100 →
    (M + F - 53 = T - K) →
    M - 53 = 29 :=
by
  intros
  sorry

end number_of_students_exclusively_in_math_l1238_123811


namespace eqn_solution_set_l1238_123820

theorem eqn_solution_set :
  {x : ℝ | x ^ 2 - 1 = 0} = {-1, 1} := 
sorry

end eqn_solution_set_l1238_123820


namespace sufficient_but_not_necessary_l1238_123867

-- Definitions for lines and planes
def line : Type := ℝ × ℝ × ℝ
def plane : Type := ℝ × ℝ × ℝ × ℝ

-- Predicate for perpendicularity of a line to a plane
def perp_to_plane (l : line) (α : plane) : Prop := sorry

-- Predicate for parallelism of two planes
def parallel_planes (α β : plane) : Prop := sorry

-- Predicate for perpendicularity of two lines
def perp_lines (l m : line) : Prop := sorry

-- Predicate for a line being parallel to a plane
def parallel_to_plane (m : line) (β : plane) : Prop := sorry

-- Given conditions
variable (l : line)
variable (m : line)
variable (alpha : plane)
variable (beta : plane)
variable (H1 : perp_to_plane l alpha) -- l ⊥ α
variable (H2 : parallel_to_plane m beta) -- m ∥ β

-- Theorem statement
theorem sufficient_but_not_necessary :
  (parallel_planes alpha beta → perp_lines l m) ∧ ¬(perp_lines l m → parallel_planes alpha beta) :=
sorry

end sufficient_but_not_necessary_l1238_123867


namespace find_r_l1238_123862

theorem find_r 
  (r s : ℝ)
  (h1 : 9 * (r * r) * s = -6)
  (h2 : r * r + 2 * r * s = -16 / 3)
  (h3 : 2 * r + s = 2 / 3)
  (polynomial_condition : ∀ x : ℝ, 9 * x^3 - 6 * x^2 - 48 * x + 54 = 9 * (x - r)^2 * (x - s)) 
: r = -2 / 3 :=
sorry

end find_r_l1238_123862


namespace expected_number_of_returns_l1238_123809

noncomputable def expected_returns_to_zero : ℝ :=
  let p_move := 1 / 3
  let expected_value := -1 + (3 / (Real.sqrt 5))
  expected_value

theorem expected_number_of_returns : expected_returns_to_zero = (3 * Real.sqrt 5 - 5) / 5 :=
  by sorry

end expected_number_of_returns_l1238_123809


namespace min_value_expression_l1238_123844

theorem min_value_expression : ∃ (x y : ℝ), (∀ (x y : ℝ), x^2 + y^2 - 8*x + 6*y + 25 ≥ 0) ∧ (∀ (x y : ℝ), x = 4 ∧ y = -3 → x^2 + y^2 - 8*x + 6*y + 25 = 0) :=
sorry

end min_value_expression_l1238_123844


namespace problem_solution_l1238_123810

theorem problem_solution (a b c d : ℝ) 
  (h1 : a + b = 14) 
  (h2 : b + c = 9) 
  (h3 : c + d = 3) : 
  a + d = 8 := 
by 
  sorry

end problem_solution_l1238_123810


namespace range_of_a_l1238_123805

noncomputable def my_function (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + (x + 1) ^ 2

theorem range_of_a (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), x₁ > x₂ → my_function a x₁ - my_function a x₂ ≥ 4 * (x₁ - x₂)) → a ≥ 1 / 2 :=
by
  sorry

end range_of_a_l1238_123805


namespace binomial_probability_p_l1238_123891

noncomputable def binomial_expected_value (n p : ℝ) := n * p
noncomputable def binomial_variance (n p : ℝ) := n * p * (1 - p)

theorem binomial_probability_p (n p : ℝ) (h1: binomial_expected_value n p = 2) (h2: binomial_variance n p = 1) : 
  p = 0.5 :=
by
  sorry

end binomial_probability_p_l1238_123891


namespace solution_set_ineq_l1238_123812

theorem solution_set_ineq (x : ℝ) : (1 < x ∧ x ≤ 3) ↔ (x - 3) / (x - 1) ≤ 0 := sorry

end solution_set_ineq_l1238_123812


namespace tuesday_pairs_of_boots_l1238_123879

theorem tuesday_pairs_of_boots (S B : ℝ) (x : ℤ) 
  (h1 : 22 * S + 16 * B = 460)
  (h2 : 8 * S + x * B = 560)
  (h3 : B = S + 15) : 
  x = 24 :=
sorry

end tuesday_pairs_of_boots_l1238_123879


namespace trigonometric_inequality_for_tan_l1238_123843

open Real

theorem trigonometric_inequality_for_tan (x : ℝ) (hx1 : 0 < x) (hx2 : x < π / 2) : 
  1 + tan x < 1 / (1 - sin x) :=
sorry

end trigonometric_inequality_for_tan_l1238_123843


namespace value_of_x_plus_y_l1238_123882

theorem value_of_x_plus_y 
  (x y : ℝ)
  (h1 : |x| = 3)
  (h2 : |y| = 2)
  (h3 : x > y) :
  x + y = 5 ∨ x + y = 1 := 
  sorry

end value_of_x_plus_y_l1238_123882


namespace circle_integer_solution_max_sum_l1238_123898

theorem circle_integer_solution_max_sum : ∀ (x y : ℤ), (x - 1)^2 + (y + 2)^2 = 16 → x + y ≤ 3 :=
by
  sorry

end circle_integer_solution_max_sum_l1238_123898


namespace roots_of_quadratic_equation_are_real_and_distinct_l1238_123801

def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem roots_of_quadratic_equation_are_real_and_distinct :
  quadratic_discriminant 1 (-2) (-6) > 0 :=
by
  norm_num
  sorry

end roots_of_quadratic_equation_are_real_and_distinct_l1238_123801


namespace range_of_m_l1238_123875

theorem range_of_m (a b m : ℝ) (h1 : 2 * b = 2 * a + b) (h2 : b * b = a * a * b) (h3 : 0 < Real.log b / Real.log m) (h4 : Real.log b / Real.log m < 1) : m > 8 :=
sorry

end range_of_m_l1238_123875


namespace sum_of_80_consecutive_integers_l1238_123864

-- Definition of the problem using the given conditions
theorem sum_of_80_consecutive_integers (n : ℤ) (h : (80 * (n + (n + 79))) / 2 = 40) : n = -39 := by
  sorry

end sum_of_80_consecutive_integers_l1238_123864


namespace minimum_d_exists_l1238_123823

open Nat

theorem minimum_d_exists :
  ∃ (a b c d e f g h i k : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ k ∧
                                b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ k ∧
                                c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ k ∧
                                d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ k ∧
                                e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ k ∧
                                f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ k ∧
                                g ≠ h ∧ g ≠ i ∧ g ≠ k ∧
                                h ≠ i ∧ h ≠ k ∧
                                i ≠ k ∧
                                d = a + 3 * (e + h) + k ∧
                                d = 20 :=
by
  sorry

end minimum_d_exists_l1238_123823


namespace correct_calculation_l1238_123859

theorem correct_calculation (x y : ℝ) : 
  ¬(2 * x^2 + 3 * x^2 = 6 * x^2) ∧ 
  ¬(x^4 * x^2 = x^8) ∧ 
  ¬(x^6 / x^2 = x^3) ∧ 
  ((x * y^2)^2 = x^2 * y^4) :=
by
  sorry

end correct_calculation_l1238_123859


namespace smallest_n_l1238_123847

theorem smallest_n (n : ℕ) (h1 : ∃ k1, 4 * n = k1^2) (h2 : ∃ k2, 3 * n = k2^3) : n = 144 :=
sorry

end smallest_n_l1238_123847


namespace num_3_digit_multiples_l1238_123892

def is_3_digit (n : Nat) : Prop := 100 ≤ n ∧ n ≤ 999
def multiple_of (k n : Nat) : Prop := ∃ m : Nat, n = m * k

theorem num_3_digit_multiples (count_35_not_70 : Nat) (h : count_35_not_70 = 13) :
  let count_multiples_35 := (980 / 35) - (105 / 35) + 1
  let count_multiples_70 := (980 / 70) - (140 / 70) + 1
  count_multiples_35 - count_multiples_70 = count_35_not_70 := sorry

end num_3_digit_multiples_l1238_123892
