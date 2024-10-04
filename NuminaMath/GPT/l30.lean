import Mathlib

namespace volume_frustum_correct_l30_30740

noncomputable def volume_of_frustum 
  (base_edge_orig : ℝ) 
  (altitude_orig : ℝ) 
  (base_edge_small : ℝ) 
  (altitude_small : ℝ) : ℝ :=
  let volume_ratio := (base_edge_small / base_edge_orig) ^ 3
  let base_area_orig := (Real.sqrt 3 / 4) * base_edge_orig ^ 2
  let volume_orig := (1 / 3) * base_area_orig * altitude_orig
  let volume_small := volume_ratio * volume_orig
  let volume_frustum := volume_orig - volume_small
  volume_frustum

theorem volume_frustum_correct :
  volume_of_frustum 18 9 9 3 = 212.625 * Real.sqrt 3 :=
sorry

end volume_frustum_correct_l30_30740


namespace determinant_value_l30_30522

theorem determinant_value (t₁ t₂ : ℤ)
    (h₁ : t₁ = 2 * 3 + 3 * 5)
    (h₂ : t₂ = 5) :
    Matrix.det ![
      ![1, -1, t₁],
      ![0, 1, -1],
      ![-1, t₂, -6]
    ] = 14 := by
  rw [h₁, h₂]
  -- Actual proof would go here
  sorry

end determinant_value_l30_30522


namespace missing_number_is_6630_l30_30025

theorem missing_number_is_6630 (x : ℕ) (h : 815472 / x = 123) : x = 6630 :=
by {
  sorry
}

end missing_number_is_6630_l30_30025


namespace cannot_tile_regular_pentagon_l30_30737

theorem cannot_tile_regular_pentagon :
  ¬ (∃ n : ℕ, 360 % (180 - (360 / 5 : ℕ)) = 0) :=
by sorry

end cannot_tile_regular_pentagon_l30_30737


namespace gold_coins_l30_30444

theorem gold_coins (n c : Nat) : 
  n = 9 * (c - 2) → n = 6 * c + 3 → n = 45 :=
by 
  intros h1 h2 
  sorry

end gold_coins_l30_30444


namespace sum_reciprocal_eq_l30_30709

theorem sum_reciprocal_eq :
  ∃ (a b : ℕ), a + b = 45 ∧ Nat.lcm a b = 120 ∧ Nat.gcd a b = 5 ∧ 
  (1/a + 1/b = (3 : ℚ) / 40) := by
  sorry

end sum_reciprocal_eq_l30_30709


namespace find_y_l30_30810

variables (ABC ACB BAC : ℝ)
variables (CDE ADE EAD AED DEB y : ℝ)

-- Conditions
axiom angle_ABC : ABC = 45
axiom angle_ACB : ACB = 90
axiom angle_BAC_eq : BAC = 180 - ABC - ACB
axiom angle_CDE : CDE = 72
axiom angle_ADE_eq : ADE = 180 - CDE
axiom angle_EAD : EAD = 45
axiom angle_AED_eq : AED = 180 - ADE - EAD
axiom angle_DEB_eq : DEB = 180 - AED
axiom y_eq : y = DEB

-- Goal
theorem find_y : y = 153 :=
by {
  -- Here we would proceed with the proof using the established axioms.
  sorry
}

end find_y_l30_30810


namespace spaceship_travel_distance_l30_30493

-- Define each leg of the journey
def distance1 := 0.5
def distance2 := 0.1
def distance3 := 0.1

-- Define the total distance traveled
def total_distance := distance1 + distance2 + distance3

-- The statement to prove
theorem spaceship_travel_distance : total_distance = 0.7 := sorry

end spaceship_travel_distance_l30_30493


namespace average_weight_of_whole_class_l30_30892

theorem average_weight_of_whole_class (n_a n_b : ℕ) (w_a w_b : ℕ) (avg_w_a avg_w_b : ℕ)
  (h_a : n_a = 36) (h_b : n_b = 24) (h_avg_a : avg_w_a = 30) (h_avg_b : avg_w_b = 30) :
  ((n_a * avg_w_a + n_b * avg_w_b) / (n_a + n_b) = 30) := 
by
  sorry

end average_weight_of_whole_class_l30_30892


namespace total_worth_of_presents_l30_30563

-- Definitions of the costs
def costOfRing : ℕ := 4000
def costOfCar : ℕ := 2000
def costOfBracelet : ℕ := 2 * costOfRing

-- Theorem statement
theorem total_worth_of_presents : 
  costOfRing + costOfCar + costOfBracelet = 14000 :=
begin
  -- by using the given definitions and the provided conditions, we assert the statement
  sorry
end

end total_worth_of_presents_l30_30563


namespace regular_polygon_sides_l30_30324

-- Definitions based on conditions in the problem
def exterior_angle (n : ℕ) : ℝ := 360 / n

theorem regular_polygon_sides (n : ℕ) (h : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l30_30324


namespace area_ratio_of_circles_l30_30227

theorem area_ratio_of_circles (R_A R_B : ℝ) 
  (h1 : (60 / 360) * (2 * Real.pi * R_A) = (40 / 360) * (2 * Real.pi * R_B)) :
  (Real.pi * R_A ^ 2) / (Real.pi * R_B ^ 2) = 9 / 4 := 
sorry

end area_ratio_of_circles_l30_30227


namespace norm_two_u_l30_30514

variable {E : Type*} [NormedAddCommGroup E]

theorem norm_two_u (u : E) (h : ∥u∥ = 5) : ∥2 • u∥ = 10 := sorry

end norm_two_u_l30_30514


namespace prob_two_ones_in_twelve_dice_l30_30872

theorem prob_two_ones_in_twelve_dice :
  (∃ p : ℚ, p ≈ 0.230) := 
begin
  let p := (nat.choose 12 2 : ℚ) * (1 / 6)^2 * (5 / 6)^10,
  have h : p ≈ 0.230,
  { sorry }, 
  exact ⟨p, h⟩,
end

end prob_two_ones_in_twelve_dice_l30_30872


namespace ratio_x_y_l30_30174

variable (x y : ℝ)

-- Conditions:
-- 1. lengths of pieces
def is_square (x : ℝ) : Prop := ∃ s, x = 4 * s
def is_pentagon (y : ℝ) : Prop := ∃ t, y = 5 * t
def equal_perimeter (x y : ℝ) : Prop := x = y

-- Theorem to prove
theorem ratio_x_y (hx : is_square x) (hy : is_pentagon y) (h_perimeter : equal_perimeter x y) : x / y = 1 :=
by {
  -- Implementation of the proof
  sorry
}

end ratio_x_y_l30_30174


namespace factor_expression_l30_30923

theorem factor_expression (x : ℝ) : 5 * x * (x - 2) + 9 * (x - 2) = (x - 2) * (5 * x + 9) := 
by 
sorry

end factor_expression_l30_30923


namespace regular_polygon_sides_l30_30323

-- Definitions based on conditions in the problem
def exterior_angle (n : ℕ) : ℝ := 360 / n

theorem regular_polygon_sides (n : ℕ) (h : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l30_30323


namespace consecutive_integers_sum_l30_30672

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l30_30672


namespace parabola_coefficients_l30_30504

theorem parabola_coefficients
    (vertex : (ℝ × ℝ))
    (passes_through : (ℝ × ℝ))
    (vertical_axis_of_symmetry : Prop)
    (hv : vertex = (2, -3))
    (hp : passes_through = (0, 1))
    (has_vertical_axis : vertical_axis_of_symmetry) :
    ∃ a b c : ℝ, ∀ x : ℝ, (x = 0 → (a * x^2 + b * x + c = 1)) ∧ (x = 2 → (a * x^2 + b * x + c = -3)) := sorry

end parabola_coefficients_l30_30504


namespace second_grade_girls_l30_30290

theorem second_grade_girls (G : ℕ) 
  (h1 : ∃ boys_2nd : ℕ, boys_2nd = 20)
  (h2 : ∃ students_3rd : ℕ, students_3rd = 2 * (20 + G))
  (h3 : 20 + G + (2 * (20 + G)) = 93) :
  G = 11 :=
by
  sorry

end second_grade_girls_l30_30290


namespace consecutive_integer_sum_l30_30610

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l30_30610


namespace opposite_event_is_at_least_one_hit_l30_30313

def opposite_event_of_missing_both_times (hit1 hit2 : Prop) : Prop :=
  ¬(¬hit1 ∧ ¬hit2)

theorem opposite_event_is_at_least_one_hit (hit1 hit2 : Prop) :
  opposite_event_of_missing_both_times hit1 hit2 = (hit1 ∨ hit2) :=
by
  sorry

end opposite_event_is_at_least_one_hit_l30_30313


namespace expectation_of_two_fair_dice_l30_30714

noncomputable def E_X : ℝ :=
  (2 * (1/36) + 3 * (2/36) + 4 * (3/36) + 5 * (4/36) + 6 * (5/36) + 7 * (6/36) + 
   8 * (5/36) + 9 * (4/36) + 10 * (3/36) + 11 * (2/36) + 12 * (1/36))

theorem expectation_of_two_fair_dice : E_X = 7 := by
  sorry

end expectation_of_two_fair_dice_l30_30714


namespace big_bottles_sold_percentage_l30_30899

-- Definitions based on conditions
def small_bottles_initial : ℕ := 5000
def big_bottles_initial : ℕ := 12000
def small_bottles_sold_percentage : ℝ := 0.15
def total_bottles_remaining : ℕ := 14090

-- Question in Lean 4
theorem big_bottles_sold_percentage : 
  (12000 - (12000 * x / 100) + 5000 - (5000 * 15 / 100)) = 14090 → x = 18 :=
by
  intros h
  sorry

end big_bottles_sold_percentage_l30_30899


namespace combined_stickers_count_l30_30424

theorem combined_stickers_count :
  let initial_june_stickers := 76
  let initial_bonnie_stickers := 63
  let stickers_given := 25
  let june_total := initial_june_stickers + stickers_given
  let bonnie_total := initial_bonnie_stickers + stickers_given
  june_total + bonnie_total = 189 :=
by
  -- Definitions
  let initial_june_stickers := 76
  let initial_bonnie_stickers := 63
  let stickers_given := 25
  
  -- Calculations
  let june_total := initial_june_stickers + stickers_given
  let bonnie_total := initial_bonnie_stickers + stickers_given

  -- Proof is omitted
  sorry

end combined_stickers_count_l30_30424


namespace correct_integer_with_7_divisors_l30_30179

theorem correct_integer_with_7_divisors (n : ℕ) (p : ℕ) (h_prime : Prime p) 
  (h_3_divisors : ∃ (d : ℕ), d = 3 ∧ n = p^2) : n = 4 :=
by
-- Proof omitted
sorry

end correct_integer_with_7_divisors_l30_30179


namespace problem_l30_30631

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l30_30631


namespace parabola_focus_l30_30353

theorem parabola_focus :
  ∃ f, (∀ x : ℝ, (x^2 + (2*x^2 - f)^2 = (2*x^2 - (-f + 1/4))^2)) ∧ f = 1/8 :=
by
  sorry

end parabola_focus_l30_30353


namespace total_books_count_l30_30050

def Darla_books := 6
def Katie_books := Darla_books / 2
def Darla_Katie_combined_books := Darla_books + Katie_books
def Gary_books := 5 * Darla_Katie_combined_books
def total_books := Darla_books + Katie_books + Gary_books

theorem total_books_count :
  total_books = 54 :=
by
  simp [Darla_books, Katie_books, Darla_Katie_combined_books, Gary_books, total_books]
  sorry

end total_books_count_l30_30050


namespace consecutive_integers_sum_l30_30665

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l30_30665


namespace sum_of_consecutive_integers_with_product_812_l30_30689

theorem sum_of_consecutive_integers_with_product_812 :
  ∃ x : ℕ, (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) := 
begin
  sorry
end

end sum_of_consecutive_integers_with_product_812_l30_30689


namespace sum_of_consecutive_integers_with_product_812_l30_30660

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l30_30660


namespace enrique_shredder_pages_l30_30758

theorem enrique_shredder_pages (total_contracts : ℕ) (num_times : ℕ) (pages_per_time : ℕ) :
  total_contracts = 2132 ∧ num_times = 44 → pages_per_time = 48 :=
by
  intros h
  sorry

end enrique_shredder_pages_l30_30758


namespace find_cost_of_fourth_cd_l30_30712

variables (cost1 cost2 cost3 cost4 : ℕ)
variables (h1 : (cost1 + cost2 + cost3) / 3 = 15)
variables (h2 : (cost1 + cost2 + cost3 + cost4) / 4 = 16)

theorem find_cost_of_fourth_cd : cost4 = 19 := 
by 
  sorry

end find_cost_of_fourth_cd_l30_30712


namespace area_inside_quadrilateral_BCDE_outside_circle_l30_30432

noncomputable def hexagon_area (side_length : ℝ) : ℝ :=
  (3 * Real.sqrt 3) / 2 * side_length ^ 2

noncomputable def circle_area (radius : ℝ) : ℝ :=
  Real.pi * radius ^ 2

theorem area_inside_quadrilateral_BCDE_outside_circle :
  let side_length := 2
  let hex_area := hexagon_area side_length
  let hex_area_large := hexagon_area (2 * side_length)
  let circle_radius := 3
  let circle_area_A := circle_area circle_radius
  let total_area_of_interest := hex_area_large - circle_area_A
  let area_of_one_region := total_area_of_interest / 6
  area_of_one_region = 4 * Real.sqrt 3 - (3 / 2) * Real.pi :=
by
  sorry

end area_inside_quadrilateral_BCDE_outside_circle_l30_30432


namespace maximize_quadratic_function_l30_30467

theorem maximize_quadratic_function (x : ℝ) :
  (∀ x, -2 * x ^ 2 - 8 * x + 18 ≤ 26) ∧ (-2 * (-2) ^ 2 - 8 * (-2) + 18 = 26) :=
by (
  sorry
)

end maximize_quadratic_function_l30_30467


namespace angle_C_measure_l30_30582

theorem angle_C_measure 
  (p q : Prop) 
  (h1 : p) (h2 : q) 
  (A B C : ℝ) 
  (h_parallel : p = q) 
  (h_A_B : A = B / 10) 
  (h_straight_line : B + C = 180) 
  : C = 16.36 := 
sorry

end angle_C_measure_l30_30582


namespace consecutive_integer_product_sum_l30_30647

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l30_30647


namespace negation_abs_val_statement_l30_30133

theorem negation_abs_val_statement (x : ℝ) :
  ¬ (|x| ≤ 3 ∨ |x| > 5) ↔ (|x| > 3 ∧ |x| ≤ 5) :=
by sorry

end negation_abs_val_statement_l30_30133


namespace train_speed_l30_30724

theorem train_speed (D T : ℝ) (h1 : D = 160) (h2 : T = 16) : D / T = 10 :=
by 
  -- given D = 160 and T = 16, we need to prove D / T = 10
  sorry

end train_speed_l30_30724


namespace base_conversion_unique_b_l30_30278

theorem base_conversion_unique_b (b : ℕ) (h_b_pos : 0 < b) :
  (1 * 5^2 + 3 * 5^1 + 2 * 5^0) = (2 * b^2 + b) → b = 4 :=
by
  sorry

end base_conversion_unique_b_l30_30278


namespace arithmetic_sum_24_l30_30553

theorem arithmetic_sum_24 {a : ℕ → ℤ} {d : ℤ} 
  (h_arith_seq : ∀ n : ℕ, a n = a 0 + n * d)
  (h_sum_condition : a 5 + a 10 + a 15 + a 20 = 20) : 
  let S24 := 12 * (a 0 + a 23) in
  S24 = 132 := 
by {
  -- Use h_arith_seq and h_sum_condition to obtain S24
  sorry
}

end arithmetic_sum_24_l30_30553


namespace tennis_tournament_matches_l30_30331

theorem tennis_tournament_matches (num_players : ℕ) (total_days : ℕ) (rest_days : ℕ)
  (num_matches_per_day : ℕ) (matches_per_player : ℕ)
  (h1 : num_players = 10)
  (h2 : total_days = 9)
  (h3 : rest_days = 1)
  (h4 : num_matches_per_day = 5)
  (h5 : matches_per_player = 1)
  : (num_players * (num_players - 1) / 2) - (num_matches_per_day * (total_days - rest_days)) = 40 :=
by
  sorry

end tennis_tournament_matches_l30_30331


namespace remainder_3211_div_103_l30_30144

theorem remainder_3211_div_103 :
  3211 % 103 = 18 :=
by
  sorry

end remainder_3211_div_103_l30_30144


namespace gcd_euclidean_120_168_gcd_subtraction_459_357_l30_30140

theorem gcd_euclidean_120_168 : Nat.gcd 120 168 = 24 := by
  sorry

theorem gcd_subtraction_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_euclidean_120_168_gcd_subtraction_459_357_l30_30140


namespace probability_from_first_to_last_floor_l30_30481

noncomputable def probability_of_open_path (n : ℕ) : ℚ :=
  let totalDoors := 2 * (n - 1)
  let halfDoors := n - 1
  let totalWays := Nat.choose totalDoors halfDoors
  let favorableWays := 2 ^ halfDoors
  favorableWays / totalWays

theorem probability_from_first_to_last_floor (n : ℕ) (h : n > 1) :
  probability_of_open_path n = 2 ^ (n - 1) / (Nat.choose (2 * (n - 1)) (n - 1)) := sorry

end probability_from_first_to_last_floor_l30_30481


namespace problem_statement_l30_30067

theorem problem_statement (x y : ℝ) 
  (hA : A = (x + y) * (y - 3 * x))
  (hB : B = (x - y)^4 / (x - y)^2)
  (hCond : 2 * y + A = B - 6) :
  y = 2 * x^2 - 3 ∧ (y + 3)^2 - 2 * x * (x * y - 3) - 6 * x * (x + 1) = 0 :=
by
  sorry

end problem_statement_l30_30067


namespace sandra_total_beignets_l30_30268

variable (beignets_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ)

def daily_consumption (beignets_per_day : ℕ) := beignets_per_day
def weekly_consumption (beignets_per_day days_per_week : ℕ) := beignets_per_day * days_per_week
def total_consumption (beignets_per_day days_per_week weeks : ℕ) := weekly_consumption beignets_per_day days_per_week * weeks

theorem sandra_total_beignets :
  daily_consumption 3 = 3 →
  days_per_week = 7 →
  weeks = 16 →
  total_consumption 3 7 16 = 336 :=
by
  intros h1 h2 h3
  sorry

end sandra_total_beignets_l30_30268


namespace train_route_l30_30834

-- Definition of letter positions
def letter_position : Char → Nat
| 'A' => 1
| 'B' => 2
| 'K' => 11
| 'L' => 12
| 'U' => 21
| 'V' => 22
| _ => 0

-- Definition of decode function
def decode (s : List Nat) : String :=
match s with
| [21, 2, 12, 21] => "Baku"
| [21, 22, 12, 21] => "Ufa"
| _ => ""

-- Assert encoded strings
def departure_encoded : List Nat := [21, 2, 12, 21]
def arrival_encoded : List Nat := [21, 22, 12, 21]

-- Theorem statement
theorem train_route :
  decode departure_encoded = "Ufa" ∧ decode arrival_encoded = "Baku" :=
by
  sorry

end train_route_l30_30834


namespace chandra_monster_hunt_l30_30187

theorem chandra_monster_hunt :
    let d0 := 2   -- monsters on the first day
    let d1 := 2 * d0   -- monsters on the second day
    let d2 := 2 * d1   -- monsters on the third day
    let d3 := 2 * d2   -- monsters on the fourth day
    let d4 := 2 * d3   -- monsters on the fifth day
in d0 + d1 + d2 + d3 + d4 = 62 := by
  sorry

end chandra_monster_hunt_l30_30187


namespace range_of_m_value_of_m_l30_30945

variables (m p x : ℝ)

-- Conditions: The quadratic equation x^2 - 2x + m - 1 = 0 must have two real roots.
def discriminant (m : ℝ) := (-2)^2 - 4 * 1 * (m - 1)

-- Part 1: Finding the range of values for m
theorem range_of_m (h : discriminant m ≥ 0) : m ≤ 2 := 
by sorry

-- Additional Condition: p is a real root of the equation x^2 - 2x + m - 1 = 0
def is_root (p m : ℝ) := p^2 - 2 * p + m - 1 = 0

-- Another condition: (p^2 - 2p + 3)(m + 4) = 7
def satisfies_condition (p m : ℝ) := (p^2 - 2 * p + 3) * (m + 4) = 7

-- Part 2: Finding the value of m given p is a real root and satisfies (p^2 - 2p + 3)(m + 4) = 7
theorem value_of_m (h1 : is_root p m) (h2 : satisfies_condition p m) : m = -3 := 
by sorry

end range_of_m_value_of_m_l30_30945


namespace combined_percentage_of_students_preferring_tennis_is_39_l30_30498

def total_students_north : ℕ := 1800
def percentage_tennis_north : ℚ := 25 / 100
def total_students_south : ℕ := 3000
def percentage_tennis_south : ℚ := 50 / 100
def total_students_valley : ℕ := 800
def percentage_tennis_valley : ℚ := 30 / 100

def students_prefer_tennis_north : ℚ := total_students_north * percentage_tennis_north
def students_prefer_tennis_south : ℚ := total_students_south * percentage_tennis_south
def students_prefer_tennis_valley : ℚ := total_students_valley * percentage_tennis_valley

def total_students : ℕ := total_students_north + total_students_south + total_students_valley
def total_students_prefer_tennis : ℚ := students_prefer_tennis_north + students_prefer_tennis_south + students_prefer_tennis_valley

def percentage_students_prefer_tennis : ℚ := (total_students_prefer_tennis / total_students) * 100

theorem combined_percentage_of_students_preferring_tennis_is_39 :
  percentage_students_prefer_tennis = 39 := by
  sorry

end combined_percentage_of_students_preferring_tennis_is_39_l30_30498


namespace magnification_factor_is_correct_l30_30744

theorem magnification_factor_is_correct
    (diameter_magnified_image : ℝ)
    (actual_diameter_tissue : ℝ)
    (diameter_magnified_image_eq : diameter_magnified_image = 2)
    (actual_diameter_tissue_eq : actual_diameter_tissue = 0.002) :
  diameter_magnified_image / actual_diameter_tissue = 1000 := by
  -- Theorem and goal statement
  sorry

end magnification_factor_is_correct_l30_30744


namespace frog_weight_difference_l30_30281

theorem frog_weight_difference
  (large_frog_weight : ℕ)
  (small_frog_weight : ℕ)
  (h1 : large_frog_weight = 10 * small_frog_weight)
  (h2 : large_frog_weight = 120) :
  large_frog_weight - small_frog_weight = 108 :=
by
  sorry

end frog_weight_difference_l30_30281


namespace scientific_notation_of_2200_l30_30599

-- Define scientific notation criteria
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  x = a * 10^n ∧ 1 ≤ a ∧ a < 10

-- Problem statement
theorem scientific_notation_of_2200 : ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n 2200 ∧ a = 2.2 ∧ n = 3 :=
by {
  -- Proof can be added here.
  sorry
}

end scientific_notation_of_2200_l30_30599


namespace day_of_week_proof_l30_30095

def day_of_week_17th_2003 := "Wednesday"
def day_of_week_305th_2003 := "Thursday"

theorem day_of_week_proof (d17 : day_of_week_17th_2003 = "Wednesday") : day_of_week_305th_2003 = "Thursday" := 
sorry

end day_of_week_proof_l30_30095


namespace train_length_l30_30494

theorem train_length (L : ℝ) (h1 : L + 110 / 15 = (L + 250) / 20) : L = 310 := 
sorry

end train_length_l30_30494


namespace convex_quadrilateral_probability_l30_30519

noncomputable def probability_convex_quadrilateral (n : ℕ) : ℚ :=
  if n = 6 then (Nat.choose 6 4 : ℚ) / (Nat.choose 15 4 : ℚ) else 0

theorem convex_quadrilateral_probability :
  probability_convex_quadrilateral 6 = 1 / 91 :=
by
  sorry

end convex_quadrilateral_probability_l30_30519


namespace distance_and_area_of_triangle_l30_30768

theorem distance_and_area_of_triangle :
  let p1 := (0, 6)
  let p2 := (8, 0)
  let origin := (0, 0)
  let distance := Real.sqrt ((8 - 0)^2 + (0 - 6)^2)
  let area := (1 / 2 : ℝ) * 8 * 6
  distance = 10 ∧ area = 24 :=
by
  let p1 := (0, 6)
  let p2 := (8, 0)
  let origin := (0, 0)
  let distance := Real.sqrt ((8 - 0)^2 + (0 - 6)^2)
  let area := (1 / 2 : ℝ) * 8 * 6
  have h_dist : distance = 10 := sorry
  have h_area : area = 24 := sorry
  exact ⟨h_dist, h_area⟩

end distance_and_area_of_triangle_l30_30768


namespace probability_exactly_two_ones_l30_30874

theorem probability_exactly_two_ones :
  let n := 12
  let p := 1 / 6
  let q := 5 / 6
  let k := 2
  let binom := @Nat.choose n k
  let prob := binom * (p ^ k) * (q ^ (n - k))
  abs (prob - 0.138) < 0.001 :=
by 
  sorry

end probability_exactly_two_ones_l30_30874


namespace total_acorns_l30_30271

theorem total_acorns (x y : ℝ) :
  let sheila_acorns := 5.3 * x
  let danny_acorns := sheila_acorns + y
  x + sheila_acorns + danny_acorns = 11.6 * x + y :=
by
  sorry

end total_acorns_l30_30271


namespace line_intercepts_and_slope_l30_30851

theorem line_intercepts_and_slope :
  ∀ (x y : ℝ), (4 * x - 5 * y - 20 = 0) → 
  ∃ (x_intercept : ℝ) (y_intercept : ℝ) (slope : ℝ), 
    x_intercept = 5 ∧ y_intercept = -4 ∧ slope = 4 / 5 :=
by
  sorry

end line_intercepts_and_slope_l30_30851


namespace measure_of_C_angle_maximum_area_triangle_l30_30547

-- Proof Problem 1: Measure of angle C
theorem measure_of_C_angle (A B C : ℝ) (a b c : ℝ)
  (h1 : 0 < C ∧ C < Real.pi)
  (m n : ℝ × ℝ)
  (h2 : m = (Real.sin A, Real.sin B))
  (h3 : n = (Real.cos B, Real.cos A))
  (h4 : m.1 * n.1 + m.2 * n.2 = -Real.sin (2 * C)) :
  C = 2 * Real.pi / 3 :=
sorry

-- Proof Problem 2: Maximum area of triangle ABC
theorem maximum_area_triangle (A B C : ℝ) (a b c S : ℝ)
  (h1 : c = 2 * Real.sqrt 3)
  (h2 : Real.cos C = -1 / 2)
  (h3 : S = 1 / 2 * a * b * Real.sin (2 * Real.pi / 3)): 
  S ≤ Real.sqrt 3 :=
sorry

end measure_of_C_angle_maximum_area_triangle_l30_30547


namespace part1_part2_l30_30381

-- Part 1: Prove values of m and n.
theorem part1 (m n : ℝ) :
  (∀ x : ℝ, |x - m| ≤ n ↔ 0 ≤ x ∧ x ≤ 4) → m = 2 ∧ n = 2 :=
by
  intro h
  -- Proof omitted
  sorry

-- Part 2: Prove the minimum value of a + b.
theorem part2 (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a * b = 2) :
  a + b = (2 / a) + (2 / b) → a + b ≥ 2 * Real.sqrt 2 :=
by
  intro h
  -- Proof omitted
  sorry

end part1_part2_l30_30381


namespace sum_first_odd_numbers_not_prime_l30_30235

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_first_odd_numbers_not_prime :
  ¬ (is_prime (1 + 3)) ∧
  ¬ (is_prime (1 + 3 + 5)) ∧
  ¬ (is_prime (1 + 3 + 5 + 7)) ∧
  ¬ (is_prime (1 + 3 + 5 + 7 + 9)) :=
by
  sorry

end sum_first_odd_numbers_not_prime_l30_30235


namespace larger_value_algebraic_expression_is_2_l30_30371

noncomputable def algebraic_expression (a b c d x : ℝ) : ℝ :=
  x^2 + a + b + c * d * x

theorem larger_value_algebraic_expression_is_2
  (a b c d : ℝ) (x : ℝ)
  (h1 : a + b = 0)
  (h2 : c * d = 1)
  (h3 : x = 1 ∨ x = -1) :
  max (algebraic_expression a b c d 1) (algebraic_expression a b c d (-1)) = 2 :=
by
  -- Proof is omitted.
  sorry

end larger_value_algebraic_expression_is_2_l30_30371


namespace find_p_of_five_l30_30310

-- Define the cubic polynomial and the conditions
def cubic_poly (p : ℝ → ℝ) :=
  ∀ x, ∃ a b c d, p x = a * x^3 + b * x^2 + c * x + d

def satisfies_conditions (p : ℝ → ℝ) :=
  p 1 = 1 ^ 2 ∧
  p 2 = 2 ^ 2 ∧
  p 3 = 3 ^ 2 ∧
  p 4 = 4 ^ 2

-- Theorem statement to be proved
theorem find_p_of_five (p : ℝ → ℝ) (hcubic : cubic_poly p) (hconditions : satisfies_conditions p) : p 5 = 25 :=
by
  sorry

end find_p_of_five_l30_30310


namespace find_original_denominator_l30_30332

variable (d : ℕ)

theorem find_original_denominator
  (h1 : ∀ n : ℕ, n = 3)
  (h2 : 3 + 7 = 10)
  (h3 : (10 : ℕ) = 1 * (d + 7) / 3) :
  d = 23 := by
  sorry

end find_original_denominator_l30_30332


namespace fraction_subtraction_l30_30773

theorem fraction_subtraction : (5 / 6 + 1 / 4 - 2 / 3) = (5 / 12) := by
  sorry

end fraction_subtraction_l30_30773


namespace range_of_m_l30_30082

theorem range_of_m (m : ℝ) : (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) → (m ≤ 2 ∧ m ≠ -2) :=
begin
  sorry
end

end range_of_m_l30_30082


namespace dawn_wash_dishes_time_l30_30337

theorem dawn_wash_dishes_time (D : ℕ) : 2 * D + 6 = 46 → D = 20 :=
by
  intro h
  sorry

end dawn_wash_dishes_time_l30_30337


namespace minimum_area_integer_triangle_l30_30478

theorem minimum_area_integer_triangle :
  ∃ (p q : ℤ), p ≠ 0 ∧ q ≠ 0 ∧ (∃ (p q : ℤ), 2 ∣ (16 * p - 30 * q)) 
  → (∃ (area : ℝ), area = (1/2 : ℝ) * |16 * p - 30 * q| ∧ area = 1) :=
by
  sorry

end minimum_area_integer_triangle_l30_30478


namespace equilateral_triangle_vertex_distance_l30_30019

noncomputable def distance_vertex_to_center (l r : ℝ) : ℝ :=
  Real.sqrt (r^2 + (l^2 / 4))

theorem equilateral_triangle_vertex_distance
  (l r : ℝ)
  (h1 : l > 0)
  (h2 : r > 0) :
  distance_vertex_to_center l r = Real.sqrt (r^2 + (l^2 / 4)) :=
sorry

end equilateral_triangle_vertex_distance_l30_30019


namespace consecutive_integers_sum_l30_30668

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l30_30668


namespace tv_selection_l30_30988

theorem tv_selection (A B : ℕ) (hA : A = 4) (hB : B = 5) : 
  ∃ n, n = 3 ∧ (∃ k, k = 70 ∧ 
    (n = 1 ∧ k = A * (B * (B - 1) / 2) + A * (A - 1) / 2 * B)) :=
sorry

end tv_selection_l30_30988


namespace find_m_l30_30202

open Real

def vec := (ℝ × ℝ)

def dot_product (v1 v2 : vec) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def a : vec := (-1, 2)
def b (m : ℝ) : vec := (3, m)
def sum (m : ℝ) : vec := (a.1 + (b m).1, a.2 + (b m).2)

theorem find_m (m : ℝ) (h : dot_product a (sum m) = 0) : m = -1 :=
by {
  sorry
}

end find_m_l30_30202


namespace initial_food_days_l30_30138

theorem initial_food_days (x : ℕ) (h : 760 * (x - 2) = 3040 * 5) : x = 22 := by
  sorry

end initial_food_days_l30_30138


namespace regular_polygon_has_20_sides_l30_30317

-- Definition of a regular polygon with an exterior angle of 18 degrees
def regular_polygon_sides (external_angle : ℝ) : ℕ :=
  if external_angle > 0 then (360 / external_angle).toInt else 0

theorem regular_polygon_has_20_sides :
  regular_polygon_sides 18 = 20 :=
by
  sorry

end regular_polygon_has_20_sides_l30_30317


namespace units_digit_of_odd_product_between_20_130_l30_30149

-- Define the range and the set of odd integers within the specified range.
def odd_ints_between (a b : ℕ) : Set ℕ := { n | a < n ∧ n < b ∧ n % 2 = 1}

-- Define the product of all elements in a set.
def product (s : Set ℕ) : ℕ := s.toFinset.prod id

-- The main theorem to prove that the units digit of our specific product is 5.
theorem units_digit_of_odd_product_between_20_130 : 
  Nat.unitsDigit (product (odd_ints_between 20 130)) = 5 :=
  sorry

end units_digit_of_odd_product_between_20_130_l30_30149


namespace mary_biking_time_l30_30247

-- Define the conditions and the task
def total_time_away := 570 -- in minutes
def time_in_classes := 7 * 45 -- in minutes
def lunch_time := 40 -- in minutes
def additional_activities := 105 -- in minutes
def time_in_school_activities := time_in_classes + lunch_time + additional_activities

-- Define the total biking time based on given conditions
theorem mary_biking_time : 
  total_time_away - time_in_school_activities = 110 :=
by 
-- sorry is used to skip the proof step.
  sorry

end mary_biking_time_l30_30247


namespace find_distance_PQ_of_polar_coords_l30_30241

theorem find_distance_PQ_of_polar_coords (α β : ℝ) (h : β - α = 2 * Real.pi / 3) :
  let P := (5, α)
  let Q := (12, β)
  dist P Q = Real.sqrt 229 :=
by
  sorry

end find_distance_PQ_of_polar_coords_l30_30241


namespace two_point_distribution_p_value_l30_30545

noncomputable def X : Type := ℕ -- discrete random variable (two-point)
def p (E_X2 : ℝ): ℝ := E_X2 -- p == E(X)

theorem two_point_distribution_p_value (var_X : ℝ) (E_X : ℝ) (E_X2 : ℝ) 
    (h1 : var_X = 2 / 9) 
    (h2 : E_X = p E_X2) 
    (h3 : E_X2 = E_X): 
    E_X = 1 / 3 ∨ E_X = 2 / 3 :=
by
  sorry

end two_point_distribution_p_value_l30_30545


namespace exists_irrational_an_l30_30445

theorem exists_irrational_an (a : ℕ → ℝ) (h1 : ∀ n, a n > 0)
  (h2 : ∀ n ≥ 1, a (n + 1)^2 = a n + 1) :
  ∃ n, ¬ ∃ q : ℚ, a n = q :=
sorry

end exists_irrational_an_l30_30445


namespace monkey_climbing_time_l30_30736

theorem monkey_climbing_time (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) (final_hop : ℕ) (net_gain : ℕ) :
  tree_height = 19 →
  hop_distance = 3 →
  slip_distance = 2 →
  net_gain = hop_distance - slip_distance →
  final_hop = hop_distance →
  (tree_height - final_hop) % net_gain = 0 →
  18 / net_gain + 1 = (tree_height - final_hop) / net_gain + 1 := 
by {
  sorry
}

end monkey_climbing_time_l30_30736


namespace lisa_caffeine_l30_30293

theorem lisa_caffeine (caffeine_per_cup : ℕ) (daily_goal : ℕ) (cups_drank : ℕ) : caffeine_per_cup = 80 → daily_goal = 200 → cups_drank = 3 → (caffeine_per_cup * cups_drank - daily_goal) = 40 :=
by
  -- This is a theorem statement, thus no proof is provided here.
  sorry

end lisa_caffeine_l30_30293


namespace find_f_l30_30798

theorem find_f {f : ℝ → ℝ} (h : ∀ x, f (1/x) = x / (1 - x)) : ∀ x, f x = 1 / (x - 1) :=
by
  sorry

end find_f_l30_30798


namespace v3_at_2_is_15_l30_30301

-- Define the polynomial f(x)
def f (x : ℝ) := x^4 + 2*x^3 + x^2 - 3*x - 1

-- Define v3 using Horner's Rule at x
def v3 (x : ℝ) := ((x + 2) * x + 1) * x - 3

-- Prove that v3 at x = 2 equals 15
theorem v3_at_2_is_15 : v3 2 = 15 :=
by
  -- Skipping the proof with sorry
  sorry

end v3_at_2_is_15_l30_30301


namespace largest_frog_weight_difference_l30_30282

def frog_weight_difference (largest_frog_weight : ℕ) (weight_ratio : ℕ) : ℕ :=
  let smallest_frog_weight := largest_frog_weight / weight_ratio
  largest_frog_weight - smallest_frog_weight

theorem largest_frog_weight_difference :
  frog_weight_difference 120 10 = 108 :=
begin
  -- Definitions and conditions have been set.
  -- Proof is not required as per the instructions.
  sorry
end

end largest_frog_weight_difference_l30_30282


namespace solve_for_a_l30_30529

-- Define the line equation and the condition of equal intercepts
def line_eq (a x y : ℝ) : Prop :=
  a * x + y - 2 - a = 0

def equal_intercepts (a : ℝ) : Prop :=
  (∀ x, line_eq a x 0 → x = 2 + a) ∧ (∀ y, line_eq a 0 y → y = 2 + a)

-- State the problem to prove the value of 'a'
theorem solve_for_a (a : ℝ) : equal_intercepts a → (a = -2 ∨ a = 1) :=
by
  sorry

end solve_for_a_l30_30529


namespace consecutive_integers_sum_l30_30615

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l30_30615


namespace domain_of_function_l30_30128

-- Definitions based on conditions
def function_domain (x : ℝ) : Prop := (x > -1) ∧ (x ≠ 1)

-- Prove the domain is the desired set
theorem domain_of_function :
  ∀ x, function_domain x ↔ ((-1 < x ∧ x < 1) ∨ (1 < x)) :=
  by
    sorry

end domain_of_function_l30_30128


namespace sum_of_consecutive_integers_with_product_812_l30_30691

theorem sum_of_consecutive_integers_with_product_812 :
  ∃ x : ℕ, (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) := 
begin
  sorry
end

end sum_of_consecutive_integers_with_product_812_l30_30691


namespace ratio_of_perimeters_l30_30841

theorem ratio_of_perimeters (s1 s2 : ℝ) (h : s1^2 / s2^2 = 16 / 81) : 
    s1 / s2 = 4 / 9 :=
by
  -- Proof goes here
  sorry

end ratio_of_perimeters_l30_30841


namespace largest_s_for_angle_ratio_l30_30819

theorem largest_s_for_angle_ratio (r s : ℕ) (hr : r ≥ 3) (hs : s ≥ 3) (h_angle_ratio : (130 * (r - 2)) * s = (131 * (s - 2)) * r) :
  s ≤ 260 :=
by 
  sorry

end largest_s_for_angle_ratio_l30_30819


namespace convert_spherical_to_cartesian_l30_30366

theorem convert_spherical_to_cartesian :
  let ρ := 5
  let θ₁ := 3 * Real.pi / 4
  let φ₁ := 9 * Real.pi / 5
  let φ' := 2 * Real.pi - φ₁
  let θ' := θ₁ + Real.pi
  ∃ (θ : ℝ) (φ : ℝ),
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    0 ≤ φ ∧ φ ≤ Real.pi ∧
    (∃ (x y z : ℝ),
      x = ρ * Real.sin φ' * Real.cos θ' ∧
      y = ρ * Real.sin φ' * Real.sin θ' ∧
      z = ρ * Real.cos φ') ∧
    θ = θ' ∧ φ = φ' :=
by
  sorry

end convert_spherical_to_cartesian_l30_30366


namespace Tom_runs_60_miles_per_week_l30_30867

theorem Tom_runs_60_miles_per_week
  (days_per_week : ℕ := 5)
  (hours_per_day : ℝ := 1.5)
  (speed_mph : ℝ := 8) :
  (days_per_week * hours_per_day * speed_mph = 60) := by
  sorry

end Tom_runs_60_miles_per_week_l30_30867


namespace exactly_one_passes_l30_30885

theorem exactly_one_passes (P_A P_B : ℚ) (hA : P_A = 3 / 5) (hB : P_B = 1 / 3) : 
  (1 - P_A) * P_B + P_A * (1 - P_B) = 8 / 15 :=
by
  -- skipping the proof as per requirement
  sorry

end exactly_one_passes_l30_30885


namespace sum_of_two_digit_numbers_with_odd_digits_l30_30860

-- Define a two-digit number whose both digits are odd
def isTwoDigitBothOddDigits (n : ℕ) : Prop :=
  n / 10 % 2 = 1 ∧ n % 10 % 2 = 1 ∧ 10 ≤ n ∧ n < 100

-- Define the sum of all two-digit numbers whose digits are both odd
def sumTwoDigitsOdd : ℕ :=
  ∑ n in Finset.range 100, if isTwoDigitBothOddDigits n then n else 0

-- State the theorem that the sum of all two-digit numbers whose digits are both odd is 1375
theorem sum_of_two_digit_numbers_with_odd_digits : sumTwoDigitsOdd = 1375 :=
by
  sorry

end sum_of_two_digit_numbers_with_odd_digits_l30_30860


namespace sum_mod_17_eq_0_l30_30929

theorem sum_mod_17_eq_0 :
  (82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17 = 0 :=
by
  sorry

end sum_mod_17_eq_0_l30_30929


namespace ratio_of_first_to_fourth_term_l30_30751

theorem ratio_of_first_to_fourth_term (a d : ℝ) (h1 : (a + d) + (a + 3 * d) = 6 * a) (h2 : a + 2 * d = 10) :
  a / (a + 3 * d) = 1 / 4 :=
by
  sorry

end ratio_of_first_to_fourth_term_l30_30751


namespace triangle_subsegment_length_l30_30706

noncomputable def length_of_shorter_subsegment (PQ QR PR PS SR : ℝ) :=
  PQ < QR ∧ 
  PR = 15 ∧ 
  PQ / QR = 1 / 5 ∧ 
  PS + SR = PR ∧ 
  PS = PQ / QR * SR → 
  PS = 5 / 2

theorem triangle_subsegment_length (PQ QR PR PS SR : ℝ) 
  (h1 : PQ < QR) 
  (h2 : PR = 15) 
  (h3 : PQ / QR = 1 / 5) 
  (h4 : PS + SR = PR) 
  (h5 : PS = PQ / QR * SR) : 
  length_of_shorter_subsegment PQ QR PR PS SR := 
sorry

end triangle_subsegment_length_l30_30706


namespace other_student_in_sample_18_l30_30482

theorem other_student_in_sample_18 (class_size sample_size : ℕ) (all_students : Finset ℕ) (sample_students : List ℕ)
  (h_class_size : class_size = 60)
  (h_sample_size : sample_size = 4)
  (h_all_students : all_students = Finset.range 60) -- students are numbered from 1 to 60
  (h_sample : sample_students = [3, 33, 48])
  (systematic_sampling : ℕ → ℕ → List ℕ) -- systematic_sampling function that generates the sample based on first element and k
  (k : ℕ) (h_k : k = class_size / sample_size) :
  systematic_sampling 3 k = [3, 18, 33, 48] := 
  sorry

end other_student_in_sample_18_l30_30482


namespace consecutive_integer_product_sum_l30_30648

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l30_30648


namespace soccer_club_girls_count_l30_30037

theorem soccer_club_girls_count
  (total_members : ℕ)
  (attended : ℕ)
  (B G : ℕ)
  (h1 : B + G = 30)
  (h2 : (1/3 : ℚ) * G + B = 18) : G = 18 := by
  sorry

end soccer_club_girls_count_l30_30037


namespace range_of_m_l30_30130

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x > 0 → (y = 1 - 3 * m / x) → y > 0) ↔ (m > 1 / 3) :=
sorry

end range_of_m_l30_30130


namespace sum_of_consecutive_integers_l30_30633

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l30_30633


namespace cards_with_1_count_l30_30365

theorem cards_with_1_count (m k : ℕ) 
  (h1 : k = m + 100) 
  (sum_of_products : (m * (m - 1) / 2) + (k * (k - 1) / 2) - m * k = 1000) : 
  m = 3950 :=
by
  sorry

end cards_with_1_count_l30_30365


namespace no_prime_degree_measure_l30_30434

theorem no_prime_degree_measure :
  ∀ n, 10 ≤ n ∧ n < 20 → ¬ Nat.Prime (180 * (n - 2) / n) :=
by
  intros n h1 h2 
  sorry

end no_prime_degree_measure_l30_30434


namespace kelly_initial_apples_l30_30817

theorem kelly_initial_apples : ∀ (T P I : ℕ), T = 105 → P = 49 → I + P = T → I = 56 :=
by
  intros T P I ht hp h
  rw [ht, hp] at h
  linarith

end kelly_initial_apples_l30_30817


namespace john_total_time_l30_30108

noncomputable def total_time_spent : ℝ :=
  let landscape_pictures := 10
  let landscape_drawing_time := 2
  let landscape_coloring_time := landscape_drawing_time * 0.7
  let landscape_enhancing_time := 0.75
  let total_landscape_time := (landscape_drawing_time + landscape_coloring_time + landscape_enhancing_time) * landscape_pictures
  
  let portrait_pictures := 15
  let portrait_drawing_time := 3
  let portrait_coloring_time := portrait_drawing_time * 0.75
  let portrait_enhancing_time := 1.0
  let total_portrait_time := (portrait_drawing_time + portrait_coloring_time + portrait_enhancing_time) * portrait_pictures
  
  let abstract_pictures := 20
  let abstract_drawing_time := 1.5
  let abstract_coloring_time := abstract_drawing_time * 0.6
  let abstract_enhancing_time := 0.5
  let total_abstract_time := (abstract_drawing_time + abstract_coloring_time + abstract_enhancing_time) * abstract_pictures
  
  total_landscape_time + total_portrait_time + total_abstract_time

theorem john_total_time : total_time_spent = 193.25 :=
by sorry

end john_total_time_l30_30108


namespace combined_work_days_l30_30725

-- Definitions for the conditions
def work_rate (days : ℕ) : ℚ := 1 / days
def combined_work_rate (days_a days_b : ℕ) : ℚ :=
  work_rate days_a + work_rate days_b

-- Theorem to prove
theorem combined_work_days (days_a days_b : ℕ) (ha : days_a = 15) (hb : days_b = 30) :
  1 / (combined_work_rate days_a days_b) = 10 :=
by
  rw [ha, hb]
  sorry

end combined_work_days_l30_30725


namespace largest_non_prime_sum_l30_30886

theorem largest_non_prime_sum (a b n : ℕ) (h1 : a ≥ 1) (h2 : b < 47) (h3 : n = 47 * a + b) (h4 : ∀ b, b < 47 → ¬Nat.Prime b → b = 43) : 
  n = 90 :=
by
  sorry

end largest_non_prime_sum_l30_30886


namespace sum_of_consecutive_integers_with_product_812_l30_30686

theorem sum_of_consecutive_integers_with_product_812 :
  ∃ x : ℕ, (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) := 
begin
  sorry
end

end sum_of_consecutive_integers_with_product_812_l30_30686


namespace consecutive_integer_sum_l30_30605

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l30_30605


namespace sum_of_consecutive_integers_with_product_812_l30_30675

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l30_30675


namespace infinite_primes_dividing_expression_l30_30441

theorem infinite_primes_dividing_expression (k : ℕ) (hk : k > 0) : 
  ∃ᶠ p in Filter.atTop, ∃ n : ℕ, p ∣ (2017^n + k) :=
sorry

end infinite_primes_dividing_expression_l30_30441


namespace coolers_total_capacity_l30_30420

theorem coolers_total_capacity :
  ∃ (C1 C2 C3 : ℕ), 
    C1 = 100 ∧ 
    C2 = C1 + (C1 / 2) ∧ 
    C3 = C2 / 2 ∧ 
    (C1 + C2 + C3 = 325) :=
sorry

end coolers_total_capacity_l30_30420


namespace f_zero_f_odd_f_range_l30_30904

-- Condition 1: The function f is defined on ℝ
-- Condition 2: f(x + y) = f(x) + f(y)
-- Condition 3: f(1/3) = 1
-- Condition 4: f(x) < 0 when x > 0

variables (f : ℝ → ℝ)
axiom f_add : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_third : f (1/3) = 1
axiom f_neg_positive : ∀ x : ℝ, 0 < x → f x < 0

-- Question 1: Find the value of f(0)
theorem f_zero : f 0 = 0 := sorry

-- Question 2: Prove that f is an odd function
theorem f_odd : ∀ x : ℝ, f (-x) = -f x := sorry

-- Question 3: Find the range of x where f(x) + f(2 + x) < 2
theorem f_range : ∀ x : ℝ, f x + f (2 + x) < 2 → -2/3 < x := sorry

end f_zero_f_odd_f_range_l30_30904


namespace solve_a_b_c_d_l30_30910

theorem solve_a_b_c_d (n a b c d : ℕ) (h0 : 0 ≤ a) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : 2^n = a^2 + b^2 + c^2 + d^2) : 
  (a, b, c, d) ∈ {p | p = (↑0, ↑0, ↑0, 2^n.div (↑4)) ∨
                  p = (↑0, ↑0, 2^n.div (↑4), 2^n.div (↑4)) ∨
                  p = (2^n.div (↑4), 2^n.div (↑4), 2^n.div (↑4), 2^n.div (↑4)) ∨
                  p = (2^n.div (↑4), 0, 0, 0) ∨
                  p = (0, 2^n.div (↑4), 0, 0) ∨
                  p = (0, 0, 2^n.div (↑4), 0) ∨
                  p = (0, 0, 0, 2^n.div (↑4))} :=
sorry

end solve_a_b_c_d_l30_30910


namespace consecutive_integers_sum_l30_30613

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l30_30613


namespace total_points_l30_30889

theorem total_points (zach_points ben_points : ℝ) (h₁ : zach_points = 42.0) (h₂ : ben_points = 21.0) : zach_points + ben_points = 63.0 :=
  by sorry

end total_points_l30_30889


namespace students_interested_in_both_l30_30483

def numberOfStudentsInterestedInBoth (T S M N: ℕ) : ℕ := 
  S + M - (T - N)

theorem students_interested_in_both (T S M N: ℕ) (hT : T = 55) (hS : S = 43) (hM : M = 34) (hN : N = 4) : 
  numberOfStudentsInterestedInBoth T S M N = 26 := 
by 
  rw [hT, hS, hM, hN]
  sorry

end students_interested_in_both_l30_30483


namespace janet_pairs_of_2_l30_30549

def total_pairs (x y z : ℕ) : Prop := x + y + z = 18

def total_cost (x y z : ℕ) : Prop := 2 * x + 5 * y + 7 * z = 60

theorem janet_pairs_of_2 (x y z : ℕ) (h1 : total_pairs x y z) (h2 : total_cost x y z) (hz : z = 3) : x = 12 :=
by
  -- Proof is currently skipped
  sorry

end janet_pairs_of_2_l30_30549


namespace max_value_arithmetic_sequence_l30_30552

theorem max_value_arithmetic_sequence
  (a : ℕ → ℝ)
  (a1 d : ℝ)
  (h1 : a 1 = a1)
  (h_diff : ∀ n : ℕ, a (n + 1) = a n + d)
  (ha1_pos : a1 > 0)
  (hd_pos : d > 0)
  (h1_2 : a1 + (a1 + d) ≤ 60)
  (h2_3 : (a1 + d) + (a1 + 2 * d) ≤ 100) :
  5 * a1 + (a1 + 4 * d) ≤ 200 :=
sorry

end max_value_arithmetic_sequence_l30_30552


namespace water_formed_l30_30061

theorem water_formed (CaOH2 CO2 CaCO3 H2O : Nat) 
  (h_balanced : ∀ n, n * CaOH2 + n * CO2 = n * CaCO3 + n * H2O)
  (h_initial : CaOH2 = 2 ∧ CO2 = 2) : 
  H2O = 2 :=
by
  sorry

end water_formed_l30_30061


namespace triangle_equilateral_l30_30256

noncomputable def is_equilateral {R p : ℝ} (A B C : ℝ) : Prop :=
  R * (Real.tan A + Real.tan B + Real.tan C) = 2 * p  →
  ∀ {a b c : ℝ}, a = b ∧ b = c ∧ c = a

theorem triangle_equilateral
  {A B C : ℝ}
  {R p : ℝ}
  (h : R * (Real.tan A + Real.tan B + Real.tan C) = 2 * p) :
  ∀ {a b c : ℝ}, a = b ∧ b = c ∧ c = a :=
sorry

end triangle_equilateral_l30_30256


namespace no_three_distinct_positive_perfect_squares_sum_to_100_l30_30103

theorem no_three_distinct_positive_perfect_squares_sum_to_100 :
  ¬∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ (∃ (m n p : ℕ), a = m^2 ∧ b = n^2 ∧ c = p^2) ∧ a + b + c = 100 :=
by
  sorry

end no_three_distinct_positive_perfect_squares_sum_to_100_l30_30103


namespace find_number_of_rational_roots_l30_30034

noncomputable def number_of_rational_roots (p : Polynomial ℤ) : ℕ := sorry

theorem find_number_of_rational_roots :
  ∀ (b4 b3 b2 b1 : ℤ), (number_of_rational_roots (8 * Polynomial.X ^ 5 
      + b4 * Polynomial.X ^ 4 
      + b3 * Polynomial.X ^ 3 
      + b2 * Polynomial.X ^ 2 
      + b1 * Polynomial.X 
      + 24) = 28) := 
by
  intro b4 b3 b2 b1
  sorry

end find_number_of_rational_roots_l30_30034


namespace vector_sum_is_correct_l30_30532

-- Definitions for vectors a and b
def vector_a := (1, -2)
def vector_b (m : ℝ) := (2, m)

-- Condition for parallel vectors a and b
def parallel_vectors (m : ℝ) : Prop :=
  1 * m - (-2) * 2 = 0

-- Defining the target calculation for given m
def calculate_sum (m : ℝ) : ℝ × ℝ :=
  let a := vector_a
  let b := vector_b m
  (3 * a.1 + 2 * b.1, 3 * a.2 + 2 * b.2)

-- Statement of the theorem to be proved
theorem vector_sum_is_correct (m : ℝ) (h : parallel_vectors m) : calculate_sum m = (7, -14) :=
by sorry

end vector_sum_is_correct_l30_30532


namespace inequality_proof_l30_30982

theorem inequality_proof (a b : ℝ) (h : a + b > 0) : 
  a / b^2 + b / a^2 ≥ 1 / a + 1 / b := 
sorry

end inequality_proof_l30_30982


namespace eccentricity_of_hyperbola_is_e_l30_30214

-- Definitions and given conditions
variable (a b c : ℝ)
variable (h_a_pos : a > 0) (h_b_pos : b > 0)
variable (h_hyperbola : ∀ x y : ℝ, (x^2)/(a^2) - (y^2)/(b^2) = 1)
variable (h_left_focus : ∀ F : ℝ × ℝ, F = (-c, 0))
variable (h_circle : ∀ E : ℝ × ℝ, E.1^2 + E.2^2 = a^2)
variable (h_parabola : ∀ P : ℝ × ℝ, P.2^2 = 4*c*P.1)
variable (h_midpoint : ∀ E P F : ℝ × ℝ, E = (F.1 + P.1) / 2 ∧ E.2 = (F.2 + P.2) / 2)

-- The statement to be proved
theorem eccentricity_of_hyperbola_is_e :
    ∃ e : ℝ, e = (Real.sqrt 5 + 1) / 2 :=
sorry

end eccentricity_of_hyperbola_is_e_l30_30214


namespace batsman_average_after_17th_inning_l30_30152

theorem batsman_average_after_17th_inning :
  ∃ x : ℤ, (63 + (16 * x) = 17 * (x + 3)) ∧ (x + 3 = 17) :=
by
  sorry

end batsman_average_after_17th_inning_l30_30152


namespace mixed_number_sum_l30_30156

theorem mixed_number_sum : 
  (4/5 + 9 * 4/5 + 99 * 4/5 + 999 * 4/5 + 9999 * 4/5 + 1 = 11111) := by
  sorry

end mixed_number_sum_l30_30156


namespace probability_of_two_approvals_in_four_l30_30960

-- Conditions
def prob_approval : ℝ := 0.6
def prob_disapproval : ℝ := 1 - prob_approval

-- Proof statement
theorem probability_of_two_approvals_in_four :
  (4.choose 2) * (prob_approval^2 * prob_disapproval^2) = 0.3456 :=
by
  sorry

end probability_of_two_approvals_in_four_l30_30960


namespace incorrect_options_l30_30222

variable (a b : ℚ) (h : a / b = 5 / 6)

theorem incorrect_options :
  (2 * a - b ≠ b * 6 / 4) ∧
  (a + 3 * b ≠ 2 * a * 19 / 10) :=
by
  sorry

end incorrect_options_l30_30222


namespace sum_of_reciprocals_eq_two_l30_30862

theorem sum_of_reciprocals_eq_two (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 20) : 1 / x + 1 / y = 2 := by
  sorry

end sum_of_reciprocals_eq_two_l30_30862


namespace pos_int_solutions_3x_2y_841_l30_30451

theorem pos_int_solutions_3x_2y_841 :
  {n : ℕ // ∃ (x y : ℕ), 3 * x + 2 * y = 841 ∧ x > 0 ∧ y > 0} =
  {n : ℕ // n = 140} := 
sorry

end pos_int_solutions_3x_2y_841_l30_30451


namespace ratio_of_perimeters_l30_30840

theorem ratio_of_perimeters (s1 s2 : ℝ) (h : s1^2 / s2^2 = 16 / 81) : 
    s1 / s2 = 4 / 9 :=
by
  -- Proof goes here
  sorry

end ratio_of_perimeters_l30_30840


namespace find_f_l30_30797

theorem find_f {f : ℝ → ℝ} (h : ∀ x, f (1/x) = x / (1 - x)) : ∀ x, f x = 1 / (x - 1) :=
by
  sorry

end find_f_l30_30797


namespace num_squares_less_than_1000_with_ones_digit_2_3_or_4_l30_30219

-- Define a function that checks if the one's digit of a number is one of 2, 3, or 4.
def ends_in (n : ℕ) (d : ℕ) : Prop := n % 10 = d

-- Define the main theorem to prove
theorem num_squares_less_than_1000_with_ones_digit_2_3_or_4 : 
  ∃ n, n = 6 ∧ ∀ m < 1000, ∃ k, m = k^2 → ends_in m 2 ∨ ends_in m 3 ∨ ends_in m 4 :=
sorry

end num_squares_less_than_1000_with_ones_digit_2_3_or_4_l30_30219


namespace problem1_solution_problem2_solution_l30_30461

-- Conditions for Problem 1
def problem1_condition (x : ℝ) : Prop := 
  5 * (x - 20) + 2 * x = 600

-- Proof for Problem 1 Goal
theorem problem1_solution (x : ℝ) (h : problem1_condition x) : x = 100 := 
by sorry

-- Conditions for Problem 2
def problem2_condition (m : ℝ) : Prop :=
  (360 / m) + (540 / (1.2 * m)) = (900 / 100)

-- Proof for Problem 2 Goal
theorem problem2_solution (m : ℝ) (h : problem2_condition m) : m = 90 := 
by sorry

end problem1_solution_problem2_solution_l30_30461


namespace fernandez_family_children_l30_30839

-- Conditions definition
variables (m : ℕ) -- age of the mother
variables (x : ℕ) -- number of children
variables (y : ℕ) -- average age of the children

-- Given conditions
def average_age_family (m : ℕ) (x : ℕ) (y : ℕ) : Prop :=
  (m + 50 + 70 + x * y) / (3 + x) = 25

def average_age_mother_children (m : ℕ) (x : ℕ) (y : ℕ) : Prop :=
  (m + x * y) / (1 + x) = 18

-- Goal statement
theorem fernandez_family_children
  (m : ℕ) (x : ℕ) (y : ℕ)
  (h1 : average_age_family m x y)
  (h2 : average_age_mother_children m x y) :
  x = 9 :=
sorry

end fernandez_family_children_l30_30839


namespace coefficient_x_squared_in_expansion_l30_30846

theorem coefficient_x_squared_in_expansion :
  (∃ c : ℤ, (1 + x)^6 * (1 - x) = c * x^2 + b * x + a) → c = 9 :=
by
  sorry

end coefficient_x_squared_in_expansion_l30_30846


namespace Riku_stickers_more_times_l30_30257

theorem Riku_stickers_more_times (Kristoff_stickers Riku_stickers : ℕ) 
  (h1 : Kristoff_stickers = 85) (h2 : Riku_stickers = 2210) : 
  Riku_stickers / Kristoff_stickers = 26 := 
by
  sorry

end Riku_stickers_more_times_l30_30257


namespace consecutive_integers_sum_l30_30699

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l30_30699


namespace ratio_x_y_l30_30172

variable (x y : ℝ)

-- Conditions:
-- 1. lengths of pieces
def is_square (x : ℝ) : Prop := ∃ s, x = 4 * s
def is_pentagon (y : ℝ) : Prop := ∃ t, y = 5 * t
def equal_perimeter (x y : ℝ) : Prop := x = y

-- Theorem to prove
theorem ratio_x_y (hx : is_square x) (hy : is_pentagon y) (h_perimeter : equal_perimeter x y) : x / y = 1 :=
by {
  -- Implementation of the proof
  sorry
}

end ratio_x_y_l30_30172


namespace gcd_660_924_l30_30717

theorem gcd_660_924 : Nat.gcd 660 924 = 132 := by
  sorry

end gcd_660_924_l30_30717


namespace combined_distance_l30_30813

-- Definitions based on the conditions
def JulienDailyDistance := 50
def SarahDailyDistance := 2 * JulienDailyDistance
def JamirDailyDistance := SarahDailyDistance + 20
def Days := 7

-- Combined weekly distances
def JulienWeeklyDistance := JulienDailyDistance * Days
def SarahWeeklyDistance := SarahDailyDistance * Days
def JamirWeeklyDistance := JamirDailyDistance * Days

-- Theorem statement with the combined distance
theorem combined_distance :
  JulienWeeklyDistance + SarahWeeklyDistance + JamirWeeklyDistance = 1890 := by
  sorry

end combined_distance_l30_30813


namespace angle_XYZ_of_excircle_circumcircle_incircle_l30_30299

theorem angle_XYZ_of_excircle_circumcircle_incircle 
  (a b c x y z : ℝ) 
  (hA : a = 50)
  (hB : b = 70)
  (hC : c = 60) 
  (triangleABC : a + b + c = 180) 
  (excircle_Omega : Prop) 
  (incircle_Gamma : Prop) 
  (circumcircle_Omega_triangleXYZ : Prop) 
  (X_on_BC : Prop)
  (Y_on_AB : Prop) 
  (Z_on_CA : Prop): 
  x = 115 := 
by 
  sorry

end angle_XYZ_of_excircle_circumcircle_incircle_l30_30299


namespace worth_of_presents_l30_30567

def ring_value := 4000
def car_value := 2000
def bracelet_value := 2 * ring_value

def total_worth := ring_value + car_value + bracelet_value

theorem worth_of_presents : total_worth = 14000 := by
  sorry

end worth_of_presents_l30_30567


namespace asymptotes_of_hyperbola_l30_30596

theorem asymptotes_of_hyperbola : 
  ∀ (x y : ℝ), 9 * y^2 - 25 * x^2 = 169 → (y = (5/3) * x ∨ y = -(5/3) * x) :=
by 
  sorry

end asymptotes_of_hyperbola_l30_30596


namespace p_at_5_l30_30163

noncomputable def p (x : ℝ) : ℝ :=
  sorry

def p_cond (n : ℝ) : Prop :=
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) → p n = 1 / n^3

theorem p_at_5 : (∀ n, p_cond n) → p 5 = -149 / 1500 :=
by
  intros
  sorry

end p_at_5_l30_30163


namespace probability_x_plus_y_lt_4_l30_30315

open MeasureTheory

-- Define the vertices of the square
def square : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.1 ≤ 3 ∧ p.2 ≥ 0 ∧ p.2 ≤ 3}

-- Define the predicate x + y < 4
def condition (p : ℝ × ℝ) : Prop := p.1 + p.2 < 4

-- Define the probability measure uniform over the square
noncomputable def uniform_square : Measure (ℝ × ℝ) :=
  MeasureTheory.Measure.Uniform (Icc (0, 0) (3, 3))

-- Define the probability of the condition x + y < 4
noncomputable def prob_condition : ennreal :=
  MeasureTheory.MeasureTheory.toOuterMeasure uniform_square {p | condition p} / 
  MeasureTheory.MeasureTheory.toOuterMeasure uniform_square square

-- Statement to prove
theorem probability_x_plus_y_lt_4 : prob_condition = (7 / 9 : ℝ) :=
  sorry

end probability_x_plus_y_lt_4_l30_30315


namespace molecular_weight_C4H10_l30_30020

theorem molecular_weight_C4H10 (molecular_weight_six_moles : ℝ) (h : molecular_weight_six_moles = 390) :
  molecular_weight_six_moles / 6 = 65 :=
by
  -- proof to be filled in here
  sorry

end molecular_weight_C4H10_l30_30020


namespace angle_bisector_median_ineq_l30_30433

variables {A B C : Type} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variables (l_a l_b l_c m_a m_b m_c : ℝ)

theorem angle_bisector_median_ineq
  (hl_a : l_a > 0) (hl_b : l_b > 0) (hl_c : l_c > 0)
  (hm_a : m_a > 0) (hm_b : m_b > 0) (hm_c : m_c > 0) :
  l_a / m_a + l_b / m_b + l_c / m_c > 1 :=
sorry

end angle_bisector_median_ineq_l30_30433


namespace fractional_eq_range_m_l30_30083

theorem fractional_eq_range_m (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) → m ≤ 2 ∧ m ≠ -2 :=
by
  sorry

end fractional_eq_range_m_l30_30083


namespace remainder_a_cubed_l30_30573

theorem remainder_a_cubed {a n : ℤ} (hn : 0 < n) (hinv : a * a ≡ 1 [ZMOD n]) (ha : a ≡ -1 [ZMOD n]) : a^3 ≡ -1 [ZMOD n] := 
sorry

end remainder_a_cubed_l30_30573


namespace lana_total_pages_l30_30112

theorem lana_total_pages (lana_initial_pages : ℕ) (duane_total_pages : ℕ) :
  lana_initial_pages = 8 ∧ duane_total_pages = 42 →
  (lana_initial_pages + duane_total_pages / 2) = 29 :=
by
  sorry

end lana_total_pages_l30_30112


namespace probability_five_cards_one_from_each_suit_and_extra_l30_30093

/--
Given five cards chosen with replacement from a standard 52-card deck, 
the probability of having exactly one card from each suit, plus one 
additional card from any suit, is 3/32.
-/
theorem probability_five_cards_one_from_each_suit_and_extra 
  (cards : ℕ) (total_suits : ℕ)
  (prob_first_diff_suit : ℚ) 
  (prob_second_diff_suit : ℚ) 
  (prob_third_diff_suit : ℚ) 
  (prob_fourth_diff_suit : ℚ) 
  (prob_any_suit : ℚ) 
  (total_prob : ℚ) :
  cards = 5 ∧ total_suits = 4 ∧ 
  prob_first_diff_suit = 3 / 4 ∧ 
  prob_second_diff_suit = 1 / 2 ∧ 
  prob_third_diff_suit = 1 / 4 ∧ 
  prob_fourth_diff_suit = 1 ∧ 
  prob_any_suit = 1 →
  total_prob = 3 / 32 :=
by {
  sorry
}

end probability_five_cards_one_from_each_suit_and_extra_l30_30093


namespace celine_erasers_collected_l30_30186

theorem celine_erasers_collected (G C J E : ℕ) 
    (hC : C = 2 * G)
    (hJ : J = 4 * G)
    (hE : E = 12 * G)
    (h_total : G + C + J + E = 151) : 
    C = 16 := 
by 
  -- Proof steps skipped, proof body not required as per instructions
  sorry

end celine_erasers_collected_l30_30186


namespace sum_of_consecutive_integers_l30_30635

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l30_30635


namespace shelves_needed_l30_30830

def total_books : ℕ := 46
def books_taken_by_librarian : ℕ := 10
def books_per_shelf : ℕ := 4
def remaining_books : ℕ := total_books - books_taken_by_librarian
def needed_shelves : ℕ := 9

theorem shelves_needed : remaining_books / books_per_shelf = needed_shelves := by
  sorry

end shelves_needed_l30_30830


namespace trim_length_l30_30594

theorem trim_length {π : ℝ} (r : ℝ)
  (π_approx : π = 22 / 7)
  (area : π * r^2 = 616) :
  2 * π * r + 5 = 93 :=
by
  sorry

end trim_length_l30_30594


namespace probability_at_least_one_woman_selected_l30_30392

open Classical

noncomputable def probability_of_selecting_at_least_one_woman : ℚ :=
  1 - (10 / 15) * (9 / 14) * (8 / 13) * (7 / 12) * (6 / 11)

theorem probability_at_least_one_woman_selected :
  probability_of_selecting_at_least_one_woman = 917 / 1001 :=
sorry

end probability_at_least_one_woman_selected_l30_30392


namespace cookies_baked_total_l30_30385

   -- Definitions based on the problem conditions
   def cookies_yesterday : ℕ := 435
   def cookies_this_morning : ℕ := 139

   -- The theorem we want to prove
   theorem cookies_baked_total : cookies_yesterday + cookies_this_morning = 574 :=
   by sorry
   
end cookies_baked_total_l30_30385


namespace expand_and_simplify_product_l30_30196

theorem expand_and_simplify_product :
  5 * (x + 6) * (x + 2) * (x + 7) = 5 * x^3 + 75 * x^2 + 340 * x + 420 := 
by
  sorry

end expand_and_simplify_product_l30_30196


namespace sum_of_consecutive_integers_l30_30640

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l30_30640


namespace correct_expression_l30_30469

theorem correct_expression :
  ¬ (|4| = -4) ∧
  ¬ (|4| = -4) ∧
  (-(4^2) ≠ 16)  ∧
  ((-4)^2 = 16) := by
  sorry

end correct_expression_l30_30469


namespace inequality_am_gm_l30_30245

theorem inequality_am_gm (a b : ℝ) (h₀ : 0 < a) (h₁ : a < 1) (h₂ : 0 < b) (h₃ : b < 1) :
  1 + a + b > 3 * Real.sqrt (a * b) :=
by
  sorry

end inequality_am_gm_l30_30245


namespace operation_eval_l30_30580

def my_operation (a b : ℤ) := a * (b + 2) + a * (b + 1)

theorem operation_eval : my_operation 3 (-1) = 3 := by
  sorry

end operation_eval_l30_30580


namespace cory_packs_l30_30507

theorem cory_packs (total_money_needed cost_per_pack : ℕ) (h1 : total_money_needed = 98) (h2 : cost_per_pack = 49) : total_money_needed / cost_per_pack = 2 :=
by 
  sorry

end cory_packs_l30_30507


namespace anna_ate_cupcakes_l30_30496

-- Given conditions
def total_cupcakes : Nat := 60
def cupcakes_given_away (total : Nat) : Nat := (4 * total) / 5
def cupcakes_remaining (total : Nat) : Nat := total - cupcakes_given_away total
def anna_cupcakes_left : Nat := 9

-- Proving the number of cupcakes Anna ate
theorem anna_ate_cupcakes : cupcakes_remaining total_cupcakes - anna_cupcakes_left = 3 := by
  sorry

end anna_ate_cupcakes_l30_30496


namespace mr_lee_broke_even_l30_30251

theorem mr_lee_broke_even (sp1 sp2 : ℝ) (p1_loss2 : ℝ) (c1 c2 : ℝ) (h1 : sp1 = 1.50) (h2 : sp2 = 1.50) 
    (h3 : c1 = sp1 / 1.25) (h4 : c2 = sp2 / 0.8333) (h5 : p1_loss2 = (sp1 - c1) + (sp2 - c2)) : 
  p1_loss2 = 0 :=
by 
  sorry

end mr_lee_broke_even_l30_30251


namespace chimney_bricks_l30_30500

variable (h : ℕ)

/-- Brenda would take 8 hours to build a chimney alone. 
    Brandon would take 12 hours to build it alone. 
    When they work together, their efficiency is diminished by 15 bricks per hour due to their chatting. 
    If they complete the chimney in 6 hours when working together, then the total number of bricks in the chimney is 360. -/
theorem chimney_bricks
  (h : ℕ)
  (Brenda_rate : ℕ)
  (Brandon_rate : ℕ)
  (effective_rate : ℕ)
  (completion_time : ℕ)
  (h_eq : Brenda_rate = h / 8)
  (h_eq_alt : Brandon_rate = h / 12)
  (effective_rate_eq : effective_rate = (Brenda_rate + Brandon_rate) - 15)
  (completion_eq : 6 * effective_rate = h) :
  h = 360 := by 
  sorry

end chimney_bricks_l30_30500


namespace monotonic_intervals_range_of_a_min_value_of_c_l30_30086

noncomputable def f (a c x : ℝ) : ℝ :=
  a * Real.log x + (x - c) * abs (x - c)

-- 1. Monotonic intervals
theorem monotonic_intervals (a c : ℝ) (ha : a = -3 / 4) (hc : c = 1 / 4) :
  ((∀ x, 0 < x ∧ x < 3 / 4 → f a c x > f a c (x - 1)) ∧ (∀ x, 3 / 4 < x → f a c x > f a c (x - 1))) :=
sorry

-- 2. Range of values for a
theorem range_of_a (a c : ℝ) (hc : c = a / 2 + 1) (h : ∀ x > c, f a c x ≥ 1 / 4) :
  -2 < a ∧ a ≤ -1 :=
sorry

-- 3. Minimum value of c
theorem min_value_of_c (a c x1 x2 : ℝ) (hx1 : x1 = Real.sqrt (-a / 2)) (hx2 : x2 = c)
  (h_tangents_perpendicular : f a c x1 * f a c x2 = -1) :
  c = 3 * Real.sqrt 3 / 2 :=
sorry

end monotonic_intervals_range_of_a_min_value_of_c_l30_30086


namespace james_savings_l30_30560

-- Define the conditions
def cost_vest : ℝ := 250
def weight_plates_pounds : ℕ := 200
def cost_per_pound : ℝ := 1.2
def original_weight_vest_cost : ℝ := 700
def discount : ℝ := 100

-- Define the derived quantities based on conditions
def cost_weight_plates : ℝ := weight_plates_pounds * cost_per_pound
def total_cost_setup : ℝ := cost_vest + cost_weight_plates
def discounted_weight_vest_cost : ℝ := original_weight_vest_cost - discount
def savings : ℝ := discounted_weight_vest_cost - total_cost_setup

-- The statement to prove the savings
theorem james_savings : savings = 110 := by
  sorry

end james_savings_l30_30560


namespace consecutive_integers_sum_l30_30669

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l30_30669


namespace number_of_planks_needed_l30_30748

-- Definitions based on conditions
def bed_height : ℕ := 2
def bed_width : ℕ := 2
def bed_length : ℕ := 8
def plank_width : ℕ := 1
def lumber_length : ℕ := 8
def num_beds : ℕ := 10

-- The theorem statement
theorem number_of_planks_needed : (2 * (bed_length / lumber_length) * bed_height) + (2 * ((bed_width * bed_height) / lumber_length) * lumber_length / 4) * num_beds = 60 :=
  by sorry

end number_of_planks_needed_l30_30748


namespace Tom_runs_60_miles_in_a_week_l30_30866

variable (days_per_week : ℕ) (hours_per_day : ℝ) (speed : ℝ)
variable (h_days_per_week : days_per_week = 5)
variable (h_hours_per_day : hours_per_day = 1.5)
variable (h_speed : speed = 8)

theorem Tom_runs_60_miles_in_a_week : (days_per_week * hours_per_day * speed) = 60 := by
  sorry

end Tom_runs_60_miles_in_a_week_l30_30866


namespace no_positive_integers_satisfy_l30_30440

theorem no_positive_integers_satisfy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : 
  x^5 + y^5 + 1 ≠ (x + 2)^5 + (y - 3)^5 :=
sorry

end no_positive_integers_satisfy_l30_30440


namespace pos_sum_inequality_l30_30374

theorem pos_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) ≥ (a + b + c) / 2 := 
sorry

end pos_sum_inequality_l30_30374


namespace avg_pages_hr_difference_l30_30113

noncomputable def avg_pages_hr_diff (total_pages_ryan : ℕ) (hours_ryan : ℕ) (books_brother : ℕ) (pages_per_book : ℕ) (hours_brother : ℕ) : ℚ :=
  (total_pages_ryan / hours_ryan : ℚ) - (books_brother * pages_per_book / hours_brother : ℚ)

theorem avg_pages_hr_difference :
  avg_pages_hr_diff 4200 78 15 250 90 = 12.18 :=
by
  sorry

end avg_pages_hr_difference_l30_30113


namespace number_of_books_is_10_l30_30548

def costPerBookBeforeDiscount : ℝ := 5
def discountPerBook : ℝ := 0.5
def totalPayment : ℝ := 45

theorem number_of_books_is_10 (n : ℕ) (h : (costPerBookBeforeDiscount - discountPerBook) * n = totalPayment) : n = 10 := by
  sorry

end number_of_books_is_10_l30_30548


namespace percentage_difference_l30_30198

open scoped Classical

theorem percentage_difference (original_number new_number : ℕ) (h₀ : original_number = 60) (h₁ : new_number = 30) :
  (original_number - new_number) / original_number * 100 = 50 :=
by
      sorry

end percentage_difference_l30_30198


namespace prec_property_l30_30893

noncomputable def prec (a b : ℕ) : Prop :=
  sorry -- The construction of the relation from the problem

axiom prec_total : ∀ a b : ℕ, (prec a b ∨ prec b a ∨ a = b)
axiom prec_trans : ∀ a b c : ℕ, (prec a b ∧ prec b c) → prec a c

theorem prec_property : ∀ a b c : ℕ, (prec a b ∧ prec b c) → 2 * b ≠ a + c :=
by
  sorry

end prec_property_l30_30893


namespace amount_lent_to_B_l30_30487

theorem amount_lent_to_B
  (rate_of_interest_per_annum : ℝ)
  (P_C : ℝ)
  (years_C : ℝ)
  (total_interest : ℝ)
  (years_B : ℝ)
  (IB : ℝ)
  (IC : ℝ)
  (P_B : ℝ):
  (rate_of_interest_per_annum = 10) →
  (P_C = 3000) →
  (years_C = 4) →
  (total_interest = 2200) →
  (years_B = 2) →
  (IC = (P_C * rate_of_interest_per_annum * years_C) / 100) →
  (IB = (P_B * rate_of_interest_per_annum * years_B) / 100) →
  (total_interest = IB + IC) →
  P_B = 5000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end amount_lent_to_B_l30_30487


namespace aunt_may_milk_leftover_l30_30045

noncomputable def milk_leftover : Real :=
let morning_milk := 5 * 13 + 4 * 0.5 + 10 * 0.25
let evening_milk := 5 * 14 + 4 * 0.6 + 10 * 0.2

let morning_spoiled := morning_milk * 0.1
let cheese_produced := morning_milk * 0.15
let remaining_morning_milk := morning_milk - morning_spoiled - cheese_produced
let ice_cream_sale := remaining_morning_milk * 0.7

let evening_spoiled := evening_milk * 0.05
let remaining_evening_milk := evening_milk - evening_spoiled
let cheese_shop_sale := remaining_evening_milk * 0.8

let leftover_previous_day := 15
let remaining_morning_after_sale := remaining_morning_milk - ice_cream_sale
let remaining_evening_after_sale := remaining_evening_milk - cheese_shop_sale

leftover_previous_day + remaining_morning_after_sale + remaining_evening_after_sale

theorem aunt_may_milk_leftover : 
  milk_leftover = 44.7735 := 
sorry

end aunt_may_milk_leftover_l30_30045


namespace arithmetic_example_l30_30184

theorem arithmetic_example : 15 * 30 + 45 * 15 = 1125 := by
  sorry

end arithmetic_example_l30_30184


namespace area_increase_percentage_area_percentage_increase_length_to_width_ratio_l30_30852

open Real

-- Part (a)
theorem area_increase_percentage (a b : ℝ) :
  (1.12 * a) * (1.15 * b) = 1.288 * (a * b) :=
  sorry

theorem area_percentage_increase (a b : ℝ) :
  ((1.12 * a) * (1.15 * b)) / (a * b) = 1.288 :=
  sorry

-- Part (b)
theorem length_to_width_ratio (a b : ℝ) (h : 2 * ((1.12 * a) + (1.15 * b)) = 1.13 * 2 * (a + b)) :
  a = 2 * b :=
  sorry

end area_increase_percentage_area_percentage_increase_length_to_width_ratio_l30_30852


namespace total_pennies_donated_l30_30501

theorem total_pennies_donated:
  let cassandra_pennies := 5000
  let james_pennies := cassandra_pennies - 276
  let stephanie_pennies := 2 * james_pennies
  cassandra_pennies + james_pennies + stephanie_pennies = 19172 :=
by
  sorry

end total_pennies_donated_l30_30501


namespace min_max_of_f_l30_30346

def f (x : ℝ) : ℝ := -2 * x + 1

-- defining the minimum and maximum values
def min_val : ℝ := -3
def max_val : ℝ := 5

theorem min_max_of_f :
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f x ≥ min_val) ∧ 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f x ≤ max_val) :=
by 
  sorry

end min_max_of_f_l30_30346


namespace sqrt_and_cbrt_eq_self_l30_30234

theorem sqrt_and_cbrt_eq_self (x : ℝ) (h1 : x = Real.sqrt x) (h2 : x = x^(1/3)) : x = 0 := by
  sorry

end sqrt_and_cbrt_eq_self_l30_30234


namespace quadratic_two_distinct_real_roots_l30_30211

theorem quadratic_two_distinct_real_roots (a : ℝ) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a * x1^2 + 2 * x1 - 3 = 0) ∧ (a * x2^2 + 2 * x2 - 3 = 0)) ↔ a > -1 / 3 := by
  sorry

end quadratic_two_distinct_real_roots_l30_30211


namespace car_rental_cost_l30_30583

theorem car_rental_cost (D R M P C : ℝ) (hD : D = 5) (hR : R = 30) (hM : M = 500) (hP : P = 0.25) 
(hC : C = (R * D) + (P * M)) : C = 275 :=
by
  rw [hD, hR, hM, hP] at hC
  sorry

end car_rental_cost_l30_30583


namespace max_perimeter_of_triangle_l30_30100

theorem max_perimeter_of_triangle (A B C a b c p : ℝ) 
  (h_angle_A : A = 2 * Real.pi / 3)
  (h_a : a = 3)
  (h_perimeter : p = a + b + c) 
  (h_sine_law : b = 2 * Real.sqrt 3 * Real.sin B ∧ c = 2 * Real.sqrt 3 * Real.sin C) :
  p ≤ 3 + 2 * Real.sqrt 3 :=
sorry

end max_perimeter_of_triangle_l30_30100


namespace red_envelope_probability_l30_30912

def wechat_red_envelope_prob {A B : Type} [DecidableEq A] [DecidableEq B]
  (total_amount : ℝ) (distribution : Finset ℝ) (num_people : ℕ)
  (amount_snatched_by_A_and_B : Finset (ℝ × ℝ)) : Prop :=
  total_amount = 8 ∧
  distribution = {1.72, 1.83, 2.28, 1.55, 0.62} ∧
  num_people = 5 ∧
  amount_snatched_by_A_and_B.filter (λ (x : ℝ × ℝ), x.1 + x.2 ≥ 3).card = 6

theorem red_envelope_probability :
  wechat_red_envelope_prob 8 {1.72, 1.83, 2.28, 1.55, 0.62} 5 { (1.72, 1.83), (1.72, 2.28), (1.72, 1.55), (1.83, 2.28), (1.83, 1.55), (2.28, 1.55) } →
  (6 / 10 : ℝ) = 3 / 5 :=
by
  sorry

end red_envelope_probability_l30_30912


namespace sum_of_consecutive_integers_with_product_812_l30_30688

theorem sum_of_consecutive_integers_with_product_812 :
  ∃ x : ℕ, (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) := 
begin
  sorry
end

end sum_of_consecutive_integers_with_product_812_l30_30688


namespace find_a_l30_30942

theorem find_a 
  (x : ℤ) 
  (a : ℤ) 
  (h1 : x = 2) 
  (h2 : y = a) 
  (h3 : 2 * x - 3 * y = 5) : a = -1 / 3 := 
by 
  sorry

end find_a_l30_30942


namespace evaluate_expression_l30_30916

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end evaluate_expression_l30_30916


namespace triangle_vertices_l30_30154

theorem triangle_vertices : 
  (∃ (x y : ℚ), 2 * x + y = 6 ∧ x - y = -4 ∧ x = 2 / 3 ∧ y = 14 / 3) ∧ 
  (∃ (x y : ℚ), x - y = -4 ∧ y = -1 ∧ x = -5) ∧
  (∃ (x y : ℚ), 2 * x + y = 6 ∧ y = -1 ∧ x = 7 / 2) :=
by
  sorry

end triangle_vertices_l30_30154


namespace find_sum_of_m1_m2_l30_30971

-- Define the quadratic equation and the conditions
def quadratic (m : ℂ) (x : ℂ) : ℂ := m * x^2 - (3 * m - 2) * x + 7

-- Define the roots a and b
def are_roots (m a b : ℂ) : Prop := quadratic m a = 0 ∧ quadratic m b = 0

-- The condition given in the problem
def root_condition (a b : ℂ) : Prop := a / b + b / a = 3 / 2

-- Main theorem to be proved
theorem find_sum_of_m1_m2 (m1 m2 a1 a2 b1 b2 : ℂ) 
  (h1 : are_roots m1 a1 b1) 
  (h2 : are_roots m2 a2 b2) 
  (hc1 : root_condition a1 b1) 
  (hc2 : root_condition a2 b2) : 
  m1 + m2 = 73 / 18 :=
by sorry

end find_sum_of_m1_m2_l30_30971


namespace find_b_l30_30276

theorem find_b (b : ℤ) (h1 : 0 ≤ b) (h2 : b ≤ 20)
  (h3 : (5 + 4 * 83 + 6 * 83^2 + 3 * 83^3 + 7 * 83^4 + 5 * 83^5 + 2 * 83^6 - b) % 17 = 0) :
  b = 8 :=
sorry

end find_b_l30_30276


namespace consecutive_integer_product_sum_l30_30652

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l30_30652


namespace min_diagonal_length_of_trapezoid_l30_30593

theorem min_diagonal_length_of_trapezoid (a b h d1 d2 : ℝ) 
  (h_area : a * h + b * h = 2)
  (h_diag : d1^2 + d2^2 = h^2 + (a + b)^2) 
  : d1 ≥ Real.sqrt 2 :=
sorry

end min_diagonal_length_of_trapezoid_l30_30593


namespace permutation_sum_integer_l30_30932

theorem permutation_sum_integer (n : ℕ) (h : n > 0) : 
  ∃ s_n, (∀ (a : Finset (Fin n)), 
    (\sum i in a, ((a.val i).val + 1)/((i.val + 1) : ℕ)) ∈ ℤ) ∧ s_n >= n :=
sorry

end permutation_sum_integer_l30_30932


namespace find_c_l30_30802

theorem find_c (c : ℝ) :
  (∃ (infinitely_many_y : ℝ → Prop), (∀ y, infinitely_many_y y ↔ 3 * (5 + 2 * c * y) = 18 * y + 15))
  → c = 3 :=
by
  sorry

end find_c_l30_30802


namespace problem_l30_30624

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l30_30624


namespace direct_proportion_function_l30_30888

-- Definitions of the given functions
def fA (x : ℝ) : ℝ := 3 * x - 4
def fB (x : ℝ) : ℝ := -2 * x + 1
def fC (x : ℝ) : ℝ := 3 * x
def fD (x : ℝ) : ℝ := 4

-- Direct proportion function definition
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∀ x, f 0 = 0 ∧ (f x) / x = f 1 / 1

-- Prove that fC (x) is the only direct proportion function among the given options
theorem direct_proportion_function :
  is_direct_proportion fC ∧ ¬ is_direct_proportion fA ∧ ¬ is_direct_proportion fB ∧ ¬ is_direct_proportion fD :=
by
  sorry

end direct_proportion_function_l30_30888


namespace ellipse_focus_distance_l30_30071

theorem ellipse_focus_distance (m : ℝ) (a b c : ℝ)
  (ellipse_eq : ∀ x y : ℝ, x^2 / m + y^2 / 16 = 1)
  (focus_distance : ∀ P : ℝ × ℝ, ∃ F1 F2 : ℝ × ℝ, dist P F1 = 3 ∧ dist P F2 = 7) :
  m = 25 := 
  sorry

end ellipse_focus_distance_l30_30071


namespace num_integer_distance_pairs_5x5_grid_l30_30446

-- Define the problem conditions
def grid_size : ℕ := 5

-- Define a function to calculate the number of pairs of vertices with integer distances
noncomputable def count_integer_distance_pairs (n : ℕ) : ℕ := sorry

-- The theorem to prove
theorem num_integer_distance_pairs_5x5_grid : count_integer_distance_pairs grid_size = 108 :=
by
  sorry

end num_integer_distance_pairs_5x5_grid_l30_30446


namespace line_has_equal_intercepts_find_a_l30_30527

theorem line_has_equal_intercepts (a : ℝ) :
  (∃ l : ℝ, (l = 0 → ax + y - 2 - a = 0) ∧ (l = 1 → (a = 1 ∨ a = -2))) := sorry

-- formalizing the problem
theorem find_a (a : ℝ) (h_eq_intercepts : ∀ x y : ℝ, (a * x + y - 2 - a = 0 ↔ (x = 2 + a ∧ y = -2 - a))) :
  a = 1 ∨ a = -2 := sorry

end line_has_equal_intercepts_find_a_l30_30527


namespace slope_and_intercept_of_line_l30_30707

theorem slope_and_intercept_of_line :
  ∀ (x y : ℝ), 3 * x + 2 * y + 6 = 0 → y = - (3 / 2) * x - 3 :=
by
  intros x y h
  sorry

end slope_and_intercept_of_line_l30_30707


namespace earliest_year_exceeds_target_l30_30978

/-- Define the initial deposit and annual interest rate -/
def initial_deposit : ℝ := 100000
def annual_interest_rate : ℝ := 0.10

/-- Define the amount in the account after n years -/
def amount_after_years (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r)^n

/-- Define the target amount to exceed -/
def target_amount : ℝ := 150100

/-- Define the year the initial deposit is made -/
def initial_year : ℕ := 2021

/-- Prove that the earliest year the amount exceeds the target is 2026 -/
theorem earliest_year_exceeds_target :
  ∃ n : ℕ, n > 0 ∧ amount_after_years initial_deposit annual_interest_rate n > target_amount ∧ (initial_year + n) = 2026 :=
by
  sorry

end earliest_year_exceeds_target_l30_30978


namespace range_of_a_l30_30215

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) (h : p a ∧ q a) : a ≤ -2 ∨ a = 1 := 
sorry

end range_of_a_l30_30215


namespace total_books_is_595_l30_30934

-- Definitions of the conditions
def satisfies_conditions (a : ℕ) : Prop :=
  ∃ R L : ℕ, a = 12 * R + 7 ∧ a = 25 * L - 5 ∧ 500 < a ∧ a < 650

-- The theorem statement
theorem total_books_is_595 : ∃ a : ℕ, satisfies_conditions a ∧ a = 595 :=
by
  use 595
  split
  · apply exists.intro 49, exists.intro 24, split
    -- a = 12R + 7
    · exact rfl
    -- a = 25L - 5
    · exact rfl
  -- Next check 500 < a and a < 650
  split
  · exact nat.lt_of_le_of_lt (by norm_num) (by norm_num)
  · exact nat.lt_of_le_of_lt (by norm_num) (by norm_num)

end total_books_is_595_l30_30934


namespace tan_theta_l30_30091

theorem tan_theta (θ : ℝ) (h : Real.sin (θ / 2) - 2 * Real.cos (θ / 2) = 0) : Real.tan θ = -4 / 3 :=
sorry

end tan_theta_l30_30091


namespace max_value_frac_sixth_roots_eq_two_l30_30964

noncomputable def max_value_frac_sixth_roots (α β : ℝ) (t : ℝ) (q : ℝ) : ℝ :=
  if α + β = t ∧ α^2 + β^2 = t ∧ α^3 + β^3 = t ∧ α^4 + β^4 = t ∧ α^5 + β^5 = t then
    max (1 / α^6 + 1 / β^6) 2
  else
    0

theorem max_value_frac_sixth_roots_eq_two (α β : ℝ) (t : ℝ) (q : ℝ) :
  (α + β = t ∧ α^2 + β^2 = t ∧ α^3 + β^3 = t ∧ α^4 + β^4 = t ∧ α^5 + β^5 = t) →
  ∃ m, max_value_frac_sixth_roots α β t q = m ∧ m = 2 :=
sorry

end max_value_frac_sixth_roots_eq_two_l30_30964


namespace consecutive_integers_sum_l30_30694

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l30_30694


namespace sufficient_but_not_necessary_condition_for_q_l30_30939

def proposition_p (a : ℝ) := (1 / a) > (1 / 4)
def proposition_q (a : ℝ) := ∀ x : ℝ, (a * x^2 + a * x + 1) > 0

theorem sufficient_but_not_necessary_condition_for_q (a : ℝ) :
  proposition_p a → proposition_q a → (∃ a : ℝ, 0 < a ∧ a < 4) ∧ (∃ a : ℝ, 0 < a ∧ a < 4 ∧ ¬ proposition_p a) 
  := sorry

end sufficient_but_not_necessary_condition_for_q_l30_30939


namespace perpendicular_lines_l30_30756

theorem perpendicular_lines (a : ℝ) : 
  (3 * y + x + 4 = 0) → 
  (4 * y + a * x + 5 = 0) → 
  (∀ x y, x ≠ 0 ∧ y ≠ 0 → - (1 / 3 : ℝ) * - (a / 4 : ℝ) = -1) → 
  a = -12 := 
by
  intros h1 h2 h_perpendicularity
  sorry

end perpendicular_lines_l30_30756


namespace find_13_numbers_l30_30924

theorem find_13_numbers :
  ∃ (a : Fin 13 → ℕ),
    (∀ i, a i % 21 = 0) ∧
    (∀ i j, i ≠ j → ¬(a i ∣ a j) ∧ ¬(a j ∣ a i)) ∧
    (∀ i j, i ≠ j → (a i ^ 5) % (a j ^ 4) = 0) :=
sorry

end find_13_numbers_l30_30924


namespace Jane_indisposed_days_l30_30421

-- Definitions based on conditions
def John_completion_days := 18
def Jane_completion_days := 12
def total_task_days := 10.8
def work_per_day_by_john := 1 / John_completion_days
def work_per_day_by_jane := 1 / Jane_completion_days
def work_per_day_together := work_per_day_by_john + work_per_day_by_jane

-- Equivalent proof problem
theorem Jane_indisposed_days : 
  ∃ (x : ℝ), 
    (10.8 - x) * work_per_day_together + x * work_per_day_by_john = 1 ∧
    x = 6 := 
by 
  sorry

end Jane_indisposed_days_l30_30421


namespace equilibrium_constant_l30_30465

theorem equilibrium_constant (C_NO2 C_O2 C_NO : ℝ) (h_NO2 : C_NO2 = 0.4) (h_O2 : C_O2 = 0.3) (h_NO : C_NO = 0.2) :
  (C_NO2^2 / (C_O2 * C_NO^2)) = 13.3 := by
  rw [h_NO2, h_O2, h_NO]
  sorry

end equilibrium_constant_l30_30465


namespace combined_stickers_l30_30422

def initial_stickers_june : ℕ := 76
def initial_stickers_bonnie : ℕ := 63
def birthday_stickers : ℕ := 25

theorem combined_stickers : 
  (initial_stickers_june + birthday_stickers) + (initial_stickers_bonnie + birthday_stickers) = 189 := 
by
  sorry

end combined_stickers_l30_30422


namespace least_subtracted_correct_second_num_correct_l30_30887

-- Define the given numbers
def given_num : ℕ := 1398
def remainder : ℕ := 5
def num1 : ℕ := 7
def num2 : ℕ := 9
def num3 : ℕ := 11

-- Least number to subtract to satisfy the condition
def least_subtracted : ℕ := 22

-- Second number in the sequence
def second_num : ℕ := 2069

-- Define the hypotheses and statements to be proved
theorem least_subtracted_correct : given_num - least_subtracted ≡ remainder [MOD num1]
∧ given_num - least_subtracted ≡ remainder [MOD num2]
∧ given_num - least_subtracted ≡ remainder [MOD num3] := sorry

theorem second_num_correct : second_num ≡ remainder [MOD num1 * num2 * num3] := sorry

end least_subtracted_correct_second_num_correct_l30_30887


namespace polynomial_solution_l30_30058
-- Import necessary library

-- Define the property to be checked
def polynomial_property (P : Real → Real) : Prop :=
  ∀ a b c : Real, 
    P (a + b - 2 * c) + P (b + c - 2 * a) + P (c + a - 2 * b) = 
    3 * P (a - b) + 3 * P (b - c) + 3 * P (c - a)

-- The statement that needs to be proven
theorem polynomial_solution (a b : Real) : polynomial_property (λ x => a * x^2 + b * x) := 
by
  sorry

end polynomial_solution_l30_30058


namespace consecutive_integers_sum_l30_30696

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l30_30696


namespace parts_supplier_total_amount_received_l30_30726

noncomputable def total_amount_received (total_packages: ℕ) (price_per_package: ℚ) (discount_factor: ℚ)
  (X_percentage: ℚ) (Y_percentage: ℚ) : ℚ :=
  let X_packages := X_percentage * total_packages
  let Y_packages := Y_percentage * total_packages
  let Z_packages := total_packages - X_packages - Y_packages
  let discounted_price := discount_factor * price_per_package
  let cost_X := X_packages * price_per_package
  let cost_Y := Y_packages * price_per_package
  let cost_Z := 10 * price_per_package + (Z_packages - 10) * discounted_price
  cost_X + cost_Y + cost_Z

-- Given conditions
def total_packages : ℕ := 60
def price_per_package : ℚ := 20
def discount_factor : ℚ := 4 / 5
def X_percentage : ℚ := 0.20
def Y_percentage : ℚ := 0.15

theorem parts_supplier_total_amount_received :
  total_amount_received total_packages price_per_package discount_factor X_percentage Y_percentage = 1084 := 
by 
  -- Here we need the proof, but we put sorry to skip it as per instructions
  sorry

end parts_supplier_total_amount_received_l30_30726


namespace lana_pages_after_adding_duane_l30_30111

theorem lana_pages_after_adding_duane :
  ∀ (lana_initial_pages duane_total_pages : ℕ), 
  lana_initial_pages = 8 → 
  duane_total_pages = 42 → 
  lana_initial_pages + (duane_total_pages / 2) = 29 :=
by
  intros lana_initial_pages duane_total_pages h_lana h_duane
  rw [h_lana, h_duane]
  norm_num

end lana_pages_after_adding_duane_l30_30111


namespace simplify_tax_suitable_for_leonid_l30_30818

structure BusinessSetup where
  sellsFlowers : Bool
  noPriorExperience : Bool
  worksIndependently : Bool

def LeonidSetup : BusinessSetup := {
  sellsFlowers := true,
  noPriorExperience := true,
  worksIndependently := true
}

def isSimplifiedTaxSystemSuitable (setup : BusinessSetup) : Prop :=
  setup.sellsFlowers = true ∧ setup.noPriorExperience = true ∧ setup.worksIndependently = true

theorem simplify_tax_suitable_for_leonid (setup : BusinessSetup) :
  isSimplifiedTaxSystemSuitable setup := by
  sorry

#eval simplify_tax_suitable_for_leonid LeonidSetup

end simplify_tax_suitable_for_leonid_l30_30818


namespace dot_product_a_a_sub_2b_l30_30523

-- Define the vectors a and b
def a : (ℝ × ℝ) := (2, 3)
def b : (ℝ × ℝ) := (-1, 2)

-- Define the subtraction of vector a and 2 * vector b
def a_sub_2b : (ℝ × ℝ) := (a.1 - 2 * b.1, a.2 - 2 * b.2)

-- Define the dot product of two vectors
def dot_product (u v : (ℝ × ℝ)) : ℝ := u.1 * v.1 + u.2 * v.2

-- State that the dot product of a and (a - 2b) is 5
theorem dot_product_a_a_sub_2b : dot_product a a_sub_2b = 5 := 
by 
  -- proof omitted
  sorry

end dot_product_a_a_sub_2b_l30_30523


namespace inequality_neg_3_l30_30940

theorem inequality_neg_3 (a b : ℝ) : a < b → -3 * a > -3 * b :=
by
  sorry

end inequality_neg_3_l30_30940


namespace add_base8_l30_30038

-- Define the base 8 numbers 5_8 and 16_8
def five_base8 : ℕ := 5
def sixteen_base8 : ℕ := 1 * 8 + 6

-- Convert the result to base 8 from the sum in base 10
def sum_base8 (a b : ℕ) : ℕ :=
  let sum_base10 := a + b
  let d1 := sum_base10 / 8
  let d0 := sum_base10 % 8
  d1 * 10 + d0 

theorem add_base8 (x y : ℕ) (hx : x = five_base8) (hy : y = sixteen_base8) :
  sum_base8 x y = 23 :=
by
  sorry

end add_base8_l30_30038


namespace two_quadratic_residue_mod_p_p_congruent_one_mod_2_pow_n_plus_two_l30_30142

variable (n : ℕ) (F : ℕ → ℕ) (p : ℕ)

-- Condition: F_n = 2^{2^n} + 1
def F_n (n : ℕ) : ℕ := 2^(2^n) + 1

-- Assuming n >= 2
def n_ge_two (n : ℕ) : Prop := n ≥ 2

-- Assuming p is a prime factor of F_n
def prime_factor_of_F_n (p : ℕ) (n : ℕ) : Prop := p ∣ (F_n n) ∧ Prime p

-- Part a: 2 is a quadratic residue modulo p
theorem two_quadratic_residue_mod_p (n : ℕ) (p : ℕ) (hn : n_ge_two n) (hp : prime_factor_of_F_n p n) :
  ∃ x : ℕ, x^2 ≡ 2 [MOD p] := sorry

-- Part b: p ≡ 1 (mod 2^(n+2))
theorem p_congruent_one_mod_2_pow_n_plus_two (n : ℕ) (p : ℕ) (hn : n_ge_two n) (hp : prime_factor_of_F_n p n) :
  p ≡ 1 [MOD 2^(n+2)] := sorry

end two_quadratic_residue_mod_p_p_congruent_one_mod_2_pow_n_plus_two_l30_30142


namespace part1_solution_set_part2_comparison_l30_30380

noncomputable def f (x : ℝ) := -|x| - |x + 2|

theorem part1_solution_set (x : ℝ) : f x < -4 ↔ x < -3 ∨ x > 1 :=
by sorry

theorem part2_comparison (a b x : ℝ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + b = Real.sqrt 5) : 
  a^2 + b^2 / 4 ≥ f x + 3 :=
by sorry

end part1_solution_set_part2_comparison_l30_30380


namespace range_of_m_l30_30097

noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.sin x

theorem range_of_m (m : ℝ) 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_incr : ∀ x y, x < y → f x < f y) : 
  f (2 * m - 1) + f (3 - m) > 0 ↔ m > -2 := 
by 
  sorry

end range_of_m_l30_30097


namespace percentage_very_satisfactory_l30_30738

-- Definitions based on conditions
def total_parents : ℕ := 120
def needs_improvement_count : ℕ := 6
def excellent_percentage : ℕ := 15
def satisfactory_remaining_percentage : ℕ := 80

-- Theorem statement
theorem percentage_very_satisfactory 
  (total_parents : ℕ) 
  (needs_improvement_count : ℕ) 
  (excellent_percentage : ℕ) 
  (satisfactory_remaining_percentage : ℕ) 
  (result : ℕ) : result = 16 :=
by
  sorry

end percentage_very_satisfactory_l30_30738


namespace regular_polygon_sides_l30_30321

theorem regular_polygon_sides (θ : ℝ) (n : ℕ) (h1 : θ = 18) (h2 : 360 = θ * n) : n = 20 :=
by {
  sorry
}

end regular_polygon_sides_l30_30321


namespace unit_prices_min_number_of_A_l30_30010

theorem unit_prices (x y : ℝ)
  (h1 : 3 * x + 4 * y = 580)
  (h2 : 6 * x + 5 * y = 860) :
  x = 60 ∧ y = 100 :=
by
  sorry

theorem min_number_of_A (x y a : ℝ)
  (x_h : x = 60)
  (y_h : y = 100)
  (h1 : 3 * x + 4 * y = 580)
  (h2 : 6 * x + 5 * y = 860)
  (trash_can_condition : a + 200 - a = 200)
  (cost_condition : 60 * a + 100 * (200 - a) ≤ 15000) :
  a ≥ 125 :=
by
  sorry

end unit_prices_min_number_of_A_l30_30010


namespace computation_of_difference_of_squares_l30_30503

theorem computation_of_difference_of_squares : (65^2 - 35^2) = 3000 := sorry

end computation_of_difference_of_squares_l30_30503


namespace regular_polygon_sides_l30_30328

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (k : ℕ), (k : ℕ) * 18 = 360) : n = 20 :=
by
  -- Proof body here
  sorry

end regular_polygon_sides_l30_30328


namespace sum_of_odd_integers_l30_30005

theorem sum_of_odd_integers (n : ℕ) (h : n * (n + 1) = 4970) : (n * n = 4900) :=
by sorry

end sum_of_odd_integers_l30_30005


namespace add_base3_numbers_l30_30742

-- Definitions to represent the numbers in base 3
def base3_num1 := (2 : ℕ) -- 2_3
def base3_num2 := (2 * 3 + 2 : ℕ) -- 22_3
def base3_num3 := (2 * 3^2 + 0 * 3 + 2 : ℕ) -- 202_3
def base3_num4 := (2 * 3^3 + 0 * 3^2 + 2 * 3 + 2 : ℕ) -- 2022_3

-- Summing the numbers in base 10 first
def sum_base10 := base3_num1 + base3_num2 + base3_num3 + base3_num4

-- Expected result in base 10 for 21010_3
def result_base10 := 2 * 3^4 + 1 * 3^3 + 0 * 3^2 + 1 * 3 + 0

-- Proof statement
theorem add_base3_numbers : sum_base10 = result_base10 :=
by {
  -- Proof not required, so we skip it using sorry
  sorry
}

end add_base3_numbers_l30_30742


namespace find_theta_l30_30723

theorem find_theta (Theta : ℕ) (h1 : 1 ≤ Theta ∧ Theta ≤ 9)
  (h2 : 294 / Theta = (30 + Theta) + 3 * Theta) : Theta = 6 :=
by sorry

end find_theta_l30_30723


namespace max_value_seq_l30_30216

noncomputable def a_n (n : ℕ) : ℝ := n / (n^2 + 90)

theorem max_value_seq : ∃ n : ℕ, a_n n = 1 / 19 :=
by
  sorry

end max_value_seq_l30_30216


namespace pies_not_eaten_with_forks_l30_30961

variables (apple_pe_forked peach_pe_forked cherry_pe_forked chocolate_pe_forked lemon_pe_forked : ℤ)
variables (total_pies types_of_pies : ℤ)

def pies_per_type (total_pies types_of_pies : ℤ) : ℤ :=
  total_pies / types_of_pies

def not_eaten_with_forks (percentage_forked : ℤ) (pies : ℤ) : ℤ :=
  pies - (pies * percentage_forked) / 100

noncomputable def apple_not_forked  := not_eaten_with_forks apple_pe_forked (pies_per_type total_pies types_of_pies)
noncomputable def peach_not_forked  := not_eaten_with_forks peach_pe_forked (pies_per_type total_pies types_of_pies)
noncomputable def cherry_not_forked := not_eaten_with_forks cherry_pe_forked (pies_per_type total_pies types_of_pies)
noncomputable def chocolate_not_forked := not_eaten_with_forks chocolate_pe_forked (pies_per_type total_pies types_of_pies)
noncomputable def lemon_not_forked := not_eaten_with_forks lemon_pe_forked (pies_per_type total_pies types_of_pies)

theorem pies_not_eaten_with_forks :
  (apple_not_forked = 128) ∧
  (peach_not_forked = 112) ∧
  (cherry_not_forked = 84) ∧
  (chocolate_not_forked = 76) ∧
  (lemon_not_forked = 140) :=
by sorry

end pies_not_eaten_with_forks_l30_30961


namespace lara_bouncy_house_time_l30_30569

theorem lara_bouncy_house_time :
  let run1_time := (3 * 60 + 45) + (2 * 60 + 10) + (1 * 60 + 28)
  let door_time := 73
  let run2_time := (2 * 60 + 55) + (1 * 60 + 48) + (1 * 60 + 15)
  run1_time + door_time + run2_time = 874 := by
    let run1_time := 225 + 130 + 88
    let door_time := 73
    let run2_time := 175 + 108 + 75
    sorry

end lara_bouncy_house_time_l30_30569


namespace smallest_value_of_3a_plus_2_l30_30949

theorem smallest_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 10 * a + 6 = 2) : 3 * a + 2 = -1 :=
by
  sorry

end smallest_value_of_3a_plus_2_l30_30949


namespace probability_two_ones_in_twelve_dice_l30_30882

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def probability_two_ones (n : ℕ) (p : ℚ) : ℚ :=
  (binomial n 2) * (p^2) * ((1 - p)^(n - 2))

theorem probability_two_ones_in_twelve_dice : 
  probability_two_ones 12 (1/6) ≈ 0.293 := 
  by
    sorry

end probability_two_ones_in_twelve_dice_l30_30882


namespace ratio_of_doctors_lawyers_engineers_l30_30845

variables (d l e : ℕ)

-- Conditions
def average_age_per_group (d l e : ℕ) : Prop :=
  (40 * d + 55 * l + 35 * e) = 45 * (d + l + e)

-- Theorem
theorem ratio_of_doctors_lawyers_engineers
  (h : average_age_per_group d l e) :
  l = d + 2 * e :=
by sorry

end ratio_of_doctors_lawyers_engineers_l30_30845


namespace sin_13pi_over_6_equals_half_l30_30762

noncomputable def sin_13pi_over_6 : ℝ := Real.sin (13 * Real.pi / 6)

theorem sin_13pi_over_6_equals_half : sin_13pi_over_6 = 1 / 2 := by
  sorry

end sin_13pi_over_6_equals_half_l30_30762


namespace adam_initial_books_l30_30178

theorem adam_initial_books (B : ℕ) (h1 : B - 11 + 23 = 45) : B = 33 := 
by
  sorry

end adam_initial_books_l30_30178


namespace fractional_eq_range_m_l30_30084

theorem fractional_eq_range_m (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) → m ≤ 2 ∧ m ≠ -2 :=
by
  sorry

end fractional_eq_range_m_l30_30084


namespace combined_stickers_count_l30_30425

theorem combined_stickers_count :
  let initial_june_stickers := 76
  let initial_bonnie_stickers := 63
  let stickers_given := 25
  let june_total := initial_june_stickers + stickers_given
  let bonnie_total := initial_bonnie_stickers + stickers_given
  june_total + bonnie_total = 189 :=
by
  -- Definitions
  let initial_june_stickers := 76
  let initial_bonnie_stickers := 63
  let stickers_given := 25
  
  -- Calculations
  let june_total := initial_june_stickers + stickers_given
  let bonnie_total := initial_bonnie_stickers + stickers_given

  -- Proof is omitted
  sorry

end combined_stickers_count_l30_30425


namespace find_width_of_room_l30_30062

theorem find_width_of_room (length room_cost cost_per_sqm total_cost width W : ℕ) 
  (h1 : length = 13)
  (h2 : cost_per_sqm = 12)
  (h3 : total_cost = 1872)
  (h4 : room_cost = length * W * cost_per_sqm)
  (h5 : total_cost = room_cost) : 
  W = 12 := 
by sorry

end find_width_of_room_l30_30062


namespace simplify_expression_l30_30918

theorem simplify_expression (a b : ℕ) (h₁ : a = 2999) (h₂ : b = 3000) :
  b^3 - a * b^2 - a^2 * b + a^3 = 5999 :=
by 
  sorry

end simplify_expression_l30_30918


namespace supplement_greater_than_complement_l30_30368

variable (angle1 : ℝ)

def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

theorem supplement_greater_than_complement (h : is_acute angle1) :
  180 - angle1 = 90 + (90 - angle1) :=
by {
  sorry
}

end supplement_greater_than_complement_l30_30368


namespace parabola_focus_l30_30352

theorem parabola_focus :
  ∃ f, (∀ x : ℝ, (x^2 + (2*x^2 - f)^2 = (2*x^2 - (-f + 1/4))^2)) ∧ f = 1/8 :=
by
  sorry

end parabola_focus_l30_30352


namespace total_cans_l30_30162

theorem total_cans (c o : ℕ) (h1 : c = 8) (h2 : o = 2 * c) : c + o = 24 := by
  sorry

end total_cans_l30_30162


namespace meaningful_expression_l30_30393

-- Definition stating the meaningfulness of the expression (condition)
def is_meaningful (a : ℝ) : Prop := (a - 1) ≠ 0

-- Theorem stating that for the expression to be meaningful, a ≠ 1
theorem meaningful_expression (a : ℝ) : is_meaningful a ↔ a ≠ 1 :=
by sorry

end meaningful_expression_l30_30393


namespace equal_ratios_l30_30176

variable (x y : ℝ)

-- Conditions
def wire_split_to_form_square_and_pentagon (x y : ℝ) : Prop :=
  4 * (x / 4) = 5 * (y / 5)

-- Theorem to prove
theorem equal_ratios (x y : ℝ) (h : wire_split_to_form_square_and_pentagon x y) : x / y = 1 :=
  sorry

end equal_ratios_l30_30176


namespace fourth_number_in_first_set_88_l30_30132

theorem fourth_number_in_first_set_88 (x y : ℝ)
  (h1 : (28 + x + 70 + y + 104) / 5 = 67)
  (h2 : (50 + 62 + 97 + 124 + x) / 5 = 75.6) :
  y = 88 :=
by
  sorry

end fourth_number_in_first_set_88_l30_30132


namespace eq_from_conditions_l30_30287

theorem eq_from_conditions (a b : ℂ) :
  (1 / (a + b)) ^ 2003 = 1 ∧ (-a + b) ^ 2005 = 1 → a ^ 2003 + b ^ 2004 = 1 := 
by
  sorry

end eq_from_conditions_l30_30287


namespace stadium_capacity_l30_30464

theorem stadium_capacity 
  (C : ℕ)
  (entry_fee : ℕ := 20)
  (three_fourth_full_fees : ℕ := 3 / 4 * C * entry_fee)
  (full_fees : ℕ := C * entry_fee)
  (fee_difference : ℕ := full_fees - three_fourth_full_fees)
  (h : fee_difference = 10000) :
  C = 2000 :=
by
  sorry

end stadium_capacity_l30_30464


namespace exists_m_for_division_l30_30065

theorem exists_m_for_division (n : ℕ) (h : 0 < n) : ∃ m : ℕ, n ∣ (2016 ^ m + m) := by
  sorry

end exists_m_for_division_l30_30065


namespace point_in_second_quadrant_l30_30965

-- Define the point in question
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Given conditions based on the problem statement
def P (x : ℝ) : Point :=
  Point.mk (-2) (x^2 + 1)

-- The theorem we aim to prove
theorem point_in_second_quadrant (x : ℝ) : (P x).x < 0 ∧ (P x).y > 0 → 
  -- This condition means that the point is in the second quadrant
  (P x).x < 0 ∧ (P x).y > 0 :=
by
  sorry

end point_in_second_quadrant_l30_30965


namespace unique_cube_coloring_l30_30129

-- Definition of vertices at the bottom of the cube with specific colors
inductive Color 
| Red | Green | Blue | Purple

open Color

def bottom_colors : Fin 4 → Color
| 0 => Red
| 1 => Green
| 2 => Blue
| 3 => Purple

-- Definition of the property that ensures each face of the cube has different colored corners
def all_faces_different_colors (top_colors : Fin 4 → Color) : Prop :=
  (top_colors 0 ≠ Red) ∧ (top_colors 0 ≠ Green) ∧ (top_colors 0 ≠ Blue) ∧
  (top_colors 1 ≠ Green) ∧ (top_colors 1 ≠ Blue) ∧ (top_colors 1 ≠ Purple) ∧
  (top_colors 2 ≠ Red) ∧ (top_colors 2 ≠ Blue) ∧ (top_colors 2 ≠ Purple) ∧
  (top_colors 3 ≠ Red) ∧ (top_colors 3 ≠ Green) ∧ (top_colors 3 ≠ Purple)

-- Prove there is exactly one way to achieve this coloring of the top corners
theorem unique_cube_coloring : ∃! (top_colors : Fin 4 → Color), all_faces_different_colors top_colors :=
sorry

end unique_cube_coloring_l30_30129


namespace negation_proposition_l30_30283

theorem negation_proposition : 
  ¬ (∃ x_0 : ℝ, x_0^2 + x_0 + 1 < 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≥ 0 :=
by {
  sorry
}

end negation_proposition_l30_30283


namespace problem_statement_l30_30820

theorem problem_statement (a b c : ℝ) (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = 11) (h3 : a * b * c = 6) :
  (a * b / c) + (b * c / a) + (c * a / b) = 49 / 6 := 
by sorry

end problem_statement_l30_30820


namespace consecutive_integers_sum_l30_30700

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l30_30700


namespace expression_divisibility_l30_30981

theorem expression_divisibility (x y : ℝ) : 
  ∃ P : ℝ, (x^2 - x * y + y^2)^3 + (x^2 + x * y + y^2)^3 = (2 * x^2 + 2 * y^2) * P := 
by 
  sorry

end expression_divisibility_l30_30981


namespace line_has_equal_intercepts_find_a_l30_30526

theorem line_has_equal_intercepts (a : ℝ) :
  (∃ l : ℝ, (l = 0 → ax + y - 2 - a = 0) ∧ (l = 1 → (a = 1 ∨ a = -2))) := sorry

-- formalizing the problem
theorem find_a (a : ℝ) (h_eq_intercepts : ∀ x y : ℝ, (a * x + y - 2 - a = 0 ↔ (x = 2 + a ∧ y = -2 - a))) :
  a = 1 ∨ a = -2 := sorry

end line_has_equal_intercepts_find_a_l30_30526


namespace crocodile_can_move_anywhere_iff_even_l30_30730

def is_even (n : ℕ) : Prop := n % 2 = 0

def can_move_to_any_square (N : ℕ) : Prop :=
∀ (x1 y1 x2 y2 : ℤ), ∃ (k : ℕ), 
(x1 + k * (N + 1) = x2 ∨ y1 + k * (N + 1) = y2)

theorem crocodile_can_move_anywhere_iff_even (N : ℕ) : can_move_to_any_square N ↔ is_even N :=
sorry

end crocodile_can_move_anywhere_iff_even_l30_30730


namespace coefficient_of_x5_in_expansion_l30_30554

theorem coefficient_of_x5_in_expansion :
  let f := λ x : ℚ, x ^ 2 - 2 / x
  let general_term := λ (n a x : ℚ), n.choose a * ((-2)^a) * x^(2*(n-a) - a)
  let coeff := general_term 7 3
  coeff = -280 :=
by
  sorry

end coefficient_of_x5_in_expansion_l30_30554


namespace sum_of_consecutive_integers_with_product_812_l30_30674

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l30_30674


namespace probability_of_selecting_male_is_three_fifths_l30_30303

-- Define the number of male and female students
def num_male_students : ℕ := 6
def num_female_students : ℕ := 4

-- Define the total number of students
def total_students : ℕ := num_male_students + num_female_students

-- Define the probability of selecting a male student's ID
def probability_male_student : ℚ := num_male_students / total_students

-- Theorem: The probability of selecting a male student's ID is 3/5
theorem probability_of_selecting_male_is_three_fifths : probability_male_student = 3 / 5 :=
by
  -- Proof to be filled in
  sorry

end probability_of_selecting_male_is_three_fifths_l30_30303


namespace factor_expression_l30_30921

theorem factor_expression (x : ℝ) : 5 * x * (x - 2) + 9 * (x - 2) = (x - 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l30_30921


namespace last_digit_of_one_over_729_l30_30718

def last_digit_of_decimal_expansion (n : ℕ) : ℕ := (n % 10)

theorem last_digit_of_one_over_729 : last_digit_of_decimal_expansion (1 / 729) = 9 :=
sorry

end last_digit_of_one_over_729_l30_30718


namespace point_in_fourth_quadrant_l30_30406

def x : ℝ := 8
def y : ℝ := -3

theorem point_in_fourth_quadrant (h1 : x > 0) (h2 : y < 0) : (x > 0 ∧ y < 0) :=
by {
  sorry
}

end point_in_fourth_quadrant_l30_30406


namespace find_value_of_expression_l30_30930

theorem find_value_of_expression :
  3 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2400 :=
by
  sorry

end find_value_of_expression_l30_30930


namespace complement_of_S_in_U_l30_30824

variable (U : Set ℕ)
variable (S : Set ℕ)

theorem complement_of_S_in_U (hU : U = {1, 2, 3, 4}) (hS : S = {1, 3}) : U \ S = {2, 4} := by
  sorry

end complement_of_S_in_U_l30_30824


namespace range_of_m_l30_30081

theorem range_of_m (m : ℝ) : (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) → (m ≤ 2 ∧ m ≠ -2) :=
begin
  sorry
end

end range_of_m_l30_30081


namespace flat_fee_l30_30035

theorem flat_fee (f n : ℝ) (h1 : f + 4 * n = 320) (h2 : f + 7 * n = 530) : f = 40 := by
  -- Proof goes here
  sorry

end flat_fee_l30_30035


namespace jackson_running_l30_30414

variable (x : ℕ)

theorem jackson_running (h : x + 4 = 7) : x = 3 := by
  sorry

end jackson_running_l30_30414


namespace move_line_left_and_up_l30_30992

/--
The equation of the line obtained by moving the line y = 2x - 3
2 units to the left and then 3 units up is y = 2x + 4.
-/
theorem move_line_left_and_up :
  ∀ (x y : ℝ), y = 2*x - 3 → ∃ x' y', x' = x + 2 ∧ y' = y + 3 ∧ y' = 2*x' + 4 :=
by
  sorry

end move_line_left_and_up_l30_30992


namespace least_square_of_conditions_l30_30452

theorem least_square_of_conditions :
  ∃ (a x y : ℕ), 0 < a ∧ 0 < x ∧ 0 < y ∧ 
  (15 * a + 165 = x^2) ∧ 
  (16 * a - 155 = y^2) ∧ 
  (min (x^2) (y^2) = 481) := 
sorry

end least_square_of_conditions_l30_30452


namespace net_effect_on_sale_value_l30_30098

theorem net_effect_on_sale_value 
  (P Original_Sales_Volume : ℝ) 
  (reduced_by : ℝ := 0.18) 
  (sales_increase : ℝ := 0.88) 
  (additional_tax : ℝ := 0.12) :
  P * Original_Sales_Volume * ((1 - reduced_by) * (1 + additional_tax) * (1 + sales_increase) - 1) = P * Original_Sales_Volume * 0.7184 :=
  by
  sorry

end net_effect_on_sale_value_l30_30098


namespace consecutive_integers_sum_l30_30697

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l30_30697


namespace problem_l30_30627

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l30_30627


namespace transport_in_neg_20_repr_transport_out_20_l30_30401

theorem transport_in_neg_20_repr_transport_out_20
  (out_recording : ∀ x : ℝ, transporting_out x → recording (-x))
  (in_recording  : ∀ x : ℝ, transporting_in x → recording x) :
  recording (-(-20)) = recording (20) := by
  sorry

end transport_in_neg_20_repr_transport_out_20_l30_30401


namespace max_area_of_rect_l30_30540

theorem max_area_of_rect (x y : ℝ) (h1 : x + y = 10) : 
  x * y ≤ 25 :=
by 
  sorry

end max_area_of_rect_l30_30540


namespace sin_cos_sum_l30_30457

theorem sin_cos_sum (α x y r : ℝ) (h1 : x = 2) (h2 : y = -1) (h3 : r = Real.sqrt 5)
    (h4 : ∀ θ, x = r * Real.cos θ) (h5 : ∀ θ, y = r * Real.sin θ) : 
    Real.sin α + Real.cos α = (- 1 / Real.sqrt 5) + (2 / Real.sqrt 5) :=
by
  sorry

end sin_cos_sum_l30_30457


namespace xy_in_m_of_comm_l30_30428

open Complex
open Matrix

def M2C := Matrix (Fin 2) (Fin 2) ℂ
def I2 : M2C := 1

def inM (A : M2C) : Prop := 
  ∀ z : ℂ, det (A - z • I2) = 0 → abs z < 1

theorem xy_in_m_of_comm {X Y : M2C} (hx : inM X) (hy : inM Y) (hxy : X ⬝ Y = Y ⬝ X) : inM (X ⬝ Y) := 
sorry

end xy_in_m_of_comm_l30_30428


namespace true_statements_proved_l30_30509

-- Conditions
def A : Prop := ∃ n : ℕ, 25 = 5 * n
def B : Prop := (∃ m1 : ℕ, 209 = 19 * m1) ∧ (¬ ∃ m2 : ℕ, 63 = 19 * m2)
def C : Prop := (¬ ∃ k1 : ℕ, 90 = 30 * k1) ∧ (¬ ∃ k2 : ℕ, 49 = 30 * k2)
def D : Prop := (∃ l1 : ℕ, 34 = 17 * l1) ∧ (¬ ∃ l2 : ℕ, 68 = 17 * l2)
def E : Prop := ∃ q : ℕ, 140 = 7 * q

-- Correct statements
def TrueStatements : Prop := A ∧ B ∧ E ∧ ¬C ∧ ¬D

-- Lean statement to prove
theorem true_statements_proved : TrueStatements := 
by
  sorry

end true_statements_proved_l30_30509


namespace find_a_geometric_sequence_l30_30429

theorem find_a_geometric_sequence (a : ℤ) (T : ℕ → ℤ) (b : ℕ → ℤ) :
  (∀ n, T n = 3 ^ n + a) →
  b 1 = T 1 →
  (∀ n, n ≥ 2 → b n = T n - T (n - 1)) →
  (∀ n, n ≥ 2 → (∃ r, r * b n = b (n - 1))) →
  a = -1 :=
by
  sorry

end find_a_geometric_sequence_l30_30429


namespace probability_ratio_l30_30761

-- Conditions definitions
def total_choices := Nat.choose 50 5
def p := 10 / total_choices
def q := (Nat.choose 10 2 * Nat.choose 5 2 * Nat.choose 5 3) / total_choices

-- Statement to prove
theorem probability_ratio : q / p = 450 := by
  sorry  -- proof is omitted

end probability_ratio_l30_30761


namespace eiffel_tower_scale_l30_30492

theorem eiffel_tower_scale (height_model : ℝ) (height_actual : ℝ) (h_model : height_model = 30) (h_actual : height_actual = 984) : 
  height_actual / height_model = 32.8 := by
  sorry

end eiffel_tower_scale_l30_30492


namespace portion_divided_equally_for_efforts_l30_30017

-- Definitions of conditions
def tom_investment : ℝ := 700
def jerry_investment : ℝ := 300
def tom_more_than_jerry : ℝ := 800
def total_profit : ℝ := 3000

-- Theorem stating what we need to prove
theorem portion_divided_equally_for_efforts (T J R E : ℝ) 
  (h1 : T = tom_investment)
  (h2 : J = jerry_investment)
  (h3 : total_profit = R)
  (h4 : (E / 2) + (7 / 10) * (R - E) - (E / 2 + (3 / 10) * (R - E)) = tom_more_than_jerry) 
  : E = 1000 :=
by
  sorry

end portion_divided_equally_for_efforts_l30_30017


namespace shortest_hypotenuse_max_inscribed_circle_radius_l30_30462

variable {a b c r : ℝ}

-- Condition 1: The perimeter of the right-angled triangle is 1 meter.
def perimeter_condition (a b : ℝ) : Prop :=
  a + b + Real.sqrt (a^2 + b^2) = 1

-- Problem 1: Prove the shortest length of the hypotenuse is √2 - 1.
theorem shortest_hypotenuse (a b : ℝ) (h : perimeter_condition a b) :
  Real.sqrt (a^2 + b^2) = Real.sqrt 2 - 1 :=
sorry

-- Problem 2: Prove the maximum value of the inscribed circle radius is 3/2 - √2.
theorem max_inscribed_circle_radius (a b r : ℝ) (h : perimeter_condition a b) :
  (a * b = r) → r = 3/2 - Real.sqrt 2 :=
sorry

end shortest_hypotenuse_max_inscribed_circle_radius_l30_30462


namespace ratio_of_money_l30_30238

-- Conditions
def amount_given := 14
def cost_of_gift := 28

-- Theorem statement to prove
theorem ratio_of_money (h1 : amount_given = 14) (h2 : cost_of_gift = 28) :
  amount_given / cost_of_gift = 1 / 2 := by
  sorry

end ratio_of_money_l30_30238


namespace find_distance_of_post_office_from_village_l30_30160

-- Conditions
def rate_to_post_office : ℝ := 12.5
def rate_back_village : ℝ := 2
def total_time : ℝ := 5.8

-- Statement of the theorem
theorem find_distance_of_post_office_from_village (D : ℝ) 
  (travel_time_to : D / rate_to_post_office = D / 12.5) 
  (travel_time_back : D / rate_back_village = D / 2)
  (journey_time_total : D / 12.5 + D / 2 = total_time) : 
  D = 10 := 
sorry

end find_distance_of_post_office_from_village_l30_30160


namespace abes_total_budget_l30_30741

theorem abes_total_budget
    (B : ℝ)
    (h1 : B = (1/3) * B + (1/4) * B + 1250) :
    B = 3000 :=
sorry

end abes_total_budget_l30_30741


namespace arithmetic_sequence_third_term_l30_30804

theorem arithmetic_sequence_third_term 
    (a d : ℝ) 
    (h1 : a = 2)
    (h2 : (a + d) + (a + 3 * d) = 10) : 
    a + 2 * d = 5 := 
by
  sorry

end arithmetic_sequence_third_term_l30_30804


namespace find_a4_b4_l30_30913

theorem find_a4_b4
  (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ)
  (h1 : a1 * b1 + a2 * b3 = 1)
  (h2 : a1 * b2 + a2 * b4 = 0)
  (h3 : a3 * b1 + a4 * b3 = 0)
  (h4 : a3 * b2 + a4 * b4 = 1)
  (h5 : a2 * b3 = 7) : 
  a4 * b4 = -6 :=
sorry

end find_a4_b4_l30_30913


namespace problem_statement_l30_30345

def binary_op (a b : ℚ) : ℚ := (a^2 + b^2) / (a^2 - b^2)

theorem problem_statement : binary_op (binary_op 8 6) 2 = 821 / 429 := 
by sorry

end problem_statement_l30_30345


namespace combined_average_speed_l30_30296

-- Definitions based on conditions
def distance_A : ℕ := 250
def time_A : ℕ := 4

def distance_B : ℕ := 480
def time_B : ℕ := 6

def distance_C : ℕ := 390
def time_C : ℕ := 5

def total_distance : ℕ := distance_A + distance_B + distance_C
def total_time : ℕ := time_A + time_B + time_C

-- Prove combined average speed
theorem combined_average_speed : (total_distance : ℚ) / (total_time : ℚ) = 74.67 :=
  by
    sorry

end combined_average_speed_l30_30296


namespace total_conference_games_scheduled_l30_30447

-- Definitions of the conditions
def num_divisions : ℕ := 2
def teams_per_division : ℕ := 6
def intradivision_games_per_pair : ℕ := 3
def interdivision_games_per_pair : ℕ := 2

-- The statement to prove the total number of conference games
theorem total_conference_games_scheduled : 
  (num_divisions * (teams_per_division * (teams_per_division - 1) * intradivision_games_per_pair) / 2) 
  + (teams_per_division * teams_per_division * interdivision_games_per_pair) = 162 := 
by
  sorry

end total_conference_games_scheduled_l30_30447


namespace quadratic_sum_constants_l30_30454

-- Define the quadratic expression
def quadratic (x : ℝ) : ℝ := -3 * x^2 + 27 * x + 135

-- Define the representation of the quadratic in the form a(x + b)^2 + c
def quadratic_rewritten (a b c : ℝ) (x : ℝ) : ℝ := a * (x + b)^2 + c

-- Theorem statement
theorem quadratic_sum_constants :
  ∃ a b c, (∀ x, quadratic x = quadratic_rewritten a b c x) ∧ a + b + c = 197.75 :=
by
  sorry

end quadratic_sum_constants_l30_30454


namespace max_gcd_is_121_l30_30999

-- Definitions from the given problem
def a (n : ℕ) : ℕ := 120 + n^2
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

-- The statement we want to prove
theorem max_gcd_is_121 : ∃ n : ℕ, d n = 121 := sorry

end max_gcd_is_121_l30_30999


namespace angle_between_north_and_south_southeast_l30_30161

-- Given a circular floor pattern with 12 equally spaced rays
def num_rays : ℕ := 12
def total_degrees : ℕ := 360

-- Proving each central angle measure
def central_angle_measure : ℕ := total_degrees / num_rays

-- Define rays of interest
def segments_between_rays : ℕ := 5

-- Prove the angle between the rays pointing due North and South-Southeast
theorem angle_between_north_and_south_southeast :
  (segments_between_rays * central_angle_measure) = 150 := by
  sorry

end angle_between_north_and_south_southeast_l30_30161


namespace function_equality_l30_30799

theorem function_equality (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 / x) = x / (1 - x)) :
  ∀ x : ℝ, f x = 1 / (x - 1) :=
by
  sorry

end function_equality_l30_30799


namespace valeries_thank_you_cards_l30_30716

variables (T R J B : ℕ)

theorem valeries_thank_you_cards :
  B = 2 →
  R = B + 3 →
  J = 2 * R →
  T + (B + 1) + R + J = 21 →
  T = 3 :=
by
  intros hB hR hJ hTotal
  sorry

end valeries_thank_you_cards_l30_30716


namespace maria_baggies_l30_30028

-- Definitions of the conditions
def total_cookies (chocolate_chip : Nat) (oatmeal : Nat) : Nat :=
  chocolate_chip + oatmeal

def cookies_per_baggie : Nat :=
  3

def number_of_baggies (total_cookies : Nat) (cookies_per_baggie : Nat) : Nat :=
  total_cookies / cookies_per_baggie

-- Proof statement
theorem maria_baggies :
  number_of_baggies (total_cookies 2 16) cookies_per_baggie = 6 := 
sorry

end maria_baggies_l30_30028


namespace blackboard_length_is_meters_pencil_case_price_is_yuan_campus_area_is_hectares_fingernail_area_is_square_centimeters_l30_30197

variables (length magnitude : ℕ)
variable (price : ℝ)
variable (area : ℕ)

-- Definitions based on the conditions
def length_is_about_4 (length : ℕ) : Prop := length = 4
def price_is_about_9_50 (price : ℝ) : Prop := price = 9.50
def large_area_is_about_3 (area : ℕ) : Prop := area = 3
def small_area_is_about_1 (area : ℕ) : Prop := area = 1

-- Proof problem statements
theorem blackboard_length_is_meters : length_is_about_4 length → length = 4 := by sorry
theorem pencil_case_price_is_yuan : price_is_about_9_50 price → price = 9.50 := by sorry
theorem campus_area_is_hectares : large_area_is_about_3 area → area = 3 := by sorry
theorem fingernail_area_is_square_centimeters : small_area_is_about_1 area → area = 1 := by sorry

end blackboard_length_is_meters_pencil_case_price_is_yuan_campus_area_is_hectares_fingernail_area_is_square_centimeters_l30_30197


namespace Alex_final_silver_tokens_l30_30039

variable (x y : ℕ)

def final_red_tokens (x y : ℕ) : ℕ := 90 - 3 * x + 2 * y
def final_blue_tokens (x y : ℕ) : ℕ := 65 + 2 * x - 4 * y
def silver_tokens (x y : ℕ) : ℕ := x + y

theorem Alex_final_silver_tokens (h1 : final_red_tokens x y < 3)
                                 (h2 : final_blue_tokens x y < 4) :
  silver_tokens x y = 67 := 
sorry

end Alex_final_silver_tokens_l30_30039


namespace correct_calculation_result_l30_30489

theorem correct_calculation_result (x : ℝ) (h : x / 12 = 8) : 12 * x = 1152 :=
sorry

end correct_calculation_result_l30_30489


namespace solve_linear_system_l30_30275

theorem solve_linear_system :
  ∃ (x1 x2 x3 : ℚ), 
  (2 * x1 + 5 * x2 - 4 * x3 = 8) ∧ 
  (3 * x1 + 15 * x2 - 9 * x3 = 5) ∧ 
  (5 * x1 + 5 * x2 - 7 * x3 = 27) ∧
  (x1 = 19 / 3 + x3) ∧ 
  (x2 = -14 / 15 + 2 / 5 * x3) := 
by 
  sorry

end solve_linear_system_l30_30275


namespace geometric_figure_area_l30_30546

theorem geometric_figure_area :
  (∀ (z : ℂ),
     (0 < (z.re / 20)) ∧ ((z.re / 20) < 1) ∧ 
     (0 < (z.im / 20)) ∧ ((z.im / 20) < 1) ∧ 
     (0 < (20 / z.re)) ∧ ((20 / z.re) < 1) ∧ 
     (0 < (20 / z.im)) ∧ ((20 / z.im) < 1)) →
     (∃ (area : ℝ), area = 400 - 50 * Real.pi) :=
by
  sorry

end geometric_figure_area_l30_30546


namespace compare_neg_two_and_neg_one_l30_30048

theorem compare_neg_two_and_neg_one : -2 < -1 :=
by {
  -- Proof is omitted
  sorry
}

end compare_neg_two_and_neg_one_l30_30048


namespace probability_of_two_ones_in_twelve_dice_rolls_l30_30869

noncomputable def binomial_prob_exact_two_show_one (n : ℕ) (p : ℚ) : ℚ :=
  let choose := (Nat.factorial n) / ((Nat.factorial 2) * (Nat.factorial (n - 2)))
  let prob := choose * (p^2) * ((1 - p)^(n - 2)) in 
  prob

theorem probability_of_two_ones_in_twelve_dice_rolls :
  binomial_prob_exact_two_show_one 12 (1 / 6) = 0.296 := by
  sorry

end probability_of_two_ones_in_twelve_dice_rolls_l30_30869


namespace find_x_l30_30750

theorem find_x (x : ℝ) (h : (x + 8 + 5 * x + 4 + 2 * x + 7) / 3 = 3 * x - 10) : x = 49 :=
sorry

end find_x_l30_30750


namespace palmer_first_week_photos_l30_30253

theorem palmer_first_week_photos :
  ∀ (X : ℕ), 
    100 + X + 2 * X + 80 = 380 →
    X = 67 :=
by
  intros X h
  -- h represents the condition 100 + X + 2 * X + 80 = 380
  sorry

end palmer_first_week_photos_l30_30253


namespace goldfish_graph_finite_set_of_points_l30_30788

-- Define the cost function for goldfish including the setup fee
def cost (n : ℕ) : ℝ := 20 * n + 5

-- Define the condition
def n_values := {n : ℕ | 1 ≤ n ∧ n ≤ 12}

-- The Lean statement to prove the nature of the graph
theorem goldfish_graph_finite_set_of_points :
  ∀ n ∈ n_values, ∃ k : ℝ, (k = cost n) :=
by
  sorry

end goldfish_graph_finite_set_of_points_l30_30788


namespace train_speed_is_18_kmh_l30_30997

noncomputable def speed_of_train (length_of_bridge length_of_train time : ℝ) : ℝ :=
  (length_of_bridge + length_of_train) / time * 3.6

theorem train_speed_is_18_kmh
  (length_of_bridge : ℝ)
  (length_of_train : ℝ)
  (time : ℝ)
  (h1 : length_of_bridge = 200)
  (h2 : length_of_train = 100)
  (h3 : time = 60) :
  speed_of_train length_of_bridge length_of_train time = 18 :=
by
  sorry

end train_speed_is_18_kmh_l30_30997


namespace green_tractor_price_l30_30016

-- Define the conditions
def salary_based_on_sales (r_ct : Nat) (r_price : ℝ) (g_ct : Nat) (g_price : ℝ) : ℝ :=
  0.1 * r_ct * r_price + 0.2 * g_ct * g_price

-- Define the problem's Lean statement
theorem green_tractor_price
  (r_ct : Nat) (g_ct : Nat)
  (r_price : ℝ) (total_salary : ℝ)
  (h_rct : r_ct = 2)
  (h_gct : g_ct = 3)
  (h_rprice : r_price = 20000)
  (h_salary : total_salary = 7000) :
  ∃ g_price : ℝ, salary_based_on_sales r_ct r_price g_ct g_price = total_salary ∧ g_price = 5000 :=
by
  sorry

end green_tractor_price_l30_30016


namespace least_positive_integer_is_4619_l30_30719

noncomputable def least_positive_integer (N : ℕ) : Prop :=
  N % 4 = 3 ∧
  N % 5 = 4 ∧
  N % 6 = 5 ∧
  N % 7 = 6 ∧
  N % 11 = 10 ∧
  ∀ M : ℕ, (M % 4 = 3 ∧ M % 5 = 4 ∧ M % 6 = 5 ∧ M % 7 = 6 ∧ M % 11 = 10) → N ≤ M

theorem least_positive_integer_is_4619 : least_positive_integer 4619 :=
  sorry

end least_positive_integer_is_4619_l30_30719


namespace quoted_price_of_shares_l30_30735

theorem quoted_price_of_shares (investment : ℝ) (face_value : ℝ) (rate_dividend : ℝ) (annual_income : ℝ) (num_shares : ℝ) (quoted_price : ℝ) :
  investment = 4455 ∧ face_value = 10 ∧ rate_dividend = 0.12 ∧ annual_income = 648 ∧ num_shares = annual_income / (rate_dividend * face_value) →
  quoted_price = investment / num_shares :=
by sorry

end quoted_price_of_shares_l30_30735


namespace total_prime_ending_starting_numerals_l30_30534

def single_digit_primes : List ℕ := [2, 3, 5, 7]
def number_of_possible_digits := 10

def count_3digit_numerals : ℕ :=
  4 * number_of_possible_digits * 4

def count_4digit_numerals : ℕ :=
  4 * number_of_possible_digits * number_of_possible_digits * 4

theorem total_prime_ending_starting_numerals : 
  count_3digit_numerals + count_4digit_numerals = 1760 := by
sorry

end total_prime_ending_starting_numerals_l30_30534


namespace math_books_count_l30_30710

theorem math_books_count (total_books : ℕ) (history_books : ℕ) (geography_books : ℕ) (math_books : ℕ) 
  (h1 : total_books = 100) 
  (h2 : history_books = 32) 
  (h3 : geography_books = 25) 
  (h4 : math_books = total_books - history_books - geography_books) 
  : math_books = 43 := 
by 
  rw [h1, h2, h3] at h4;
  exact h4;
-- use 'sorry' to skip the proof if needed
-- sorry

end math_books_count_l30_30710


namespace Mark_sold_1_box_less_than_n_l30_30436

variable (M A n : ℕ)

theorem Mark_sold_1_box_less_than_n (h1 : n = 8)
 (h2 : A = n - 2)
 (h3 : M + A < n)
 (h4 : M ≥ 1) 
 (h5 : A ≥ 1)
 : M = 1 := 
sorry

end Mark_sold_1_box_less_than_n_l30_30436


namespace consecutive_integer_product_sum_l30_30649

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l30_30649


namespace inches_repaired_before_today_l30_30731

-- Definitions and assumptions based on the conditions.
def total_inches_repaired : ℕ := 4938
def inches_repaired_today : ℕ := 805

-- Target statement that needs to be proven.
theorem inches_repaired_before_today : total_inches_repaired - inches_repaired_today = 4133 :=
by
  sorry

end inches_repaired_before_today_l30_30731


namespace area_of_quadrilateral_l30_30828

/-- The area of the quadrilateral defined by the system of inequalities is 15/7. -/
theorem area_of_quadrilateral : 
  (∃ (x y : ℝ), 3 * x + 2 * y ≤ 6 ∧ x + 3 * y ≥ 3 ∧ x ≥ 0 ∧ y ≥ 0) →
  (∃ (area : ℝ), area = 15 / 7) :=
by
  sorry

end area_of_quadrilateral_l30_30828


namespace range_of_t_l30_30066

theorem range_of_t (a b t : ℝ) (h1 : a * (-1)^2 + b * (-1) + 1 / 2 = 0)
    (h2 : (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = a * x^2 + b * x + 1 / 2))
    (h3 : t = 2 * a + b) : 
    -1 < t ∧ t < 1 / 2 :=
  sorry

end range_of_t_l30_30066


namespace smallest_M_inequality_l30_30911

theorem smallest_M_inequality :
  ∃ M : ℝ, 
  M = 9 / (16 * Real.sqrt 2) ∧
  ∀ a b c : ℝ, 
    |a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)| 
    ≤ M * (a^2 + b^2 + c^2)^2 :=
by
  use 9 / (16 * Real.sqrt 2)
  sorry

end smallest_M_inequality_l30_30911


namespace total_wheels_at_station_l30_30006

theorem total_wheels_at_station (trains carriages rows wheels : ℕ) 
  (h_trains : trains = 4)
  (h_carriages : carriages = 4)
  (h_rows : rows = 3)
  (h_wheels : wheels = 5) : 
  trains * carriages * rows * wheels = 240 := 
by 
  rw [h_trains, h_carriages, h_rows, h_wheels]
  exact Nat.mul_eq_iff_eq_div.mpr rfl

end total_wheels_at_station_l30_30006


namespace smallest_positive_multiple_of_3_4_5_is_60_l30_30148

theorem smallest_positive_multiple_of_3_4_5_is_60 :
  ∃ n : ℕ, n > 0 ∧ (n % 3 = 0) ∧ (n % 4 = 0) ∧ (n % 5 = 0) ∧ n = 60 :=
by
  use 60
  sorry

end smallest_positive_multiple_of_3_4_5_is_60_l30_30148


namespace quadratic_root_property_l30_30943

theorem quadratic_root_property (m p : ℝ) 
  (h1 : (p^2 - 2 * p + m - 1 = 0)) 
  (h2 : (p^2 - 2 * p + 3) * (m + 4) = 7)
  (h3 : ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ r1^2 - 2 * r1 + m - 1 = 0 ∧ r2^2 - 2 * r2 + m - 1 = 0) : 
  m = -3 :=
by 
  sorry

end quadratic_root_property_l30_30943


namespace original_price_of_article_l30_30474

theorem original_price_of_article (x : ℝ) (h : 0.80 * x = 620) : x = 775 := 
by 
  sorry

end original_price_of_article_l30_30474


namespace divisibility_equivalence_l30_30430

theorem divisibility_equivalence (a b c d : ℤ) (h : a ≠ c) :
  (a - c) ∣ (a * b + c * d) ↔ (a - c) ∣ (a * d + b * c) :=
by
  sorry

end divisibility_equivalence_l30_30430


namespace probability_of_inverse_proportion_l30_30786

def points : List (ℝ × ℝ) :=
  [(0.5, -4.5), (1, -4), (1.5, -3.5), (2, -3), (2.5, -2.5), (3, -2), (3.5, -1.5),
   (4, -1), (4.5, -0.5), (5, 0)]

def inverse_proportion_pairs : List ((ℝ × ℝ) × (ℝ × ℝ)) :=
  [((0.5, -4.5), (4.5, -0.5)), ((1, -4), (4, -1)), ((1.5, -3.5), (3.5, -1.5)), ((2, -3), (3, -2))]

theorem probability_of_inverse_proportion:
  let num_pairs := List.length points * (List.length points - 1)
  let favorable_pairs := 2 * List.length inverse_proportion_pairs
  favorable_pairs / num_pairs = (4 : ℚ) / 45 := by
  sorry

end probability_of_inverse_proportion_l30_30786


namespace exists_good_pair_for_each_m_l30_30826

def is_good_pair (m n : ℕ) : Prop :=
  ∃ (a b : ℕ), m * n = a^2 ∧ (m + 1) * (n + 1) = b^2

theorem exists_good_pair_for_each_m : ∀ m : ℕ, ∃ n : ℕ, m < n ∧ is_good_pair m n := by
  intro m
  let n := m * (4 * m + 3)^2
  use n
  have h1 : m < n := sorry -- Proof that m < n
  have h2 : is_good_pair m n := sorry -- Proof that (m, n) is a good pair
  exact ⟨h1, h2⟩

end exists_good_pair_for_each_m_l30_30826


namespace find_ordered_pair_l30_30906

theorem find_ordered_pair (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^y + 1 = y^x) (h2 : 2 * x^y = y^x + 13) : (x = 2 ∧ y = 4) :=
by {
  sorry
}

end find_ordered_pair_l30_30906


namespace number_of_books_in_library_l30_30935

theorem number_of_books_in_library 
  (a : ℕ) 
  (R L : ℕ) 
  (h1 : a = 12 * R + 7) 
  (h2 : a = 25 * L - 5) 
  (h3 : 500 < a ∧ a < 650) : 
  a = 595 :=
begin
  sorry
end

end number_of_books_in_library_l30_30935


namespace bridget_heavier_than_martha_l30_30183

def bridget_weight := 39
def martha_weight := 2

theorem bridget_heavier_than_martha :
  bridget_weight - martha_weight = 37 :=
by
  sorry

end bridget_heavier_than_martha_l30_30183


namespace sum_of_consecutive_integers_with_product_812_l30_30681

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l30_30681


namespace intersection_points_eq_2_l30_30053

def eq1 (x y : ℝ) : Prop := (x - 2 * y + 3) * (4 * x + y - 5) = 0
def eq2 (x y : ℝ) : Prop := (x + 2 * y - 3) * (3 * x - 4 * y + 6) = 0

theorem intersection_points_eq_2 : ∃ (points : Finset (ℝ × ℝ)), 
  (∀ p ∈ points, eq1 p.1 p.2 ∧ eq2 p.1 p.2) ∧ points.card = 2 := 
sorry

end intersection_points_eq_2_l30_30053


namespace total_worth_of_presents_l30_30562

-- Definitions of the costs
def costOfRing : ℕ := 4000
def costOfCar : ℕ := 2000
def costOfBracelet : ℕ := 2 * costOfRing

-- Theorem statement
theorem total_worth_of_presents : 
  costOfRing + costOfCar + costOfBracelet = 14000 :=
begin
  -- by using the given definitions and the provided conditions, we assert the statement
  sorry
end

end total_worth_of_presents_l30_30562


namespace log_sufficient_not_necessary_l30_30780

theorem log_sufficient_not_necessary (a b: ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) :
  (log 10 a > log 10 b) ↔ (a > b) := 
sorry

end log_sufficient_not_necessary_l30_30780


namespace alfonso_initial_money_l30_30495

def daily_earnings : ℕ := 6
def days_per_week : ℕ := 5
def total_weeks : ℕ := 10
def cost_of_helmet : ℕ := 340

theorem alfonso_initial_money :
  let weekly_earnings := daily_earnings * days_per_week
  let total_earnings := weekly_earnings * total_weeks
  cost_of_helmet - total_earnings = 40 :=
by
  let weekly_earnings := daily_earnings * days_per_week
  let total_earnings := weekly_earnings * total_weeks
  show cost_of_helmet - total_earnings = 40
  sorry

end alfonso_initial_money_l30_30495


namespace derivative_at_one_l30_30785

theorem derivative_at_one (f : ℝ → ℝ) (df : ℝ → ℝ) 
  (h₁ : ∀ x, f x = x^2) 
  (h₂ : ∀ x, df x = 2 * x) : 
  df 1 = 2 :=
by sorry

end derivative_at_one_l30_30785


namespace determine_a_l30_30767

theorem determine_a (a : ℝ) :
  (∃ (x y : ℝ), (|y - 10| + |x + 3| - 2) * (x^2 + y^2 - 6) = 0 ∧ (x + 3)^2 + (y - 5)^2 = a) →
  (a = 49 ∨ a = 40 - 4 * Real.sqrt 51) :=
by sorry

end determine_a_l30_30767


namespace gumballs_per_package_l30_30789

theorem gumballs_per_package (total_gumballs : ℕ) (packages : ℝ) (h1 : total_gumballs = 100) (h2 : packages = 20.0) :
  total_gumballs / packages = 5 :=
by sorry

end gumballs_per_package_l30_30789


namespace donald_juice_l30_30510

variable (P D : ℕ)

theorem donald_juice (h1 : P = 3) (h2 : D = 2 * P + 3) : D = 9 := by
  sorry

end donald_juice_l30_30510


namespace total_water_capacity_of_coolers_l30_30417

theorem total_water_capacity_of_coolers :
  ∀ (first_cooler second_cooler third_cooler : ℕ), 
  first_cooler = 100 ∧ 
  second_cooler = first_cooler + first_cooler / 2 ∧ 
  third_cooler = second_cooler / 2 -> 
  first_cooler + second_cooler + third_cooler = 325 := 
by
  intros first_cooler second_cooler third_cooler H
  cases' H with H1 H2
  cases' H2 with H3 H4
  sorry

end total_water_capacity_of_coolers_l30_30417


namespace range_of_m_value_of_m_l30_30946

variables (m p x : ℝ)

-- Conditions: The quadratic equation x^2 - 2x + m - 1 = 0 must have two real roots.
def discriminant (m : ℝ) := (-2)^2 - 4 * 1 * (m - 1)

-- Part 1: Finding the range of values for m
theorem range_of_m (h : discriminant m ≥ 0) : m ≤ 2 := 
by sorry

-- Additional Condition: p is a real root of the equation x^2 - 2x + m - 1 = 0
def is_root (p m : ℝ) := p^2 - 2 * p + m - 1 = 0

-- Another condition: (p^2 - 2p + 3)(m + 4) = 7
def satisfies_condition (p m : ℝ) := (p^2 - 2 * p + 3) * (m + 4) = 7

-- Part 2: Finding the value of m given p is a real root and satisfies (p^2 - 2p + 3)(m + 4) = 7
theorem value_of_m (h1 : is_root p m) (h2 : satisfies_condition p m) : m = -3 := 
by sorry

end range_of_m_value_of_m_l30_30946


namespace consecutive_integers_sum_l30_30622

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l30_30622


namespace xy_difference_l30_30226

theorem xy_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end xy_difference_l30_30226


namespace fractional_equation_solution_l30_30075

theorem fractional_equation_solution (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) →
  (m ≤ 2 ∧ m ≠ -2) :=
by
  sorry

end fractional_equation_solution_l30_30075


namespace point_in_fourth_quadrant_l30_30404

theorem point_in_fourth_quadrant (x : ℝ) (y : ℝ) (hx : x = 8) (hy : y = -3) : x > 0 ∧ y < 0 :=
by
  sorry

end point_in_fourth_quadrant_l30_30404


namespace quadratic_has_two_distinct_real_roots_l30_30289

-- Define the quadratic equation and its coefficients
def a := 1
def b := -4
def c := -3

-- Define the discriminant function for a quadratic equation ax^2 + bx + c = 0
def discriminant (a b c : ℤ) : ℤ := b^2 - 4 * a * c

-- State the problem in Lean: Prove that the quadratic equation x^2 - 4x - 3 = 0 has a positive discriminant.
theorem quadratic_has_two_distinct_real_roots : discriminant a b c > 0 :=
by
  sorry -- This is where the proof would go

end quadratic_has_two_distinct_real_roots_l30_30289


namespace unique_paintings_count_l30_30233

-- Given the conditions of the problem:
-- - N = 6 disks
-- - 3 disks are blue
-- - 2 disks are red
-- - 1 disk is green
-- - Two paintings that can be obtained from one another by a rotation or a reflection are considered the same

-- Define a theorem to calculate the number of unique paintings.
theorem unique_paintings_count : 
    ∃ n : ℕ, n = 13 :=
sorry

end unique_paintings_count_l30_30233


namespace rice_in_each_container_l30_30727

variable (weight_in_pounds : ℚ := 35 / 2)
variable (num_containers : ℕ := 4)
variable (pound_to_oz : ℕ := 16)

theorem rice_in_each_container :
  (weight_in_pounds * pound_to_oz) / num_containers = 70 :=
by
  sorry

end rice_in_each_container_l30_30727


namespace problem_l30_30629

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l30_30629


namespace sum_of_consecutive_integers_with_product_812_l30_30682

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l30_30682


namespace sum_of_digits_of_smallest_N_l30_30970

-- Defining the conditions
def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k
def P (N : ℕ) : ℚ := ((2/3 : ℚ) * N * (1/3 : ℚ) * N) / ((N + 2) * (N + 3))
def S (n : ℕ) : ℕ := (n % 10) + ((n / 10) % 10) + (n / 100)

-- The statement of the problem
theorem sum_of_digits_of_smallest_N :
  ∃ N : ℕ, is_multiple_of_6 N ∧ P N < (4/5 : ℚ) ∧ S N = 6 :=
sorry

end sum_of_digits_of_smallest_N_l30_30970


namespace number_of_sixth_graders_who_bought_more_pens_than_seventh_graders_l30_30042

theorem number_of_sixth_graders_who_bought_more_pens_than_seventh_graders 
  (p : ℕ) (h1 : 178 % p = 0) (h2 : 252 % p = 0) :
  (252 / p) - (178 / p) = 5 :=
sorry

end number_of_sixth_graders_who_bought_more_pens_than_seventh_graders_l30_30042


namespace find_a_plus_b_l30_30224

theorem find_a_plus_b (a b : ℝ) 
  (h1 : 1 = a - b) 
  (h2 : 5 = a - b / 5) : a + b = 11 :=
by
  sorry

end find_a_plus_b_l30_30224


namespace maximize_product_minimize_product_l30_30056

-- Define the numbers that need to be arranged
def numbers : List ℕ := [2, 4, 6, 8]

-- Prove that 82 * 64 is the maximum product arrangement
theorem maximize_product : ∃ a b c d : ℕ, (a = 8) ∧ (b = 2) ∧ (c = 6) ∧ (d = 4) ∧ 
  (a * 10 + b) * (c * 10 + d) = 5248 :=
by
  existsi 8, 2, 6, 4
  constructor; constructor
  repeat {assumption}
  sorry

-- Prove that 28 * 46 is the minimum product arrangement
theorem minimize_product : ∃ a b c d : ℕ, (a = 2) ∧ (b = 8) ∧ (c = 4) ∧ (d = 6) ∧ 
  (a * 10 + b) * (c * 10 + d) = 1288 :=
by
  existsi 2, 8, 4, 6
  constructor; constructor
  repeat {assumption}
  sorry

end maximize_product_minimize_product_l30_30056


namespace dice_probability_l30_30884

noncomputable def binomial_coefficient : ℕ → ℕ → ℕ
| 0, k       := if k = 0 then 1 else 0
| (n+1), 0   := 1
| (n+1), (k+1) := binomial_coefficient n k + binomial_coefficient n (k+1)

def probability_two_ones_in_twelve_rolls: ℝ :=
(binomial_coefficient 12 2) * (1/6)^2 * (5/6)^10

theorem dice_probability :
  real.round (1000 * probability_two_ones_in_twelve_rolls) / 1000 = 0.293 :=
by sorry

end dice_probability_l30_30884


namespace complement_A_union_B_m_eq_4_B_nonempty_and_subset_A_range_m_l30_30242

-- Definitions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | (m + 1) ≤ x ∧ x ≤ (2 * m - 1)}

-- Part (Ⅰ)
theorem complement_A_union_B_m_eq_4 :
  (m = 4) → compl (A ∪ B 4) = {x | x < -2} ∪ {x | x > 7} := 
by
  sorry

-- Part (Ⅱ)
theorem B_nonempty_and_subset_A_range_m :
  (∃ x, x ∈ B m) ∧ (B m ⊆ A) → 2 ≤ m ∧ m ≤ 3 :=
by
  sorry

end complement_A_union_B_m_eq_4_B_nonempty_and_subset_A_range_m_l30_30242


namespace parity_of_magazines_and_celebrities_l30_30333

-- Define the main problem statement using Lean 4

theorem parity_of_magazines_and_celebrities {m c : ℕ}
  (h1 : ∀ i, i < m → ∃ d_i, d_i % 2 = 1)
  (h2 : ∀ j, j < c → ∃ e_j, e_j % 2 = 1) :
  (m % 2 = c % 2) ∧ (∃ ways, ways = 2 ^ ((m - 1) * (c - 1))) :=
by
  sorry

end parity_of_magazines_and_celebrities_l30_30333


namespace consecutive_integers_sum_l30_30619

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l30_30619


namespace smallest_five_digit_congruent_two_mod_seventeen_l30_30147

theorem smallest_five_digit_congruent_two_mod_seventeen : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 2 ∧ n = 10013 :=
by
  sorry

end smallest_five_digit_congruent_two_mod_seventeen_l30_30147


namespace ratio_of_horns_l30_30341

def charlie_flutes := 1
def charlie_horns := 2
def charlie_harps := 1

def carli_flutes := 2 * charlie_flutes
def carli_harps := 0

def total_instruments := 7

def charlie_instruments := charlie_flutes + charlie_horns + charlie_harps
def carli_instruments := total_instruments - charlie_instruments

def carli_horns := carli_instruments - carli_flutes

theorem ratio_of_horns : (carli_horns : ℚ) / charlie_horns = 1 / 2 := by
  sorry

end ratio_of_horns_l30_30341


namespace athlete_difference_is_30_l30_30033

def initial_athletes : ℕ := 600
def leaving_rate : ℕ := 35
def leaving_duration : ℕ := 6
def arrival_rate : ℕ := 20
def arrival_duration : ℕ := 9

def athletes_left : ℕ := leaving_rate * leaving_duration
def new_athletes : ℕ := arrival_rate * arrival_duration
def remaining_athletes : ℕ := initial_athletes - athletes_left
def final_athletes : ℕ := remaining_athletes + new_athletes
def athlete_difference : ℕ := initial_athletes - final_athletes

theorem athlete_difference_is_30 : athlete_difference = 30 :=
by
  show athlete_difference = 30
  -- Proof goes here
  sorry

end athlete_difference_is_30_l30_30033


namespace remainder_sum_div_8_l30_30092

theorem remainder_sum_div_8 (n : ℤ) : (((8 - n) + (n + 5)) % 8) = 5 := 
by {
  sorry
}

end remainder_sum_div_8_l30_30092


namespace initial_pencils_correct_l30_30863

variable (initial_pencils : ℕ)
variable (pencils_added : ℕ := 45)
variable (total_pencils : ℕ := 72)

theorem initial_pencils_correct (h : total_pencils = initial_pencils + pencils_added) : initial_pencils = 27 := by
  sorry

end initial_pencils_correct_l30_30863


namespace parabola_line_intersect_l30_30205

theorem parabola_line_intersect (a : ℝ) (b : ℝ) (h1 : a ≠ 0) (h2 : ∀ x : ℝ, (y = a * x^2) ↔ (y = 2 * x - 3) → (x, y) = (1, -1)) :
  a = -1 ∧ b = -1 ∧ ((x, y) = (-3, -9) ∨ (x, y) = (1, -1)) := by
  sorry

end parabola_line_intersect_l30_30205


namespace sam_paint_cans_l30_30121

theorem sam_paint_cans : 
  ∀ (cans_per_room : ℝ) (initial_cans remaining_cans : ℕ),
    initial_cans * cans_per_room = 40 ∧
    remaining_cans * cans_per_room = 30 ∧
    initial_cans - remaining_cans = 4 →
    remaining_cans = 12 :=
by sorry

end sam_paint_cans_l30_30121


namespace total_investment_is_10000_l30_30903

open Real

-- Definitions of conditions
def interest_rate_8 : Real := 0.08
def interest_rate_9 : Real := 0.09
def combined_interest : Real := 840
def investment_8 : Real := 6000
def total_interest (x : Real) : Real := (interest_rate_8 * investment_8 + interest_rate_9 * x)
def investment_9 : Real := 4000

-- Theorem stating the problem
theorem total_investment_is_10000 :
    (∀ x : Real,
        total_interest x = combined_interest → x = investment_9) →
    investment_8 + investment_9 = 10000 := 
by
    intros
    sorry

end total_investment_is_10000_l30_30903


namespace difference_in_dimes_l30_30986

variables (q : ℝ)

def samantha_quarters : ℝ := 3 * q + 2
def bob_quarters : ℝ := 2 * q + 8
def quarter_to_dimes : ℝ := 2.5

theorem difference_in_dimes :
  quarter_to_dimes * (samantha_quarters q - bob_quarters q) = 2.5 * q - 15 :=
by sorry

end difference_in_dimes_l30_30986


namespace black_to_white_ratio_l30_30189

theorem black_to_white_ratio (initial_black initial_white new_black new_white : ℕ) 
  (h1 : initial_black = 7) (h2 : initial_white = 18)
  (h3 : new_black = 31) (h4 : new_white = 18) :
  (new_black : ℚ) / new_white = 31 / 18 :=
by
  sorry

end black_to_white_ratio_l30_30189


namespace sum_of_consecutive_integers_with_product_812_l30_30658

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l30_30658


namespace steven_has_15_more_peaches_than_jill_l30_30415

-- Definitions based on conditions
def peaches_jill : ℕ := 12
def peaches_jake : ℕ := peaches_jill - 1
def peaches_steven : ℕ := peaches_jake + 16

-- The proof problem
theorem steven_has_15_more_peaches_than_jill : peaches_steven - peaches_jill = 15 := by
  sorry

end steven_has_15_more_peaches_than_jill_l30_30415


namespace find_radii_l30_30409

-- Definitions based on the problem conditions
def tangent_lengths (TP T'Q r r' PQ: ℝ) : Prop :=
  TP = 6 ∧ T'Q = 10 ∧ PQ = 16 ∧ r < r'

-- The main theorem to prove the radii are 15 and 5
theorem find_radii (TP T'Q r r' PQ: ℝ) 
  (h : tangent_lengths TP T'Q r r' PQ) :
  r = 15 ∧ r' = 5 :=
sorry

end find_radii_l30_30409


namespace root_sum_abs_gt_6_l30_30223

variables (r1 r2 p : ℝ)

theorem root_sum_abs_gt_6 
  (h1 : r1 + r2 = -p)
  (h2 : r1 * r2 = 9)
  (h3 : p^2 > 36) :
  |r1 + r2| > 6 :=
by sorry

end root_sum_abs_gt_6_l30_30223


namespace total_drawing_sheets_l30_30237

-- Definitions based on the conditions given
def brown_sheets := 28
def yellow_sheets := 27

-- The statement we need to prove
theorem total_drawing_sheets : brown_sheets + yellow_sheets = 55 := by
  sorry

end total_drawing_sheets_l30_30237


namespace planes_parallel_l30_30229

variables (α β : Type)
variables (n : ℝ → ℝ → ℝ → Prop) (u v : ℝ × ℝ × ℝ)

-- Conditions: 
def normal_vector_plane_alpha (u : ℝ × ℝ × ℝ) := u = (1, 2, -1)
def normal_vector_plane_beta (v : ℝ × ℝ × ℝ) := v = (-3, -6, 3)

-- Proof Problem: Prove that alpha is parallel to beta
theorem planes_parallel (h1 : normal_vector_plane_alpha u)
                        (h2 : normal_vector_plane_beta v) :
  v = -3 • u :=
by sorry

end planes_parallel_l30_30229


namespace average_percentage_of_kernels_popped_l30_30979

theorem average_percentage_of_kernels_popped :
  let bag1_popped := 60
  let bag1_total := 75
  let bag2_popped := 42
  let bag2_total := 50
  let bag3_popped := 82
  let bag3_total := 100
  let percentage (popped total : ℕ) := (popped : ℚ) / total * 100
  let p1 := percentage bag1_popped bag1_total
  let p2 := percentage bag2_popped bag2_total
  let p3 := percentage bag3_popped bag3_total
  let avg := (p1 + p2 + p3) / 3
  avg = 82 :=
by
  sorry

end average_percentage_of_kernels_popped_l30_30979


namespace parallelogram_sides_l30_30857

theorem parallelogram_sides (x y : ℝ) 
    (h1 : 4 * y + 2 = 12) 
    (h2 : 6 * x - 2 = 10)
    (h3 : 10 + 12 + (6 * x - 2) + (4 * y + 2) = 68) :
    x + y = 4.5 := 
by
  -- Proof to be provided
  sorry

end parallelogram_sides_l30_30857


namespace green_tractor_price_l30_30014

variable (S : ℕ) (r g R G : ℕ) (R_price G_price : ℝ)
variable (h1 : S = 7000)
variable (h2 : r = 2)
variable (h3 : g = 3)
variable (h4 : R_price = 20000)
variable (h5 : G_price = 5000)
variable (h6 : ∀ (x : ℝ), Tobias_earning_red : ℝ) (h7 : ∀ (y : ℝ), Tobias_earning_green : ℝ)

def earning_percentage_red : ℝ := 0.10
def earning_percentage_green : ℝ := 0.20

theorem green_tractor_price :
  S = Tobias_earning_red + Tobias_earning_green →
  Tobias_earning_red = (earning_percentage_red * R_price) * r →
  Tobias_earning_green = (earning_percentage_green * G_price) * g →
  G_price = 5000 :=
  sorry

end green_tractor_price_l30_30014


namespace sum_of_consecutive_integers_with_product_812_l30_30677

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l30_30677


namespace find_ages_l30_30307

theorem find_ages (P J G : ℕ)
  (h1 : P - 10 = 1 / 3 * (J - 10))
  (h2 : J = P + 12)
  (h3 : G = 1 / 2 * (P + J)) :
  P = 16 ∧ G = 22 :=
by
  sorry

end find_ages_l30_30307


namespace proposition_correctness_l30_30784

theorem proposition_correctness :
  (∀ a b : ℝ, a < b ∧ b < 0 → ¬ (1 / a < 1 / b)) ∧
  (∀ a b : ℝ, a > 0 ∧ b > 0 → (a + b) / 2 ≥ Real.sqrt (a * b) ∧ Real.sqrt (a * b) ≥ a * b / (a + b)) ∧
  (∀ a b : ℝ, a < b ∧ b < 0 → a^2 > a * b ∧ a * b > b^2) ∧
  (Real.log 9 * Real.log 11 < 1) ∧
  (∀ a b : ℝ, a > b ∧ 1 / a > 1 / b → a > 0 ∧ b < 0) ∧
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / x + 1 / y = 1 → ¬(x + 2 * y = 6)) :=
sorry

end proposition_correctness_l30_30784


namespace consecutive_integers_sum_l30_30666

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l30_30666


namespace correct_operation_l30_30470

theorem correct_operation (a b : ℝ) : 
  (3 * Real.sqrt 7 + 7 * Real.sqrt 3 ≠ 10 * Real.sqrt 10) ∧ 
  (Real.sqrt (2 * a) * Real.sqrt (3) * a = Real.sqrt (6) * a) ∧ 
  (Real.sqrt a - Real.sqrt b ≠ Real.sqrt (a - b)) ∧ 
  (Real.sqrt (20 / 45) ≠ 4 / 9) :=
by
  sorry

end correct_operation_l30_30470


namespace regular_polygon_sides_l30_30319

theorem regular_polygon_sides (θ : ℝ) (n : ℕ) (h1 : θ = 18) (h2 : 360 = θ * n) : n = 20 :=
by {
  sorry
}

end regular_polygon_sides_l30_30319


namespace probability_x_plus_y_lt_4_inside_square_l30_30314

def square_area : ℝ := 9
def triangle_area : ℝ := 2
def probability : ℝ := 7 / 9

theorem probability_x_plus_y_lt_4_inside_square :
  ∀ (x y : ℝ), 
  (0 ≤ x ∧ x ≤ 3) ∧ (0 ≤ y ∧ y ≤ 3) ∧ (x + y < 4) → 
  (triangle_area = 2) ∧ (square_area = 9) ∧ (probability = 7 / 9) :=
by
  intros x y h
  sorry

end probability_x_plus_y_lt_4_inside_square_l30_30314


namespace monotonic_increasing_interval_of_f_l30_30854

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.logb (1/2) (x^2))

theorem monotonic_increasing_interval_of_f : 
  (∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ < 0 ∧ -1 ≤ x₂ ∧ x₂ < 0 ∧ x₁ ≤ x₂ → f x₁ ≤ f x₂) ∧ 
  (∀ x : ℝ, f x ≥ 0) := sorry

end monotonic_increasing_interval_of_f_l30_30854


namespace trash_can_prices_and_minimum_A_can_purchase_l30_30012

theorem trash_can_prices_and_minimum_A_can_purchase 
  (x y : ℕ) 
  (h₁ : 3 * x + 4 * y = 580)
  (h₂ : 6 * x + 5 * y = 860)
  (total_trash_cans : ℕ)
  (total_cost : ℕ)
  (cond₃ : total_trash_cans = 200)
  (cond₄ : 60 * (total_trash_cans - x) + 100 * x ≤ 15000) : 
  x = 60 ∧ y = 100 ∧ x ≥ 125 := 
sorry

end trash_can_prices_and_minimum_A_can_purchase_l30_30012


namespace sector_radius_l30_30096

theorem sector_radius (α S r : ℝ) (h1 : α = 3/4 * Real.pi) (h2 : S = 3/2 * Real.pi) :
  S = 1/2 * r^2 * α → r = 2 :=
by
  sorry

end sector_radius_l30_30096


namespace norm_two_u_l30_30513

noncomputable def vector_u : ℝ × ℝ := sorry

theorem norm_two_u {u : ℝ × ℝ} (hu : ∥u∥ = 5) : ∥(2 : ℝ) • u∥ = 10 := by
  sorry

end norm_two_u_l30_30513


namespace total_monsters_l30_30188

theorem total_monsters (a1 a2 a3 a4 a5 : ℕ) 
  (h1 : a1 = 2) 
  (h2 : a2 = 2 * a1) 
  (h3 : a3 = 2 * a2) 
  (h4 : a4 = 2 * a3) 
  (h5 : a5 = 2 * a4) : 
  a1 + a2 + a3 + a4 + a5 = 62 :=
by
  sorry

end total_monsters_l30_30188


namespace probability_two_dice_show_1_l30_30879

theorem probability_two_dice_show_1 :
  let n := 12 in
  let p := 1/6 in
  let k := 2 in
  let binom := Nat.choose 12 2 in
  let prob := binom * (p^2) * ((1 - p)^(n - k)) in
  Float.toNearest prob = 0.294 := 
by 
  let n := 12
  let p := 1/6
  let k := 2
  let binom := Nat.choose 12 2
  let prob := binom * (p^2) * ((1 - p)^(n - k))
  have answer : Float.toNearest prob = 0.294 := sorry
  exact answer

end probability_two_dice_show_1_l30_30879


namespace arithmetic_progression_x_value_l30_30505

theorem arithmetic_progression_x_value :
  ∀ (x : ℝ), (3 * x + 2) - (2 * x - 4) = (5 * x - 1) - (3 * x + 2) → x = 9 :=
by
  intros x h
  sorry

end arithmetic_progression_x_value_l30_30505


namespace sum_r_j_eq_3_l30_30856

variable (p r j : ℝ)

theorem sum_r_j_eq_3
  (h : (6 * p^2 - 4 * p + r) * (2 * p^2 + j * p - 7) = 12 * p^4 - 34 * p^3 - 19 * p^2 + 28 * p - 21) :
  r + j = 3 := by
  sorry

end sum_r_j_eq_3_l30_30856


namespace evaluate_expression_l30_30760

theorem evaluate_expression : 8^8 * 27^8 * 8^27 * 27^27 = 216^35 :=
by sorry

end evaluate_expression_l30_30760


namespace even_function_periodicity_l30_30781

noncomputable def f : ℝ → ℝ :=
sorry -- The actual function definition is not provided here but assumed to exist.

theorem even_function_periodicity (x : ℝ) (h1 : 1 ≤ x ∧ x ≤ 2)
  (h2 : f (x + 2) = f x)
  (hf_even : ∀ x, f x = f (-x))
  (hf_segment : ∀ x, 1 ≤ x ∧ x ≤ 2 → f x = x^2 + 2*x - 1) :
  ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x^2 - 6*x + 7 :=
sorry

end even_function_periodicity_l30_30781


namespace equal_ratios_l30_30177

variable (x y : ℝ)

-- Conditions
def wire_split_to_form_square_and_pentagon (x y : ℝ) : Prop :=
  4 * (x / 4) = 5 * (y / 5)

-- Theorem to prove
theorem equal_ratios (x y : ℝ) (h : wire_split_to_form_square_and_pentagon x y) : x / y = 1 :=
  sorry

end equal_ratios_l30_30177


namespace probability_two_dice_showing_1_l30_30877

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_two_dice_showing_1 : 
  binomial_probability 12 2 (1/6) ≈ 0.295 := 
by 
  sorry

end probability_two_dice_showing_1_l30_30877


namespace haniMoreSitupsPerMinute_l30_30533

-- Define the conditions given in the problem
def totalSitups : Nat := 110
def situpsByDiana : Nat := 40
def rateDianaPerMinute : Nat := 4

-- Define the derived conditions from the solution steps
def timeDianaMinutes := situpsByDiana / rateDianaPerMinute -- 10 minutes
def situpsByHani := totalSitups - situpsByDiana -- 70 situps
def rateHaniPerMinute := situpsByHani / timeDianaMinutes -- 7 situps per minute

-- The theorem we need to prove
theorem haniMoreSitupsPerMinute : rateHaniPerMinute - rateDianaPerMinute = 3 :=
by
  -- Placeholder for the actual proof
  sorry

end haniMoreSitupsPerMinute_l30_30533


namespace initial_people_count_l30_30164

theorem initial_people_count (left remaining total : ℕ) (h1 : left = 6) (h2 : remaining = 5) : total = 11 :=
  by
  sorry

end initial_people_count_l30_30164


namespace lisa_caffeine_l30_30292

theorem lisa_caffeine (caffeine_per_cup : ℕ) (daily_goal : ℕ) (cups_drank : ℕ) : caffeine_per_cup = 80 → daily_goal = 200 → cups_drank = 3 → (caffeine_per_cup * cups_drank - daily_goal) = 40 :=
by
  -- This is a theorem statement, thus no proof is provided here.
  sorry

end lisa_caffeine_l30_30292


namespace math_problem_l30_30117

theorem math_problem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a / (1 + a) + b / (1 + b) = 1) :
  a / (1 + b^2) - b / (1 + a^2) = a - b :=
sorry

end math_problem_l30_30117


namespace total_people_at_gathering_l30_30181

theorem total_people_at_gathering (total_wine : ℕ) (total_soda : ℕ) (both_wine_soda : ℕ) 
    (H1 : total_wine = 26) (H2 : total_soda = 22) (H3 : both_wine_soda = 17) : 
    total_wine - both_wine_soda + total_soda - both_wine_soda + both_wine_soda = 31 := 
by
  rw [H1, H2, H3]
  exact Nat.correct_answer = 31 -- combining results
  rw [Nat.sub_add_cancel (Nat.le_of_lt (sorry))] -- just using properties
  exact nat.add_comm 17 9 -- final proof step
  sorry -- ending suggestion

end total_people_at_gathering_l30_30181


namespace parallel_vectors_m_eq_neg3_l30_30947

theorem parallel_vectors_m_eq_neg3 : 
  ∀ m : ℝ, (∀ (a b : ℝ × ℝ), a = (1, -2) → b = (1 + m, 1 - m) → a.1 * b.2 - a.2 * b.1 = 0) → m = -3 :=
by
  intros m h_par
  specialize h_par (1, -2) (1 + m, 1 - m) rfl rfl
  -- We need to show m = -3
  sorry

end parallel_vectors_m_eq_neg3_l30_30947


namespace monomial_properties_l30_30472

noncomputable def monomial_coeff : ℚ := -(3/5 : ℚ)

def monomial_degree (x y : ℤ) : ℕ :=
  1 + 2

theorem monomial_properties (x y : ℤ) :
  monomial_coeff = -(3/5) ∧ monomial_degree x y = 3 :=
by
  -- Proof is to be filled here
  sorry

end monomial_properties_l30_30472


namespace product_gcd_lcm_l30_30770

-- Conditions.
def num1 : ℕ := 12
def num2 : ℕ := 9

-- Theorem to prove.
theorem product_gcd_lcm (a b : ℕ) (h1 : a = num1) (h2 : b = num2) :
  (Nat.gcd a b) * (Nat.lcm a b) = 108 :=
by
  sorry

end product_gcd_lcm_l30_30770


namespace stratified_sampling_correct_l30_30600

-- Definitions based on the conditions
def total_employees : ℕ := 300
def over_40 : ℕ := 50
def between_30_and_40 : ℕ := 150
def under_30 : ℕ := 100
def sample_size : ℕ := 30
def stratified_ratio : ℕ := 1 / 10  -- sample_size / total_employees

-- Function to compute the number of individuals sampled from each age group
def sampled_from_age_group (group_size : ℕ) : ℕ :=
  group_size * stratified_ratio

-- Mathematical properties to be proved
theorem stratified_sampling_correct :
  sampled_from_age_group over_40 = 5 ∧ 
  sampled_from_age_group between_30_and_40 = 15 ∧ 
  sampled_from_age_group under_30 = 10 := by
  sorry

end stratified_sampling_correct_l30_30600


namespace quadratic_root_property_l30_30944

theorem quadratic_root_property (m p : ℝ) 
  (h1 : (p^2 - 2 * p + m - 1 = 0)) 
  (h2 : (p^2 - 2 * p + 3) * (m + 4) = 7)
  (h3 : ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ r1^2 - 2 * r1 + m - 1 = 0 ∧ r2^2 - 2 * r2 + m - 1 = 0) : 
  m = -3 :=
by 
  sorry

end quadratic_root_property_l30_30944


namespace simplify_polynomial_problem_l30_30272

theorem simplify_polynomial_problem (p : ℝ) :
  (5 * p^4 - 4 * p^3 + 3 * p + 2) + (-3 * p^4 + 2 * p^3 - 7 * p^2 + 8) = 2 * p^4 - 2 * p^3 - 7 * p^2 + 3 * p + 10 := 
by
  sorry

end simplify_polynomial_problem_l30_30272


namespace abs_inequality_solution_set_l30_30859

theorem abs_inequality_solution_set (x : ℝ) :
  |x - 1| + |x + 2| < 5 ↔ -3 < x ∧ x < 2 :=
by {
  sorry
}

end abs_inequality_solution_set_l30_30859


namespace cylinder_volume_ratio_l30_30485

theorem cylinder_volume_ratio (s : ℝ) :
  let r := s / 2
  let h := s
  let V_cylinder := π * r^2 * h
  let V_cube := s^3
  V_cylinder / V_cube = π / 4 :=
by
  sorry

end cylinder_volume_ratio_l30_30485


namespace consecutive_integers_sum_l30_30693

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l30_30693


namespace consecutive_integer_sum_l30_30603

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l30_30603


namespace solve_equation_l30_30360

theorem solve_equation (x : ℝ) (h : 16 * x^2 = 81) : x = 9 / 4 ∨ x = - (9 / 4) :=
by
  sorry

end solve_equation_l30_30360


namespace sin_13pi_over_6_equals_half_l30_30763

noncomputable def sin_13pi_over_6 : ℝ := Real.sin (13 * Real.pi / 6)

theorem sin_13pi_over_6_equals_half : sin_13pi_over_6 = 1 / 2 := by
  sorry

end sin_13pi_over_6_equals_half_l30_30763


namespace cone_lateral_surface_area_l30_30803

theorem cone_lateral_surface_area (r V: ℝ) (h : ℝ) (l : ℝ) (L: ℝ):
  r = 3 →
  V = 12 * Real.pi →
  V = (1 / 3) * Real.pi * r^2 * h →
  l = Real.sqrt (r^2 + h^2) →
  L = Real.pi * r * l →
  L = 15 * Real.pi :=
by
  intros hr hv hV hl hL
  rw [hr, hv] at hV
  sorry

end cone_lateral_surface_area_l30_30803


namespace time_to_fill_bucket_l30_30397

theorem time_to_fill_bucket (t : ℝ) (h : 2/3 = 2 / t) : t = 3 :=
by
  sorry

end time_to_fill_bucket_l30_30397


namespace john_total_distance_traveled_l30_30109

theorem john_total_distance_traveled :
  let d1 := 45 * 2.5
  let d2 := 60 * 3.5
  let d3 := 40 * 2
  let d4 := 55 * 3
  d1 + d2 + d3 + d4 = 567.5 := by
  sorry

end john_total_distance_traveled_l30_30109


namespace consecutive_integers_sum_l30_30618

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l30_30618


namespace arjun_becca_3_different_colors_l30_30497

open Classical

noncomputable def arjun_becca_probability : ℚ := 
  let arjun_initial := [2, 1, 1, 1] -- 2 red, 1 green, 1 yellow, 1 violet
  let becca_initial := [2, 1] -- 2 black, 1 orange
  
  -- possible cases represented as a list of probabilities
  let cases := [
    (2/5) * (1/4) * (3/5),    -- Case 1: Arjun does move a red ball to Becca, and then processes accordingly
    (3/5) * (1/2) * (1/5),    -- Case 2a: Arjun moves a non-red ball, followed by Becca moving a black ball, concluding in the defined manner
    (3/5) * (1/2) * (3/5)     -- Case 2b: Arjun moves a non-red ball, followed by Becca moving a non-black ball, again concluding appropriately
  ]
  
  -- sum of cases representing the total probability
  let total_probability := List.sum cases
  
  total_probability

theorem arjun_becca_3_different_colors : arjun_becca_probability = 3/10 := 
  by
    simp [arjun_becca_probability]
    sorry

end arjun_becca_3_different_colors_l30_30497


namespace consecutive_integer_sum_l30_30604

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l30_30604


namespace oliver_shelves_needed_l30_30833

-- Definitions based on conditions
def total_books : ℕ := 46
def books_taken_by_librarian : ℕ := 10
def books_remaining (total books_taken : ℕ) : ℕ := total - books_taken
def books_per_shelf : ℕ := 4

-- Theorem statement
theorem oliver_shelves_needed :
  books_remaining total_books books_taken_by_librarian / books_per_shelf = 9 := by
  sorry

end oliver_shelves_needed_l30_30833


namespace alyssa_cookie_count_l30_30336

variable (Aiyanna_cookies Alyssa_cookies : ℕ)
variable (h1 : Aiyanna_cookies = 140)
variable (h2 : Aiyanna_cookies = Alyssa_cookies + 11)

theorem alyssa_cookie_count : Alyssa_cookies = 129 := by
  -- We can use the given conditions to prove the theorem
  sorry

end alyssa_cookie_count_l30_30336


namespace sum_of_consecutive_integers_with_product_812_l30_30659

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l30_30659


namespace number_of_linear_eqs_l30_30524

def is_linear_eq_in_one_var (eq : String) : Bool :=
  match eq with
  | "0.3x = 1" => true
  | "x/2 = 5x + 1" => true
  | "x = 6" => true
  | _ => false

theorem number_of_linear_eqs :
  let eqs := ["x - 2 = 2 / x", "0.3x = 1", "x/2 = 5x + 1", "x^2 - 4x = 3", "x = 6", "x + 2y = 0"]
  (eqs.filter is_linear_eq_in_one_var).length = 3 :=
by
  sorry

end number_of_linear_eqs_l30_30524


namespace arithmetic_seq_term_298_eq_100_l30_30775

-- Define the arithmetic sequence
def arithmetic_seq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Define the specific sequence given in the problem
def a_n (n : ℕ) : ℕ := arithmetic_seq 1 3 n

-- State the theorem
theorem arithmetic_seq_term_298_eq_100 : a_n 100 = 298 :=
by
  -- Proof will be filled in
  sorry

end arithmetic_seq_term_298_eq_100_l30_30775


namespace probability_two_ones_in_twelve_dice_l30_30876
noncomputable theory

def probability_of_exactly_two_ones (n : ℕ) (p : ℚ) :=
  (nat.choose n 2 : ℚ) * p^2 * (1 - p)^(n - 2)

theorem probability_two_ones_in_twelve_dice :
  probability_of_exactly_two_ones 12 (1/6) ≈ 0.298 :=
by
  sorry

end probability_two_ones_in_twelve_dice_l30_30876


namespace problem_l30_30626

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l30_30626


namespace rationalize_denominator_and_product_l30_30984

theorem rationalize_denominator_and_product :
  let A := -11
  let B := -5
  let C := 5
  let expr := (3 + Real.sqrt 5) / (2 - Real.sqrt 5)
  (expr * (2 + Real.sqrt 5) / (2 + Real.sqrt 5) = A + B * Real.sqrt C) ∧ (A * B * C = 275) :=
by
  sorry

end rationalize_denominator_and_product_l30_30984


namespace problem_statement_l30_30973

noncomputable def α : ℝ := 3 + 2 * Real.sqrt 2
noncomputable def β : ℝ := 3 - 2 * Real.sqrt 2
noncomputable def x : ℝ := α ^ 50
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := by
  sorry

end problem_statement_l30_30973


namespace sum_of_consecutive_integers_with_product_812_l30_30662

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l30_30662


namespace cubics_inequality_l30_30363

theorem cubics_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) : a^3 + b^3 ≥ a^2 * b + a * b^2 :=
by
  sorry

end cubics_inequality_l30_30363


namespace percentage_of_students_choose_harvard_l30_30232

theorem percentage_of_students_choose_harvard
  (total_applicants : ℕ)
  (acceptance_rate : ℝ)
  (students_attend_harvard : ℕ)
  (students_attend_other : ℝ)
  (percentage_attended_harvard : ℝ) :
  total_applicants = 20000 →
  acceptance_rate = 0.05 →
  students_attend_harvard = 900 →
  students_attend_other = 0.10 →
  percentage_attended_harvard = ((students_attend_harvard / (total_applicants * acceptance_rate)) * 100) →
  percentage_attended_harvard = 90 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end percentage_of_students_choose_harvard_l30_30232


namespace largest_side_of_rectangle_l30_30047

theorem largest_side_of_rectangle (l w : ℝ) 
    (h1 : 2 * l + 2 * w = 240) 
    (h2 : l * w = 1920) : 
    max l w = 101 := 
sorry

end largest_side_of_rectangle_l30_30047


namespace sin_240_eq_neg_sqrt3_over_2_l30_30157

theorem sin_240_eq_neg_sqrt3_over_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_240_eq_neg_sqrt3_over_2_l30_30157


namespace trash_can_prices_and_minimum_A_can_purchase_l30_30013

theorem trash_can_prices_and_minimum_A_can_purchase 
  (x y : ℕ) 
  (h₁ : 3 * x + 4 * y = 580)
  (h₂ : 6 * x + 5 * y = 860)
  (total_trash_cans : ℕ)
  (total_cost : ℕ)
  (cond₃ : total_trash_cans = 200)
  (cond₄ : 60 * (total_trash_cans - x) + 100 * x ≤ 15000) : 
  x = 60 ∧ y = 100 ∧ x ≥ 125 := 
sorry

end trash_can_prices_and_minimum_A_can_purchase_l30_30013


namespace functions_not_necessarily_equal_l30_30957

-- Define the domain and range
variables {α β : Type*}

-- Define two functions f and g with the same domain and range
variables (f g : α → β)

-- Lean statement for the given mathematical problem
theorem functions_not_necessarily_equal (h_domain : ∀ x : α, (∃ x : α, true))
  (h_range : ∀ y : β, (∃ y : β, true)) : ¬(f = g) :=
sorry

end functions_not_necessarily_equal_l30_30957


namespace female_democrats_l30_30476

/-
There are 810 male and female participants in a meeting.
Half of the female participants and one-quarter of the male participants are Democrats.
One-third of all the participants are Democrats.
Prove that the number of female Democrats is 135.
-/

theorem female_democrats (F M : ℕ) (h : F + M = 810)
  (female_democrats : F / 2 = F / 2)
  (male_democrats : M / 4 = M / 4)
  (total_democrats : (F / 2 + M / 4) = 810 / 3) : 
  F / 2 = 135 := by
  sorry

end female_democrats_l30_30476


namespace proof_problem_l30_30079

variable (m : ℝ)

theorem proof_problem 
  (h1 : ∀ x, (m / (x - 2) + 1 = x / (2 - x)) → x ≠ 2 ∧ x ≥ 0) :
  m ≤ 2 ∧ m ≠ -2 := by
  sorry

end proof_problem_l30_30079


namespace proof_problem_l30_30077

variable (m : ℝ)

theorem proof_problem 
  (h1 : ∀ x, (m / (x - 2) + 1 = x / (2 - x)) → x ≠ 2 ∧ x ≥ 0) :
  m ≤ 2 ∧ m ≠ -2 := by
  sorry

end proof_problem_l30_30077


namespace sandra_beignets_l30_30264

theorem sandra_beignets (beignets_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) (h1 : beignets_per_day = 3) (h2 : days_per_week = 7) (h3: weeks = 16) : 
  (beignets_per_day * days_per_week * weeks) = 336 :=
by {
  -- the proof goes here
  sorry
}

end sandra_beignets_l30_30264


namespace consecutive_integer_sum_l30_30612

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l30_30612


namespace four_sq_geq_prod_sum_l30_30305

variable {α : Type*} [LinearOrderedField α]

theorem four_sq_geq_prod_sum (a b c d : α) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) :=
sorry

end four_sq_geq_prod_sum_l30_30305


namespace consecutive_integer_product_sum_l30_30644

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l30_30644


namespace find_area_of_triangle_l30_30072
  
def point (α : Type) := α × α

variables (A B P₁ P₂ P₃ : point ℝ)
variables (l₁ l₂ l₃ : point ℝ → Prop)

def minimize_sum_of_squared_distances (p: point ℝ) :=
  let x1 := p.1, y1 := p.2,
      a1 := A.1, a2 := A.2, 
      b1 := B.1, b2 := B.2 in
  (a1 - x1)^2 + (a2 - y1)^2 + (b1 - x1)^2 + (b2 - y1)^2

def line_l1 (p : point ℝ) : Prop := p.1 = 0
def line_l2 (p : point ℝ) : Prop := p.2 = 0
def line_l3 (p : point ℝ) : Prop := p.1 + 3*p.2 - 1 = 0

theorem find_area_of_triangle : 
  ∃ (P₁ P₂ P₃ : point ℝ), (
  line_l1 P₁ ∧
  line_l2 P₂ ∧
  line_l3 P₃ ∧
  (minimize_sum_of_squared_distances A P₁ + minimize_sum_of_squared_distances B P₁) ≤
  min (minimize_sum_of_squared_distances A P₂ + minimize_sum_of_squared_distances B P₂)
  (minimize_sum_of_squared_distances A P₃ + minimize_sum_of_squared_distances B P₃) ∧
  let
  P₁ := (0, 3),
  P₂ := (2, 0),
  P₃ := (1, 0) in 
  1 / 2 * abs ((1 - 0) * (3 - 0)) = 3 / 2
  ) := sorry

end find_area_of_triangle_l30_30072


namespace height_of_screen_is_100_l30_30001

-- Definitions for the conditions and the final proof statement
def side_length_of_square_paper := 20 -- cm

def perimeter_of_square_paper (s : ℕ) : ℕ := 4 * s

def height_of_computer_screen (P : ℕ) := P + 20

theorem height_of_screen_is_100 :
  let s := side_length_of_square_paper in
  let P := perimeter_of_square_paper s in
  height_of_computer_screen P = 100 :=
by
  sorry

end height_of_screen_is_100_l30_30001


namespace value_of_a_star_b_l30_30602

variable (a b : ℤ)

def operation_star (a b : ℤ) : ℚ :=
  1 / a + 1 / b

theorem value_of_a_star_b (h1 : a + b = 7) (h2 : a * b = 12) :
  operation_star a b = 7 / 12 := by
  sorry

end value_of_a_star_b_l30_30602


namespace total_number_of_crayons_l30_30396

def number_of_blue_crayons := 3
def number_of_red_crayons := 4 * number_of_blue_crayons
def number_of_green_crayons := 2 * number_of_red_crayons
def number_of_yellow_crayons := number_of_green_crayons / 2

theorem total_number_of_crayons :
  number_of_blue_crayons + number_of_red_crayons + number_of_green_crayons + number_of_yellow_crayons = 51 :=
by 
  -- Proof is not required
  sorry

end total_number_of_crayons_l30_30396


namespace tenth_term_of_arithmetic_progression_l30_30350

variable (a d n T_n : ℕ)

def arithmetic_progression (a d n : ℕ) : ℕ := a + (n - 1) * d

theorem tenth_term_of_arithmetic_progression :
  arithmetic_progression 8 2 10 = 26 :=
  by
  sorry

end tenth_term_of_arithmetic_progression_l30_30350


namespace graph_of_equation_is_hyperbola_l30_30049

theorem graph_of_equation_is_hyperbola (x y : ℝ):
  (x^2 - 9 * y^2 + 6 * x = 0) → ∃ a b h k : ℝ, ∀ x y : ℝ, (x + 3)^2 / 9 - y^2 = 1 :=
by
  sorry

end graph_of_equation_is_hyperbola_l30_30049


namespace ariana_average_speed_l30_30259

theorem ariana_average_speed
  (sadie_speed : ℝ)
  (sadie_time : ℝ)
  (ariana_time : ℝ)
  (sarah_speed : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (sadie_speed_eq : sadie_speed = 3)
  (sadie_time_eq : sadie_time = 2)
  (ariana_time_eq : ariana_time = 0.5)
  (sarah_speed_eq : sarah_speed = 4)
  (total_time_eq : total_time = 4.5)
  (total_distance_eq : total_distance = 17) :
  ∃ ariana_speed : ℝ, ariana_speed = 6 :=
by {
  sorry
}

end ariana_average_speed_l30_30259


namespace base_8_add_sub_l30_30358

-- Definitions of the numbers in base 8
def n1 : ℕ := 4 * 8^2 + 5 * 8^1 + 1 * 8^0
def n2 : ℕ := 1 * 8^2 + 6 * 8^1 + 2 * 8^0
def n3 : ℕ := 1 * 8^2 + 2 * 8^1 + 3 * 8^0

-- Convert the result to base 8
def to_base_8 (n : ℕ) : ℕ :=
  let d2 := n / 64
  let rem1 := n % 64
  let d1 := rem1 / 8
  let d0 := rem1 % 8
  d2 * 100 + d1 * 10 + d0

-- Proof statement
theorem base_8_add_sub :
  to_base_8 ((n1 + n2) - n3) = to_base_8 (5 * 8^2 + 1 * 8^1 + 0 * 8^0) :=
by
  sorry

end base_8_add_sub_l30_30358


namespace min_sum_of_angles_l30_30811

theorem min_sum_of_angles (A B C : ℝ) (hABC : A + B + C = 180) (h_sin : Real.sin A + Real.sin B + Real.sin C ≤ 1) : 
  min (A + B) (min (B + C) (C + A)) < 30 := 
sorry

end min_sum_of_angles_l30_30811


namespace range_of_a_l30_30954

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (a ∈ [-1, 3]) := 
by
  sorry

end range_of_a_l30_30954


namespace leah_coins_value_l30_30239

theorem leah_coins_value
  (p n : ℕ)
  (h₁ : n + p = 15)
  (h₂ : n + 2 = p) : p + 5 * n = 38 :=
by
  -- definitions used in converting conditions
  sorry

end leah_coins_value_l30_30239


namespace find_second_number_l30_30488

-- Define the given number
def given_number := 220070

-- Define the constants in the problem
def constant_555 := 555
def remainder := 70

-- Define the second number (our unknown)
variable (x : ℕ)

-- Define the condition as an equation
def condition : Prop :=
  given_number = (constant_555 + x) * 2 * (x - constant_555) + remainder

-- The theorem to prove that the second number is 343
theorem find_second_number : ∃ x : ℕ, condition x ∧ x = 343 :=
sorry

end find_second_number_l30_30488


namespace point_in_fourth_quadrant_l30_30405

theorem point_in_fourth_quadrant (x : ℝ) (y : ℝ) (hx : x = 8) (hy : y = -3) : x > 0 ∧ y < 0 :=
by
  sorry

end point_in_fourth_quadrant_l30_30405


namespace value_of_a_minus_b_l30_30537

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 6) (h2 : |b| = 2) (h3 : a + b > 0) :
  a - b = 4 ∨ a - b = 8 :=
  sorry

end value_of_a_minus_b_l30_30537


namespace range_of_a_l30_30115

open Set

theorem range_of_a (a x : ℝ) (p : ℝ → Prop) (q : ℝ → ℝ → Prop)
    (hp : p x → |x - a| > 3)
    (hq : q x a → (x + 1) * (2 * x - 1) ≥ 0)
    (hsuff : ∀ x, ¬p x → q x a) :
    {a | ∀ x, (¬ (|x - a| > 3) → (x + 1) * (2 * x - 1) ≥ 0) → (( a ≤ -4) ∨ (a ≥ 7 / 2))} :=
by
  sorry

end range_of_a_l30_30115


namespace total_students_l30_30806

theorem total_students (S F G B N : ℕ) 
  (hF : F = 41) 
  (hG : G = 22) 
  (hB : B = 9) 
  (hN : N = 24) 
  (h_total : S = (F + G - B) + N) : 
  S = 78 :=
by
  sorry

end total_students_l30_30806


namespace count_distinct_four_digit_numbers_from_2025_l30_30794

-- Define the digits 2, 0, 2, and 5 as a multiset
def digits : Multiset ℕ := {2, 0, 2, 5}

-- Define the set of valid four-digit numbers formed from the digits
def four_digit_numbers (d : Multiset ℕ) : Finset ℕ :=
  Finset.filter (λ x : ℕ, 1000 ≤ x ∧ x < 10000) (Multiset.permutations d).to_finset.map
    (λ ds, ds.foldr (λ a b, a + 10 * b) 0)

-- The theorem we aim to prove
theorem count_distinct_four_digit_numbers_from_2025 : 
  (four_digit_numbers digits).card = 7 :=
sorry

end count_distinct_four_digit_numbers_from_2025_l30_30794


namespace total_worth_of_presents_l30_30565

-- Define the costs as given in the conditions
def ring_cost : ℕ := 4000
def car_cost : ℕ := 2000
def bracelet_cost : ℕ := 2 * ring_cost

-- Define the total worth of the presents
def total_worth : ℕ := ring_cost + car_cost + bracelet_cost

-- Statement: Prove the total worth is 14000
theorem total_worth_of_presents : total_worth = 14000 :=
by
  -- Here is the proof statement
  sorry

end total_worth_of_presents_l30_30565


namespace find_b_plus_c_l30_30389

variable {a b c d : ℝ}

theorem find_b_plus_c
  (h1 : a + b = 4)
  (h2 : c + d = 3)
  (h3 : a + d = 2) :
  b + c = 5 := 
  by
  sorry

end find_b_plus_c_l30_30389


namespace total_arrangement_methods_l30_30054

variable (M F: Type) [Fintype M] [Fintype F]

def num_doctors := 4
def num_male := 2
def num_female := 2
def num_hospitals := 3
def at_least_one_doctor (h : Finset (M ⊕ F)) : Prop := 
  0 < h.card

theorem total_arrangement_methods :
  let male_doctors : Finset M := Finset.univ.filter (λ (m : M), m ∈ univ)
  let female_doctors : Finset F := Finset.univ.filter (λ (f : F), f ∈ univ)
  let hospitals : Finset (M ⊕ F) := univ

  (∀ (h : Finset (M ⊕ F)), at_least_one_doctor h) →
  (∀ (m1 m2 : M), m1 ≠ m2 → none ∈ hospitals \ (finset.filter m1 hospitals) ∩ finset.filter m2 hospitals) →
  Fintype.card {arrangement // ∃ (A : Finset M) (B : Finset F)
      (C : Finset (M ⊕ F)), 
      A ∪ (C \ (C.filter A)) ∪ (C \ (C.union (A.filter B))) ∈ univ 
      ∧ (∀ i ∈ A, i ∈ hospitals \ B) 
      ∧ ∀ (j : F), j ∈ B} = 18 := 
sorry

end total_arrangement_methods_l30_30054


namespace ratio_of_perimeters_l30_30842

theorem ratio_of_perimeters (A1 A2 : ℝ) (h : A1 / A2 = 16 / 81) : 
  let s1 := real.sqrt A1 
  let s2 := real.sqrt A2 
  (4 * s1) / (4 * s2) = 4 / 9 :=
by {
  sorry
}

end ratio_of_perimeters_l30_30842


namespace bob_needs_50_planks_l30_30746

-- Define the raised bed dimensions and requirements
structure RaisedBedDimensions where
  height : ℕ -- in feet
  width : ℕ  -- in feet
  length : ℕ -- in feet

def plank_length : ℕ := 8  -- length of each plank in feet
def plank_width : ℕ := 1  -- width of each plank in feet
def num_beds : ℕ := 10

def planks_needed (bed : RaisedBedDimensions) : ℕ :=
  let long_sides := 2  -- 2 long sides per bed
  let short_sides := 2 * (bed.width / plank_length)  -- 1/4 plank per short side if width is 2 feet
  let total_sides := long_sides + short_sides
  let stacked_sides := total_sides * (bed.height / plank_width)  -- stacked to match height
  stacked_sides

def raised_bed : RaisedBedDimensions := {height := 2, width := 2, length := 8}

theorem bob_needs_50_planks : planks_needed raised_bed * num_beds = 50 := by
  sorry

end bob_needs_50_planks_l30_30746


namespace factorial_sum_power_of_two_l30_30766

theorem factorial_sum_power_of_two (a b c n : ℕ) (h : a ≤ b ∧ b ≤ c) :
  a! + b! + c! = 2^n →
  (a = 1 ∧ b = 1 ∧ c = 2) ∨
  (a = 1 ∧ b = 1 ∧ c = 3) ∨
  (a = 2 ∧ b = 3 ∧ c = 4) ∨
  (a = 2 ∧ b = 3 ∧ c = 5) :=
by
  sorry

end factorial_sum_power_of_two_l30_30766


namespace symmetry_center_2tan_2x_sub_pi_div_4_l30_30848

theorem symmetry_center_2tan_2x_sub_pi_div_4 (k : ℤ) :
  ∃ (x : ℝ), 2 * (x) - π / 4 = k * π / 2 ∧ x = k * π / 4 + π / 8 :=
by
  sorry

end symmetry_center_2tan_2x_sub_pi_div_4_l30_30848


namespace initial_winnings_l30_30561

theorem initial_winnings (X : ℝ) 
  (h1 : X - 0.25 * X = 0.75 * X)
  (h2 : 0.75 * X - 0.10 * (0.75 * X) = 0.675 * X)
  (h3 : 0.675 * X - 0.15 * (0.675 * X) = 0.57375 * X)
  (h4 : 0.57375 * X = 240) :
  X = 418 := by
  sorry

end initial_winnings_l30_30561


namespace probability_different_colors_l30_30231

theorem probability_different_colors :
  let total_chips := 18
  let blue_chips := 7
  let red_chips := 6
  let yellow_chips := 5
  let prob_first_blue := blue_chips / total_chips
  let prob_first_red := red_chips / total_chips
  let prob_first_yellow := yellow_chips / total_chips
  let prob_second_not_blue := (red_chips + yellow_chips) / (total_chips - 1)
  let prob_second_not_red := (blue_chips + yellow_chips) / (total_chips - 1)
  let prob_second_not_yellow := (blue_chips + red_chips) / (total_chips - 1)
  (
    prob_first_blue * prob_second_not_blue +
    prob_first_red * prob_second_not_red +
    prob_first_yellow * prob_second_not_yellow
  ) = 122 / 153 :=
by sorry

end probability_different_colors_l30_30231


namespace math_problem_l30_30228

noncomputable def problem_statement (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (6 - a) * (6 - b) * (6 - c) * (6 - d) = 9

theorem math_problem
  (a b c d : ℕ)
  (h1 : a ≠ b)
  (h2 : a ≠ c)
  (h3 : a ≠ d)
  (h4 : b ≠ c)
  (h5 : b ≠ d)
  (h6 : c ≠ d)
  (h7 : (6 - a) * (6 - b) * (6 - c) * (6 - d) = 9) :
  a + b + c + d = 24 :=
sorry

end math_problem_l30_30228


namespace counterexample_to_strict_inequality_l30_30559

theorem counterexample_to_strict_inequality :
  ∃ (a1 a2 b1 b2 c1 c2 d1 d2 : ℕ),
  (0 < a1) ∧ (0 < a2) ∧ (0 < b1) ∧ (0 < b2) ∧ (0 < c1) ∧ (0 < c2) ∧ (0 < d1) ∧ (0 < d2) ∧
  (a1 * b2 < a2 * b1) ∧ (c1 * d2 < c2 * d1) ∧ ¬ (a1 + c1) * (b2 + d2) < (a2 + c2) * (b1 + d1) :=
sorry

end counterexample_to_strict_inequality_l30_30559


namespace fencing_cost_per_meter_l30_30490

-- Definitions based on given conditions
def area : ℚ := 1200
def short_side : ℚ := 30
def total_cost : ℚ := 1800

-- Definition to represent the length of the long side
def long_side := area / short_side

-- Definition to represent the diagonal of the rectangle
def diagonal := (long_side^2 + short_side^2).sqrt

-- Definition to represent the total length of the fence
def total_length := long_side + short_side + diagonal

-- Definition to represent the cost per meter
def cost_per_meter := total_cost / total_length

-- Theorem statement asserting that cost_per_meter == 15
theorem fencing_cost_per_meter : cost_per_meter = 15 := 
by 
  sorry

end fencing_cost_per_meter_l30_30490


namespace complement_intersection_l30_30384

-- Define the universal set U.
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the set M.
def M : Set ℕ := {2, 3}

-- Define the set N.
def N : Set ℕ := {1, 3}

-- Define the complement of set M in U.
def complement_U_M : Set ℕ := {x ∈ U | x ∉ M}

-- Define the complement of set N in U.
def complement_U_N : Set ℕ := {x ∈ U | x ∉ N}

-- The statement to be proven.
theorem complement_intersection :
  (complement_U_M ∩ complement_U_N) = {4, 5, 6} :=
sorry

end complement_intersection_l30_30384


namespace largest_tangential_quadrilaterals_l30_30572

-- Definitions and conditions
def convex_ngon {n : ℕ} (h : n ≥ 5) : Type := sorry -- Placeholder for defining a convex n-gon with ≥ 5 sides
def tangential_quadrilateral {n : ℕ} (h : n ≥ 5) (k : ℕ) : Prop := 
  -- Placeholder for the property that exactly k quadrilaterals out of all possible ones 
  -- in a convex n-gon have an inscribed circle
  sorry

theorem largest_tangential_quadrilaterals {n : ℕ} (h : n ≥ 5) : 
  ∃ k : ℕ, tangential_quadrilateral h k ∧ k = n / 2 :=
sorry

end largest_tangential_quadrilaterals_l30_30572


namespace speed_in_still_water_l30_30312

theorem speed_in_still_water (U D : ℝ) (hU : U = 15) (hD : D = 25) : (U + D) / 2 = 20 :=
by
  rw [hU, hD]
  norm_num

end speed_in_still_water_l30_30312


namespace toothpicks_pattern_100th_stage_l30_30598

theorem toothpicks_pattern_100th_stage :
  let a_1 := 5
  let d := 4
  let n := 100
  (a_1 + (n - 1) * d) = 401 := by
  sorry

end toothpicks_pattern_100th_stage_l30_30598


namespace min_side_length_is_isosceles_l30_30837

-- Let a denote the side length BC
-- Let b denote the side length AB
-- Let c denote the side length AC

theorem min_side_length_is_isosceles (α : ℝ) (S : ℝ) (a b c : ℝ) :
  (a^2 = b^2 + c^2 - 2 * b * c * Real.cos α ∧ S = 0.5 * b * c * Real.sin α) →
  a = Real.sqrt (((b - c)^2 + (4 * S * (1 - Real.cos α)) / Real.sin α)) →
  b = c :=
by
  intros h1 h2
  sorry

end min_side_length_is_isosceles_l30_30837


namespace total_worth_of_presents_l30_30564

-- Define the costs as given in the conditions
def ring_cost : ℕ := 4000
def car_cost : ℕ := 2000
def bracelet_cost : ℕ := 2 * ring_cost

-- Define the total worth of the presents
def total_worth : ℕ := ring_cost + car_cost + bracelet_cost

-- Statement: Prove the total worth is 14000
theorem total_worth_of_presents : total_worth = 14000 :=
by
  -- Here is the proof statement
  sorry

end total_worth_of_presents_l30_30564


namespace unit_prices_min_number_of_A_l30_30011

theorem unit_prices (x y : ℝ)
  (h1 : 3 * x + 4 * y = 580)
  (h2 : 6 * x + 5 * y = 860) :
  x = 60 ∧ y = 100 :=
by
  sorry

theorem min_number_of_A (x y a : ℝ)
  (x_h : x = 60)
  (y_h : y = 100)
  (h1 : 3 * x + 4 * y = 580)
  (h2 : 6 * x + 5 * y = 860)
  (trash_can_condition : a + 200 - a = 200)
  (cost_condition : 60 * a + 100 * (200 - a) ≤ 15000) :
  a ≥ 125 :=
by
  sorry

end unit_prices_min_number_of_A_l30_30011


namespace fractional_equation_solution_l30_30074

theorem fractional_equation_solution (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) →
  (m ≤ 2 ∧ m ≠ -2) :=
by
  sorry

end fractional_equation_solution_l30_30074


namespace minimum_value_expression_l30_30373

theorem minimum_value_expression {x1 x2 x3 x4 : ℝ} (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) (hx4 : 0 < x4) (h_sum : x1 + x2 + x3 + x4 = Real.pi) :
  (2 * (Real.sin x1)^2 + 1 / (Real.sin x1)^2) * (2 * (Real.sin x2)^2 + 1 / (Real.sin x2)^2) * (2 * (Real.sin x3)^2 + 1 / (Real.sin x3)^2) * (2 * (Real.sin x4)^2 + 1 / (Real.sin x4)^2) ≥ 81 :=
by {
  sorry
}

end minimum_value_expression_l30_30373


namespace f_23_plus_f_neg14_l30_30388

noncomputable def f : ℝ → ℝ := sorry

axiom periodic_f : ∀ x, f (x + 5) = f x
axiom odd_f : ∀ x, f (-x) = -f x
axiom f_one : f 1 = 1
axiom f_two : f 2 = 2

theorem f_23_plus_f_neg14 : f 23 + f (-14) = -1 := by
  sorry

end f_23_plus_f_neg14_l30_30388


namespace best_player_total_hits_l30_30308

theorem best_player_total_hits
  (team_avg_hits_per_game : ℕ)
  (games_played : ℕ)
  (total_players : ℕ)
  (other_players_avg_hits_next_6_games : ℕ)
  (correct_answer : ℕ)
  (h1 : team_avg_hits_per_game = 15)
  (h2 : games_played = 5)
  (h3 : total_players = 11)
  (h4 : other_players_avg_hits_next_6_games = 6)
  (h5 : correct_answer = 25) :
  ∃ total_hits_of_best_player : ℕ,
  total_hits_of_best_player = correct_answer := by
  sorry

end best_player_total_hits_l30_30308


namespace consecutive_integers_sum_l30_30667

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l30_30667


namespace average_age_of_girls_l30_30962

theorem average_age_of_girls (total_students : ℕ) (boys_avg_age : ℝ) (school_avg_age : ℚ)
    (girls_count : ℕ) (total_age_school : ℝ) (boys_count : ℕ) 
    (total_age_boys : ℝ) (total_age_girls : ℝ): (total_students = 640) →
    (boys_avg_age = 12) →
    (school_avg_age = 47 / 4) →
    (girls_count = 160) →
    (total_students - girls_count = boys_count) →
    (boys_avg_age * boys_count = total_age_boys) →
    (school_avg_age * total_students = total_age_school) →
    (total_age_school - total_age_boys = total_age_girls) →
    total_age_girls / girls_count = 11 :=
by
  intros h_total_students h_boys_avg_age h_school_avg_age h_girls_count 
         h_boys_count h_total_age_boys h_total_age_school h_total_age_girls
  sorry

end average_age_of_girls_l30_30962


namespace largest_possible_percent_error_l30_30246

theorem largest_possible_percent_error 
  (r : ℝ) (delta : ℝ) (h_r : r = 15) (h_delta : delta = 0.1) : 
  ∃(error : ℝ), error = 0.21 :=
by
  -- The proof would go here
  sorry

end largest_possible_percent_error_l30_30246


namespace sandy_age_l30_30269

theorem sandy_age (S M : ℕ) 
  (h1 : M = S + 16) 
  (h2 : (↑S : ℚ) / ↑M = 7 / 9) : 
  S = 56 :=
by sorry

end sandy_age_l30_30269


namespace consecutive_integers_sum_l30_30616

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l30_30616


namespace shelves_needed_l30_30831

def total_books : ℕ := 46
def books_taken_by_librarian : ℕ := 10
def books_per_shelf : ℕ := 4
def remaining_books : ℕ := total_books - books_taken_by_librarian
def needed_shelves : ℕ := 9

theorem shelves_needed : remaining_books / books_per_shelf = needed_shelves := by
  sorry

end shelves_needed_l30_30831


namespace loraine_total_wax_l30_30976

-- Conditions
def large_animal_wax := 4
def small_animal_wax := 2
def small_animal_count := 12 / small_animal_wax
def large_animal_count := small_animal_count / 3
def total_wax := 12 + (large_animal_count * large_animal_wax)

-- The proof problem
theorem loraine_total_wax : total_wax = 20 := by
  sorry

end loraine_total_wax_l30_30976


namespace sum_of_consecutive_integers_with_product_812_l30_30676

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l30_30676


namespace min_value_expression_l30_30203

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1) : 
  (1 / a + 2) * (1 / b + 2) ≥ 16 :=
sorry

end min_value_expression_l30_30203


namespace sum_of_consecutive_integers_with_product_812_l30_30692

theorem sum_of_consecutive_integers_with_product_812 :
  ∃ x : ℕ, (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) := 
begin
  sorry
end

end sum_of_consecutive_integers_with_product_812_l30_30692


namespace operation_B_is_correct_l30_30471

theorem operation_B_is_correct (a b x : ℝ) : 
  2 * (a^2) * b * 4 * a * (b^3) = 8 * (a^3) * (b^4) :=
by
  sorry

-- Conditions for incorrect operations
lemma operation_A_is_incorrect (x : ℝ) : 
  x^8 / x^2 ≠ x^4 :=
by
  sorry

lemma operation_C_is_incorrect (x : ℝ) : 
  (-x^5)^4 ≠ -x^20 :=
by
  sorry

lemma operation_D_is_incorrect (a b : ℝ) : 
  (a + b)^2 ≠ a^2 + b^2 :=
by
  sorry

end operation_B_is_correct_l30_30471


namespace sum_of_ten_distinct_numbers_lt_75_l30_30254

theorem sum_of_ten_distinct_numbers_lt_75 :
  ∃ (S : Finset ℕ), S.card = 10 ∧
  (∃ (S_div_5 : Finset ℕ), S_div_5 ⊆ S ∧ S_div_5.card = 3 ∧ ∀ x ∈ S_div_5, 5 ∣ x) ∧
  (∃ (S_div_4 : Finset ℕ), S_div_4 ⊆ S ∧ S_div_4.card = 4 ∧ ∀ x ∈ S_div_4, 4 ∣ x) ∧
  S.sum id < 75 :=
by { 
  sorry 
}

end sum_of_ten_distinct_numbers_lt_75_l30_30254


namespace consecutive_integer_product_sum_l30_30645

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l30_30645


namespace negation_universal_to_existential_l30_30855

theorem negation_universal_to_existential :
  ¬ (∀ x : ℝ, x^2 - x ≥ 0) ↔ ∃ x : ℝ, x^2 - x < 0 :=
by
  sorry

end negation_universal_to_existential_l30_30855


namespace B_pow_2024_l30_30240

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![Real.cos (Real.pi / 4), 0, -Real.sin (Real.pi / 4)],
    ![0, 1, 0],
    ![Real.sin (Real.pi / 4), 0, Real.cos (Real.pi / 4)]
  ]

theorem B_pow_2024 :
  B ^ 2024 = ![
    ![-1, 0, 0],
    ![0, 1, 0],
    ![0, 0, -1]
  ] :=
by
  sorry

end B_pow_2024_l30_30240


namespace point_in_fourth_quadrant_l30_30407

def x : ℝ := 8
def y : ℝ := -3

theorem point_in_fourth_quadrant (h1 : x > 0) (h2 : y < 0) : (x > 0 ∧ y < 0) :=
by {
  sorry
}

end point_in_fourth_quadrant_l30_30407


namespace membership_fee_increase_each_year_l30_30901

variable (fee_increase : ℕ)

def yearly_membership_fee_increase (first_year_fee sixth_year_fee yearly_increase : ℕ) : Prop :=
  yearly_increase * 5 = sixth_year_fee - first_year_fee

theorem membership_fee_increase_each_year :
  yearly_membership_fee_increase 80 130 10 :=
by
  unfold yearly_membership_fee_increase
  sorry

end membership_fee_increase_each_year_l30_30901


namespace cost_comparison_for_30_pens_l30_30151

def cost_store_a (x : ℕ) : ℝ :=
  if x > 10 then 0.9 * x + 6
  else 1.5 * x

def cost_store_b (x : ℕ) : ℝ :=
  1.2 * x

theorem cost_comparison_for_30_pens :
  cost_store_a 30 < cost_store_b 30 :=
by
  have store_a_cost : cost_store_a 30 = 0.9 * 30 + 6 := by rfl
  have store_b_cost : cost_store_b 30 = 1.2 * 30 := by rfl
  rw [store_a_cost, store_b_cost]
  sorry

end cost_comparison_for_30_pens_l30_30151


namespace find_b_l30_30073

variables (U : Set ℝ) (A : Set ℝ) (b : ℝ)

theorem find_b (hU : U = Set.univ)
               (hA : A = {x | 1 ≤ x ∧ x < b})
               (hComplA : U \ A = {x | x < 1 ∨ x ≥ 2}) :
  b = 2 :=
sorry

end find_b_l30_30073


namespace distinct_arrangements_TOOL_l30_30387

/-- The word "TOOL" consists of four letters where "O" is repeated twice. 
Prove that the number of distinct arrangements of the letters in the word is 12. -/
theorem distinct_arrangements_TOOL : 
  let total_letters := 4
  let repeated_O := 2
  (Nat.factorial total_letters / Nat.factorial repeated_O) = 12 := 
by
  sorry

end distinct_arrangements_TOOL_l30_30387


namespace sequence_integral_terms_l30_30288

theorem sequence_integral_terms (x : ℕ → ℝ) (h1 : ∀ n, x n ≠ 0)
  (h2 : ∀ n > 2, x n = (x (n - 2) * x (n - 1)) / (2 * x (n - 2) - x (n - 1))) :
  (∀ n, ∃ k : ℤ, x n = k) → x 1 = x 2 :=
by
  sorry

end sequence_integral_terms_l30_30288


namespace count_primes_between_60_and_85_l30_30221

-- Define the range of interest
def range : Finset ℕ := Finset.range (85 + 1) \ Finset.range 60

-- Define the prime subset of the range
def primes_in_range : Finset ℕ := range.filter Nat.Prime

-- The theorem we aim to prove
theorem count_primes_between_60_and_85 : primes_in_range.card = 6 := by {
  sorry
}

end count_primes_between_60_and_85_l30_30221


namespace consecutive_integers_sum_l30_30614

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l30_30614


namespace sin_thirteen_pi_over_six_l30_30764

-- Define a lean statement for the proof problem
theorem sin_thirteen_pi_over_six : Real.sin (13 * Real.pi / 6) = 1 / 2 := 
by 
  -- Add the proof later (or keep sorry if the proof is not needed)
  sorry

end sin_thirteen_pi_over_six_l30_30764


namespace parabola_focus_l30_30355

theorem parabola_focus : ∃ f : ℝ, 
  (∀ x : ℝ, (x^2 + (2*x^2 - f)^2 = (2*x^2 - (1/4 + f))^2)) ∧
  f = 1/8 := 
by
  sorry

end parabola_focus_l30_30355


namespace sum_of_consecutive_integers_with_product_812_l30_30673

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l30_30673


namespace function_passes_through_point_l30_30394

theorem function_passes_through_point (a : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (1, -1) ∧ ∀ x : ℝ, (y = a^(x-1) - 2) → y = -1 := by
  sorry

end function_passes_through_point_l30_30394


namespace find_lawn_width_l30_30491

/-- Given a rectangular lawn with a length of 80 m and roads each 10 m wide,
    one running parallel to the length and the other running parallel to the width,
    with a total travel cost of Rs. 3300 at Rs. 3 per sq m, prove that the width of the lawn is 30 m. -/
theorem find_lawn_width (w : ℕ) (h_area_road : 10 * w + 10 * 80 = 1100) : w = 30 :=
by {
  sorry
}

end find_lawn_width_l30_30491


namespace range_of_m_l30_30080

theorem range_of_m (m : ℝ) : (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) → (m ≤ 2 ∧ m ≠ -2) :=
begin
  sorry
end

end range_of_m_l30_30080


namespace regular_polygon_sides_l30_30330

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (k : ℕ), (k : ℕ) * 18 = 360) : n = 20 :=
by
  -- Proof body here
  sorry

end regular_polygon_sides_l30_30330


namespace solve_fractional_eq_l30_30274

theorem solve_fractional_eq (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) : 
  (2 / (x - 1) = 1 / x) ↔ (x = -1) :=
by 
  sorry

end solve_fractional_eq_l30_30274


namespace polynomial_remainder_correct_l30_30703

noncomputable def remainder_polynomial (x : ℝ) : ℝ := x ^ 100

def divisor_polynomial (x : ℝ) : ℝ := x ^ 2 - 3 * x + 2

def polynomial_remainder (x : ℝ) : ℝ := 2 ^ 100 * (x - 1) - (x - 2)

theorem polynomial_remainder_correct : ∀ x : ℝ, (remainder_polynomial x) % (divisor_polynomial x) = polynomial_remainder x := by
  sorry

end polynomial_remainder_correct_l30_30703


namespace ratio_x_y_l30_30173

variable (x y : ℝ)

-- Conditions:
-- 1. lengths of pieces
def is_square (x : ℝ) : Prop := ∃ s, x = 4 * s
def is_pentagon (y : ℝ) : Prop := ∃ t, y = 5 * t
def equal_perimeter (x y : ℝ) : Prop := x = y

-- Theorem to prove
theorem ratio_x_y (hx : is_square x) (hy : is_pentagon y) (h_perimeter : equal_perimeter x y) : x / y = 1 :=
by {
  -- Implementation of the proof
  sorry
}

end ratio_x_y_l30_30173


namespace find_number_l30_30297

theorem find_number (x : ℕ) (h : 3 * (x + 2) = 24 + x) : x = 9 :=
by 
  sorry

end find_number_l30_30297


namespace max_a_for_inequality_l30_30204

open Real

theorem max_a_for_inequality (a : ℝ) : (∀ x : ℝ, 0 ≤ x → exp x + sin x - 2 * x ≥ a * x^2 + 1) → a ≤ 1 / 2 :=
by
  sorry

end max_a_for_inequality_l30_30204


namespace minimum_sum_at_nine_l30_30938

noncomputable def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

noncomputable def sum_of_arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem minimum_sum_at_nine {a1 d : ℤ} (h_a1_neg : a1 < 0) 
    (h_sum_equal : sum_of_arithmetic_sequence a1 d 12 = sum_of_arithmetic_sequence a1 d 6) :
  ∀ n : ℕ, (n = 9) → sum_of_arithmetic_sequence a1 d n ≤ sum_of_arithmetic_sequence a1 d m :=
sorry

end minimum_sum_at_nine_l30_30938


namespace find_a_increasing_intervals_geq_zero_set_l30_30378

noncomputable def f (x a : ℝ) : ℝ :=
  sin (x + π / 6) + sin (x - π / 6) + cos x + a

-- Given that the maximum value of f(x) is 1, determine the value of a
theorem find_a (h : ∀ x : ℝ, f x a ≤ 1) : a = -1 :=
sorry

-- Determine the intervals where f(x) is monotonically increasing
theorem increasing_intervals (a : ℝ) (h : a = -1) :
  ∃ k : ℤ, ∀ x ∈ set.Icc ((2 * k : ℝ) * π - 2 * π / 3) ((2 * k : ℝ) * π + π / 3),
    monotoneOn (λ x, f x a) (set.Icc ((2 * k : ℝ) * π - 2 * π / 3) ((2 * k : ℝ) * π + π / 3)) :=
sorry

-- Find the set of values of x for which f(x) ≥ 0
theorem geq_zero_set (a : ℝ) (h : a = -1) :
  ∀ k : ℤ, ∀ x ∈ set.Icc ((2 * k : ℝ) * π - π / 6) ((2 * k : ℝ) * π + π / 2), 0 ≤ f x a :=
sorry

end find_a_increasing_intervals_geq_zero_set_l30_30378


namespace work_ratio_l30_30225

theorem work_ratio (m b : ℝ) (h1 : 12 * m + 16 * b = 1 / 5) (h2 : 13 * m + 24 * b = 1 / 4) : m = 2 * b :=
by sorry

end work_ratio_l30_30225


namespace circle_represents_range_l30_30849

theorem circle_represents_range (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + m * x - 2 * y + 3 = 0 → (m > 2 * Real.sqrt 2 ∨ m < -2 * Real.sqrt 2)) :=
by
  sorry

end circle_represents_range_l30_30849


namespace range_of_a_l30_30087

def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 2

noncomputable def f_prime (a x : ℝ) : ℝ := 3 * a * x^2 + 6 * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x = 0 → x < 0) → (∃ x : ℝ, f a x = 0) → a < -Real.sqrt 2 := by
  sorry

end range_of_a_l30_30087


namespace contrapositive_statement_l30_30535

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬is_even n

theorem contrapositive_statement (a b : ℕ) :
  (¬(is_odd a ∧ is_odd b) ∧ ¬(is_even a ∧ is_even b)) → ¬is_even (a + b) :=
by
  sorry

end contrapositive_statement_l30_30535


namespace plane_overtake_time_is_80_minutes_l30_30463

noncomputable def plane_overtake_time 
  (speed_a speed_b : ℝ)
  (head_start : ℝ) 
  (t : ℝ) : Prop :=
  speed_a * (t + head_start) = speed_b * t

theorem plane_overtake_time_is_80_minutes :
  plane_overtake_time 200 300 (2/3) (80 / 60)
:=
  sorry

end plane_overtake_time_is_80_minutes_l30_30463


namespace sandra_beignets_l30_30263

theorem sandra_beignets (beignets_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) (h1 : beignets_per_day = 3) (h2 : days_per_week = 7) (h3: weeks = 16) : 
  (beignets_per_day * days_per_week * weeks) = 336 :=
by {
  -- the proof goes here
  sorry
}

end sandra_beignets_l30_30263


namespace Jessica_victory_l30_30218

def bullseye_points : ℕ := 10
def other_possible_scores : Set ℕ := {0, 2, 5, 8, 10}
def minimum_score_per_shot : ℕ := 2
def shots_taken : ℕ := 40
def remaining_shots : ℕ := 40
def jessica_advantage : ℕ := 30

def victory_condition (n : ℕ) : Prop :=
  8 * n + 80 > 370

theorem Jessica_victory :
  ∃ n, victory_condition n ∧ n = 37 :=
by
  use 37
  sorry

end Jessica_victory_l30_30218


namespace sum_of_consecutive_integers_l30_30637

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l30_30637


namespace consecutive_integers_sum_l30_30695

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l30_30695


namespace sum_of_consecutive_integers_with_product_812_l30_30685

theorem sum_of_consecutive_integers_with_product_812 :
  ∃ x : ℕ, (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) := 
begin
  sorry
end

end sum_of_consecutive_integers_with_product_812_l30_30685


namespace combined_height_of_cylinders_l30_30311

/-- Given three cylinders with perimeters 6 feet, 9 feet, and 11 feet respectively,
    and rolled out on a rectangular plate with a diagonal of 19 feet,
    the combined height of the cylinders is 26 feet. -/
theorem combined_height_of_cylinders
  (p1 p2 p3 : ℝ) (d : ℝ)
  (h_p1 : p1 = 6) (h_p2 : p2 = 9) (h_p3 : p3 = 11) (h_d : d = 19) :
  p1 + p2 + p3 = 26 :=
sorry

end combined_height_of_cylinders_l30_30311


namespace consecutive_integer_sum_l30_30608

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l30_30608


namespace negation_of_existential_proposition_l30_30367

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, Real.exp x < 0) = (∀ x : ℝ, Real.exp x ≥ 0) :=
sorry

end negation_of_existential_proposition_l30_30367


namespace wire_cut_problem_l30_30170

-- Conditions
variable (x y : ℝ)
variable (h1 : x = y)
variable (hx : x > 0) -- Assuming positive lengths for the wire pieces

-- Statement to prove
theorem wire_cut_problem : x / y = 1 :=
by sorry

end wire_cut_problem_l30_30170


namespace simplify_division_l30_30123

noncomputable def simplify_expression (m : ℝ) : ℝ :=
  (m^2 - 3 * m + 1) / m + 1

noncomputable def divisor_expression (m : ℝ) : ℝ :=
  (m^2 - 1) / m

theorem simplify_division (m : ℝ) (hm1 : m ≠ 0) (hm2 : m ≠ 1) (hm3 : m ≠ -1) :
  (simplify_expression m) / (divisor_expression m) = (m - 1) / (m + 1) :=
by {
  sorry
}

end simplify_division_l30_30123


namespace consecutive_integer_product_sum_l30_30643

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l30_30643


namespace sum_of_consecutive_integers_with_product_812_l30_30656

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l30_30656


namespace focus_of_parabola_y_2x2_l30_30357

theorem focus_of_parabola_y_2x2 :
  ∃ f, f = 1 / 8 ∧ (∀ x, sqrt (x^2 + (2*x^2 - f)^2) = abs (2*x^2 - (-f)))
:= sorry

end focus_of_parabola_y_2x2_l30_30357


namespace combined_payment_is_correct_l30_30473

-- Define the conditions for discounts
def discount_scheme (amount : ℕ) : ℕ :=
  if amount ≤ 100 then amount
  else if amount ≤ 300 then (amount * 90) / 100
  else (amount * 80) / 100

-- Given conditions for Wang Bo's purchases
def first_purchase := 80
def second_purchase_with_discount_applied := 252

-- Two possible original amounts for the second purchase
def possible_second_purchases : Set ℕ :=
  { x | discount_scheme x = second_purchase_with_discount_applied }

-- Total amount to be considered for combined buys with discounts
def total_amount_paid := {x + first_purchase | x ∈ possible_second_purchases}

-- discount applied on the combined amount
def discount_applied_amount (combined : ℕ) : ℕ :=
  discount_scheme combined

-- Prove the combined amount is either 288 or 316
theorem combined_payment_is_correct :
  ∃ combined ∈ total_amount_paid, discount_applied_amount combined = 288 ∨ discount_applied_amount combined = 316 :=
sorry

end combined_payment_is_correct_l30_30473


namespace oliver_shelves_needed_l30_30832

-- Definitions based on conditions
def total_books : ℕ := 46
def books_taken_by_librarian : ℕ := 10
def books_remaining (total books_taken : ℕ) : ℕ := total - books_taken
def books_per_shelf : ℕ := 4

-- Theorem statement
theorem oliver_shelves_needed :
  books_remaining total_books books_taken_by_librarian / books_per_shelf = 9 := by
  sorry

end oliver_shelves_needed_l30_30832


namespace transport_equivalence_l30_30402

theorem transport_equivalence (f : ℤ → ℤ) (x y : ℤ) (h : f x = -x) :
  f (-y) = y :=
by
  sorry

end transport_equivalence_l30_30402


namespace people_in_each_bus_l30_30896

-- Definitions and conditions
def num_vans : ℕ := 2
def num_buses : ℕ := 3
def people_per_van : ℕ := 8
def total_people : ℕ := 76

-- Theorem statement to prove the number of people in each bus
theorem people_in_each_bus : (total_people - num_vans * people_per_van) / num_buses = 20 :=
by
    -- The actual proof would go here
    sorry

end people_in_each_bus_l30_30896


namespace trigonometric_identity_l30_30782

theorem trigonometric_identity (α : ℝ) (h : Real.sin α = 1 / 3) : 
  Real.cos (Real.pi / 4 + α) * Real.cos (Real.pi / 4 - α) = 7 / 18 :=
by sorry

end trigonometric_identity_l30_30782


namespace consecutive_integers_sum_l30_30620

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l30_30620


namespace max_value_of_function_l30_30213

/-- Let y(x) = a^(2*x) + 2 * a^x - 1 for a positive real number a and x in [-1, 1].
    Prove that the maximum value of y on the interval [-1, 1] is 14 when a = 1/3 or a = 3. -/
theorem max_value_of_function (a : ℝ) (a_pos : 0 < a) (h : a = 1 / 3 ∨ a = 3) : 
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^(2*x) + 2 * a^x - 1 = 14 := 
sorry

end max_value_of_function_l30_30213


namespace sausages_fried_l30_30427

def num_eggs : ℕ := 6
def time_per_sausage : ℕ := 5
def time_per_egg : ℕ := 4
def total_time : ℕ := 39
def time_per_sauteurs (S : ℕ) : ℕ := S * time_per_sausage

theorem sausages_fried (S : ℕ) (h : num_eggs * time_per_egg + S * time_per_sausage = total_time) : S = 3 :=
by
  sorry

end sausages_fried_l30_30427


namespace library_books_l30_30933

theorem library_books (a : ℕ) (R L : ℕ) :
  (∃ R, a = 12 * R + 7) ∧ (∃ L, a = 25 * L - 5) ∧ 500 < a ∧ a < 650 → a = 595 :=
by
  sorry

end library_books_l30_30933


namespace consecutive_integers_sum_l30_30664

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l30_30664


namespace partial_fraction_decomposition_l30_30349

theorem partial_fraction_decomposition (A B C : ℚ) :
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
    (-x^2 + 5*x - 6) / (x^3 - x) = A / x + (B*x + C) / (x^2 - 1)) →
  A = 6 ∧ B = -7 ∧ C = 5 :=
by
  intro h
  sorry

end partial_fraction_decomposition_l30_30349


namespace salesperson_commission_l30_30166

noncomputable def commission (sale_price : ℕ) (rate : ℚ) : ℚ :=
  rate * sale_price

noncomputable def total_commission (machines_sold : ℕ) (first_rate : ℚ) (second_rate : ℚ) (sale_price : ℕ) : ℚ :=
  let first_commission := commission sale_price first_rate * 100
  let second_commission := commission sale_price second_rate * (machines_sold - 100)
  first_commission + second_commission

theorem salesperson_commission :
  total_commission 130 0.03 0.04 10000 = 42000 := by
  sorry

end salesperson_commission_l30_30166


namespace grandmother_dolls_l30_30985

-- Define the conditions
variable (S G : ℕ)

-- Rene has three times as many dolls as her sister
def rene_dolls : ℕ := 3 * S

-- The sister has two more dolls than their grandmother
def sister_dolls_eq : Prop := S = G + 2

-- Together they have a total of 258 dolls
def total_dolls : Prop := (rene_dolls S) + S + G = 258

-- Prove that the grandmother has 50 dolls given the conditions
theorem grandmother_dolls : sister_dolls_eq S G → total_dolls S G → G = 50 :=
by
  intros h1 h2
  sorry

end grandmother_dolls_l30_30985


namespace population_in_scientific_notation_l30_30342

theorem population_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 1370540000 = a * 10^n ∧ a = 1.37054 ∧ n = 9 :=
by
  sorry

end population_in_scientific_notation_l30_30342


namespace sum_of_consecutive_integers_with_product_812_l30_30661

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l30_30661


namespace sum_even_numbers_l30_30064

def is_even (n : ℕ) : Prop := n % 2 = 0

def largest_even_less_than_or_equal (n m : ℕ) : ℕ :=
if h : m % 2 = 0 ∧ m ≤ n then m else
if h : m % 2 = 1 ∧ (m - 1) ≤ n then m - 1 else 0

def smallest_even_less_than_or_equal (n : ℕ) : ℕ :=
if h : 2 ≤ n then 2 else 0

theorem sum_even_numbers (n : ℕ) (h : n = 49) :
  largest_even_less_than_or_equal n 48 + smallest_even_less_than_or_equal n = 50 :=
by sorry

end sum_even_numbers_l30_30064


namespace probability_two_dice_showing_1_l30_30878

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_two_dice_showing_1 : 
  binomial_probability 12 2 (1/6) ≈ 0.295 := 
by 
  sorry

end probability_two_dice_showing_1_l30_30878


namespace necessary_and_sufficient_condition_l30_30477

theorem necessary_and_sufficient_condition {a b : ℝ} :
  (a > b) ↔ (a^3 > b^3) := sorry

end necessary_and_sufficient_condition_l30_30477


namespace symmetric_shading_additional_squares_l30_30907

theorem symmetric_shading_additional_squares :
  let initial_shaded : List (ℕ × ℕ) := [(1, 1), (2, 4), (4, 3)]
  let required_horizontal_symmetry := [(4, 1), (1, 6), (4, 6)]
  let required_vertical_symmetry := [(2, 3), (1, 3)]
  let total_additional_squares := required_horizontal_symmetry ++ required_vertical_symmetry
  let final_shaded := initial_shaded ++ total_additional_squares
  ∀ s ∈ total_additional_squares, s ∉ initial_shaded →
    final_shaded.length - initial_shaded.length = 5 :=
by
  sorry

end symmetric_shading_additional_squares_l30_30907


namespace sum_of_digits_l30_30555

def digits (n : ℕ) : Prop := n ≥ 0 ∧ n < 10

def P := 1
def Q := 0
def R := 2
def S := 5
def T := 6

theorem sum_of_digits :
  digits P ∧ digits Q ∧ digits R ∧ digits S ∧ digits T ∧ 
  (10000 * P + 1000 * Q + 100 * R + 10 * S + T) * 4 = 41024 →
  P + Q + R + S + T = 14 :=
by
  sorry

end sum_of_digits_l30_30555


namespace abs_quotient_eq_sqrt_7_div_2_l30_30094

theorem abs_quotient_eq_sqrt_7_div_2 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 5 * a * b) :
  abs ((a + b) / (a - b)) = Real.sqrt (7 / 2) :=
by
  sorry

end abs_quotient_eq_sqrt_7_div_2_l30_30094


namespace cost_price_is_92_percent_l30_30126

noncomputable def cost_price_percentage_of_selling_price (profit_percentage : ℝ) : ℝ :=
  let CP := (1 / ((profit_percentage / 100) + 1))
  CP * 100

theorem cost_price_is_92_percent (profit_percentage : ℝ) (h : profit_percentage = 8.695652173913043) :
  cost_price_percentage_of_selling_price profit_percentage = 92 :=
by
  rw [h]
  -- now we need to show that cost_price_percentage_of_selling_price 8.695652173913043 = 92
  -- by definition, cost_price_percentage_of_selling_price 8.695652173913043 is:
  -- let CP := 1 / (8.695652173913043 / 100 + 1)
  -- CP * 100 = (1 / (8.695652173913043 / 100 + 1)) * 100
  sorry

end cost_price_is_92_percent_l30_30126


namespace employee_discount_percentage_l30_30309

theorem employee_discount_percentage:
  let purchase_price := 500
  let markup_percentage := 0.15
  let savings := 57.5
  let retail_price := purchase_price * (1 + markup_percentage)
  let discount_percentage := (savings / retail_price) * 100
  discount_percentage = 10 :=
by
  sorry

end employee_discount_percentage_l30_30309


namespace original_volume_l30_30335

variable (V : ℝ)

theorem original_volume (h1 : (1/4) * V = V₁)
                       (h2 : (1/4) * V₁ = V₂)
                       (h3 : (1/3) * V₂ = 0.4) : 
                       V = 19.2 := 
by 
  sorry

end original_volume_l30_30335


namespace inequality_proof_l30_30118

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) : 
    (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 := 
by
  sorry

end inequality_proof_l30_30118


namespace square_area_multiplier_l30_30592

theorem square_area_multiplier 
  (perimeter_square : ℝ) (length_rectangle : ℝ) (width_rectangle : ℝ)
  (perimeter_square_eq : perimeter_square = 800) 
  (length_rectangle_eq : length_rectangle = 125) 
  (width_rectangle_eq : width_rectangle = 64)
  : (perimeter_square / 4) ^ 2 / (length_rectangle * width_rectangle) = 5 := 
by
  sorry

end square_area_multiplier_l30_30592


namespace max_rectangle_area_l30_30088

noncomputable def curve_parametric_equation (θ : ℝ) :
    ℝ × ℝ :=
  (1 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

theorem max_rectangle_area :
  ∃ (θ : ℝ), (θ ∈ Set.Icc 0 (2 * Real.pi)) ∧
  ∀ (x y : ℝ), (x, y) = curve_parametric_equation θ →
  |(1 + 2 * Real.cos θ) * (1 + 2 * Real.sin θ)| = 3 + 2 * Real.sqrt 2 :=
sorry

end max_rectangle_area_l30_30088


namespace sum_of_fractions_is_correct_l30_30466

-- Definitions from the conditions
def half_of_third := (1 : ℚ) / 2 * (1 : ℚ) / 3
def third_of_quarter := (1 : ℚ) / 3 * (1 : ℚ) / 4
def quarter_of_fifth := (1 : ℚ) / 4 * (1 : ℚ) / 5
def sum_fractions := half_of_third + third_of_quarter + quarter_of_fifth

-- The theorem to prove
theorem sum_of_fractions_is_correct : sum_fractions = (3 : ℚ) / 10 := by
  sorry

end sum_of_fractions_is_correct_l30_30466


namespace solution_set_A_solution_set_B_subset_A_l30_30379

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

theorem solution_set_A :
  {x : ℝ | f x > 6} = {x : ℝ | x < -1 ∨ x > 2} :=
sorry

theorem solution_set_B_subset_A {a : ℝ} :
  (∀ x, f x > |a-1| → x < -1 ∨ x > 2) → a ≤ -5 ∨ a ≥ 7 :=
sorry

end solution_set_A_solution_set_B_subset_A_l30_30379


namespace transformed_expression_value_l30_30475

theorem transformed_expression_value :
  (240 / 80) * 60 / 40 + 10 = 14.5 :=
by
  sorry

end transformed_expression_value_l30_30475


namespace swimming_pool_distance_l30_30814

theorem swimming_pool_distance (julien_daily_distance : ℕ) (sarah_multi_factor : ℕ)
    (jamir_additional_distance : ℕ) (week_days : ℕ) 
    (julien_weekly_distance : ℕ) (sarah_weekly_distance : ℕ) (jamir_weekly_distance : ℕ) 
    (total_combined_distance : ℕ) : 
    julien_daily_distance = 50 → 
    sarah_multi_factor = 2 →
    jamir_additional_distance = 20 →
    week_days = 7 →
    julien_weekly_distance = julien_daily_distance * week_days →
    sarah_weekly_distance = (sarah_multi_factor * julien_daily_distance) * week_days →
    jamir_weekly_distance = ((sarah_multi_factor * julien_daily_distance) + jamir_additional_distance) * week_days →
    total_combined_distance = julien_weekly_distance + sarah_weekly_distance + jamir_weekly_distance →
    total_combined_distance = 1890 := 
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [h1, h2, h3, h4] at *
  rw [h5, h6, h7, h8]
  sorry

end swimming_pool_distance_l30_30814


namespace sum_of_consecutive_integers_l30_30642

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l30_30642


namespace quadratic_eq_real_roots_roots_diff_l30_30377

theorem quadratic_eq_real_roots (m : ℝ) : 
  ∃ x y : ℝ, x ≠ y ∧ 
  (x^2 + (m-2)*x - m = 0) ∧
  (y^2 + (m-2)*y - m = 0) := sorry

theorem roots_diff (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0)
  (h_roots : (m^2 + (m-2)*m - m = 0) ∧ (n^2 + (m-2)*n - m = 0)) :
  m - n = 5/2 := sorry

end quadratic_eq_real_roots_roots_diff_l30_30377


namespace measure_four_messzely_l30_30898

theorem measure_four_messzely (c3 c5 : ℕ) (hc3 : c3 = 3) (hc5 : c5 = 5) : 
  ∃ (x y z : ℕ), x = 4 ∧ x + y * c3 + z * c5 = 4 := 
sorry

end measure_four_messzely_l30_30898


namespace period_fraction_sum_nines_l30_30286

theorem period_fraction_sum_nines (q : ℕ) (p : ℕ) (N N1 N2 : ℕ) (n : ℕ) (t : ℕ) 
  (hq_prime : Nat.Prime q) (hq_gt_5 : q > 5) (hp_lt_q : p < q)
  (ht_eq_2n : t = 2 * n) (h_period : 10^t ≡ 1 [MOD q])
  (hN_eq_concat : (N = N1 * 10^n + N2) ∧ (N % 10^n = N2))
  : N1 + N2 = (10^n - 1) := 
sorry

end period_fraction_sum_nines_l30_30286


namespace sum_of_consecutive_integers_with_product_812_l30_30655

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l30_30655


namespace train_length_l30_30900

theorem train_length (speed_kph : ℝ) (time_sec : ℝ) (speed_mps : ℝ) (length_m : ℝ) 
  (h1 : speed_kph = 60) 
  (h2 : time_sec = 42) 
  (h3 : speed_mps = speed_kph * 1000 / 3600) 
  (h4 : length_m = speed_mps * time_sec) :
  length_m = 700.14 :=
by
  sorry

end train_length_l30_30900


namespace function_equality_l30_30800

theorem function_equality (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 / x) = x / (1 - x)) :
  ∀ x : ℝ, f x = 1 / (x - 1) :=
by
  sorry

end function_equality_l30_30800


namespace range_of_reciprocal_sum_l30_30779

theorem range_of_reciprocal_sum {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : x + 4 * y + 1 / x + 1 / y = 10) :
  1 ≤ 1 / x + 1 / y ∧ 1 / x + 1 / y ≤ 9 := 
sorry

end range_of_reciprocal_sum_l30_30779


namespace compute_b_l30_30395

theorem compute_b (x y b : ℝ) (h1 : 4 * x + 2 * y = b) (h2 : 3 * x + 7 * y = 3 * b) (hx : x = 3) : b = 66 :=
sorry

end compute_b_l30_30395


namespace coolers_total_capacity_l30_30419

theorem coolers_total_capacity :
  ∃ (C1 C2 C3 : ℕ), 
    C1 = 100 ∧ 
    C2 = C1 + (C1 / 2) ∧ 
    C3 = C2 / 2 ∧ 
    (C1 + C2 + C3 = 325) :=
sorry

end coolers_total_capacity_l30_30419


namespace binom_1500_1_eq_1500_l30_30344

theorem binom_1500_1_eq_1500 : binom 1500 1 = 1500 :=
by
  sorry

end binom_1500_1_eq_1500_l30_30344


namespace sum_of_digits_M_is_9_l30_30822

-- Definition for the number of divisors
def d (n : ℕ) : ℕ := (List.range (n + 1)).count (λ m, m > 0 ∧ n % m = 0)

-- Definition for the function g
def g (n : ℕ) :  ℚ := ((d n)^2 : ℚ) / (n : ℚ)^(1/4)

-- Definition for the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Hypothesis that leads to the maximum M
def M : ℕ := 432

-- The final theorem statement
theorem sum_of_digits_M_is_9 : sum_of_digits M = 9 := by
  sorry

end sum_of_digits_M_is_9_l30_30822


namespace apples_given_to_Larry_l30_30110

-- Define the initial conditions
def initial_apples : ℕ := 75
def remaining_apples : ℕ := 23

-- The statement that we need to prove
theorem apples_given_to_Larry : initial_apples - remaining_apples = 52 :=
by
  -- skip the proof
  sorry

end apples_given_to_Larry_l30_30110


namespace evaluate_expression_l30_30915

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end evaluate_expression_l30_30915


namespace find_some_number_l30_30539

theorem find_some_number (a : ℕ) (h1 : a = 105) (h2 : a^3 = some_number * 35 * 45 * 35) : some_number = 1 := by
  sorry

end find_some_number_l30_30539


namespace bob_needs_50_planks_l30_30745

-- Define the raised bed dimensions and requirements
structure RaisedBedDimensions where
  height : ℕ -- in feet
  width : ℕ  -- in feet
  length : ℕ -- in feet

def plank_length : ℕ := 8  -- length of each plank in feet
def plank_width : ℕ := 1  -- width of each plank in feet
def num_beds : ℕ := 10

def planks_needed (bed : RaisedBedDimensions) : ℕ :=
  let long_sides := 2  -- 2 long sides per bed
  let short_sides := 2 * (bed.width / plank_length)  -- 1/4 plank per short side if width is 2 feet
  let total_sides := long_sides + short_sides
  let stacked_sides := total_sides * (bed.height / plank_width)  -- stacked to match height
  stacked_sides

def raised_bed : RaisedBedDimensions := {height := 2, width := 2, length := 8}

theorem bob_needs_50_planks : planks_needed raised_bed * num_beds = 50 := by
  sorry

end bob_needs_50_planks_l30_30745


namespace solve_ode_l30_30838

noncomputable def x (t : ℝ) : ℝ :=
  -((1 : ℝ) / 18) * Real.exp (-t) +
  (25 / 54) * Real.exp (5 * t) -
  (11 / 27) * Real.exp (-4 * t)

theorem solve_ode :
  ∀ t : ℝ, 
    (deriv^[2] x t) - (deriv x t) - 20 * x t = Real.exp (-t) ∧
    x 0 = 0 ∧
    (deriv x 0) = 4 :=
by
  sorry

end solve_ode_l30_30838


namespace train_length_l30_30541

theorem train_length (x : ℕ) (h1 : (310 + x) / 18 = x / 8) : x = 248 :=
  sorry

end train_length_l30_30541


namespace probability_two_dice_show_1_l30_30880

theorem probability_two_dice_show_1 :
  let n := 12 in
  let p := 1/6 in
  let k := 2 in
  let binom := Nat.choose 12 2 in
  let prob := binom * (p^2) * ((1 - p)^(n - k)) in
  Float.toNearest prob = 0.294 := 
by 
  let n := 12
  let p := 1/6
  let k := 2
  let binom := Nat.choose 12 2
  let prob := binom * (p^2) * ((1 - p)^(n - k))
  have answer : Float.toNearest prob = 0.294 := sorry
  exact answer

end probability_two_dice_show_1_l30_30880


namespace sum_of_consecutive_integers_with_product_812_l30_30687

theorem sum_of_consecutive_integers_with_product_812 :
  ∃ x : ℕ, (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) := 
begin
  sorry
end

end sum_of_consecutive_integers_with_product_812_l30_30687


namespace arithmetic_sequence_sum_first_five_terms_l30_30808

theorem arithmetic_sequence_sum_first_five_terms:
  ∀ (a : ℕ → ℤ), a 2 = 1 → a 4 = 7 → (a 1 + a 5 = a 2 + a 4) → (5 * (a 1 + a 5) / 2 = 20) :=
by
  intros a h1 h2 h3
  sorry

end arithmetic_sequence_sum_first_five_terms_l30_30808


namespace range_of_a_l30_30506

variable {a x : ℝ}

def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def proposition_q (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 1 → (4 * x^2 - a * x) ≥ (4 * (x + 1)^2 - a * (x + 1))

theorem range_of_a (h : ¬ proposition_p a ∧ (proposition_p a ∨ proposition_q a)) : a ≤ 0 ∨ 4 ≤ a ∧ a ≤ 8 :=
sorry

end range_of_a_l30_30506


namespace total_balloons_is_18_l30_30776

-- Define the number of balloons each person has
def Fred_balloons : Nat := 5
def Sam_balloons : Nat := 6
def Mary_balloons : Nat := 7

-- Define the total number of balloons
def total_balloons : Nat := Fred_balloons + Sam_balloons + Mary_balloons

-- The theorem statement to prove
theorem total_balloons_is_18 : total_balloons = 18 := sorry

end total_balloons_is_18_l30_30776


namespace melissa_games_played_l30_30438

-- Define the conditions mentioned:
def points_per_game := 12
def total_points := 36

-- State the proof problem:
theorem melissa_games_played : total_points / points_per_game = 3 :=
by sorry

end melissa_games_played_l30_30438


namespace problem_l30_30244

def f (x : ℚ) : ℚ := (4 * x^2 + 6 * x + 10) / (x^2 - 2 * x + 5)
def g (x : ℚ) : ℚ := x - 2

theorem problem : f (g 2) + g (f 2) = 38 / 5 :=
by
  sorry

end problem_l30_30244


namespace card_probability_l30_30743

-- Definitions of the conditions
def is_multiple (n d : ℕ) : Prop := d ∣ n

def count_multiples (d m : ℕ) : ℕ := (m / d)

def multiples_in_range (n : ℕ) : ℕ := 
  count_multiples 2 n + count_multiples 3 n + count_multiples 5 n
  - count_multiples 6 n - count_multiples 10 n - count_multiples 15 n 
  + count_multiples 30 n

def probability_of_multiples_in_range (n : ℕ) : ℚ := 
  multiples_in_range n / n 

-- Proof statement
theorem card_probability (n : ℕ) (h : n = 120) : probability_of_multiples_in_range n = 11 / 15 :=
  sorry

end card_probability_l30_30743


namespace combined_distance_l30_30812

-- Definitions based on the conditions
def JulienDailyDistance := 50
def SarahDailyDistance := 2 * JulienDailyDistance
def JamirDailyDistance := SarahDailyDistance + 20
def Days := 7

-- Combined weekly distances
def JulienWeeklyDistance := JulienDailyDistance * Days
def SarahWeeklyDistance := SarahDailyDistance * Days
def JamirWeeklyDistance := JamirDailyDistance * Days

-- Theorem statement with the combined distance
theorem combined_distance :
  JulienWeeklyDistance + SarahWeeklyDistance + JamirWeeklyDistance = 1890 := by
  sorry

end combined_distance_l30_30812


namespace xyz_neg_l30_30520

theorem xyz_neg {a b c x y z : ℝ} 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) 
  (h : |x - a| + |y - b| + |z - c| = 0) : 
  x * y * z < 0 :=
by 
  -- to be proven
  sorry

end xyz_neg_l30_30520


namespace sum_eq_twenty_x_l30_30391

variable {R : Type*} [CommRing R] (x y z : R)

theorem sum_eq_twenty_x (h1 : y = 3 * x) (h2 : z = 3 * y) : 2 * x + 3 * y + z = 20 * x := by
  sorry

end sum_eq_twenty_x_l30_30391


namespace inequality_true_for_all_real_l30_30588

theorem inequality_true_for_all_real (a : ℝ) : 
  3 * (1 + a^2 + a^4) ≥ (1 + a + a^2)^2 :=
sorry

end inequality_true_for_all_real_l30_30588


namespace player_A_wins_if_n_equals_9_l30_30158

-- Define the conditions
def drawing_game (n : ℕ) : Prop :=
  ∃ strategy : ℕ → ℕ,
    strategy 0 = 1 ∧ -- Player A always starts by drawing 1 ball
    (∀ k, 1 ≤ strategy k ∧ strategy k ≤ 3) ∧ -- Players draw between 1 and 3 balls
    ∀ b, 1 ≤ b → b ≤ 3 → (n - 1 - strategy (b - 1)) ≤ 3 → (strategy (n - 1 - (b - 1)) = n - (b - 1) - 1)

-- State the problem to prove Player A has a winning strategy if n = 9
theorem player_A_wins_if_n_equals_9 : drawing_game 9 :=
sorry

end player_A_wins_if_n_equals_9_l30_30158


namespace exponent_relation_l30_30536

theorem exponent_relation (a : ℝ) (m n : ℕ) (h1 : a^m = 9) (h2 : a^n = 3) : a^(m - n) = 3 := 
sorry

end exponent_relation_l30_30536


namespace initial_scooter_value_l30_30458

theorem initial_scooter_value (V : ℝ) (h : V * (3/4)^2 = 22500) : V = 40000 :=
by
  sorry

end initial_scooter_value_l30_30458


namespace prove_k_range_l30_30936

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x - b * Real.log x

theorem prove_k_range (a b k : ℝ) (h1 : a - b = 1) (h2 : f 1 a b = 2) :
  (∀ x ≥ 1, f x a b ≥ k * x) → k ≤ 2 - 1 / Real.exp 1 :=
by
  sorry

end prove_k_range_l30_30936


namespace range_of_m_l30_30230

theorem range_of_m (x y m : ℝ) (h1 : x + 2 * y = 1 + m) (h2 : 2 * x + y = 3) (h3 : x + y > 0) : m > -4 := by
  sorry

end range_of_m_l30_30230


namespace diagonal_of_rectangle_l30_30252

theorem diagonal_of_rectangle (a b d : ℝ)
  (h_side : a = 15)
  (h_area : a * b = 120)
  (h_diag : a^2 + b^2 = d^2) :
  d = 17 :=
by
  sorry

end diagonal_of_rectangle_l30_30252


namespace solve_quadratic_equation_l30_30991

theorem solve_quadratic_equation : 
  ∃ (a b c : ℤ), (0 < a) ∧ (64 * x^2 + 48 * x - 36 = 0) ∧ ((a * x + b)^2 = c) ∧ (a + b + c = 56) := 
by
  sorry

end solve_quadratic_equation_l30_30991


namespace infinite_powers_of_two_in_sequence_l30_30589

theorem infinite_powers_of_two_in_sequence :
  ∃ᶠ n in at_top, ∃ k : ℕ, ∃ a : ℕ, (a = ⌊n * Real.sqrt 2⌋ ∧ a = 2^k) :=
sorry

end infinite_powers_of_two_in_sequence_l30_30589


namespace maxwell_meets_brad_l30_30248

variable (t : ℝ) -- time in hours
variable (distance_between_homes : ℝ) -- total distance
variable (maxwell_speed : ℝ) -- Maxwell's walking speed
variable (brad_speed : ℝ) -- Brad's running speed
variable (brad_delay : ℝ) -- Brad's start time delay

theorem maxwell_meets_brad 
  (hb: brad_delay = 1)
  (d: distance_between_homes = 34)
  (v_m: maxwell_speed = 4)
  (v_b: brad_speed = 6)
  (h : 4 * t + 6 * (t - 1) = distance_between_homes) :
  t = 4 := 
  sorry

end maxwell_meets_brad_l30_30248


namespace sum_modulo_remainder_l30_30926

theorem sum_modulo_remainder :
  ((82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17) = 12 :=
by
  sorry

end sum_modulo_remainder_l30_30926


namespace sandra_beignets_16_weeks_l30_30261

-- Define the constants used in the problem
def beignets_per_morning : ℕ := 3
def days_per_week : ℕ := 7
def weeks : ℕ := 16

-- Define the number of beignets Sandra eats in 16 weeks
def beignets_in_16_weeks : ℕ := beignets_per_morning * days_per_week * weeks

-- State the theorem
theorem sandra_beignets_16_weeks : beignets_in_16_weeks = 336 :=
by
  -- Provide a placeholder for the proof
  sorry

end sandra_beignets_16_weeks_l30_30261


namespace consecutive_integer_sum_l30_30609

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l30_30609


namespace sum_of_consecutive_integers_with_product_812_l30_30678

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l30_30678


namespace age_of_B_is_23_l30_30891

-- Definitions of conditions
variable (A B C : ℕ)
variable (h1 : A + B + C = 87)
variable (h2 : A + C = 64)

-- Statement of the problem
theorem age_of_B_is_23 : B = 23 :=
by { sorry }

end age_of_B_is_23_l30_30891


namespace probability_of_red_ball_and_removed_red_balls_l30_30550

-- Conditions for the problem
def initial_red_balls : Nat := 10
def initial_yellow_balls : Nat := 2
def initial_blue_balls : Nat := 8
def total_balls : Nat := initial_red_balls + initial_yellow_balls + initial_blue_balls

-- Problem statement in Lean
theorem probability_of_red_ball_and_removed_red_balls :
  (initial_red_balls / total_balls = 1 / 2) ∧
  (∃ (x : Nat), -- Number of red balls removed
    ((initial_yellow_balls + x) / total_balls = 2 / 5) ∧
    (initial_red_balls - x = 10 - 6)) := 
by
  -- Lean will need the proofs here; we use sorry for now.
  sorry

end probability_of_red_ball_and_removed_red_balls_l30_30550


namespace sum_of_first_five_terms_l30_30941

def a (n : ℕ) : ℚ := 1 / (n * (n + 1))

theorem sum_of_first_five_terms :
  (a 1 + a 2 + a 3 + a 4 + a 5) = 5 / 6 := 
by 
  unfold a
  -- sorry is used as a placeholder for the actual proof
  sorry

end sum_of_first_five_terms_l30_30941


namespace false_statement_E_l30_30139

theorem false_statement_E
  (A B C : Type)
  (a b c : ℝ)
  (ha_gt_hb : a > b)
  (hb_gt_hc : b > c)
  (AB BC : ℝ)
  (hAB : AB = a - b → True)
  (hBC : BC = b + c → True)
  (hABC : AB + BC > a + b + c → True)
  (hAC : AB + BC > a - c → True) : False := sorry

end false_statement_E_l30_30139


namespace sandra_beignets_16_weeks_l30_30260

-- Define the constants used in the problem
def beignets_per_morning : ℕ := 3
def days_per_week : ℕ := 7
def weeks : ℕ := 16

-- Define the number of beignets Sandra eats in 16 weeks
def beignets_in_16_weeks : ℕ := beignets_per_morning * days_per_week * weeks

-- State the theorem
theorem sandra_beignets_16_weeks : beignets_in_16_weeks = 336 :=
by
  -- Provide a placeholder for the proof
  sorry

end sandra_beignets_16_weeks_l30_30260


namespace solve_for_A_l30_30431

-- Define the functions f and g
def f (A B x : ℝ) : ℝ := A * x^2 - 3 * B^2
def g (B x : ℝ) : ℝ := B * x^2

-- A Lean theorem that formalizes the given math problem.
theorem solve_for_A (A B : ℝ) (h₁ : B ≠ 0) (h₂ : f A B (g B 1) = 0) : A = 3 :=
by {
  sorry
}

end solve_for_A_l30_30431


namespace regular_polygon_sides_l30_30322

-- Definitions based on conditions in the problem
def exterior_angle (n : ℕ) : ℝ := 360 / n

theorem regular_polygon_sides (n : ℕ) (h : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l30_30322


namespace worth_of_presents_l30_30566

def ring_value := 4000
def car_value := 2000
def bracelet_value := 2 * ring_value

def total_worth := ring_value + car_value + bracelet_value

theorem worth_of_presents : total_worth = 14000 := by
  sorry

end worth_of_presents_l30_30566


namespace necessary_and_sufficient_condition_l30_30370

-- Define the conditions and question in Lean 4
variable (a : ℝ) 

-- State the theorem based on the conditions and the correct answer
theorem necessary_and_sufficient_condition :
  (a > 0) ↔ (
    let z := (⟨-a, -5⟩ : ℂ)
    ∃ (x y : ℝ), (z = x + y * I) ∧ x < 0 ∧ y < 0
  ) := by
  sorry

end necessary_and_sufficient_condition_l30_30370


namespace factor_expression_l30_30920

theorem factor_expression (x : ℝ) : 5 * x * (x - 2) + 9 * (x - 2) = (x - 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l30_30920


namespace sin_thirteen_pi_over_six_l30_30765

-- Define a lean statement for the proof problem
theorem sin_thirteen_pi_over_six : Real.sin (13 * Real.pi / 6) = 1 / 2 := 
by 
  -- Add the proof later (or keep sorry if the proof is not needed)
  sorry

end sin_thirteen_pi_over_six_l30_30765


namespace rooks_placement_possible_l30_30413

/-- 
  It is possible to place 8 rooks on a chessboard such that they do not attack each other
  and each rook stands on cells of different colors, given that the chessboard is divided 
  into 32 colors with exactly two cells of each color.
-/
theorem rooks_placement_possible :
  ∃ (placement : Fin 8 → Fin 8 × Fin 8),
    (∀ i j, i ≠ j → (placement i).fst ≠ (placement j).fst ∧ (placement i).snd ≠ (placement j).snd) ∧
    (∀ i j, i ≠ j → (placement i ≠ placement j)) ∧
    (∀ c : Fin 32, ∃! p1 p2, placement p1 = placement p2 ∧ (placement p1).fst ≠ (placement p2).fst 
                        ∧ (placement p1).snd ≠ (placement p2).snd) :=
by
  sorry

end rooks_placement_possible_l30_30413


namespace probability_two_ones_in_twelve_dice_l30_30875
noncomputable theory

def probability_of_exactly_two_ones (n : ℕ) (p : ℚ) :=
  (nat.choose n 2 : ℚ) * p^2 * (1 - p)^(n - 2)

theorem probability_two_ones_in_twelve_dice :
  probability_of_exactly_two_ones 12 (1/6) ≈ 0.298 :=
by
  sorry

end probability_two_ones_in_twelve_dice_l30_30875


namespace tom_age_ratio_l30_30298

-- Definitions of given conditions
variables (T N : ℕ) -- Tom's age (T) and number of years ago (N)

-- Tom's age is T years
-- The sum of the ages of Tom's three children is also T
-- N years ago, Tom's age was twice the sum of his children's ages then

theorem tom_age_ratio (h1 : T - N = 2 * (T - 3 * N)) : T / N = 5 :=
sorry

end tom_age_ratio_l30_30298


namespace evaluate_fractional_exponent_l30_30195

theorem evaluate_fractional_exponent : 64^(2/3 : ℝ) = 16 := by
  have h1 : (64 : ℝ) = 2^6 := by
    norm_num
  rw [h1]
  have h2 : (2^6 : ℝ)^(2/3) = 2^(6 * (2/3)) := by
    rw [← Real.rpow_mul (by norm_num : 0 ≤ 2)] -- Using exponent properties
  rw [h2]
  calc 2^(6 * (2/3)) = 2^4 : by congr; ring
                ...  = 16  : by norm_num

end evaluate_fractional_exponent_l30_30195


namespace proof_x_exists_l30_30190

noncomputable def find_x : ℝ := 33.33

theorem proof_x_exists (A B C : ℝ) (h1 : A = (1 + find_x / 100) * B) (h2 : C = 0.75 * A) (h3 : A > C) (h4 : C > B) :
  find_x = 33.33 := 
by
  -- Proof steps
  sorry

end proof_x_exists_l30_30190


namespace solve_for_D_d_Q_R_l30_30796

theorem solve_for_D_d_Q_R (D d Q R : ℕ) 
    (h1 : D = d * Q + R) 
    (h2 : d * Q = 135) 
    (h3 : R = 2 * d) : 
    D = 165 ∧ d = 15 ∧ Q = 9 ∧ R = 30 :=
by
  sorry

end solve_for_D_d_Q_R_l30_30796


namespace area_of_polygon_ABLFKJ_l30_30410

theorem area_of_polygon_ABLFKJ 
  (side_length : ℝ) (area_square : ℝ) (midpoint_l : ℝ) (area_triangle : ℝ)
  (remaining_area_each_square : ℝ) (total_area : ℝ)
  (h1 : side_length = 6)
  (h2 : area_square = side_length * side_length)
  (h3 : midpoint_l = side_length / 2)
  (h4 : area_triangle = 0.5 * side_length * midpoint_l)
  (h5 : remaining_area_each_square = area_square - 2 * area_triangle)
  (h6 : total_area = 3 * remaining_area_each_square)
  : total_area = 54 :=
by
  sorry

end area_of_polygon_ABLFKJ_l30_30410


namespace half_angle_quadrant_second_quadrant_l30_30208

theorem half_angle_quadrant_second_quadrant
  (θ : Real)
  (h1 : π < θ ∧ θ < 3 * π / 2) -- θ is in the third quadrant
  (h2 : Real.cos (θ / 2) < 0) : -- cos (θ / 2) < 0
  π / 2 < θ / 2 ∧ θ / 2 < π := -- θ / 2 is in the second quadrant
sorry

end half_angle_quadrant_second_quadrant_l30_30208


namespace probability_of_two_ones_in_twelve_dice_rolls_l30_30870

noncomputable def binomial_prob_exact_two_show_one (n : ℕ) (p : ℚ) : ℚ :=
  let choose := (Nat.factorial n) / ((Nat.factorial 2) * (Nat.factorial (n - 2)))
  let prob := choose * (p^2) * ((1 - p)^(n - 2)) in 
  prob

theorem probability_of_two_ones_in_twelve_dice_rolls :
  binomial_prob_exact_two_show_one 12 (1 / 6) = 0.296 := by
  sorry

end probability_of_two_ones_in_twelve_dice_rolls_l30_30870


namespace parabola_standard_equation_l30_30135

theorem parabola_standard_equation :
  ∃ m : ℝ, (∀ x y : ℝ, (x^2 = 2 * m * y ↔ (0, -6) ∈ ({p | 3 * p.1 - 4 * p.2 - 24 = 0}))) → 
  (x^2 = -24 * y) := 
by {
  sorry
}

end parabola_standard_equation_l30_30135


namespace regular_polygon_sides_l30_30320

theorem regular_polygon_sides (θ : ℝ) (n : ℕ) (h1 : θ = 18) (h2 : 360 = θ * n) : n = 20 :=
by {
  sorry
}

end regular_polygon_sides_l30_30320


namespace sequence_formula_l30_30361

theorem sequence_formula (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, a (n + 1) - a n = 3^n) :
  ∀ n : ℕ, a n = (3^n - 1) / 2 :=
sorry

end sequence_formula_l30_30361


namespace sum_of_modified_numbers_l30_30300

theorem sum_of_modified_numbers (x y R : ℝ) (h : x + y = R) : 
  2 * (x + 4) + 2 * (y + 5) = 2 * R + 18 :=
by
  sorry

end sum_of_modified_numbers_l30_30300


namespace estimate_larger_than_difference_l30_30963

theorem estimate_larger_than_difference
  (u v δ γ : ℝ)
  (huv : u > v)
  (hδ : δ > 0)
  (hγ : γ > 0)
  (hδγ : δ > γ) : (u + δ) - (v - γ) > u - v := by
  sorry

end estimate_larger_than_difference_l30_30963


namespace difference_of_triangular_23_and_21_l30_30041

def triangular (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem difference_of_triangular_23_and_21 : triangular 23 - triangular 21 = 45 :=
sorry

end difference_of_triangular_23_and_21_l30_30041


namespace contrapositive_of_squared_sum_eq_zero_l30_30847

theorem contrapositive_of_squared_sum_eq_zero (a b : ℝ) :
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ (a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0) :=
by
  sorry

end contrapositive_of_squared_sum_eq_zero_l30_30847


namespace increasing_sequence_k_range_l30_30206

theorem increasing_sequence_k_range (k : ℝ) (a : ℕ → ℝ) (h : ∀ n : ℕ, a n = n^2 + k * n) :
  (∀ n : ℕ, a (n + 1) > a n) → (k ≥ -3) :=
  sorry

end increasing_sequence_k_range_l30_30206


namespace product_lcm_gcd_eq_108_l30_30771

def lcm (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

theorem product_lcm_gcd_eq_108 (a b : ℕ) (h1 : a = 12) (h2 : b = 9) :
  (lcm a b) * (Nat.gcd a b) = 108 := by
  rw [h1, h2] -- replace a and b with 12 and 9
  have lcm_12_9 : lcm 12 9 = 36 := sorry -- find the LCM of 12 and 9
  have gcd_12_9 : Nat.gcd 12 9 = 3 := sorry -- find the GCD of 12 and 9
  rw [lcm_12_9, gcd_12_9]
  norm_num -- simplifies the multiplication
  exact eq.refl 108

end product_lcm_gcd_eq_108_l30_30771


namespace consecutive_integers_sum_l30_30617

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l30_30617


namespace correct_calculation_l30_30168

theorem correct_calculation (N : ℤ) (h : 41 - N = 12) : 41 + N = 70 := 
by 
  sorry

end correct_calculation_l30_30168


namespace minimum_workers_required_l30_30752

theorem minimum_workers_required (total_days : ℕ) (days_elapsed : ℕ) (initial_workers : ℕ) (job_fraction_done : ℚ)
  (remaining_work_fraction : job_fraction_done < 1) 
  (worker_productivity_constant : Prop) : 
  total_days = 40 → days_elapsed = 10 → initial_workers = 10 → job_fraction_done = (1/4) →
  (total_days - days_elapsed) * initial_workers * job_fraction_done = (1 - job_fraction_done) →
  job_fraction_done = 1 → initial_workers = 10 :=
by
  intros;
  sorry

end minimum_workers_required_l30_30752


namespace sum_of_consecutive_integers_with_product_812_l30_30657

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l30_30657


namespace quadratic_solution_product_l30_30574

theorem quadratic_solution_product :
  let r := 9 / 2
  let s := -11
  (r + 4) * (s + 4) = -119 / 2 :=
by
  -- Define the quadratic equation and its solutions
  let r := 9 / 2
  let s := -11

  -- Prove the statement
  sorry

end quadratic_solution_product_l30_30574


namespace water_added_l30_30398

theorem water_added (initial_volume : ℕ) (ratio_milk_water_initial : ℚ) 
  (ratio_milk_water_final : ℚ) (w : ℕ)
  (initial_volume_eq : initial_volume = 45)
  (ratio_milk_water_initial_eq : ratio_milk_water_initial = 4 / 1)
  (ratio_milk_water_final_eq : ratio_milk_water_final = 6 / 5)
  (final_ratio_eq : ratio_milk_water_final = 36 / (9 + w)) :
  w = 21 := 
sorry

end water_added_l30_30398


namespace evaluate_fractional_exponent_l30_30194

theorem evaluate_fractional_exponent : 64^(2/3 : ℝ) = 16 := by
  have h1 : (64 : ℝ) = 2^6 := by
    norm_num
  rw [h1]
  have h2 : (2^6 : ℝ)^(2/3) = 2^(6 * (2/3)) := by
    rw [← Real.rpow_mul (by norm_num : 0 ≤ 2)] -- Using exponent properties
  rw [h2]
  calc 2^(6 * (2/3)) = 2^4 : by congr; ring
                ...  = 16  : by norm_num

end evaluate_fractional_exponent_l30_30194


namespace Keenan_essay_length_l30_30426

-- Given conditions
def words_per_hour_first_two_hours : ℕ := 400
def first_two_hours : ℕ := 2
def words_per_hour_later : ℕ := 200
def later_hours : ℕ := 2

-- Total words written in 4 hours
def total_words : ℕ := words_per_hour_first_two_hours * first_two_hours + words_per_hour_later * later_hours

-- Theorem statement
theorem Keenan_essay_length : total_words = 1200 := by
  sorry

end Keenan_essay_length_l30_30426


namespace sum_of_coefficients_l30_30210

theorem sum_of_coefficients (n : ℕ) (h₀ : n > 0) (h₁ : Nat.choose n 2 = Nat.choose n 7) : 
  (1 - 2 : ℝ) ^ n = -1 := by
  -- Proof
  sorry

end sum_of_coefficients_l30_30210


namespace distinct_roots_iff_l30_30983

theorem distinct_roots_iff (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m * x1 + m + 3 = 0 ∧ x2^2 + m * x2 + m + 3 = 0) ↔ (m < -2 ∨ m > 6) := 
sorry

end distinct_roots_iff_l30_30983


namespace solve_for_P_l30_30124

noncomputable def sqrt16_81 : ℝ := real.rpow 81 (1/16)
noncomputable def cube_root_of_59049 := 27 * real.rpow 3 (1 / 3)

theorem solve_for_P (P : ℝ) (h : real.rpow P (3/4) = 81 * sqrt16_81) : 
  P = cube_root_of_59049 :=
begin
  sorry
end

end solve_for_P_l30_30124


namespace find_A_l30_30023

theorem find_A (A B : ℕ) (A_digit : A < 10) (B_digit : B < 10) :
  let fourteenA := 100 * 1 + 10 * 4 + A
  let Bseventy3 := 100 * B + 70 + 3
  fourteenA + Bseventy3 = 418 → A = 5 :=
by
  sorry

end find_A_l30_30023


namespace sum_of_consecutive_integers_with_product_812_l30_30690

theorem sum_of_consecutive_integers_with_product_812 :
  ∃ x : ℕ, (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) := 
begin
  sorry
end

end sum_of_consecutive_integers_with_product_812_l30_30690


namespace coffee_shop_lattes_l30_30279

theorem coffee_shop_lattes (x : ℕ) (number_of_teas number_of_lattes : ℕ)
  (h1 : number_of_teas = 6)
  (h2 : number_of_lattes = 32)
  (h3 : number_of_lattes = x * number_of_teas + 8) :
  x = 4 :=
by
  sorry

end coffee_shop_lattes_l30_30279


namespace total_water_capacity_of_coolers_l30_30418

theorem total_water_capacity_of_coolers :
  ∀ (first_cooler second_cooler third_cooler : ℕ), 
  first_cooler = 100 ∧ 
  second_cooler = first_cooler + first_cooler / 2 ∧ 
  third_cooler = second_cooler / 2 -> 
  first_cooler + second_cooler + third_cooler = 325 := 
by
  intros first_cooler second_cooler third_cooler H
  cases' H with H1 H2
  cases' H2 with H3 H4
  sorry

end total_water_capacity_of_coolers_l30_30418


namespace greatest_q_minus_r_l30_30285

theorem greatest_q_minus_r : 
  ∃ (q r : ℕ), 1013 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ (q - r = 39) := 
by
  sorry

end greatest_q_minus_r_l30_30285


namespace blue_candy_count_l30_30459

def total_pieces : ℕ := 3409
def red_pieces : ℕ := 145
def blue_pieces : ℕ := total_pieces - red_pieces

theorem blue_candy_count :
  blue_pieces = 3264 := by
  sorry

end blue_candy_count_l30_30459


namespace problem_l30_30625

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l30_30625


namespace sum_of_consecutive_integers_with_product_812_l30_30654

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l30_30654


namespace tangent_parabola_points_l30_30002

theorem tangent_parabola_points (a b : ℝ) (h_circle : a^2 + b^2 = 1) (h_discriminant : a^2 - 4 * b * (b - 1) = 0) :
    (a = 0 ∧ b = 1) ∨ 
    (a = 2 * Real.sqrt 6 / 5 ∧ b = -1 / 5) ∨ 
    (a = -2 * Real.sqrt 6 / 5 ∧ b = -1 / 5) := sorry

end tangent_parabola_points_l30_30002


namespace number_of_lucky_numbers_l30_30400

-- Defining the concept of sequence with even number of digit 8
def is_lucky (seq : List ℕ) : Prop :=
  seq.count 8 % 2 = 0

-- Define S(n) recursive formula
noncomputable def S : ℕ → ℝ
| 0 => 0
| n+1 => 4 * (1 - (1 / (2 ^ (n+1))))

theorem number_of_lucky_numbers (n : ℕ) :
  ∀ (seq : List ℕ), (seq.length ≤ n) → is_lucky seq → S n = 4 * (1 - 1 / (2 ^ n)) :=
sorry

end number_of_lucky_numbers_l30_30400


namespace mark_card_sum_l30_30827

/--
Mark has seven green cards numbered 1 through 7 and five red cards numbered 2 through 6.
He arranges the cards such that colors alternate and the sum of each pair of neighboring cards forms a prime.
Prove that the sum of the numbers on the last three cards in his stack is 16.
-/
theorem mark_card_sum {green_cards : Fin 7 → ℕ} {red_cards : Fin 5 → ℕ}
  (h_green_numbered : ∀ i, 1 ≤ green_cards i ∧ green_cards i ≤ 7)
  (h_red_numbered : ∀ i, 2 ≤ red_cards i ∧ red_cards i ≤ 6)
  (h_alternate : ∀ i, i < 6 → (∃ j k, green_cards j + red_cards k = prime) ∨ (red_cards j + green_cards k = prime)) :
  ∃ s, s = 16 := sorry

end mark_card_sum_l30_30827


namespace consecutive_integer_sum_l30_30606

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l30_30606


namespace consecutive_integer_product_sum_l30_30650

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l30_30650


namespace monica_usd_start_amount_l30_30250

theorem monica_usd_start_amount (x : ℕ) (H : ∃ (y : ℕ), y = 40 ∧ (8 : ℚ) / 5 * x - y = x) :
  (x / 100) + (x % 100 / 10) + (x % 10) = 2 := 
by
  sorry

end monica_usd_start_amount_l30_30250


namespace years_before_marriage_l30_30460

theorem years_before_marriage {wedding_anniversary : ℕ} 
  (current_year : ℕ) (met_year : ℕ) (years_before_dating : ℕ) :
  wedding_anniversary = 20 →
  current_year = 2025 →
  met_year = 2000 →
  years_before_dating = 2 →
  met_year + years_before_dating + (current_year - met_year - wedding_anniversary) = current_year - wedding_anniversary - years_before_dating + wedding_anniversary - current_year :=
by
  sorry

end years_before_marriage_l30_30460


namespace max_value_x_plus_y_l30_30435

theorem max_value_x_plus_y :
  ∃ x y : ℝ, 5 * x + 3 * y ≤ 10 ∧ 3 * x + 5 * y = 15 ∧ x + y = 47 / 16 :=
by
  sorry

end max_value_x_plus_y_l30_30435


namespace parabola_num_xintercepts_l30_30753

-- Defining the equation of the parabola
def parabola (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

-- The main theorem to state: the number of x-intercepts for the parabola is 2.
theorem parabola_num_xintercepts : ∃ (a b : ℝ), parabola a = 0 ∧ parabola b = 0 ∧ a ≠ b :=
by
  sorry

end parabola_num_xintercepts_l30_30753


namespace geometric_sequence_ratio_l30_30101

variable {a : ℕ → ℝ} -- Define the geometric sequence {a_n}

-- Conditions: The sequence is geometric with positive terms
variable (q : ℝ) (hq : q > 0) (hgeo : ∀ n, a (n + 1) = q * a n)

-- Additional condition: a2, 1/2 a3, and a1 form an arithmetic sequence
variable (hseq : a 1 - (1 / 2) * a 2 = (1 / 2) * a 2 - a 0)

theorem geometric_sequence_ratio :
  (a 3 + a 4) / (a 2 + a 3) = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end geometric_sequence_ratio_l30_30101


namespace max_value_of_quadratic_function_l30_30143

noncomputable def quadratic_function (x : ℝ) : ℝ := -5*x^2 + 25*x - 15

theorem max_value_of_quadratic_function : ∃ x : ℝ, quadratic_function x = 750 :=
by
-- maximum value
sorry

end max_value_of_quadratic_function_l30_30143


namespace consecutive_integers_sum_l30_30671

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l30_30671


namespace tank_empty_time_l30_30734

theorem tank_empty_time 
  (time_to_empty_leak : ℝ) 
  (inlet_rate_per_minute : ℝ) 
  (tank_volume : ℝ) 
  (net_time_to_empty : ℝ) : 
  time_to_empty_leak = 7 → 
  inlet_rate_per_minute = 6 → 
  tank_volume = 6048.000000000001 → 
  net_time_to_empty = 12 :=
by
  intros h1 h2 h3
  sorry

end tank_empty_time_l30_30734


namespace num_black_balls_l30_30959

theorem num_black_balls 
  (R W B : ℕ) 
  (R_eq : R = 30) 
  (prob_white : (W : ℝ) / 100 = 0.47) 
  (total_balls : R + W + B = 100) : B = 23 := 
by 
  sorry

end num_black_balls_l30_30959


namespace sum_of_two_digit_odd_numbers_l30_30861

-- Define the set of all two-digit numbers with both digits odd
def two_digit_odd_numbers : List ℕ := 
  [11, 13, 15, 17, 19, 31, 33, 35, 37, 39,
   51, 53, 55, 57, 59, 71, 73, 75, 77, 79,
   91, 93, 95, 97, 99]

-- Define a function to compute the sum of elements in a list
def list_sum (l : List ℕ) : ℕ := l.foldl (.+.) 0

theorem sum_of_two_digit_odd_numbers :
  list_sum two_digit_odd_numbers = 1375 :=
by
  sorry

end sum_of_two_digit_odd_numbers_l30_30861


namespace min_value_of_expression_l30_30720

theorem min_value_of_expression (x y : ℝ) : (2 * x * y - 3) ^ 2 + (x - y) ^ 2 ≥ 1 :=
sorry

end min_value_of_expression_l30_30720


namespace green_tractor_price_is_5000_l30_30015

-- Definitions based on the given conditions
def red_tractor_price : ℝ := 20000
def green_tractor_commission_rate : ℝ := 0.20
def red_tractor_commission_rate : ℝ := 0.10
def red_tractors_sold : ℕ := 2
def green_tractors_sold : ℕ := 3
def salary : ℝ := 7000

-- The theorem statement
theorem green_tractor_price_is_5000 
  (rtp : ℝ := red_tractor_price)
  (gtcr : ℝ := green_tractor_commission_rate)
  (rtcr : ℝ := red_tractor_commission_rate)
  (rts : ℕ := red_tractors_sold)
  (gts : ℕ := green_tractors_sold)
  (s : ℝ := salary) :
  let earnings_red := rts * (rtcr * rtp) in
  let earnings_green := s - earnings_red in
  let green_tractor_price := (earnings_green / gts) / gtcr in
  green_tractor_price = 5000 := sorry

end green_tractor_price_is_5000_l30_30015


namespace solve_for_a_l30_30528

-- Define the line equation and the condition of equal intercepts
def line_eq (a x y : ℝ) : Prop :=
  a * x + y - 2 - a = 0

def equal_intercepts (a : ℝ) : Prop :=
  (∀ x, line_eq a x 0 → x = 2 + a) ∧ (∀ y, line_eq a 0 y → y = 2 + a)

-- State the problem to prove the value of 'a'
theorem solve_for_a (a : ℝ) : equal_intercepts a → (a = -2 ∨ a = 1) :=
by
  sorry

end solve_for_a_l30_30528


namespace Jamie_correct_percentage_l30_30584

theorem Jamie_correct_percentage (y : ℕ) : ((8 * y - 2 * y : ℕ) / (8 * y : ℕ) : ℚ) * 100 = 75 := by
  sorry

end Jamie_correct_percentage_l30_30584


namespace total_surface_area_of_cube_l30_30711

theorem total_surface_area_of_cube (edge_sum : ℕ) (h_edge_sum : edge_sum = 180) :
  ∃ (S : ℕ), S = 1350 := 
by
  sorry

end total_surface_area_of_cube_l30_30711


namespace Tom_runs_60_miles_in_a_week_l30_30865

variable (days_per_week : ℕ) (hours_per_day : ℝ) (speed : ℝ)
variable (h_days_per_week : days_per_week = 5)
variable (h_hours_per_day : hours_per_day = 1.5)
variable (h_speed : speed = 8)

theorem Tom_runs_60_miles_in_a_week : (days_per_week * hours_per_day * speed) = 60 := by
  sorry

end Tom_runs_60_miles_in_a_week_l30_30865


namespace inequality_proof_l30_30525

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (sqrt (a^2 + 8 * b * c) / a + sqrt (b^2 + 8 * a * c) / b + sqrt (c^2 + 8 * a * b) / c) ≥ 9 :=
by 
  sorry

end inequality_proof_l30_30525


namespace sandra_total_beignets_l30_30267

variable (beignets_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ)

def daily_consumption (beignets_per_day : ℕ) := beignets_per_day
def weekly_consumption (beignets_per_day days_per_week : ℕ) := beignets_per_day * days_per_week
def total_consumption (beignets_per_day days_per_week weeks : ℕ) := weekly_consumption beignets_per_day days_per_week * weeks

theorem sandra_total_beignets :
  daily_consumption 3 = 3 →
  days_per_week = 7 →
  weeks = 16 →
  total_consumption 3 7 16 = 336 :=
by
  intros h1 h2 h3
  sorry

end sandra_total_beignets_l30_30267


namespace probability_two_ones_in_twelve_dice_l30_30881

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def probability_two_ones (n : ℕ) (p : ℚ) : ℚ :=
  (binomial n 2) * (p^2) * ((1 - p)^(n - 2))

theorem probability_two_ones_in_twelve_dice : 
  probability_two_ones 12 (1/6) ≈ 0.293 := 
  by
    sorry

end probability_two_ones_in_twelve_dice_l30_30881


namespace complex_value_of_z_six_plus_z_inv_six_l30_30538

open Complex

theorem complex_value_of_z_six_plus_z_inv_six (z : ℂ) (h : z + z⁻¹ = 1) : z^6 + (z⁻¹)^6 = 2 := by
  sorry

end complex_value_of_z_six_plus_z_inv_six_l30_30538


namespace zero_in_A_l30_30382

-- Define the set A
def A : Set ℝ := { x | x * (x - 2) = 0 }

-- State the theorem
theorem zero_in_A : 0 ∈ A :=
by {
  -- Skipping the actual proof with "sorry"
  sorry
}

end zero_in_A_l30_30382


namespace shaded_quadrilateral_area_l30_30200

noncomputable def area_of_shaded_quadrilateral : ℝ :=
  let side_lens : List ℝ := [3, 5, 7, 9]
  let total_base: ℝ := side_lens.sum
  let largest_square_height: ℝ := 9
  let height_base_ratio := largest_square_height / total_base
  let heights := side_lens.scanl (· + ·) 0 |>.tail.map (λ x => x * height_base_ratio)
  let a := heights.get! 0
  let b := heights.get! heights.length - 1
  (largest_square_height * (a + b)) / 2

theorem shaded_quadrilateral_area :
    let side_lens := [3, 5, 7, 9]
    let total_base := side_lens.sum
    let largest_square_height := 9
    let height_base_ratio := largest_square_height / total_base
    let heights := side_lens.scanl (· + ·) 0 |>.tail.map (λ x => x * height_base_ratio)
    let a := heights.get! 0
    let b := heights.get! heights.length - 1
    (largest_square_height * (a + b)) / 2 = 30.375 :=
by 
  sorry

end shaded_quadrilateral_area_l30_30200


namespace inequality_a_b_l30_30516

theorem inequality_a_b (a b : ℝ) (h : a > b ∧ b > 0) : (1/a) < (1/b) := 
by
  sorry

end inequality_a_b_l30_30516


namespace minimum_value_expr_min_value_reachable_l30_30579

noncomputable def expr (x y : ℝ) : ℝ :=
  4 * x^2 + 9 * y^2 + 16 / x^2 + 6 * y / x

theorem minimum_value_expr (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  expr x y ≥ (2 * Real.sqrt 564) / 3 :=
sorry

theorem min_value_reachable :
  ∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ expr x y = (2 * Real.sqrt 564) / 3 :=
sorry

end minimum_value_expr_min_value_reachable_l30_30579


namespace tangent_line_equation_l30_30512

noncomputable def f (x : ℝ) : ℝ := (2 + Real.sin x) / Real.cos x

theorem tangent_line_equation :
  let x0 : ℝ := 0
  let y0 : ℝ := f x0
  let m : ℝ := (2 * x0 + 1) / (Real.cos x0 ^ 2)
  ∃ (a b c : ℝ), a * x0 + b * y0 + c = 0 ∧ a = 1 ∧ b = -1 ∧ c = 2 :=
by
  sorry

end tangent_line_equation_l30_30512


namespace r_minus_s_l30_30578

-- Define the equation whose roots are r and s
def equation (x : ℝ) := (6 * x - 18) / (x ^ 2 + 4 * x - 21) = x + 3

-- Define the condition that r and s are distinct roots of the equation and r > s
def is_solution_pair (r s : ℝ) :=
  equation r ∧ equation s ∧ r ≠ s ∧ r > s

-- The main theorem we need to prove
theorem r_minus_s (r s : ℝ) (h : is_solution_pair r s) : r - s = 12 :=
by
  sorry

end r_minus_s_l30_30578


namespace calculate_v2_using_horner_method_l30_30364

def f (x : ℕ) : ℕ := x^5 + 5 * x^4 + 10 * x^3 + 10 * x^2 + 5 * x + 1

def horner_step (x b a : ℕ) := a * x + b

def horner_eval (coeffs : List ℕ) (x : ℕ) : ℕ :=
coeffs.foldr (horner_step x) 0

theorem calculate_v2_using_horner_method :
  horner_eval [1, 5, 10, 10, 5, 1] 2 = 24 :=
by
  -- This is the theorem statement, the proof is not required as per instructions
  sorry

end calculate_v2_using_horner_method_l30_30364


namespace simplify_expression_l30_30917

theorem simplify_expression (a b : ℕ) (h₁ : a = 2999) (h₂ : b = 3000) :
  b^3 - a * b^2 - a^2 * b + a^3 = 5999 :=
by 
  sorry

end simplify_expression_l30_30917


namespace num_solutions_gcd_lcm_l30_30155

noncomputable def factorial : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

theorem num_solutions_gcd_lcm (x y : ℕ) :
  (Nat.gcd x y = factorial 20) ∧ (Nat.lcm x y = factorial 30) →
  2^10 = 1024 :=
  by
  intro h
  sorry

end num_solutions_gcd_lcm_l30_30155


namespace average_glasses_is_15_l30_30044

variable (S L : ℕ)

-- Conditions:
def box1 := 12 -- One box contains 12 glasses
def box2 := 16 -- Another box contains 16 glasses
def total_glasses := 480 -- Total number of glasses
def diff_L_S := 16 -- There are 16 more larger boxes

-- Equations derived from conditions:
def eq1 : Prop := (12 * S + 16 * L = total_glasses)
def eq2 : Prop := (L = S + diff_L_S)

-- We need to prove that the average number of glasses per box is 15:
def avg_glasses_per_box := total_glasses / (S + L)

-- The statement we need to prove:
theorem average_glasses_is_15 :
  (12 * S + 16 * L = total_glasses) ∧ (L = S + diff_L_S) → avg_glasses_per_box = 15 :=
by
  sorry

end average_glasses_is_15_l30_30044


namespace highest_of_seven_consecutive_with_average_33_l30_30277

theorem highest_of_seven_consecutive_with_average_33 (x : ℤ) 
    (h : (x - 3 + x - 2 + x - 1 + x + x + 1 + x + 2 + x + 3) / 7 = 33) : 
    x + 3 = 36 := 
sorry

end highest_of_seven_consecutive_with_average_33_l30_30277


namespace hexagon_perimeter_l30_30136

theorem hexagon_perimeter (side_length : ℕ) (num_sides : ℕ) (perimeter : ℕ) 
  (h1 : num_sides = 6)
  (h2 : side_length = 7)
  (h3 : perimeter = side_length * num_sides) : perimeter = 42 := by
  sorry

end hexagon_perimeter_l30_30136


namespace roof_length_width_diff_l30_30704

variable (w l : ℝ)
variable (h1 : l = 4 * w)
variable (h2 : l * w = 676)

theorem roof_length_width_diff :
  l - w = 39 :=
by
  sorry

end roof_length_width_diff_l30_30704


namespace find_ordered_pair_l30_30853

theorem find_ordered_pair (s l : ℝ) :
  (∀ (x y : ℝ), (∃ t : ℝ, (x, y) = (-8 + t * l, s - 7 * t)) ↔ y = 2 * x - 3) →
  (s = -19 ∧ l = -7 / 2) :=
by
  intro h
  have : (∀ (x y : ℝ), (∃ t : ℝ, (x, y) = (-8 + t * l, s - 7 * t)) ↔ y = 2 * x - 3) := h
  sorry

end find_ordered_pair_l30_30853


namespace find_f_2010_l30_30114

open Nat

variable (f : ℕ → ℕ)

axiom strictly_increasing : ∀ m n : ℕ, m < n → f m < f n

axiom function_condition : ∀ n : ℕ, f (f n) = 3 * n

theorem find_f_2010 : f 2010 = 3015 := sorry

end find_f_2010_l30_30114


namespace gathering_people_total_l30_30180

theorem gathering_people_total (W S B : ℕ) (hW : W = 26) (hS : S = 22) (hB : B = 17) :
  W + S - B = 31 :=
by
  sorry

end gathering_people_total_l30_30180


namespace smallest_k_no_real_roots_l30_30302

theorem smallest_k_no_real_roots :
  ∃ k : ℤ, (∀ x : ℝ, 3 * x * (k * x - 5) - 2 * x^2 + 13 ≠ 0) ∧
  (∀ n : ℤ, n < k → ∃ x : ℝ, 3 * x * (n * x - 5) - 2 * x^2 + 13 = 0) :=
by sorry

end smallest_k_no_real_roots_l30_30302


namespace frank_spent_per_week_l30_30201

theorem frank_spent_per_week (mowing_dollars : ℕ) (weed_eating_dollars : ℕ) (weeks : ℕ) 
    (total_dollars := mowing_dollars + weed_eating_dollars) 
    (spending_rate := total_dollars / weeks) :
    mowing_dollars = 5 → weed_eating_dollars = 58 → weeks = 9 → spending_rate = 7 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end frank_spent_per_week_l30_30201


namespace negation_of_exists_l30_30284

theorem negation_of_exists (x : ℝ) : 
  ¬ (∃ x : ℝ, 2 * x^2 + 2 * x - 1 ≤ 0) ↔ ∀ x : ℝ, 2 * x^2 + 2 * x - 1 > 0 :=
by
  sorry

end negation_of_exists_l30_30284


namespace cannot_be_square_of_difference_formula_l30_30468

theorem cannot_be_square_of_difference_formula (x y c d a b m n : ℝ) :
  ¬ ((m - n) * (-m + n) = (x^2 - y^2) ∨ 
       (m - n) * (-m + n) = (c^2 - d^2) ∨ 
       (m - n) * (-m + n) = (a^2 - b^2)) :=
by sorry

end cannot_be_square_of_difference_formula_l30_30468


namespace parallelogram_sticks_l30_30542

theorem parallelogram_sticks (a : ℕ) (h₁ : ∃ l₁ l₂, l₁ = 5 ∧ l₂ = 5 ∧ 
                                (l₁ = l₂) ∧ (a = 7)) : a = 7 :=
by sorry

end parallelogram_sticks_l30_30542


namespace sufficient_but_not_necessary_l30_30521

theorem sufficient_but_not_necessary (x : ℝ) (h : x > 2) : (1/x < 1/2 ∧ (∃ y : ℝ, 1/y < 1/2 ∧ y ≤ 2)) :=
by { sorry }

end sufficient_but_not_necessary_l30_30521


namespace abc_inequality_l30_30974

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) :=
  sorry

end abc_inequality_l30_30974


namespace trapezoid_AD_BC_ratio_l30_30070

variables {A B C D M N K : Type} {AD BC CM MD NA CN : ℝ}

-- Definition of the trapezoid and the ratio conditions
def is_trapezoid (A B C D : Type) : Prop := sorry -- Assume existence of a trapezoid for lean to accept the statement
def ratio_CM_MD (CM MD : ℝ) : Prop := CM / MD = 4 / 3
def ratio_NA_CN (NA CN : ℝ) : Prop := NA / CN = 4 / 3

-- Proof statement for the given problem
theorem trapezoid_AD_BC_ratio 
  (h_trapezoid: is_trapezoid A B C D)
  (h_CM_MD: ratio_CM_MD CM MD)
  (h_NA_CN: ratio_NA_CN NA CN) :
  AD / BC = 7 / 12 :=
sorry

end trapezoid_AD_BC_ratio_l30_30070


namespace polynomial_solution_l30_30057
-- Import necessary library

-- Define the property to be checked
def polynomial_property (P : Real → Real) : Prop :=
  ∀ a b c : Real, 
    P (a + b - 2 * c) + P (b + c - 2 * a) + P (c + a - 2 * b) = 
    3 * P (a - b) + 3 * P (b - c) + 3 * P (c - a)

-- The statement that needs to be proven
theorem polynomial_solution (a b : Real) : polynomial_property (λ x => a * x^2 + b * x) := 
by
  sorry

end polynomial_solution_l30_30057


namespace probability_not_above_y_axis_l30_30836

-- Define the vertices
structure Point :=
  (x : ℝ)
  (y : ℝ)

def P := Point.mk (-1) 5
def Q := Point.mk 2 (-3)
def R := Point.mk (-5) (-3)
def S := Point.mk (-8) 5

-- Define predicate for being above the y-axis
def is_above_y_axis (p : Point) : Prop := p.y > 0

-- Define the parallelogram region (this is theoretical as defining a whole region 
-- can be complex, but we state the region as a property)
noncomputable def in_region_of_parallelogram (p : Point) : Prop := sorry

-- Define the probability calculation statement
theorem probability_not_above_y_axis (p : Point) :
  in_region_of_parallelogram p → ¬is_above_y_axis p := sorry

end probability_not_above_y_axis_l30_30836


namespace line_plane_relationship_l30_30543

variables {V : Type} [InnerProductSpace ℝ V]

theorem line_plane_relationship {a b : V} (α : Set V) [Plane α] 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ⊥ b) (h4 : a ∈ α) : 
  b ∈ α ∨ (∀ {p : V}, p ∈ α → ¬ (b ≠ 0 ∧ b - p ∈ α)) ∨ ∃ q ∈ α, b = q + c • r :=
sorry

end line_plane_relationship_l30_30543


namespace sum_of_two_integers_l30_30453

theorem sum_of_two_integers (a b : ℕ) (h1 : a * b + a + b = 113) (h2 : Nat.gcd a b = 1) (h3 : a < 25) (h4 : b < 25) : a + b = 23 := by
  sorry

end sum_of_two_integers_l30_30453


namespace smallest_integer_cube_ends_in_528_l30_30199

theorem smallest_integer_cube_ends_in_528 :
  ∃ (n : ℕ), (n^3 % 1000 = 528 ∧ ∀ m : ℕ, (m^3 % 1000 = 528) → m ≥ n) ∧ n = 428 :=
by
  sorry

end smallest_integer_cube_ends_in_528_l30_30199


namespace acres_left_untouched_l30_30732

def total_acres := 65057
def covered_acres := 64535

theorem acres_left_untouched : total_acres - covered_acres = 522 :=
by
  sorry

end acres_left_untouched_l30_30732


namespace find_x_plus_y_l30_30209

theorem find_x_plus_y
  (x y : ℝ)
  (h1 : x + Real.cos y = 2010)
  (h2 : x + 2010 * Real.sin y = 2009)
  (h3 : (π / 2) ≤ y ∧ y ≤ π) :
  x + y = 2011 + π :=
sorry

end find_x_plus_y_l30_30209


namespace john_pin_discount_l30_30107

theorem john_pin_discount :
  ∀ (n_pins price_per_pin amount_spent discount_rate : ℝ),
    n_pins = 10 →
    price_per_pin = 20 →
    amount_spent = 170 →
    discount_rate = ((n_pins * price_per_pin - amount_spent) / (n_pins * price_per_pin)) * 100 →
    discount_rate = 15 :=
by
  intros n_pins price_per_pin amount_spent discount_rate h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end john_pin_discount_l30_30107


namespace number_of_factors_l30_30191

theorem number_of_factors :
  let n := (2^3) * (3^5) * (5^4) * (7^2) * (11^6)
  let exponents := [3, 5, 4, 2, 6]
  let calc_number_of_factors (exps : List ℕ) : ℕ := exps.foldl (\prod (exp : ℕ) => prod * (exp + 1)) 1
  calc_number_of_factors exponents = 2520 :=
by
  sorry

end number_of_factors_l30_30191


namespace sum_of_digits_l30_30280

def original_sum := 943587 + 329430
def provided_sum := 1412017
def correct_sum_after_change (d e : ℕ) : ℕ := 
  let new_first := if d = 3 then 944587 else 943587
  let new_second := if d = 3 then 429430 else 329430
  new_first + new_second

theorem sum_of_digits (d e : ℕ) : d = 3 ∧ e = 4 → d + e = 7 :=
by
  intros
  exact sorry

end sum_of_digits_l30_30280


namespace train_meeting_distance_l30_30728

theorem train_meeting_distance :
  let distance := 150
  let time_x := 4
  let time_y := 3.5
  let speed_x := distance / time_x
  let speed_y := distance / time_y
  let relative_speed := speed_x + speed_y
  let time_to_meet := distance / relative_speed
  let distance_x_at_meeting := time_to_meet * speed_x
  distance_x_at_meeting = 70 := by
sorry

end train_meeting_distance_l30_30728


namespace probability_three_digit_divisible_by_3_l30_30362

def digits : List ℕ := [0, 1, 2, 3]

noncomputable def count_total_three_digit_numbers : ℕ :=
  (3 * 3 * 2) -- 3 choices for first digit (excluding 0), 3 choices for second, 2 for third

noncomputable def count_divisible_by_three : ℕ :=
  (4 + 6) -- combination of the different ways calculable

theorem probability_three_digit_divisible_by_3 :
  (count_divisible_by_three : ℚ) / (count_total_three_digit_numbers : ℚ) = 5 / 9 := 
  sorry

end probability_three_digit_divisible_by_3_l30_30362


namespace one_greater_one_smaller_l30_30106

theorem one_greater_one_smaller (a b : ℝ) (h : ( (1 + a * b) / (a + b) )^2 < 1) :
  (a > 1 ∧ -1 < b ∧ b < 1) ∨ (b > 1 ∧ -1 < a ∧ a < 1) ∨ (a < -1 ∧ -1 < b ∧ b < 1) ∨ (b < -1 ∧ -1 < a ∧ a < 1) :=
by
  sorry

end one_greater_one_smaller_l30_30106


namespace y_time_to_complete_work_l30_30304

-- Definitions of the conditions
def work_rate_x := 1 / 40
def work_done_by_x_in_8_days := 8 * work_rate_x
def remaining_work := 1 - work_done_by_x_in_8_days
def y_completion_time := 32
def work_rate_y := remaining_work / y_completion_time

-- Lean theorem
theorem y_time_to_complete_work :
  y_completion_time * work_rate_y = 1 →
  (1 / work_rate_y = 40) :=
by
  sorry

end y_time_to_complete_work_l30_30304


namespace focus_of_parabola_y_2x2_l30_30356

theorem focus_of_parabola_y_2x2 :
  ∃ f, f = 1 / 8 ∧ (∀ x, sqrt (x^2 + (2*x^2 - f)^2) = abs (2*x^2 - (-f)))
:= sorry

end focus_of_parabola_y_2x2_l30_30356


namespace simplify_fraction_l30_30590

theorem simplify_fraction (x y : ℝ) : (x - y) / (y - x) = -1 :=
sorry

end simplify_fraction_l30_30590


namespace gcd_condition_for_divisibility_l30_30909

theorem gcd_condition_for_divisibility (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (∀ (x : ℕ), x^n + x^(2*n) + ... + x^(m*n) ∣ x + x^(2) + ... + x^m) ↔ Nat.gcd(n, m+1) = 1 :=
by
  sorry

end gcd_condition_for_divisibility_l30_30909


namespace triangle_angles_30_60_90_l30_30967

-- Definition of the angles based on the given ratio
def angles_ratio (A B C : ℝ) : Prop :=
  A / B = 1 / 2 ∧ B / C = 2 / 3

-- The main statement to be proved
theorem triangle_angles_30_60_90
  (A B C : ℝ)
  (h1 : angles_ratio A B C)
  (h2 : A + B + C = 180) :
  A = 30 ∧ B = 60 ∧ C = 90 := 
sorry

end triangle_angles_30_60_90_l30_30967


namespace monogram_count_l30_30977

theorem monogram_count : (Finset.combo 12 2).card = 66 :=
by sorry

end monogram_count_l30_30977


namespace expression_evaluation_l30_30597

theorem expression_evaluation : (6 * 111) - (2 * 111) = 444 :=
by
  sorry

end expression_evaluation_l30_30597


namespace percentage_of_fish_gone_bad_l30_30416

-- Definitions based on conditions
def fish_per_roll : ℕ := 40
def total_fish_bought : ℕ := 400
def sushi_rolls_made : ℕ := 8

-- Definition of fish calculations
def total_fish_used (rolls: ℕ) (per_roll: ℕ) : ℕ := rolls * per_roll
def fish_gone_bad (total : ℕ) (used : ℕ) : ℕ := total - used
def percentage (part : ℕ) (whole : ℕ) : ℚ := (part : ℚ) / (whole : ℚ) * 100

-- Theorem to prove the percentage of bad fish
theorem percentage_of_fish_gone_bad :
  percentage (fish_gone_bad total_fish_bought (total_fish_used sushi_rolls_made fish_per_roll)) total_fish_bought = 20 := by
  sorry

end percentage_of_fish_gone_bad_l30_30416


namespace pieces_in_each_package_l30_30258

-- Definitions from conditions
def num_packages : ℕ := 5
def extra_pieces : ℕ := 6
def total_pieces : ℕ := 41

-- Statement to prove
theorem pieces_in_each_package : ∃ x : ℕ, num_packages * x + extra_pieces = total_pieces ∧ x = 7 :=
by
  -- Begin the proof with the given setup
  sorry

end pieces_in_each_package_l30_30258


namespace equal_ratios_l30_30175

variable (x y : ℝ)

-- Conditions
def wire_split_to_form_square_and_pentagon (x y : ℝ) : Prop :=
  4 * (x / 4) = 5 * (y / 5)

-- Theorem to prove
theorem equal_ratios (x y : ℝ) (h : wire_split_to_form_square_and_pentagon x y) : x / y = 1 :=
  sorry

end equal_ratios_l30_30175


namespace pencils_pens_total_l30_30403

theorem pencils_pens_total (x : ℕ) (h1 : 4 * x + 1 = 7 * (5 * x - 1)) : 4 * x + 5 * x = 45 :=
by
  sorry

end pencils_pens_total_l30_30403


namespace consecutive_integers_sum_l30_30670

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l30_30670


namespace norm_2u_equals_10_l30_30515

-- Define u as a vector in ℝ² and the function for its norm.
variable (u : ℝ × ℝ)

-- Define the condition that the norm of u is 5.
def norm_eq_5 : Prop := Real.sqrt (u.1^2 + u.2^2) = 5

-- Statement of the proof problem
theorem norm_2u_equals_10 (h : norm_eq_5 u) : Real.sqrt ((2 * u.1)^2 + (2 * u.2)^2) = 10 :=
by
  sorry

end norm_2u_equals_10_l30_30515


namespace find_cost_of_two_enchiladas_and_five_tacos_l30_30120

noncomputable def cost_of_two_enchiladas_and_five_tacos (e t : ℝ) : ℝ :=
  2 * e + 5 * t

theorem find_cost_of_two_enchiladas_and_five_tacos (e t : ℝ):
  (e + 4 * t = 3.50) → (4 * e + t = 4.20) → cost_of_two_enchiladas_and_five_tacos e t = 5.04 :=
by
  intro h1 h2
  sorry

end find_cost_of_two_enchiladas_and_five_tacos_l30_30120


namespace ab_parallel_to_x_axis_and_ac_parallel_to_y_axis_l30_30556

theorem ab_parallel_to_x_axis_and_ac_parallel_to_y_axis
  (a b : ℝ)
  (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ)
  (hA : A = (a, -1))
  (hB : B = (2, 3 - b))
  (hC : C = (-5, 4))
  (hAB_parallel_x : A.2 = B.2)
  (hAC_parallel_y : A.1 = C.1) : a + b = -1 := by
  sorry


end ab_parallel_to_x_axis_and_ac_parallel_to_y_axis_l30_30556


namespace light_travel_distance_120_years_l30_30448

theorem light_travel_distance_120_years :
  let annual_distance : ℝ := 9.46e12
  let years : ℝ := 120
  (annual_distance * years) = 1.1352e15 := 
by
  sorry

end light_travel_distance_120_years_l30_30448


namespace num_sets_l30_30601

theorem num_sets {A : Set ℕ} :
  {1} ⊆ A ∧ A ⊆ {1, 2, 3, 4, 5} → ∃ n, n = 16 := 
by
  sorry

end num_sets_l30_30601


namespace range_of_x_l30_30376

variable {f : ℝ → ℝ}

-- Define the function is_increasing
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem range_of_x (h_inc : is_increasing f) (h_ineq : ∀ x : ℝ, f x < f (2 * x - 3)) :
  ∀ x : ℝ, 3 < x → f x < f (2 * x - 3) := 
sorry

end range_of_x_l30_30376


namespace empty_set_implies_a_range_l30_30956

theorem empty_set_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ¬(a * x^2 - 2 * a * x + 1 < 0)) ↔ (0 ≤ a ∧ a < 1) := 
sorry

end empty_set_implies_a_range_l30_30956


namespace weibull_distribution_double_exponential_distribution_integer_fractional_parts_independent_integer_part_distribution_fractional_part_distribution_l30_30027

open MeasureTheory ProbabilityTheory

variables {λ : ℝ} (hλ : λ > 0) {α : ℝ} (hα : α > 0)

noncomputable def exp_pdf (λ : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 0 else λ * exp (-λ * x)

noncomputable def weibull_density (λ : ℝ) (α : ℝ) (y : ℝ) : ℝ :=
  λ * α * y^(α - 1) * exp (-(λ * y^α))

noncomputable def double_exp_density (λ : ℝ) (y : ℝ) : ℝ :=
  λ * exp (y - λ * exp y)

theorem weibull_distribution :
  ∀ y > 0, PDF (λ X, X ^ (1 / α)) (weibull_density λ α) :=
sorry

theorem double_exponential_distribution :
  ∀ y : ℝ, PDF (λ X, log X) (double_exp_density λ) :=
sorry

theorem integer_fractional_parts_independent :
  ∀ (n : ℕ) α ∈ [0, 1), independent (λ X, ⌊X⌋) (λ X, X - ⌊X⌋) :=
sorry

theorem integer_part_distribution :
  ∀ n : ℕ, P (λ X, ⌊X⌋ = n) = exp (-λ * n) * (1 - exp (-λ)) :=
sorry

theorem fractional_part_distribution :
  ∀ α ∈ [0, 1), P (λ X, fract X ≤ α) = (1 - exp (-λ * α)) / (1 - exp (-λ)) :=
sorry

end weibull_distribution_double_exponential_distribution_integer_fractional_parts_independent_integer_part_distribution_fractional_part_distribution_l30_30027


namespace combined_stickers_l30_30423

def initial_stickers_june : ℕ := 76
def initial_stickers_bonnie : ℕ := 63
def birthday_stickers : ℕ := 25

theorem combined_stickers : 
  (initial_stickers_june + birthday_stickers) + (initial_stickers_bonnie + birthday_stickers) = 189 := 
by
  sorry

end combined_stickers_l30_30423


namespace local_maximum_at_e_l30_30131

open Real

noncomputable def f (x : ℝ) : ℝ := (ln x) / x
noncomputable def f_prime (x : ℝ) : ℝ := (1 - ln x) / x^2

theorem local_maximum_at_e :
  is_local_max (λ x : ℝ, (ln x) / x) e :=
by {
  -- Proof would go here
  sorry
}

end local_maximum_at_e_l30_30131


namespace last_locker_opened_2046_l30_30480

def last_locker_opened (n : ℕ) : ℕ :=
  n - (n % 3)

theorem last_locker_opened_2046 : last_locker_opened 2048 = 2046 := by
  sorry

end last_locker_opened_2046_l30_30480


namespace geometric_sequence_problem_l30_30375

section 
variables (a : ℕ → ℝ) (r : ℝ) 

-- Condition: {a_n} is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) := ∀ n : ℕ, a (n + 1) = a n * r

-- Condition: a_4 + a_6 = 8
axiom a4_a6_sum : a 4 + a 6 = 8

-- Mathematical equivalent proof problem
theorem geometric_sequence_problem (h : is_geometric_sequence a r) : 
  a 1 * a 7 + 2 * a 3 * a 7 + a 3 * a 9 = 64 :=
sorry

end

end geometric_sequence_problem_l30_30375


namespace line_through_points_l30_30412

theorem line_through_points (m n p : ℝ) 
  (h1 : m = 4 * n + 5) 
  (h2 : m + 2 = 4 * (n + p) + 5) : 
  p = 1 / 2 := 
by 
  sorry

end line_through_points_l30_30412


namespace solve_quadratic_l30_30273

theorem solve_quadratic (x : ℝ) : (x^2 + x)^2 + (x^2 + x) - 6 = 0 ↔ x = -2 ∨ x = 1 :=
by
  sorry

end solve_quadratic_l30_30273


namespace smallest_n_exists_l30_30755

theorem smallest_n_exists (n : ℕ) (h : n ≥ 4) :
  (∃ (S : Finset ℤ), S.card = n ∧
    (∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
        (a + b - c - d) % 20 = 0))
  ↔ n = 9 := sorry

end smallest_n_exists_l30_30755


namespace original_inhabitants_l30_30902

theorem original_inhabitants (X : ℝ) (h : 0.75 * 0.9 * X = 5265) : X = 7800 :=
by
  sorry

end original_inhabitants_l30_30902


namespace josh_marbles_l30_30568

theorem josh_marbles (original_marble : ℝ) (given_marble : ℝ)
  (h1 : original_marble = 22.5) (h2 : given_marble = 20.75) :
  original_marble + given_marble = 43.25 := by
  sorry

end josh_marbles_l30_30568


namespace sum_coordinates_D_is_13_l30_30585

theorem sum_coordinates_D_is_13 
  (A B C D : ℝ × ℝ) 
  (hA : A = (4, 8))
  (hB : B = (2, 2))
  (hC : C = (6, 4))
  (hD : D = (8, 5))
  (h_mid1 : (A.1 + B.1) / 2 = 3 ∧ (A.2 + B.2) / 2 = 5)
  (h_mid2 : (B.1 + C.1) / 2 = 4 ∧ (B.2 + C.2) / 2 = 3)
  (h_mid3 : (C.1 + D.1) / 2 = 7 ∧ (C.2 + D.2) / 2 = 4.5)
  (h_mid4 : (D.1 + A.1) / 2 = 6 ∧ (D.2 + A.2) / 2 = 6.5)
  (h_square : ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (3, 5) ∧
               ((B.1 + C.1) / 2, (B.2 + C.2) / 2) = (4, 3) ∧
               ((C.1 + D.1) / 2, (C.2 + D.2) / 2) = (7, 4.5) ∧
               ((D.1 + A.1) / 2, (D.2 + A.2) / 2) = (6, 6.5))
  : (8 + 5) = 13 :=
by
  sorry

end sum_coordinates_D_is_13_l30_30585


namespace find_n_satisfying_conditions_l30_30925

noncomputable def exists_set_satisfying_conditions (n : ℕ) : Prop :=
  ∃ S : Finset ℕ, S.card = n ∧
  (∀ x ∈ S, x < 2^(n-1)) ∧
  ∀ A B : Finset ℕ, A ⊆ S → B ⊆ S → A ≠ B → A ≠ ∅ → B ≠ ∅ → A.sum id ≠ B.sum id

theorem find_n_satisfying_conditions : ∀ n : ℕ, (n ≥ 4) ↔ exists_set_satisfying_conditions n :=
sorry

end find_n_satisfying_conditions_l30_30925


namespace complete_square_form_l30_30987

noncomputable def quadratic_expr (x : ℝ) : ℝ := x^2 - 10 * x + 15

theorem complete_square_form (b c : ℤ) (h : ∀ x : ℝ, quadratic_expr x = 0 ↔ (x + b)^2 = c) :
  b + c = 5 :=
sorry

end complete_square_form_l30_30987


namespace consecutive_integers_sum_l30_30621

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l30_30621


namespace gcd_lcm_product_l30_30772

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem gcd_lcm_product (a b : ℕ) : gcd a b * lcm a b = a * b :=
begin
  apply Nat.gcd_mul_lcm,
end

example : gcd 12 9 * lcm 12 9 = 108 :=
by
  have h : gcd 12 9 * lcm 12 9 = 12 * 9 := gcd_lcm_product 12 9
  rw [show 12 * 9 = 108, by norm_num] at h
  exact h

end gcd_lcm_product_l30_30772


namespace feet_to_inches_conversion_l30_30994

-- Define the constant equivalence between feet and inches
def foot_to_inches := 12

-- Prove the conversion factor between feet and inches
theorem feet_to_inches_conversion:
  foot_to_inches = 12 :=
by
  sorry

end feet_to_inches_conversion_l30_30994


namespace evaluate_sqrt_log_expression_l30_30914

noncomputable def evaluate_log_expression : ℝ :=
  let log3 (x : ℝ) := Real.log x / Real.log 3
  let log4 (x : ℝ) := Real.log x / Real.log 4
  Real.sqrt (log3 8 + log4 8)

theorem evaluate_sqrt_log_expression : evaluate_log_expression = Real.sqrt 3 := 
by
  sorry

end evaluate_sqrt_log_expression_l30_30914


namespace factor_expression_l30_30922

theorem factor_expression (x : ℝ) : 5 * x * (x - 2) + 9 * (x - 2) = (x - 2) * (5 * x + 9) := 
by 
sorry

end factor_expression_l30_30922


namespace simplify_fraction_l30_30518

variable (a b y : ℝ)
variable (h1 : y = (a + 2 * b) / a)
variable (h2 : a ≠ -2 * b)
variable (h3 : a ≠ 0)

theorem simplify_fraction : (2 * a + 2 * b) / (a - 2 * b) = (y + 1) / (3 - y) :=
by
  sorry

end simplify_fraction_l30_30518


namespace chantel_final_bracelets_l30_30774

-- Definitions of the conditions in Lean
def initial_bracelets_7_days := 7 * 4
def after_school_giveaway := initial_bracelets_7_days - 8
def bracelets_10_days := 10 * 5
def total_after_10_days := after_school_giveaway + bracelets_10_days
def after_soccer_giveaway := total_after_10_days - 12
def crafting_club_bracelets := 4 * 6
def total_after_crafting_club := after_soccer_giveaway + crafting_club_bracelets
def weekend_trip_bracelets := 2 * 3
def total_after_weekend_trip := total_after_crafting_club + weekend_trip_bracelets
def final_total := total_after_weekend_trip - 10

-- Lean statement to prove the final total bracelets
theorem chantel_final_bracelets : final_total = 78 :=
by
  -- Note: The proof is not required, hence the sorry
  sorry

end chantel_final_bracelets_l30_30774


namespace consecutive_integer_product_sum_l30_30646

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l30_30646


namespace subset_M_N_l30_30975

-- Definition of the sets
def M : Set ℝ := {-1, 1}
def N : Set ℝ := { x | 1/x < 3 }

theorem subset_M_N : M ⊆ N :=
by
  -- sorry to skip the proof
  sorry

end subset_M_N_l30_30975


namespace sandra_total_beignets_l30_30266

variable (beignets_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ)

def daily_consumption (beignets_per_day : ℕ) := beignets_per_day
def weekly_consumption (beignets_per_day days_per_week : ℕ) := beignets_per_day * days_per_week
def total_consumption (beignets_per_day days_per_week weeks : ℕ) := weekly_consumption beignets_per_day days_per_week * weeks

theorem sandra_total_beignets :
  daily_consumption 3 = 3 →
  days_per_week = 7 →
  weeks = 16 →
  total_consumption 3 7 16 = 336 :=
by
  intros h1 h2 h3
  sorry

end sandra_total_beignets_l30_30266


namespace nonnegative_expr_interval_l30_30508

noncomputable def expr (x : ℝ) : ℝ := (2 * x - 15 * x ^ 2 + 56 * x ^ 3) / (9 - x ^ 3)

theorem nonnegative_expr_interval (x : ℝ) :
  expr x ≥ 0 ↔ 0 ≤ x ∧ x < 3 := by
  sorry

end nonnegative_expr_interval_l30_30508


namespace distinct_solutions_subtraction_l30_30576

theorem distinct_solutions_subtraction (r s : ℝ) (h_eq : ∀ x ≠ 3, (6 * x - 18) / (x^2 + 4 * x - 21) = x + 3) 
  (h_r : (6 * r - 18) / (r^2 + 4 * r - 21) = r + 3) 
  (h_s : (6 * s - 18) / (s^2 + 4 * s - 21) = s + 3) 
  (h_distinct : r ≠ s) 
  (h_order : r > s) : 
  r - s = 10 := 
by 
  sorry

end distinct_solutions_subtraction_l30_30576


namespace regular_polygon_has_20_sides_l30_30316

-- Definition of a regular polygon with an exterior angle of 18 degrees
def regular_polygon_sides (external_angle : ℝ) : ℕ :=
  if external_angle > 0 then (360 / external_angle).toInt else 0

theorem regular_polygon_has_20_sides :
  regular_polygon_sides 18 = 20 :=
by
  sorry

end regular_polygon_has_20_sides_l30_30316


namespace dihedral_angle_of_equilateral_triangle_l30_30807

theorem dihedral_angle_of_equilateral_triangle (a : ℝ) 
(ABC_eq : ∀ {A B C : ℝ}, (B - A) ^ 2 + (C - A) ^ 2 = a^2 ∧ (C - B) ^ 2 + (A - B) ^ 2 = a^2 ∧ (A - C) ^ 2 + (B - C) ^ 2 = a^2) 
(perpendicular : ∀ A B C D : ℝ, D = (B + C)/2 ∧ (B - D) * (C - D) = 0) : 
∃ θ : ℝ, θ = 60 := 
  sorry

end dihedral_angle_of_equilateral_triangle_l30_30807


namespace part_one_part_two_l30_30581

noncomputable def M := Set.Ioo (-(1 : ℝ)/2) (1/2)

namespace Problem

variables {a b : ℝ}
def in_M (x : ℝ) := x ∈ M

theorem part_one (ha : in_M a) (hb : in_M b) :
  |(1/3 : ℝ) * a + (1/6) * b| < 1/4 :=
sorry

theorem part_two (ha : in_M a) (hb : in_M b) :
  |1 - 4 * a * b| > 2 * |a - b| :=
sorry

end Problem

end part_one_part_two_l30_30581


namespace problem_l30_30630

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l30_30630


namespace sets_equal_l30_30531

-- Definitions of sets M and N
def M := { u : ℤ | ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l }
def N := { u : ℤ | ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r }

-- Theorem statement asserting M = N
theorem sets_equal : M = N :=
by sorry

end sets_equal_l30_30531


namespace problem_l30_30972

noncomputable def f (x a b : ℝ) := x^2 + a*x + b
noncomputable def g (x c d : ℝ) := x^2 + c*x + d

theorem problem (a b c d : ℝ) (h_min_f : f (-a/2) a b = -25) (h_min_g : g (-c/2) c d = -25)
  (h_intersection_f : f 50 a b = -50) (h_intersection_g : g 50 c d = -50)
  (h_root_f_of_g : g (-a/2) c d = 0) (h_root_g_of_f : f (-c/2) a b = 0) :
  a + c = -200 := by
  sorry

end problem_l30_30972


namespace units_digit_17_pow_35_l30_30721

theorem units_digit_17_pow_35 : (17 ^ 35) % 10 = 3 := by
sorry

end units_digit_17_pow_35_l30_30721


namespace sufficient_condition_for_sets_l30_30530

theorem sufficient_condition_for_sets (A B : Set ℝ) (m : ℝ) :
    (∀ x, x ∈ A → x ∈ B) → (m ≥ 3 / 4 ∨ m ≤ -3 / 4) :=
by
    have A_def : A = {y | ∃ x, y = x^2 - (3 / 2) * x + 1 ∧ (1 / 4) ≤ x ∧ x ≤ 2} := sorry
    have B_def : B = {x | x ≥ 1 - m^2} := sorry
    sorry

end sufficient_condition_for_sets_l30_30530


namespace total_books_count_l30_30008

theorem total_books_count (books_read : ℕ) (books_unread : ℕ) (h1 : books_read = 13) (h2 : books_unread = 8) : books_read + books_unread = 21 := 
by
  -- Proof omitted
  sorry

end total_books_count_l30_30008


namespace hyperbola_eccentricity_l30_30069

theorem hyperbola_eccentricity (a b x₀ y₀ : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : x₀^2 / a^2 - y₀^2 / b^2 = 1)
  (h₄ : a ≤ x₀ ∧ x₀ ≤ 2 * a)
  (h₅ : x₀ / a^2 * 0 - y₀ / b^2 * b = 1)
  (h₆ : - (a * a / (2 * b)) = 2) :
  (1 + b^2 / a^2 = 3) :=
sorry

end hyperbola_eccentricity_l30_30069


namespace sum_of_consecutive_integers_l30_30638

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l30_30638


namespace r_minus_s_l30_30577

-- Define the equation whose roots are r and s
def equation (x : ℝ) := (6 * x - 18) / (x ^ 2 + 4 * x - 21) = x + 3

-- Define the condition that r and s are distinct roots of the equation and r > s
def is_solution_pair (r s : ℝ) :=
  equation r ∧ equation s ∧ r ≠ s ∧ r > s

-- The main theorem we need to prove
theorem r_minus_s (r s : ℝ) (h : is_solution_pair r s) : r - s = 12 :=
by
  sorry

end r_minus_s_l30_30577


namespace problem_statement_l30_30951

variable (m : ℝ) -- We declare m as a real number

theorem problem_statement (h : m + 1/m = 10) : m^2 + 1/m^2 + 4 = 102 := 
by 
  sorry -- The proof is omitted

end problem_statement_l30_30951


namespace cross_number_puzzle_digit_star_l30_30729

theorem cross_number_puzzle_digit_star :
  ∃ N₁ N₂ N₃ N₄ : ℕ,
    N₁ % 1000 / 100 = 4 ∧ N₁ % 10 = 1 ∧ ∃ n : ℕ, N₁ = n ^ 2 ∧
    N₃ % 1000 / 100 = 6 ∧ ∃ m : ℕ, N₃ = m ^ 4 ∧
    ∃ p : ℕ, N₂ = 2 * p ^ 5 ∧ 100 ≤ N₂ ∧ N₂ < 1000 ∧
    N₄ % 10 = 5 ∧ ∃ q : ℕ, N₄ = q ^ 3 ∧ 100 ≤ N₄ ∧ N₄ < 1000 ∧
    (N₁ % 10 = 4) :=
by
  sorry

end cross_number_puzzle_digit_star_l30_30729


namespace regular_polygon_sides_l30_30329

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (k : ℕ), (k : ℕ) * 18 = 360) : n = 20 :=
by
  -- Proof body here
  sorry

end regular_polygon_sides_l30_30329


namespace units_digit_3_pow_2004_l30_30722

-- Definition of the observed pattern of the units digits of powers of 3.
def pattern_units_digits : List ℕ := [3, 9, 7, 1]

-- Theorem stating that the units digit of 3^2004 is 1.
theorem units_digit_3_pow_2004 : (3 ^ 2004) % 10 = 1 :=
by
  sorry

end units_digit_3_pow_2004_l30_30722


namespace mass_percentage_H3BO3_l30_30769

theorem mass_percentage_H3BO3 :
  ∃ (element : String) (mass_percent : ℝ), 
    element ∈ ["H", "B", "O"] ∧ 
    mass_percent = 4.84 ∧ 
    mass_percent = 4.84 :=
sorry

end mass_percentage_H3BO3_l30_30769


namespace problem_l30_30623

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l30_30623


namespace roots_cubic_identity_l30_30950

theorem roots_cubic_identity (r s : ℚ) (h1 : 3 * r^2 + 5 * r + 2 = 0) (h2 : 3 * s^2 + 5 * s + 2 = 0) :
  (1 / r^3) + (1 / s^3) = -27 / 35 :=
sorry

end roots_cubic_identity_l30_30950


namespace find_minimal_sum_n_l30_30369

noncomputable def minimal_sum_n {a : ℕ → ℤ} {S : ℕ → ℤ} (h1 : ∀ n, a (n + 1) = a n + d) 
    (h2 : a 1 = -9) (h3 : S 3 = S 7) : ℕ := 
     5

theorem find_minimal_sum_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n, a (n + 1) = a n + d) 
    (h2 : a 1 = -9) (h3 : S 3 = S 7) : minimal_sum_n h1 h2 h3 = 5 :=
    sorry

end find_minimal_sum_n_l30_30369


namespace range_of_a_l30_30805

theorem range_of_a (a : ℝ) : (∃ x y : ℝ, x - y + 1 = 0 ∧ (x - a)^2 + y^2 = 2) ↔ -3 ≤ a ∧ a ≤ 1 := by
  sorry

end range_of_a_l30_30805


namespace obtain_2001_from_22_l30_30165

theorem obtain_2001_from_22 :
  ∃ (f : ℕ → ℕ), (∀ n, f (n + 1) = n ∨ f (n) = n + 1) ∧ (f 22 = 2001) := 
sorry

end obtain_2001_from_22_l30_30165


namespace candies_shared_l30_30980

theorem candies_shared (y b d x : ℕ) (h1 : x = 2 * y + 10) (h2 : x = 3 * b + 18) (h3 : x = 5 * d - 55) (h4 : x + y + b + d = 2013) : x = 990 :=
by
  sorry

end candies_shared_l30_30980


namespace binom_1500_1_l30_30343

-- Define binomial coefficient
def binom (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

-- Theorem statement
theorem binom_1500_1 : binom 1500 1 = 1500 :=
by
  sorry

end binom_1500_1_l30_30343


namespace consecutive_integers_sum_l30_30698

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l30_30698


namespace fraction_half_l30_30733

theorem fraction_half {A : ℕ} (h : 8 * (A + 8) - 8 * (A - 8) = 128) (age_eq : A = 64) :
  (64 : ℚ) / (128 : ℚ) = 1 / 2 :=
by
  sorry

end fraction_half_l30_30733


namespace probability_in_dark_l30_30036

theorem probability_in_dark (rev_per_min : ℕ) (given_prob : ℝ) (h1 : rev_per_min = 3) (h2 : given_prob = 0.25) :
  given_prob = 0.25 :=
by
  sorry

end probability_in_dark_l30_30036


namespace sum_of_consecutive_integers_with_product_812_l30_30653

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l30_30653


namespace total_hours_watched_l30_30969

theorem total_hours_watched (Monday Tuesday Wednesday Thursday Friday : ℕ) (hMonday : Monday = 12) (hTuesday : Tuesday = 4) (hWednesday : Wednesday = 6) (hThursday : Thursday = (Monday + Tuesday + Wednesday) / 2) (hFriday : Friday = 19) :
  Monday + Tuesday + Wednesday + Thursday + Friday = 52 := by
  sorry

end total_hours_watched_l30_30969


namespace range_of_m_l30_30557

noncomputable def system_of_equations (x y m : ℝ) : Prop :=
  (x + 2 * y = 1 - m) ∧ (2 * x + y = 3)

variable (x y m : ℝ)

theorem range_of_m (h : system_of_equations x y m) (hxy : x + y > 0) : m < 4 :=
by
  sorry

end range_of_m_l30_30557


namespace find_MT_square_l30_30551

-- Definitions and conditions
variables (P Q R S L O M N T U : Type*)
variables (x : ℝ)
variables (PL PQ PS QR RS LO : finset ℝ)
variable (side_length_PQRS : ℝ) (area_PLQ area_QMTL area_SNUL area_RNMUT : ℝ)
variables (LO_MT_perpendicular LO_NU_perpendicular : Prop)

-- Stating the problem
theorem find_MT_square :
  (side_length_PQRS = 3) →
  (PL ⊆ PQ) →
  (PO ⊆ PS) →
  (PL = PO) →
  (PL = x) →
  (U ∈ LO) →
  (T ∈ LO) →
  (LO_MT_perpendicular) →
  (LO_NU_perpendicular) →
  (area_PLQ = 1) →
  (area_QMTL = 1) →
  (area_SNUL = 2) →
  (area_RNMUT = 2) →
  (x^2 / 2 = 1) → 
  (PL * LO = 1) →
  MT^2 = 1 / 2 :=
sorry

end find_MT_square_l30_30551


namespace first_place_friend_distance_friend_running_distance_l30_30249

theorem first_place_friend_distance (distance_mina_finish : ℕ) (halfway_condition : ∀ x, x = distance_mina_finish / 2) :
  (∃ y, y = distance_mina_finish / 2) :=
by
  sorry

-- Given conditions
def distance_mina_finish : ℕ := 200
noncomputable def first_place_friend_position := distance_mina_finish / 2

-- The theorem we need to prove
theorem friend_running_distance : first_place_friend_position = 100 :=
by
  sorry

end first_place_friend_distance_friend_running_distance_l30_30249


namespace hoseok_add_8_l30_30952

theorem hoseok_add_8 (x : ℕ) (h : 6 * x = 72) : x + 8 = 20 :=
sorry

end hoseok_add_8_l30_30952


namespace cost_of_blue_pill_l30_30749

/-
Statement:
Bob takes two blue pills and one orange pill each day for three weeks.
The cost of a blue pill is $2 more than an orange pill.
The total cost for all pills over the three weeks amounts to $966.
Prove that the cost of one blue pill is $16.
-/

theorem cost_of_blue_pill (days : ℕ) (total_cost : ℝ) (cost_orange : ℝ) (cost_blue : ℝ) 
  (h1 : days = 21) 
  (h2 : total_cost = 966) 
  (h3 : cost_blue = cost_orange + 2) 
  (daily_pill_cost : ℝ)
  (h4 : daily_pill_cost = total_cost / days)
  (h5 : daily_pill_cost = 2 * cost_blue + cost_orange) :
  cost_blue = 16 :=
by
  sorry

end cost_of_blue_pill_l30_30749


namespace number_of_planks_needed_l30_30747

-- Definitions based on conditions
def bed_height : ℕ := 2
def bed_width : ℕ := 2
def bed_length : ℕ := 8
def plank_width : ℕ := 1
def lumber_length : ℕ := 8
def num_beds : ℕ := 10

-- The theorem statement
theorem number_of_planks_needed : (2 * (bed_length / lumber_length) * bed_height) + (2 * ((bed_width * bed_height) / lumber_length) * lumber_length / 4) * num_beds = 60 :=
  by sorry

end number_of_planks_needed_l30_30747


namespace regular_polygon_has_20_sides_l30_30318

-- Definition of a regular polygon with an exterior angle of 18 degrees
def regular_polygon_sides (external_angle : ℝ) : ℕ :=
  if external_angle > 0 then (360 / external_angle).toInt else 0

theorem regular_polygon_has_20_sides :
  regular_polygon_sides 18 = 20 :=
by
  sorry

end regular_polygon_has_20_sides_l30_30318


namespace purple_ring_weight_l30_30816

def orange_ring_weight : ℝ := 0.08
def white_ring_weight : ℝ := 0.42
def total_weight : ℝ := 0.83

theorem purple_ring_weight : 
  ∃ (purple_ring_weight : ℝ), purple_ring_weight = total_weight - (orange_ring_weight + white_ring_weight) := 
  by
  use 0.33
  sorry

end purple_ring_weight_l30_30816


namespace sum_of_consecutive_integers_l30_30639

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l30_30639


namespace angle_RPS_is_1_degree_l30_30105

-- Definitions of the given angles
def angle_QRS : ℝ := 150
def angle_PQS : ℝ := 60
def angle_PSQ : ℝ := 49
def angle_QPR : ℝ := 70

-- Definition for the calculated angle QPS
def angle_QPS : ℝ := 180 - angle_PQS - angle_PSQ

-- Definition for the target angle RPS
def angle_RPS : ℝ := angle_QPS - angle_QPR

-- The theorem we aim to prove
theorem angle_RPS_is_1_degree : angle_RPS = 1 := by
  sorry

end angle_RPS_is_1_degree_l30_30105


namespace boys_meeting_problem_l30_30713

theorem boys_meeting_problem (d : ℝ) (t : ℝ)
  (speed1 speed2 : ℝ)
  (h1 : speed1 = 6) 
  (h2 : speed2 = 8) 
  (h3 : t > 0)
  (h4 : ∀ n : ℤ, n * (speed1 + speed2) * t ≠ d) : 
  0 = 0 :=
by 
  sorry

end boys_meeting_problem_l30_30713


namespace sum_of_consecutive_integers_with_product_812_l30_30680

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l30_30680


namespace TotalToysIsNinetyNine_l30_30792

def BillHasToys : ℕ := 60
def HalfOfBillToys : ℕ := BillHasToys / 2
def AdditionalToys : ℕ := 9
def HashHasToys : ℕ := HalfOfBillToys + AdditionalToys
def TotalToys : ℕ := BillHasToys + HashHasToys

theorem TotalToysIsNinetyNine : TotalToys = 99 := by
  sorry

end TotalToysIsNinetyNine_l30_30792


namespace annual_return_l30_30339

theorem annual_return (initial_price profit : ℝ) (h₁ : initial_price = 5000) (h₂ : profit = 400) : 
  ((profit / initial_price) * 100 = 8) := by
  -- Lean's substitute for proof
  sorry

end annual_return_l30_30339


namespace sum_of_consecutive_integers_with_product_812_l30_30684

theorem sum_of_consecutive_integers_with_product_812 :
  ∃ x : ℕ, (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) := 
begin
  sorry
end

end sum_of_consecutive_integers_with_product_812_l30_30684


namespace arithmetic_sequence_sum_l30_30955

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_roots : (a 3) * (a 10) - 3 * (a 3 + a 10) - 5 = 0) : a 5 + a 8 = 3 :=
sorry

end arithmetic_sequence_sum_l30_30955


namespace simplify_fraction_complex_l30_30989

open Complex

theorem simplify_fraction_complex :
  (3 - I) / (2 + 5 * I) = (1 / 29) - (17 / 29) * I := by
  sorry

end simplify_fraction_complex_l30_30989


namespace perimeter_of_plot_l30_30003

variable (length breadth : ℝ)
variable (h_ratio : length / breadth = 7 / 5)
variable (h_area : length * breadth = 5040)

theorem perimeter_of_plot (h_ratio : length / breadth = 7 / 5) (h_area : length * breadth = 5040) : 
  (2 * length + 2 * breadth = 288) :=
sorry

end perimeter_of_plot_l30_30003


namespace geometric_sequence_product_l30_30411

variable {a1 a2 a3 a4 a5 a6 : ℝ}
variable (r : ℝ)
variable (seq : ℕ → ℝ)

-- Conditions defining the terms of a geometric sequence
def is_geometric_sequence (seq : ℕ → ℝ) (a1 r : ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) = seq n * r

-- Given condition: a_3 * a_4 = 5
def given_condition (seq : ℕ → ℝ) := (seq 2 * seq 3 = 5)

-- Proving the required question: a_1 * a_2 * a_5 * a_6 = 5
theorem geometric_sequence_product
  (h_geom : is_geometric_sequence seq a1 r)
  (h_given : given_condition seq) :
  seq 0 * seq 1 * seq 4 * seq 5 = 5 :=
sorry

end geometric_sequence_product_l30_30411


namespace consecutive_integers_sum_l30_30701

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l30_30701


namespace consecutive_integers_sum_l30_30702

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l30_30702


namespace swimming_pool_distance_l30_30815

theorem swimming_pool_distance (julien_daily_distance : ℕ) (sarah_multi_factor : ℕ)
    (jamir_additional_distance : ℕ) (week_days : ℕ) 
    (julien_weekly_distance : ℕ) (sarah_weekly_distance : ℕ) (jamir_weekly_distance : ℕ) 
    (total_combined_distance : ℕ) : 
    julien_daily_distance = 50 → 
    sarah_multi_factor = 2 →
    jamir_additional_distance = 20 →
    week_days = 7 →
    julien_weekly_distance = julien_daily_distance * week_days →
    sarah_weekly_distance = (sarah_multi_factor * julien_daily_distance) * week_days →
    jamir_weekly_distance = ((sarah_multi_factor * julien_daily_distance) + jamir_additional_distance) * week_days →
    total_combined_distance = julien_weekly_distance + sarah_weekly_distance + jamir_weekly_distance →
    total_combined_distance = 1890 := 
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [h1, h2, h3, h4] at *
  rw [h5, h6, h7, h8]
  sorry

end swimming_pool_distance_l30_30815


namespace max_b_lattice_free_line_l30_30897

theorem max_b_lattice_free_line : 
  ∃ b : ℚ, (∀ (m : ℚ), (1 / 3) < m ∧ m < b → 
  ∀ x : ℤ, 0 < x ∧ x ≤ 150 → ¬ (∃ y : ℤ, y = m * x + 4)) ∧ 
  b = 50 / 147 :=
sorry

end max_b_lattice_free_line_l30_30897


namespace option_costs_more_cost_effective_x30_more_cost_effective_plan_x30_l30_30895

def racket_price : ℕ := 80
def ball_price : ℕ := 20
def discount_rate : ℕ := 90

def option_1_cost (n_rackets : ℕ) : ℕ :=
  n_rackets * racket_price

def option_2_cost (n_rackets : ℕ) (n_balls : ℕ) : ℕ :=
  (discount_rate * (n_rackets * racket_price + n_balls * ball_price)) / 100

-- Part 1: Express in Algebraic Terms
theorem option_costs (n_rackets : ℕ) (n_balls : ℕ) :
  option_1_cost n_rackets = 1600 ∧ option_2_cost n_rackets n_balls = 1440 + 18 * n_balls := 
by
  sorry

-- Part 2: For x = 30, determine more cost-effective option
theorem more_cost_effective_x30 (x : ℕ) (h : x = 30) :
  option_1_cost 20 < option_2_cost 20 x := 
by
  sorry

-- Part 3: More cost-effective Plan for x = 30
theorem more_cost_effective_plan_x30 :
  1600 + (discount_rate * (10 * ball_price)) / 100 < option_2_cost 20 30 :=
by
  sorry

end option_costs_more_cost_effective_x30_more_cost_effective_plan_x30_l30_30895


namespace difference_in_elevation_difference_in_running_time_l30_30948

structure Day :=
  (distance_km : ℝ) -- kilometers
  (pace_min_per_km : ℝ) -- minutes per kilometer
  (elevation_gain_m : ℝ) -- meters

def monday : Day := { distance_km := 9, pace_min_per_km := 6, elevation_gain_m := 300 }
def wednesday : Day := { distance_km := 4.816, pace_min_per_km := 5.5, elevation_gain_m := 150 }
def friday : Day := { distance_km := 2.095, pace_min_per_km := 7, elevation_gain_m := 50 }

noncomputable def calculate_running_time(day : Day) : ℝ :=
  day.distance_km * day.pace_min_per_km

noncomputable def total_elevation_gain(wednesday friday : Day) : ℝ :=
  wednesday.elevation_gain_m + friday.elevation_gain_m

noncomputable def total_running_time(wednesday friday : Day) : ℝ :=
  calculate_running_time wednesday + calculate_running_time friday

theorem difference_in_elevation :
  monday.elevation_gain_m - total_elevation_gain wednesday friday = 100 := by 
  sorry

theorem difference_in_running_time :
  calculate_running_time monday - total_running_time wednesday friday = 12.847 := by 
  sorry

end difference_in_elevation_difference_in_running_time_l30_30948


namespace nth_equation_l30_30255

theorem nth_equation (n : ℕ) (h : 0 < n) : (- (n : ℤ)) * (n : ℝ) / (n + 1) = - (n : ℤ) + (n : ℝ) / (n + 1) :=
sorry

end nth_equation_l30_30255


namespace least_number_of_teams_l30_30484

/-- A coach has 30 players in a team. If he wants to form teams of at most 7 players each for a tournament, we aim to prove that the least number of teams that he needs is 5. -/
theorem least_number_of_teams (players teams : ℕ) 
  (h_players : players = 30) 
  (h_teams : ∀ t, t ≤ 7 → t ∣ players) : teams = 5 := by
  sorry

end least_number_of_teams_l30_30484


namespace units_digit_p2_plus_3p_l30_30243

-- Define p
def p : ℕ := 2017^3 + 3^2017

-- Define the theorem to be proved
theorem units_digit_p2_plus_3p : (p^2 + 3^p) % 10 = 5 :=
by
  sorry -- Proof goes here

end units_digit_p2_plus_3p_l30_30243


namespace arithmetic_example_l30_30185

theorem arithmetic_example : 15 * 30 + 45 * 15 = 1125 := by
  sorry

end arithmetic_example_l30_30185


namespace find_a_plus_b_l30_30571

theorem find_a_plus_b (a b : ℕ) (positive_a : 0 < a) (positive_b : 0 < b)
  (condition : ∀ (n : ℕ), (n > 0) → (∃ m n : ℕ, n = m * a + n * b) ∨ (∃ k l : ℕ, n = 2009 + k * a + l * b))
  (not_expressible : ∃ m n : ℕ, 1776 = m * a + n * b): a + b = 133 :=
sorry

end find_a_plus_b_l30_30571


namespace simplify_expression_1_simplify_expression_2_l30_30905

section Problem1
variables (a b c : ℝ) (h1 : c ≠ 0) (h2 : a ≠ 0) (h3 : b ≠ 0)

theorem simplify_expression_1 :
  ((a^2 * b / (-c))^3 * (c^2 / (- (a * b)))^2 / (b * c / a)^4)
  = - (a^10 / (b^3 * c^7)) :=
by sorry
end Problem1

section Problem2
variables (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ a) (h3 : b ≠ 0)

theorem simplify_expression_2 :
  ((2 / (a^2 - b^2) - 1 / (a^2 - a * b)) / (a / (a + b))) = 1 / a^2 :=
by sorry
end Problem2

end simplify_expression_1_simplify_expression_2_l30_30905


namespace tenured_professors_percentage_l30_30043

noncomputable def percentage_tenured (W M T TM : ℝ) := W = 0.69 ∧ (1 - W) = M ∧ (M * 0.52) = TM ∧ (W + T - TM) = 0.90 → T = 0.7512

-- Define the mathematical entities
variables (W M T TM : ℝ)

-- The main statement
theorem tenured_professors_percentage : percentage_tenured W M T TM := by
  sorry

end tenured_professors_percentage_l30_30043


namespace total_toys_l30_30790

variable (B H : ℕ)

theorem total_toys (h1 : B = 60) (h2 : H = 9 + (B / 2)) : B + H = 99 := by
  sorry

end total_toys_l30_30790


namespace sum_of_consecutive_integers_l30_30636

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l30_30636


namespace quadratic_solution_transform_l30_30993

theorem quadratic_solution_transform (a b c : ℝ) (hA : 0 = a * (-3)^2 + b * (-3) + c) (hB : 0 = a * 4^2 + b * 4 + c) :
  (∃ x1 x2 : ℝ, a * (x1 - 1)^2 + b * (x1 - 1) + c = 0 ∧ a * (x2 - 1)^2 + b * (x2 - 1) + c = 0 ∧ x1 = -2 ∧ x2 = 5) :=
  sorry

end quadratic_solution_transform_l30_30993


namespace f_at_8_5_l30_30595

def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom odd_function_shifted : ∀ x : ℝ, f (x - 1) = -f (1 - x)
axiom f_half : f 0.5 = 9

theorem f_at_8_5 : f 8.5 = 9 := by
  sorry

end f_at_8_5_l30_30595


namespace sum_modulo_remainder_l30_30927

theorem sum_modulo_remainder :
  ((82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17) = 12 :=
by
  sorry

end sum_modulo_remainder_l30_30927


namespace solve_for_x_l30_30125

theorem solve_for_x : ∃ x : ℝ, 5 * x + 9 * x = 570 - 12 * (x - 5) ∧ x = 315 / 13 :=
by
  sorry

end solve_for_x_l30_30125


namespace perfect_squares_with_specific_ones_digit_count_l30_30220

theorem perfect_squares_with_specific_ones_digit_count : 
  ∃ n : ℕ, (∀ k : ℕ, k < 2500 → (k % 10 = 4 ∨ k % 10 = 5 ∨ k % 10 = 6) ↔ ∃ m : ℕ, m < n ∧ (m % 10 = 2 ∨ m % 10 = 8 ∨ m % 10 = 5 ∨ m % 10 = 4 ∨ m % 10 = 6) ∧ k = m * m) 
  ∧ n = 25 := 
by 
  sorry

end perfect_squares_with_specific_ones_digit_count_l30_30220


namespace scientific_notation_equivalence_l30_30193

/-- The scientific notation for 20.26 thousand hectares in square meters is equal to 2.026 × 10^9. -/
theorem scientific_notation_equivalence :
  (20.26 * 10^3 * 10^4) = 2.026 * 10^9 := 
sorry

end scientific_notation_equivalence_l30_30193


namespace ratio_markus_age_son_age_l30_30437

variable (M S G : ℕ)

theorem ratio_markus_age_son_age (h1 : G = 20) (h2 : S = 2 * G) (h3 : M + S + G = 140) : M / S = 2 := by
  sorry

end ratio_markus_age_son_age_l30_30437


namespace gasoline_price_percentage_increase_l30_30449

theorem gasoline_price_percentage_increase 
  (price_month1_euros : ℝ) (price_month3_dollars : ℝ) (exchange_rate : ℝ) 
  (price_month1 : ℝ) (percent_increase : ℝ):
  price_month1_euros = 20 →
  price_month3_dollars = 15 →
  exchange_rate = 1.2 →
  price_month1 = price_month1_euros * exchange_rate →
  percent_increase = ((price_month1 - price_month3_dollars) / price_month3_dollars) * 100 →
  percent_increase = 60 :=
by intros; sorry

end gasoline_price_percentage_increase_l30_30449


namespace range_of_y_l30_30517

theorem range_of_y (a b y : ℝ) (hab : a + b = 2) (hbl : b ≤ 2) (hy : y = a^2 + 2*a - 2) : y ≥ -2 :=
by
  sorry

end range_of_y_l30_30517


namespace compressor_station_distances_compressor_station_distances_when_a_is_30_l30_30009

theorem compressor_station_distances (a : ℝ) (h : 0 < a ∧ a < 60) :
  ∃ x y z : ℝ, x + y = 3 * z ∧ z + y = x + a ∧ x + z = 60 :=
sorry

theorem compressor_station_distances_when_a_is_30 :
  ∃ x y z : ℝ, 
  (x + y = 3 * z) ∧ (z + y = x + 30) ∧ (x + z = 60) ∧ 
  (x = 35) ∧ (y = 40) ∧ (z = 25) :=
sorry

end compressor_station_distances_compressor_station_distances_when_a_is_30_l30_30009


namespace train_speed_l30_30715

theorem train_speed (v : ℕ) :
    let distance_between_stations := 155
    let speed_of_train_from_A := 20
    let start_time_train_A := 7
    let start_time_train_B := 8
    let meet_time := 11
    let distance_traveled_by_A := speed_of_train_from_A * (meet_time - start_time_train_A)
    let remaining_distance := distance_between_stations - distance_traveled_by_A
    let traveling_time_train_B := meet_time - start_time_train_B
    v * traveling_time_train_B = remaining_distance → v = 25 :=
by
  intros
  sorry

end train_speed_l30_30715


namespace original_price_sarees_l30_30705

theorem original_price_sarees (P : ℝ) (h : 0.80 * P * 0.85 = 231.2) : P = 340 := 
by sorry

end original_price_sarees_l30_30705


namespace points_in_quadrant_I_l30_30754

theorem points_in_quadrant_I (x y : ℝ) :
  (y > 3 * x) ∧ (y > 5 - 2 * x) → (x > 0) ∧ (y > 0) := by
  sorry

end points_in_quadrant_I_l30_30754


namespace christen_potatoes_l30_30089

theorem christen_potatoes :
  let total_potatoes := 60
  let homer_rate := 4
  let christen_rate := 6
  let alex_potatoes := 2
  let homer_minutes := 6
  homer_minutes * homer_rate + christen_rate * ((total_potatoes + alex_potatoes - homer_minutes * homer_rate) / (homer_rate + christen_rate)) = 24 := 
sorry

end christen_potatoes_l30_30089


namespace milk_fraction_correct_l30_30586

def fraction_of_milk_in_coffee_cup (coffee_initial : ℕ) (milk_initial : ℕ) : ℚ :=
  let coffee_transferred := coffee_initial / 3
  let milk_cup_after_transfer := milk_initial + coffee_transferred
  let coffee_left := coffee_initial - coffee_transferred
  let total_mixed := milk_cup_after_transfer
  let transfer_back := total_mixed / 2
  let coffee_back := transfer_back * (coffee_transferred / total_mixed)
  let milk_back := transfer_back * (milk_initial / total_mixed)
  let coffee_final := coffee_left + coffee_back
  let milk_final := milk_back
  milk_final / (coffee_final + milk_final)

theorem milk_fraction_correct (coffee_initial : ℕ) (milk_initial : ℕ)
  (h_coffee : coffee_initial = 6) (h_milk : milk_initial = 3) :
  fraction_of_milk_in_coffee_cup coffee_initial milk_initial = 3 / 13 :=
by
  sorry

end milk_fraction_correct_l30_30586


namespace probability_of_red_and_blue_or_blue_and_green_is_four_ninths_l30_30291

-- Define the conditions of the problem
def total_chips := 12
def red_chips := 6
def blue_chips := 4
def green_chips := 2

-- Define the probability of drawing each specific pair of chips
def P_red_blue : ℚ := (red_chips / total_chips) * (blue_chips / total_chips)
def P_blue_red : ℚ := (blue_chips / total_chips) * (red_chips / total_chips)
def P_blue_green : ℚ := (blue_chips / total_chips) * (green_chips / total_chips)
def P_green_blue : ℚ := (green_chips / total_chips) * (blue_chips / total_chips)

-- Add these probabilities together
def P_total : ℚ := P_red_blue + P_blue_red + P_blue_green + P_green_blue

-- Statement of the problem to prove
theorem probability_of_red_and_blue_or_blue_and_green_is_four_ninths : P_total = 4 / 9 :=
by sorry

end probability_of_red_and_blue_or_blue_and_green_is_four_ninths_l30_30291


namespace probability_of_two_queens_or_at_least_one_king_l30_30390

def probability_two_queens_or_at_least_one_king : ℚ := 2 / 13

theorem probability_of_two_queens_or_at_least_one_king :
  let probability_two_queens := (4/52) * (3/51)
  let probability_exactly_one_king := (2 * (4/52) * (48/51))
  let probability_two_kings := (4/52) * (3/51)
  let probability_at_least_one_king := probability_exactly_one_king + probability_two_kings
  let total_probability := probability_two_queens + probability_at_least_one_king
  total_probability = probability_two_queens_or_at_least_one_king := 
by
  sorry

end probability_of_two_queens_or_at_least_one_king_l30_30390


namespace solution_set_for_rational_inequality_l30_30359

theorem solution_set_for_rational_inequality (x : ℝ) :
  (x - 2) / (x - 1) > 0 ↔ x < 1 ∨ x > 2 := 
sorry

end solution_set_for_rational_inequality_l30_30359


namespace carrots_weight_l30_30486

-- Let the weight of the carrots be denoted by C (in kg).
variables (C : ℕ)

-- Conditions:
-- The merchant installed 13 kg of zucchini and 8 kg of broccoli.
-- He sold only half of the total, which amounted to 18 kg, so the total weight was 36 kg.
def conditions := (C + 13 + 8 = 36)

-- Prove that the weight of the carrots installed is 15 kg.
theorem carrots_weight (H : C + 13 + 8 = 36) : C = 15 :=
by {
  sorry -- proof to be filled in
}

end carrots_weight_l30_30486


namespace grid_sum_21_proof_l30_30998

-- Define the condition that the sum of the horizontal and vertical lines are 21
def valid_grid (nums : List ℕ) (x : ℕ) : Prop :=
  nums ≠ [] ∧ (((nums.sum + x) = 42) ∧ (21 + 21 = 42))

-- Define the main theorem to prove x = 7
theorem grid_sum_21_proof (nums : List ℕ) (h : valid_grid nums 7) : 7 ∈ nums :=
  sorry

end grid_sum_21_proof_l30_30998


namespace convex_polygon_longest_sides_convex_polygon_shortest_sides_l30_30558

noncomputable def convex_polygon : Type := sorry

-- Definitions for the properties and functions used in conditions
def is_convex (P : convex_polygon) : Prop := sorry
def equal_perimeters (A B : convex_polygon) : Prop := sorry
def longest_side (P : convex_polygon) : ℝ := sorry
def shortest_side (P : convex_polygon) : ℝ := sorry

-- Problem part a
theorem convex_polygon_longest_sides (P : convex_polygon) (h_convex : is_convex P) :
  ∃ (A B : convex_polygon), equal_perimeters A B ∧ longest_side A = longest_side B :=
sorry

-- Problem part b
theorem convex_polygon_shortest_sides (P : convex_polygon) (h_convex : is_convex P) :
  ¬(∀ (A B : convex_polygon), equal_perimeters A B → shortest_side A = shortest_side B) :=
sorry

end convex_polygon_longest_sides_convex_polygon_shortest_sides_l30_30558


namespace no_intersect_M1_M2_l30_30570

theorem no_intersect_M1_M2 (A B : ℤ) : ∃ C : ℤ, 
  ∀ x y : ℤ, (x^2 + A * x + B) ≠ (2 * y^2 + 2 * y + C) := by
  sorry

end no_intersect_M1_M2_l30_30570


namespace new_students_weights_correct_l30_30150

-- Definitions of the initial conditions
def initial_student_count : ℕ := 29
def initial_avg_weight : ℚ := 28
def total_initial_weight := initial_student_count * initial_avg_weight
def new_student_counts : List ℕ := [30, 31, 32, 33]
def new_avg_weights : List ℚ := [27.2, 27.8, 27.6, 28]

-- Weights of the four new students
def W1 : ℚ := 4
def W2 : ℚ := 45.8
def W3 : ℚ := 21.4
def W4 : ℚ := 40.8

-- The proof statement
theorem new_students_weights_correct :
  total_initial_weight = 812 ∧
  W1 = 4 ∧
  W2 = 45.8 ∧
  W3 = 21.4 ∧
  W4 = 40.8 ∧
  (total_initial_weight + W1) = 816 ∧
  (total_initial_weight + W1) / new_student_counts.head! = new_avg_weights.head! ∧
  (total_initial_weight + W1 + W2) = 861.8 ∧
  (total_initial_weight + W1 + W2) / new_student_counts.tail.head! = new_avg_weights.tail.head! ∧
  (total_initial_weight + W1 + W2 + W3) = 883.2 ∧
  (total_initial_weight + W1 + W2 + W3) / new_student_counts.tail.tail.head! = new_avg_weights.tail.tail.head! ∧
  (total_initial_weight + W1 + W2 + W3 + W4) = 924 ∧
  (total_initial_weight + W1 + W2 + W3 + W4) / new_student_counts.tail.tail.tail.head! = new_avg_weights.tail.tail.tail.head! :=
by
  sorry

end new_students_weights_correct_l30_30150


namespace mass_of_man_is_120_l30_30479

def length_of_boat : ℝ := 3
def breadth_of_boat : ℝ := 2
def height_water_rise : ℝ := 0.02
def density_of_water : ℝ := 1000
def volume_displaced : ℝ := length_of_boat * breadth_of_boat * height_water_rise
def mass_of_man := density_of_water * volume_displaced

theorem mass_of_man_is_120 : mass_of_man = 120 :=
by
  -- insert the detailed proof here
  sorry

end mass_of_man_is_120_l30_30479


namespace coupon_savings_inequalities_l30_30167

variable {P : ℝ} (p : ℝ) (hP : P = 150 + p) (hp_pos : p > 0)
variable (ha : 0.15 * P > 30) (hb : 0.15 * P > 0.20 * p)
variable (cA_saving : ℝ := 0.15 * P)
variable (cB_saving : ℝ := 30)
variable (cC_saving : ℝ := 0.20 * p)

theorem coupon_savings_inequalities (h1 : 0.15 * P - 30 > 0) (h2 : 0.15 * P - 0.20 * (P - 150) > 0) :
  let x := 200
  let y := 600
  y - x = 400 :=
by
  sorry

end coupon_savings_inequalities_l30_30167


namespace bridgette_total_baths_l30_30340

def bridgette_baths (dogs baths_per_dog_per_month cats baths_per_cat_per_month birds baths_per_bird_per_month : ℕ) : ℕ :=
  (dogs * baths_per_dog_per_month * 12) + (cats * baths_per_cat_per_month * 12) + (birds * (12 / baths_per_bird_per_month))

theorem bridgette_total_baths :
  bridgette_baths 2 2 3 1 4 4 = 96 :=
by
  -- Proof omitted
  sorry

end bridgette_total_baths_l30_30340


namespace base_of_log_is_176_l30_30783

theorem base_of_log_is_176 
    (x : ℕ)
    (h : ∃ q r : ℕ, x = 19 * q + r ∧ q = 9 ∧ r = 5) :
    x = 176 :=
by
  sorry

end base_of_log_is_176_l30_30783


namespace sum_a6_to_a9_l30_30937

-- Given definitions and conditions
def sequence_sum (n : ℕ) : ℕ := n^3
def a (n : ℕ) : ℕ := sequence_sum (n + 1) - sequence_sum n

-- Theorem to be proved
theorem sum_a6_to_a9 : a 6 + a 7 + a 8 + a 9 = 604 :=
by sorry

end sum_a6_to_a9_l30_30937


namespace father_son_skating_ratio_l30_30850

theorem father_son_skating_ratio (v_f v_s : ℝ) (h1 : v_f > v_s) (h2 : (v_f + v_s) / (v_f - v_s) = 5) :
  v_f / v_s = 1.5 :=
sorry

end father_son_skating_ratio_l30_30850


namespace geese_count_l30_30439

variables (k n : ℕ)

theorem geese_count (h1 : k * n = (k + 20) * (n - 75)) (h2 : k * n = (k - 15) * (n + 100)) : n = 300 :=
by
  sorry

end geese_count_l30_30439


namespace round_to_nearest_hundredth_l30_30442

noncomputable def recurring_decimal (n : ℕ) : ℝ :=
  if n = 87 then 87 + 36 / 99 else 0 -- Defines 87.3636... for n = 87

theorem round_to_nearest_hundredth : recurring_decimal 87 = 87.36 :=
by sorry

end round_to_nearest_hundredth_l30_30442


namespace problem_l30_30632

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l30_30632


namespace hunting_season_fraction_l30_30236

noncomputable def fraction_of_year_hunting_season (hunting_times_per_month : ℕ) 
    (deers_per_hunt : ℕ) (weight_per_deer : ℕ) (fraction_kept : ℚ) 
    (total_weight_kept : ℕ) : ℚ :=
  let total_yearly_weight := total_weight_kept * 2
  let weight_per_hunt := deers_per_hunt * weight_per_deer
  let total_hunts_per_year := total_yearly_weight / weight_per_hunt
  let total_months_hunting := total_hunts_per_year / hunting_times_per_month
  let fraction_of_year := total_months_hunting / 12
  fraction_of_year

theorem hunting_season_fraction : 
  fraction_of_year_hunting_season 6 2 600 (1 / 2 : ℚ) 10800 = 1 / 4 := 
by
  simp [fraction_of_year_hunting_season]
  sorry

end hunting_season_fraction_l30_30236


namespace Tom_runs_60_miles_per_week_l30_30868

theorem Tom_runs_60_miles_per_week
  (days_per_week : ℕ := 5)
  (hours_per_day : ℝ := 1.5)
  (speed_mph : ℝ := 8) :
  (days_per_week * hours_per_day * speed_mph = 60) := by
  sorry

end Tom_runs_60_miles_per_week_l30_30868


namespace arthur_bought_hamburgers_on_first_day_l30_30338

-- Define the constants and parameters
def D : ℕ := 1
def H : ℕ := 2
def total_cost_day1 : ℕ := 10
def total_cost_day2 : ℕ := 7

-- Define the equation representing the transactions
def equation_day1 (h : ℕ) := H * h + 4 * D = total_cost_day1
def equation_day2 := 2 * H + 3 * D = total_cost_day2

-- The theorem we need to prove: the number of hamburgers h bought on the first day is 3
theorem arthur_bought_hamburgers_on_first_day (h : ℕ) (hd1 : equation_day1 h) (hd2 : equation_day2) : h = 3 := 
by 
  sorry

end arthur_bought_hamburgers_on_first_day_l30_30338


namespace sum_of_consecutive_integers_l30_30634

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l30_30634


namespace race_winner_l30_30102

-- Definitions and conditions based on the problem statement
def tortoise_speed : ℕ := 5  -- Tortoise speed in meters per minute
def hare_speed_1 : ℕ := 20  -- Hare initial speed in meters per minute
def hare_time_1 : ℕ := 3  -- Hare initial running time in minutes
def hare_speed_2 : ℕ := 10  -- Hare speed when going back in meters per minute
def hare_time_2 : ℕ := 2  -- Hare back running time in minutes
def hare_sleep_time : ℕ := 5  -- Hare sleeping time in minutes
def hare_speed_3 : ℕ := 25  -- Hare final speed in meters per minute
def track_length : ℕ := 130  -- Total length of the race track in meters

-- The problem statement
theorem race_winner :
  track_length / tortoise_speed > hare_time_1 + hare_time_2 + hare_sleep_time + (track_length - (hare_speed_1 * hare_time_1 - hare_speed_2 * hare_time_2)) / hare_speed_3 :=
sorry

end race_winner_l30_30102


namespace consecutive_integer_sum_l30_30607

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l30_30607


namespace proof_inequality_l30_30801

theorem proof_inequality (p q r : ℝ) (hr : r < 0) (hpq_ne_zero : p * q ≠ 0) (hpr_lt_qr : p * r < q * r) : 
  p < q :=
by 
  sorry

end proof_inequality_l30_30801


namespace election_votes_l30_30399

theorem election_votes (total_votes : ℕ) (h1 : (4 / 15) * total_votes = 48) : total_votes = 180 :=
sorry

end election_votes_l30_30399


namespace find_triangle_sides_l30_30000

-- Define the conditions and translate them into Lean 4
theorem find_triangle_sides :
  (∃ a b c: ℝ, a + b + c = 40 ∧ a^2 + b^2 = c^2 ∧ 
   (a + 4)^2 + (b + 1)^2 = (c + 3)^2 ∧ 
   a = 8 ∧ b = 15 ∧ c = 17) :=
by 
  sorry

end find_triangle_sides_l30_30000


namespace dragons_total_games_l30_30182

noncomputable def numberOfGames (y x : ℕ) (h1 : x = 6 * y / 10) (h2 : x + 9 = (62 * (y + 12)) / 100) : ℕ :=
y + 12

theorem dragons_total_games (y x : ℕ) (h1 : x = 6 * y / 10) (h2 : x + 9 = (62 * (y + 12)) / 100) :
  numberOfGames y x h1 h2 = 90 := 
sorry

end dragons_total_games_l30_30182


namespace car_second_hour_speed_l30_30004

theorem car_second_hour_speed (x : ℝ) 
  (first_hour_speed : ℝ := 20)
  (average_speed : ℝ := 40) 
  (total_time : ℝ := 2)
  (total_distance : ℝ := first_hour_speed + x) 
  : total_distance / total_time = average_speed → x = 60 :=
by
  intro h
  sorry

end car_second_hour_speed_l30_30004


namespace intersection_A_B_l30_30383

-- Define the sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | 3 - 2 * x > 0}

-- Prove the intersection of A and B
theorem intersection_A_B :
  (A ∩ B) = {x | x < (3 / 2)} := sorry

end intersection_A_B_l30_30383


namespace monotone_f_solve_inequality_range_of_a_l30_30372

noncomputable def e := Real.exp 1
noncomputable def f (x : ℝ) : ℝ := e^x + 1/(e^x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.log ((3 - a) * (f x - 1/e^x) + 1) - Real.log (3 * a) - 2 * x

-- Part 1: Monotonicity of f(x)
theorem monotone_f : ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 < x2 → f x1 < f x2 :=
by sorry

-- Part 2: Solving the inequality f(2x) ≥ f(x + 1)
theorem solve_inequality : ∀ x : ℝ, f (2 * x) ≥ f (x + 1) ↔ x ≥ 1 ∨ x ≤ -1 / 3 :=
by sorry

-- Part 3: Finding the range of a
theorem range_of_a : ∀ a : ℝ, (∀ x : ℝ, 0 ≤ x → g x a ≤ 0) ↔ 1 ≤ a ∧ a ≤ 3 :=
by sorry

end monotone_f_solve_inequality_range_of_a_l30_30372


namespace consecutive_integer_product_sum_l30_30651

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l30_30651


namespace inequality_proof_l30_30821

theorem inequality_proof (a b c d : ℝ) (hnonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) (hsum : a + b + c + d = 1) :
  abcd + bcda + cdab + dabc ≤ 1/27 + (176/27) * abcd :=
by
  sorry

end inequality_proof_l30_30821


namespace math_team_selection_l30_30829

theorem math_team_selection : 
  (nat.choose 6 3) * (nat.choose 8 3) = 1120 := 
by
  sorry

end math_team_selection_l30_30829


namespace sequence_property_l30_30823

theorem sequence_property {m : ℤ} (h_m : |m| ≥ 2) (a : ℕ → ℤ)
  (h_nonzero : ¬(a 1 = 0 ∧ a 2 = 0))
  (h_rec : ∀ n : ℕ, a (n+2) = a (n+1) - m * a n)
  (r s : ℕ) (h_r_s : r > s ∧ s ≥ 2) (h_eq : a r = a s ∧ a s = a 1) :
  r - s ≥ |m| := sorry

end sequence_property_l30_30823


namespace consecutive_integer_sum_l30_30611

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l30_30611


namespace saree_original_price_l30_30455

theorem saree_original_price :
  ∃ P : ℝ, (0.95 * 0.88 * P = 334.4) ∧ (P = 400) :=
by
  sorry

end saree_original_price_l30_30455


namespace slope_of_tangent_at_0_l30_30858

theorem slope_of_tangent_at_0 (f : ℝ → ℝ) (h : ∀ x, f x = Real.exp (2 * x)) : 
  (deriv f 0) = 2 :=
sorry

end slope_of_tangent_at_0_l30_30858


namespace dice_probability_l30_30883

noncomputable def binomial_coefficient : ℕ → ℕ → ℕ
| 0, k       := if k = 0 then 1 else 0
| (n+1), 0   := 1
| (n+1), (k+1) := binomial_coefficient n k + binomial_coefficient n (k+1)

def probability_two_ones_in_twelve_rolls: ℝ :=
(binomial_coefficient 12 2) * (1/6)^2 * (5/6)^10

theorem dice_probability :
  real.round (1000 * probability_two_ones_in_twelve_rolls) / 1000 = 0.293 :=
by sorry

end dice_probability_l30_30883


namespace marcella_shoes_lost_l30_30119

theorem marcella_shoes_lost (pairs_initial : ℕ) (pairs_left_max : ℕ) (individuals_initial : ℕ) (individuals_left_max : ℕ) (pairs_initial_eq : pairs_initial = 25) (pairs_left_max_eq : pairs_left_max = 20) (individuals_initial_eq : individuals_initial = pairs_initial * 2) (individuals_left_max_eq : individuals_left_max = pairs_left_max * 2) : (individuals_initial - individuals_left_max) = 10 := 
by
  sorry

end marcella_shoes_lost_l30_30119


namespace range_of_x_squared_f_x_lt_x_squared_minus_f_1_l30_30052

noncomputable def even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

noncomputable def satisfies_inequality (f f' : ℝ → ℝ) : Prop :=
∀ x : ℝ, 2 * f x + x * f' x < 2

theorem range_of_x_squared_f_x_lt_x_squared_minus_f_1 (f f' : ℝ → ℝ)
  (h_even : even_function f)
  (h_ineq : satisfies_inequality f f')
  : {x : ℝ | x^2 * f x - f 1 < x^2 - 1} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 1} :=
sorry

end range_of_x_squared_f_x_lt_x_squared_minus_f_1_l30_30052


namespace kelly_can_buy_ten_pounds_of_mangoes_l30_30835

theorem kelly_can_buy_ten_pounds_of_mangoes (h : 0.5 * 1.2 = 0.60) : 12 / (2 * 0.60) = 10 :=
  by
    sorry

end kelly_can_buy_ten_pounds_of_mangoes_l30_30835


namespace tim_out_of_pocket_cost_l30_30864

noncomputable def totalOutOfPocketCost : ℝ :=
  let mriCost := 1200
  let xrayCost := 500
  let examinationCost := 400 * (45 / 60)
  let feeForBeingSeen := 150
  let consultationFee := 75
  let physicalTherapyCost := 100 * 8
  let totalCostBeforeInsurance := mriCost + xrayCost + examinationCost + feeForBeingSeen + consultationFee + physicalTherapyCost
  let insuranceCoverage := 0.70 * totalCostBeforeInsurance
  let outOfPocketCost := totalCostBeforeInsurance - insuranceCoverage
  outOfPocketCost

theorem tim_out_of_pocket_cost : totalOutOfPocketCost = 907.50 :=
  by
    -- Proof will be provided here
    sorry

end tim_out_of_pocket_cost_l30_30864


namespace length_is_56_l30_30153

noncomputable def length_of_plot (b : ℝ) : ℝ := b + 12

theorem length_is_56 (b : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) (h_cost : cost_per_meter = 26.50) (h_total_cost : total_cost = 5300) (h_fencing : 26.50 * (4 * b + 24) = 5300) : length_of_plot b = 56 := 
by 
  sorry

end length_is_56_l30_30153


namespace simplest_fraction_sum_l30_30134

theorem simplest_fraction_sum (c d : ℕ) (h1 : 0.325 = (c:ℚ)/d) (h2 : Int.gcd c d = 1) : c + d = 53 :=
by sorry

end simplest_fraction_sum_l30_30134


namespace sounds_meet_at_x_l30_30499

theorem sounds_meet_at_x (d c s : ℝ) (h1 : 0 < d) (h2 : 0 < c) (h3 : 0 < s) :
  ∃ x : ℝ, x = d / 2 * (1 + s / c) ∧ x <= d ∧ x > 0 :=
by
  sorry

end sounds_meet_at_x_l30_30499


namespace surface_area_invisible_block_l30_30334

-- Define the given areas of the seven blocks
def A1 := 148
def A2 := 46
def A3 := 72
def A4 := 28
def A5 := 88
def A6 := 126
def A7 := 58

-- Define total surface areas of the black and white blocks
def S_black := A1 + A2 + A3 + A4
def S_white := A5 + A6 + A7

-- Define the proof problem
theorem surface_area_invisible_block : S_black - S_white = 22 :=
by
  -- This sorry allows the Lean statement to build successfully
  sorry

end surface_area_invisible_block_l30_30334


namespace TotalToysIsNinetyNine_l30_30793

def BillHasToys : ℕ := 60
def HalfOfBillToys : ℕ := BillHasToys / 2
def AdditionalToys : ℕ := 9
def HashHasToys : ℕ := HalfOfBillToys + AdditionalToys
def TotalToys : ℕ := BillHasToys + HashHasToys

theorem TotalToysIsNinetyNine : TotalToys = 99 := by
  sorry

end TotalToysIsNinetyNine_l30_30793


namespace largest_prime_divisor_for_primality_check_l30_30958

theorem largest_prime_divisor_for_primality_check (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1050) : 
  ∃ p, Prime p ∧ p ≤ Int.sqrt 1050 ∧ ∀ q, Prime q → q ≤ Int.sqrt n → q ≤ p := sorry

end largest_prime_divisor_for_primality_check_l30_30958


namespace inclination_angle_l30_30450

-- Define the line equation as a proposition in Lean.
def line_equation (x y : ℝ) : Prop := x + sqrt(3) * y - 5 = 0

theorem inclination_angle (x y : ℝ) :
  line_equation x y → ∃ θ : ℝ, 0 ≤ θ ∧ θ < 180 ∧ tan θ = -sqrt(3) / 3 ∧ θ = 150 :=
by {
  intros h,
  use 150,
  split; try {linarith},
  split; try {linarith},
  split; try {norm_num, rw [tan_def, sin_cos_iff_eq, cos_150, sin_150], 
      linarith, all_goals {rfield_tac; norm_num}},
  sorry
}

end inclination_angle_l30_30450


namespace num_boys_in_class_l30_30040

-- Definitions based on conditions
def num_positions (p1 p2 : Nat) (total : Nat) : Nat :=
  if h : p1 < p2 then p2 - p1
  else total - (p1 - p2)

theorem num_boys_in_class (p1 p2 : Nat) (total : Nat) :
  p1 = 6 ∧ p2 = 16 ∧ num_positions p1 p2 total = 10 → total = 22 :=
by
  intros h
  sorry

end num_boys_in_class_l30_30040


namespace zoo_people_l30_30894

def number_of_people (cars : ℝ) (people_per_car : ℝ) : ℝ :=
  cars * people_per_car

theorem zoo_people (h₁ : cars = 3.0) (h₂ : people_per_car = 63.0) :
  number_of_people cars people_per_car = 189.0 :=
by
  rw [h₁, h₂]
  -- multiply the numbers directly after substitution
  norm_num
  -- left this as a placeholder for now, can use calc or norm_num for final steps
  exact sorry

end zoo_people_l30_30894


namespace circle_standard_equation_l30_30456

theorem circle_standard_equation (x y : ℝ) (h : (x + 1)^2 + (y - 2)^2 = 4) : 
  (x + 1)^2 + (y - 2)^2 = 4 :=
sorry

end circle_standard_equation_l30_30456


namespace arithmetic_sequence_common_difference_l30_30809

theorem arithmetic_sequence_common_difference :
  ∃ d : ℝ, (∀ (a_n : ℕ → ℝ), a_n 1 = 3 ∧ a_n 3 = 7 ∧ (∀ n, a_n n = 3 + (n - 1) * d)) → d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l30_30809


namespace max_value_of_m_l30_30068

theorem max_value_of_m
  (a b : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : (2 / a) + (1 / b) = 1 / 4)
  (h4 : ∀ a b, 2 * a + b ≥ 9 * m) :
  m = 4 := 
sorry

end max_value_of_m_l30_30068


namespace second_divisor_203_l30_30063

theorem second_divisor_203 (x : ℕ) (h1 : 210 % 13 = 3) (h2 : 210 % x = 7) : x = 203 :=
by sorry

end second_divisor_203_l30_30063


namespace schools_participation_l30_30055

-- Definition of the problem conditions
def school_teams : ℕ := 3

-- Paula's rank p must satisfy this
def total_participants (p : ℕ) : ℕ := 2 * p - 1

-- Predicate indicating the number of participants condition:
def participants_condition (p : ℕ) : Prop := total_participants p ≥ 75

-- Translation of number of participants to number of schools
def number_of_schools (n : ℕ) : ℕ := 3 * n

-- The statement to prove:
theorem schools_participation : ∃ (n p : ℕ), participants_condition p ∧ p = 38 ∧ number_of_schools n = total_participants p ∧ n = 25 := 
by 
  sorry

end schools_participation_l30_30055


namespace consecutive_integers_sum_l30_30663

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l30_30663


namespace ratio_of_perimeters_l30_30843

theorem ratio_of_perimeters (A1 A2 : ℝ) (h : A1 / A2 = 16 / 81) : 
  let s1 := real.sqrt A1 
  let s2 := real.sqrt A2 
  (4 * s1) / (4 * s2) = 4 / 9 :=
by {
  sorry
}

end ratio_of_perimeters_l30_30843


namespace sum_of_consecutive_integers_l30_30641

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l30_30641


namespace fractional_eq_range_m_l30_30085

theorem fractional_eq_range_m (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) → m ≤ 2 ∧ m ≠ -2 :=
by
  sorry

end fractional_eq_range_m_l30_30085


namespace power_mod_five_l30_30145

theorem power_mod_five (n : ℕ) (hn : n ≡ 0 [MOD 4]): (3^2000 ≡ 1 [MOD 5]) :=
by 
  sorry

end power_mod_five_l30_30145


namespace volume_sphere_gt_cube_l30_30024

theorem volume_sphere_gt_cube (a r : ℝ) (h : 6 * a^2 = 4 * π * r^2) : 
  (4 / 3) * π * r^3 > a^3 :=
by sorry

end volume_sphere_gt_cube_l30_30024


namespace service_fee_correct_l30_30587
open Nat -- Open the natural number namespace

-- Define the conditions
def ticket_price : ℕ := 44
def num_tickets : ℕ := 3
def total_paid : ℕ := 150

-- Define the cost of tickets
def cost_of_tickets : ℕ := ticket_price * num_tickets

-- Define the service fee calculation
def service_fee : ℕ := total_paid - cost_of_tickets

-- The proof problem statement
theorem service_fee_correct : service_fee = 18 :=
by
  -- Omits the proof, providing a placeholder.
  sorry

end service_fee_correct_l30_30587


namespace sqrt_xyz_sum_l30_30116

theorem sqrt_xyz_sum {x y z : ℝ} (h₁ : y + z = 24) (h₂ : z + x = 26) (h₃ : x + y = 28) :
  Real.sqrt (x * y * z * (x + y + z)) = Real.sqrt 83655 := by
  sorry

end sqrt_xyz_sum_l30_30116


namespace roots_of_equation_l30_30908

def operation (a b : ℝ) : ℝ := a^2 * b + a * b - 1

theorem roots_of_equation :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ operation x₁ 1 = 0 ∧ operation x₂ 1 = 0 :=
by
  sorry

end roots_of_equation_l30_30908


namespace quadratic_inequality_solution_l30_30708

theorem quadratic_inequality_solution
  (a b c : ℝ)
  (h1: ∀ x : ℝ, (-1/3 < x ∧ x < 2) → (ax^2 + bx + c) > 0)
  (h2: a < 0):
  ∀ x : ℝ, ((-3 < x ∧ x < 1/2) ↔ (cx^2 + bx + a) < 0) :=
by
  sorry

end quadratic_inequality_solution_l30_30708


namespace total_toys_l30_30791

variable (B H : ℕ)

theorem total_toys (h1 : B = 60) (h2 : H = 9 + (B / 2)) : B + H = 99 := by
  sorry

end total_toys_l30_30791


namespace students_prefer_mac_l30_30029

-- Define number of students in survey, and let M be the number who prefer Mac to Windows
variables (M E no_pref windows_pref : ℕ)
-- Total number of students surveyed
variable (total_students : ℕ)
-- Define that the total number of students is 210
axiom H_total : total_students = 210
-- Define that one third as many of the students who prefer Mac equally prefer both brands
axiom H_equal_preference : E = M / 3
-- Define that 90 students had no preference
axiom H_no_pref : no_pref = 90
-- Define that 40 students preferred Windows to Mac
axiom H_windows_pref : windows_pref = 40
-- Define that the total number of students is the sum of all groups
axiom H_students_sum : M + E + no_pref + windows_pref = total_students

-- The statement we need to prove
theorem students_prefer_mac :
  M = 60 :=
by sorry

end students_prefer_mac_l30_30029


namespace sequential_inequality_probability_l30_30931

open MeasureTheory

noncomputable def uniform_distribution (i : ℕ) := 
MeasureTheory.ProbabilityMassFunction.uniform (Set.Icc 0 (i ^ 2))

def sequential_probability (n : ℕ) : ℝ := 
∏ i in Finset.range (n - 1), 
((∫ x in 0..(i ^ 2), ∫ y in x..((i + 1) ^ 2), 
  (1 / (i ^ 2)) * (1 / ((i + 1) ^ 2)) d(y) d(x)) * (1 / (1 - x)))

theorem sequential_inequality_probability : 
(∫ x in 0..1, ∫ y in x..4, (1 / 1 ^ 2) * (1 / 2 ^ 2) d(y) d(x)) *
(∫ x in 0..4, ∫ y in x..9, (1 / 2 ^ 2) * (1 / 3 ^ 2) d(y) d(x)) *
(∫ x in 0..9, ∫ y in x..16, (1 / 3 ^ 2) * (1 / 4 ^ 2) d(y) d(x)) *
(∫ x in 0..16, ∫ y in x..25, (1 / 4 ^ 2) * (1 / 5 ^ 2) d(y) d(x)) *
(∫ x in 0..25, ∫ y in x..36, (1 / 5 ^ 2) * (1 / 6 ^ 2) d(y) d(x)) *
(∫ x in 0..36, ∫ y in x..49, (1 / 6 ^ 2) * (1 / 7 ^ 2) d(y) d(x)) *
(∫ x in 0..49, ∫ y in x..64, (1 / 7 ^ 2) * (1 / 8 ^ 2) d(y) d(x)) *
(∫ x in 0..64, ∫ y in x..81, (1 / 8 ^ 2) * (1 / 9 ^ 2) d(y) d(x)) *
(∫ x in 0..81, ∫ y in x..100, (1 / 9 ^ 2) * (1 / 10 ^ 2) d(y) d(x)) ≈ 0.003679 :=
sorry

end sequential_inequality_probability_l30_30931


namespace length_of_tank_l30_30031

namespace TankProblem

def field_length : ℝ := 90
def field_breadth : ℝ := 50
def field_area : ℝ := field_length * field_breadth

def tank_breadth : ℝ := 20
def tank_depth : ℝ := 4

def earth_volume (L : ℝ) : ℝ := L * tank_breadth * tank_depth

def remaining_field_area (L : ℝ) : ℝ := field_area - L * tank_breadth

def height_increase : ℝ := 0.5

theorem length_of_tank (L : ℝ) :
  earth_volume L = remaining_field_area L * height_increase →
  L = 25 :=
by
  sorry

end TankProblem

end length_of_tank_l30_30031


namespace polynomial_equation_solution_l30_30059

open Polynomial

theorem polynomial_equation_solution (P : ℝ[X])
(h : ∀ (a b c : ℝ), P.eval (a + b - 2 * c) + P.eval (b + c - 2 * a) + P.eval (c + a - 2 * b) = 
      3 * P.eval (a - b) + 3 * P.eval (b - c) + 3 * P.eval (c - a)) : 
∃ (a b : ℝ), P = Polynomial.C a * X^2 + Polynomial.C b * X := 
sorry

end polynomial_equation_solution_l30_30059


namespace equivalent_annual_rate_8_percent_quarterly_is_8_24_l30_30192

noncomputable def quarterly_interest_rate (annual_rate : ℚ) := annual_rate / 4

noncomputable def growth_factor (interest_rate : ℚ) := 1 + interest_rate / 100

noncomputable def annual_growth_factor_from_quarterly (quarterly_factor : ℚ) := quarterly_factor ^ 4

noncomputable def equivalent_annual_interest_rate (annual_growth_factor : ℚ) := 
  ((annual_growth_factor - 1) * 100)

theorem equivalent_annual_rate_8_percent_quarterly_is_8_24 :
  let quarter_rate := quarterly_interest_rate 8
  let quarterly_factor := growth_factor quarter_rate
  let annual_factor := annual_growth_factor_from_quarterly quarterly_factor
  equivalent_annual_interest_rate annual_factor = 8.24 := by
  sorry

end equivalent_annual_rate_8_percent_quarterly_is_8_24_l30_30192


namespace wire_cut_problem_l30_30169

-- Conditions
variable (x y : ℝ)
variable (h1 : x = y)
variable (hx : x > 0) -- Assuming positive lengths for the wire pieces

-- Statement to prove
theorem wire_cut_problem : x / y = 1 :=
by sorry

end wire_cut_problem_l30_30169


namespace num_pairs_eq_seven_l30_30386

theorem num_pairs_eq_seven :
  ∃ S : Finset (Nat × Nat), 
    (∀ (a b : Nat), (a, b) ∈ S ↔ (0 < a ∧ 0 < b ∧ a + b ≤ 100 ∧ (a + 1 / b) / (1 / a + b) = 13)) ∧
    S.card = 7 :=
sorry

end num_pairs_eq_seven_l30_30386


namespace range_cos_A_l30_30968

theorem range_cos_A {A B C : ℚ} (h : 1 / (Real.tan B) + 1 / (Real.tan C) = 1 / (Real.tan A))
  (h_non_neg_A: 0 ≤ A) (h_less_pi_A: A ≤ π): 
  (Real.cos A ∈ Set.Ico (2 / 3) 1) :=
sorry

end range_cos_A_l30_30968


namespace polynomial_equation_solution_l30_30060

open Polynomial

theorem polynomial_equation_solution (P : ℝ[X])
(h : ∀ (a b c : ℝ), P.eval (a + b - 2 * c) + P.eval (b + c - 2 * a) + P.eval (c + a - 2 * b) = 
      3 * P.eval (a - b) + 3 * P.eval (b - c) + 3 * P.eval (c - a)) : 
∃ (a b : ℝ), P = Polynomial.C a * X^2 + Polynomial.C b * X := 
sorry

end polynomial_equation_solution_l30_30060


namespace probability_of_first_spade_or_ace_and_second_ace_l30_30018

theorem probability_of_first_spade_or_ace_and_second_ace :
  let deck_size := 52
  let aces := 4
  let spades := 13
  let non_ace_spades := spades - 1
  let other_aces := aces - 1
  let prob_first_non_ace_spade := non_ace_spades / deck_size
  let prob_first_ace_not_spade := (aces - 1) / deck_size
  let prob_first_ace_spade := 1 / deck_size
  let prob_second_ace_after_non_ace_spade := aces / (deck_size - 1)
  let prob_second_ace_after_ace_not_spade := (aces - 1) / (deck_size - 1)
  let prob_second_ace_after_ace_spade := (aces - 1) / (deck_size - 1)
  ((prob_first_non_ace_spade * prob_second_ace_after_non_ace_spade) +
   (prob_first_ace_not_spade * prob_second_ace_after_ace_not_spade) +
   (prob_first_ace_spade * prob_second_ace_after_ace_spade)) = 5 / 221 :=
by
  let deck_size := 52
  let aces := 4
  let spades := 13
  let non_ace_spades := spades - 1
  let other_aces := aces - 1
  let prob_first_non_ace_spade := non_ace_spades / deck_size
  let prob_first_ace_not_spade := (aces - 1) / deck_size
  let prob_first_ace_spade := 1 / deck_size
  let prob_second_ace_after_non_ace_spade := aces / (deck_size - 1)
  let prob_second_ace_after_ace_not_spade := (aces - 1) / (deck_size - 1)
  let prob_second_ace_after_ace_spade := (aces - 1) / (deck_size - 1)
  sorry

end probability_of_first_spade_or_ace_and_second_ace_l30_30018


namespace fractional_equation_solution_l30_30076

theorem fractional_equation_solution (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) →
  (m ≤ 2 ∧ m ≠ -2) :=
by
  sorry

end fractional_equation_solution_l30_30076


namespace number_of_real_z5_is_10_l30_30137

theorem number_of_real_z5_is_10 :
  ∃ S : Finset ℂ, (∀ z ∈ S, z ^ 30 = 1 ∧ (z ^ 5).im = 0) ∧ S.card = 10 :=
sorry

end number_of_real_z5_is_10_l30_30137


namespace arithmetic_geometric_mean_l30_30844

theorem arithmetic_geometric_mean (a b m : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : (a + b) / 2 = m * Real.sqrt (a * b)) :
  a / b = (m + Real.sqrt (m^2 + 1)) / (m - Real.sqrt (m^2 - 1)) :=
by
  sorry

end arithmetic_geometric_mean_l30_30844


namespace total_books_l30_30051

theorem total_books (d k g : ℕ) 
  (h1 : d = 6) 
  (h2 : k = d / 2) 
  (h3 : g = 5 * (d + k)) : 
  d + k + g = 54 :=
by
  sorry

end total_books_l30_30051


namespace village_population_rate_decrease_l30_30141

/--
Village X has a population of 78,000, which is decreasing at a certain rate \( R \) per year.
Village Y has a population of 42,000, which is increasing at the rate of 800 per year.
In 18 years, the population of the two villages will be equal.
We aim to prove that the rate of decrease in population per year for Village X is 1200.
-/
theorem village_population_rate_decrease (R : ℝ) 
  (hx : 78000 - 18 * R = 42000 + 18 * 800) : 
  R = 1200 :=
by
  sorry

end village_population_rate_decrease_l30_30141


namespace pyramid_addition_totals_l30_30739

theorem pyramid_addition_totals 
  (initial_faces : ℕ) (initial_edges : ℕ) (initial_vertices : ℕ)
  (first_pyramid_new_faces : ℕ) (first_pyramid_new_edges : ℕ) (first_pyramid_new_vertices : ℕ)
  (second_pyramid_new_faces : ℕ) (second_pyramid_new_edges : ℕ) (second_pyramid_new_vertices : ℕ)
  (cancelling_faces_first : ℕ) (cancelling_faces_second : ℕ) :
  initial_faces = 5 → 
  initial_edges = 9 → 
  initial_vertices = 6 → 
  first_pyramid_new_faces = 3 →
  first_pyramid_new_edges = 3 →
  first_pyramid_new_vertices = 1 →
  second_pyramid_new_faces = 4 →
  second_pyramid_new_edges = 4 →
  second_pyramid_new_vertices = 1 →
  cancelling_faces_first = 1 →
  cancelling_faces_second = 1 →
  initial_faces + first_pyramid_new_faces - cancelling_faces_first 
  + second_pyramid_new_faces - cancelling_faces_second 
  + initial_edges + first_pyramid_new_edges + second_pyramid_new_edges
  + initial_vertices + first_pyramid_new_vertices + second_pyramid_new_vertices 
  = 34 := by sorry

end pyramid_addition_totals_l30_30739


namespace find_r_l30_30787

noncomputable theory

def vecA : ℝ × ℝ × ℝ := (2, 3, -1)
def vecB : ℝ × ℝ × ℝ := (1, 1, 0)

def crossProduct (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

def vecC : ℝ × ℝ × ℝ := (3, 4, -1)

def scalarEquation (p q r : ℝ) : Prop :=
  (vecC.1 = p * vecA.1 + q * vecB.1 + r * (crossProduct vecA vecB).1) ∧
  (vecC.2 = p * vecA.2 + q * vecB.2 + r * (crossProduct vecA vecB).2) ∧
  (vecC.3 = p * vecA.3 + q * vecB.3 + r * (crossProduct vecA vecB).3)

theorem find_r : ∃ r : ℝ, scalarEquation 0 0 r ∧ r = 2 / 3 :=
by
  sorry

end find_r_l30_30787


namespace part1_part2_l30_30825

noncomputable def f (a x : ℝ) : ℝ := Real.log x - a * x - (a - 1) / x

theorem part1 (a : ℝ) (x : ℝ) (h1 : a ≥ 1) (h2 : x > 0) : f a x ≤ -1 :=
sorry

theorem part2 (a : ℝ) (θ : ℝ) (h1 : a ≥ 1) (h2 : 0 ≤ θ) (h3 : θ ≤ Real.pi / 2) : 
  f a (1 - Real.sin θ) ≤ f a (1 + Real.sin θ) :=
sorry

end part1_part2_l30_30825


namespace probability_exactly_two_ones_l30_30873

theorem probability_exactly_two_ones :
  let n := 12
  let p := 1 / 6
  let q := 5 / 6
  let k := 2
  let binom := @Nat.choose n k
  let prob := binom * (p ^ k) * (q ^ (n - k))
  abs (prob - 0.138) < 0.001 :=
by 
  sorry

end probability_exactly_two_ones_l30_30873


namespace increasing_interval_f_l30_30996

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 3)

theorem increasing_interval_f :
  (∀ x, x ∈ Set.Ioi 3 → f x ∈ Set.Ioi 3) := sorry

end increasing_interval_f_l30_30996


namespace inequality_solution_nonempty_l30_30099

theorem inequality_solution_nonempty (a : ℝ) :
  (∃ x : ℝ, x ^ 2 - a * x - a ≤ -3) ↔ (a ≤ -6 ∨ a ≥ 2) :=
by
  sorry

end inequality_solution_nonempty_l30_30099


namespace algebraic_expression_value_l30_30022

theorem algebraic_expression_value (x : ℝ) (h : x = Real.sqrt 19 - 1) : x^2 + 2 * x + 2 = 20 := by
  sorry

end algebraic_expression_value_l30_30022


namespace eugene_cards_in_deck_l30_30759

theorem eugene_cards_in_deck 
  (cards_used_per_card : ℕ)
  (boxes_used : ℕ)
  (toothpicks_per_box : ℕ)
  (cards_leftover : ℕ)
  (total_toothpicks_used : ℕ)
  (cards_used : ℕ)
  (total_cards_in_deck : ℕ)
  (h1 : cards_used_per_card = 75)
  (h2 : boxes_used = 6)
  (h3 : toothpicks_per_box = 450)
  (h4 : cards_leftover = 16)
  (h5 : total_toothpicks_used = boxes_used * toothpicks_per_box)
  (h6 : cards_used = total_toothpicks_used / cards_used_per_card)
  (h7 : total_cards_in_deck = cards_used + cards_leftover) :
  total_cards_in_deck = 52 :=
by 
  sorry

end eugene_cards_in_deck_l30_30759


namespace complex_pure_imaginary_solution_l30_30953

theorem complex_pure_imaginary_solution (m : ℝ) 
  (h_real_part : m^2 + 2*m - 3 = 0) 
  (h_imaginary_part : m - 1 ≠ 0) : 
  m = -3 :=
sorry

end complex_pure_imaginary_solution_l30_30953


namespace maximum_n_Sn_pos_l30_30778

def arithmetic_sequence := ℕ → ℝ

noncomputable def sum_first_n_terms (a : arithmetic_sequence) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))

axiom a1_eq : ∀ (a : arithmetic_sequence), (a 1) = 2 * (a 2) + (a 4)

axiom S5_eq_5 : ∀ (a : arithmetic_sequence), sum_first_n_terms a 5 = 5

theorem maximum_n_Sn_pos : ∀ (a : arithmetic_sequence), (∃ (n : ℕ), n < 6 ∧ sum_first_n_terms a n > 0) → n = 5 :=
  sorry

end maximum_n_Sn_pos_l30_30778


namespace integral_one_over_x_from_inv_e_to_e_l30_30511

open Real
open IntervalIntegrable

theorem integral_one_over_x_from_inv_e_to_e : 
  (∫ x in (1 : ℝ) / real.exp 1 .. real.exp 1, 1 / x) = 2 := 
by
  sorry

end integral_one_over_x_from_inv_e_to_e_l30_30511


namespace monotonicity_and_zero_range_l30_30212

noncomputable def f (a x : ℝ) : ℝ := a * exp (2 * x) + (a - 2) * exp x - x

theorem monotonicity_and_zero_range (a : ℝ) :
  (∀ x : ℝ, a ≤ 0 → deriv (f a) x ≤ 0) ∧
  (a > 0 → 
    (∀ x : ℝ, x < real.log (1 / a) → deriv (f a) x < 0) ∧
    (∀ x : ℝ, x > real.log (1 / a) → deriv (f a) x > 0)) ∧
  (∀ b : ℝ, 
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) → 
    a ∈ set.Ioo 0 1) := by sorry

end monotonicity_and_zero_range_l30_30212


namespace evaluate_expression_l30_30347

theorem evaluate_expression : (3 / (2 - (4 / (-5)))) = (15 / 14) :=
by
  sorry

end evaluate_expression_l30_30347


namespace total_wheels_at_station_l30_30007

/--
There are 4 trains at a train station.
Each train has 4 carriages.
Each carriage has 3 rows of wheels.
Each row of wheels has 5 wheels.
The total number of wheels at the train station is 240.
-/
theorem total_wheels_at_station : 
    let number_of_trains := 4
    let carriages_per_train := 4
    let rows_per_carriage := 3
    let wheels_per_row := 5
    number_of_trains * carriages_per_train * rows_per_carriage * wheels_per_row = 240 := 
by
    sorry

end total_wheels_at_station_l30_30007


namespace geometric_solid_is_tetrahedron_l30_30995

-- Definitions based on the conditions provided
def top_view_is_triangle : Prop := sorry -- Placeholder for the actual definition
def front_view_is_triangle : Prop := sorry -- Placeholder for the actual definition
def side_view_is_triangle : Prop := sorry -- Placeholder for the actual definition

-- Theorem statement to prove the geometric solid is a triangular pyramid
theorem geometric_solid_is_tetrahedron 
  (h_top : top_view_is_triangle)
  (h_front : front_view_is_triangle)
  (h_side : side_view_is_triangle) :
  -- Conclusion that the solid is a triangular pyramid (tetrahedron)
  is_tetrahedron :=
sorry

end geometric_solid_is_tetrahedron_l30_30995


namespace mary_puts_back_correct_number_of_oranges_l30_30890

namespace FruitProblem

def price_apple := 40
def price_orange := 60
def total_fruits := 10
def average_price_all := 56
def average_price_kept := 50

theorem mary_puts_back_correct_number_of_oranges :
  ∀ (A O O' T: ℕ),
  A + O = total_fruits →
  A * price_apple + O * price_orange = total_fruits * average_price_all →
  A = 2 →
  T = A + O' →
  A * price_apple + O' * price_orange = T * average_price_kept →
  O - O' = 6 :=
by
  sorry

end FruitProblem

end mary_puts_back_correct_number_of_oranges_l30_30890


namespace prob_two_ones_in_twelve_dice_l30_30871

theorem prob_two_ones_in_twelve_dice :
  (∃ p : ℚ, p ≈ 0.230) := 
begin
  let p := (nat.choose 12 2 : ℚ) * (1 / 6)^2 * (5 / 6)^10,
  have h : p ≈ 0.230,
  { sorry }, 
  exact ⟨p, h⟩,
end

end prob_two_ones_in_twelve_dice_l30_30871


namespace distinct_solutions_subtraction_l30_30575

theorem distinct_solutions_subtraction (r s : ℝ) (h_eq : ∀ x ≠ 3, (6 * x - 18) / (x^2 + 4 * x - 21) = x + 3) 
  (h_r : (6 * r - 18) / (r^2 + 4 * r - 21) = r + 3) 
  (h_s : (6 * s - 18) / (s^2 + 4 * s - 21) = s + 3) 
  (h_distinct : r ≠ s) 
  (h_order : r > s) : 
  r - s = 10 := 
by 
  sorry

end distinct_solutions_subtraction_l30_30575


namespace parabola_focus_l30_30354

theorem parabola_focus : ∃ f : ℝ, 
  (∀ x : ℝ, (x^2 + (2*x^2 - f)^2 = (2*x^2 - (1/4 + f))^2)) ∧
  f = 1/8 := 
by
  sorry

end parabola_focus_l30_30354


namespace sandra_beignets_l30_30265

theorem sandra_beignets (beignets_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) (h1 : beignets_per_day = 3) (h2 : days_per_week = 7) (h3: weeks = 16) : 
  (beignets_per_day * days_per_week * weeks) = 336 :=
by {
  -- the proof goes here
  sorry
}

end sandra_beignets_l30_30265


namespace number_of_sides_l30_30325

-- Define the conditions as variables/constants
def exterior_angle (n : ℕ) : ℝ := 18         -- Each exterior angle is 18 degrees
def sum_of_exterior_angles : ℝ := 360        -- Sum of exterior angles of any polygon is 360 degrees

-- Prove the number of sides is equal to 20 given the conditions
theorem number_of_sides : 
  ∃ n : ℕ, (exterior_angle n) * (n : ℝ) = sum_of_exterior_angles → n = 20 := 
by
  sorry

end number_of_sides_l30_30325


namespace combination_identity_l30_30046

-- Lean statement defining the proof problem
theorem combination_identity : Nat.choose 12 5 + Nat.choose 12 6 = Nat.choose 13 6 :=
  sorry

end combination_identity_l30_30046


namespace sum_mod_17_eq_0_l30_30928

theorem sum_mod_17_eq_0 :
  (82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17 = 0 :=
by
  sorry

end sum_mod_17_eq_0_l30_30928


namespace smallest_five_digit_congruent_to_2_mod_17_l30_30146

-- Definitions provided by conditions
def is_five_digit (x : ℕ) : Prop := 10000 ≤ x ∧ x < 100000
def is_congruent_to_2_mod_17 (x : ℕ) : Prop := x % 17 = 2

-- Proving the existence of the smallest five-digit integer satisfying the conditions
theorem smallest_five_digit_congruent_to_2_mod_17 : 
  ∃ x : ℕ, is_five_digit x ∧ is_congruent_to_2_mod_17 x ∧ 
  (∀ y : ℕ, is_five_digit y ∧ is_congruent_to_2_mod_17 y → x ≤ y) := 
begin
  use 10013,
  split,
  { -- Check if it's a five digit number
    unfold is_five_digit,
    exact ⟨by norm_num, by norm_num⟩ },
  split,
  { -- Check if it's congruent to 2 mod 17
    unfold is_congruent_to_2_mod_17,
    exact by norm_num },
  { -- Prove it is the smallest
    intros y hy,
    have h_congruent : y % 17 = 2 := hy.2,
    have h_five_digit : 10000 ≤ y ∧ y < 100000 := hy.1,
    sorry
  }
end

end smallest_five_digit_congruent_to_2_mod_17_l30_30146


namespace problem_l30_30628

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l30_30628


namespace proof_problem_l30_30078

variable (m : ℝ)

theorem proof_problem 
  (h1 : ∀ x, (m / (x - 2) + 1 = x / (2 - x)) → x ≠ 2 ∧ x ≥ 0) :
  m ≤ 2 ∧ m ≠ -2 := by
  sorry

end proof_problem_l30_30078


namespace magnitude_2a_sub_b_l30_30217

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-1, 2)

theorem magnitude_2a_sub_b : (‖(2 * a.1 - b.1, 2 * a.2 - b.2)‖ = 5) :=
by {
  sorry
}

end magnitude_2a_sub_b_l30_30217


namespace sum_of_consecutive_integers_with_product_812_l30_30683

theorem sum_of_consecutive_integers_with_product_812 :
  ∃ x : ℕ, (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) := 
begin
  sorry
end

end sum_of_consecutive_integers_with_product_812_l30_30683


namespace min_tablets_to_get_two_each_l30_30159

def least_tablets_to_ensure_two_each (A B : ℕ) (A_eq : A = 10) (B_eq : B = 10) : ℕ :=
  if A ≥ 2 ∧ B ≥ 2 then 4 else 12

theorem min_tablets_to_get_two_each :
  least_tablets_to_ensure_two_each 10 10 rfl rfl = 12 :=
by
  sorry

end min_tablets_to_get_two_each_l30_30159


namespace evaluate_expression_l30_30919

theorem evaluate_expression : 
  (3^2015 + 3^2013 + 3^2012) / (3^2015 - 3^2013 + 3^2012) = 31 / 25 :=
by
  sorry

end evaluate_expression_l30_30919


namespace determine_a_b_l30_30306

-- Step d) The Lean 4 statement for the transformed problem
theorem determine_a_b (a b : ℝ) (h : ∀ t : ℝ, (t^2 + t + 1) * 1^2 - 2 * (a + t)^2 * 1 + t^2 + 3 * a * t + b = 0) : 
  a = 1 ∧ b = 1 := 
sorry

end determine_a_b_l30_30306


namespace inequality_solution_l30_30757

theorem inequality_solution (x : ℝ) : 3 * x^2 - 8 * x + 3 < 0 ↔ (1 / 3 < x ∧ x < 3) := by
  sorry

end inequality_solution_l30_30757


namespace cos_105_sub_alpha_l30_30207

variable (α : ℝ)

-- Condition
def condition : Prop := Real.cos (75 * Real.pi / 180 + α) = 1 / 2

-- Statement
theorem cos_105_sub_alpha (h : condition α) : Real.cos (105 * Real.pi / 180 - α) = -1 / 2 :=
by
  sorry

end cos_105_sub_alpha_l30_30207


namespace find_a_5_in_arithmetic_sequence_l30_30104

variable {a : ℕ → ℕ}

def arithmetic_sequence (a : ℕ → ℕ) (a1 d : ℕ) : Prop :=
  a 1 = a1 ∧ ∀ n, a (n + 1) = a n + d

theorem find_a_5_in_arithmetic_sequence (h : arithmetic_sequence a 1 2) : a 5 = 9 :=
sorry

end find_a_5_in_arithmetic_sequence_l30_30104


namespace wire_cut_problem_l30_30171

-- Conditions
variable (x y : ℝ)
variable (h1 : x = y)
variable (hx : x > 0) -- Assuming positive lengths for the wire pieces

-- Statement to prove
theorem wire_cut_problem : x / y = 1 :=
by sorry

end wire_cut_problem_l30_30171


namespace number_of_sides_l30_30327

-- Define the conditions as variables/constants
def exterior_angle (n : ℕ) : ℝ := 18         -- Each exterior angle is 18 degrees
def sum_of_exterior_angles : ℝ := 360        -- Sum of exterior angles of any polygon is 360 degrees

-- Prove the number of sides is equal to 20 given the conditions
theorem number_of_sides : 
  ∃ n : ℕ, (exterior_angle n) * (n : ℝ) = sum_of_exterior_angles → n = 20 := 
by
  sorry

end number_of_sides_l30_30327


namespace multiplication_value_l30_30021

theorem multiplication_value : 725143 * 999999 = 725142274857 :=
by
  sorry

end multiplication_value_l30_30021


namespace pies_and_leftover_apples_l30_30122

theorem pies_and_leftover_apples 
  (apples : ℕ) 
  (h : apples = 55) 
  (h1 : 15/3 = 5) :
  (apples / 5 = 11) ∧ (apples - 11 * 5 = 0) :=
by
  sorry

end pies_and_leftover_apples_l30_30122


namespace caffeine_over_goal_l30_30294

theorem caffeine_over_goal (cups_per_day : ℕ) (mg_per_cup : ℕ) (caffeine_goal : ℕ) (total_cups : ℕ) :
  total_cups = 3 ->
  cups_per_day = 3 ->
  mg_per_cup = 80 ->
  caffeine_goal = 200 ->
  (cups_per_day * mg_per_cup) - caffeine_goal = 40 := by
  sorry

end caffeine_over_goal_l30_30294


namespace union_of_A_and_B_l30_30270

variable (a b : ℕ)

def A : Set ℕ := {3, 2^a}
def B : Set ℕ := {a, b}
def intersection_condition : A a ∩ B a b = {2} := by sorry

theorem union_of_A_and_B (h : A a ∩ B a b = {2}) : 
  A a ∪ B a b = {1, 2, 3} := by sorry

end union_of_A_and_B_l30_30270


namespace fourth_intersection_point_of_curve_and_circle_l30_30966

theorem fourth_intersection_point_of_curve_and_circle (h k R : ℝ)
  (h1 : (3 - h)^2 + (2 / 3 - k)^2 = R^2)
  (h2 : (-4 - h)^2 + (-1 / 2 - k)^2 = R^2)
  (h3 : (1 / 2 - h)^2 + (4 - k)^2 = R^2) :
  ∃ (x y : ℝ), xy = 2 ∧ (x, y) ≠ (3, 2 / 3) ∧ (x, y) ≠ (-4, -1 / 2) ∧ (x, y) ≠ (1 / 2, 4) ∧ 
    (x - h)^2 + (y - k)^2 = R^2 ∧ (x, y) = (2 / 3, 3) := 
sorry

end fourth_intersection_point_of_curve_and_circle_l30_30966


namespace number_of_divisors_8_factorial_l30_30090

open Nat

theorem number_of_divisors_8_factorial :
  let n := 8!
  let factorization := [(2, 7), (3, 2), (5, 1), (7, 1)]
  let numberOfDivisors := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  n = 2^7 * 3^2 * 5^1 * 7^1 ->
  n.factors.count = 4 ->
  numberOfDivisors = 96 :=
by
  sorry

end number_of_divisors_8_factorial_l30_30090


namespace number_of_sides_l30_30326

-- Define the conditions as variables/constants
def exterior_angle (n : ℕ) : ℝ := 18         -- Each exterior angle is 18 degrees
def sum_of_exterior_angles : ℝ := 360        -- Sum of exterior angles of any polygon is 360 degrees

-- Prove the number of sides is equal to 20 given the conditions
theorem number_of_sides : 
  ∃ n : ℕ, (exterior_angle n) * (n : ℝ) = sum_of_exterior_angles → n = 20 := 
by
  sorry

end number_of_sides_l30_30326


namespace length_of_second_offset_l30_30351

theorem length_of_second_offset 
  (d : ℝ) (offset1 : ℝ) (area : ℝ) (offset2 : ℝ) 
  (h1 : d = 40)
  (h2 : offset1 = 9)
  (h3 : area = 300) :
  offset2 = 6 :=
by
  sorry

end length_of_second_offset_l30_30351


namespace sum_of_ages_twins_l30_30348

-- Define that Evan has two older twin sisters and their ages are such that the product of all three ages is 162
def twin_sisters_ages (a : ℕ) (b : ℕ) (c : ℕ) : Prop :=
  a * b * c = 162

-- Given the above definition, we need to prove the sum of these ages is 20
theorem sum_of_ages_twins (a b c : ℕ) (h : twin_sisters_ages a b c) (ha : b = c) : a + b + c = 20 :=
by 
  sorry

end sum_of_ages_twins_l30_30348


namespace general_formula_a_sum_T_max_k_value_l30_30777

-- Given conditions
noncomputable def S (n : ℕ) : ℚ := (1/2 : ℚ) * n^2 + (11/2 : ℚ) * n
noncomputable def a (n : ℕ) : ℚ := if n = 1 then 6 else n + 5
noncomputable def b (n : ℕ) : ℚ := 3 / ((2 * a n - 11) * (2 * a (n + 1) - 11))
noncomputable def T (n : ℕ) : ℚ := (3 * n) / (2 * n + 1)

-- Proof statements
theorem general_formula_a (n : ℕ) : a n = if n = 1 then 6 else n + 5 :=
by sorry

theorem sum_T (n : ℕ) : T n = (3 * n) / (2 * n + 1) :=
by sorry

theorem max_k_value (k : ℕ) : k = 19 → ∀ n : ℕ, T n > k / 20 :=
by sorry

end general_formula_a_sum_T_max_k_value_l30_30777


namespace new_person_weight_l30_30026

-- Define the total number of persons and their average weight increase
def num_persons : ℕ := 9
def avg_increase : ℝ := 1.5

-- Define the weight of the person being replaced
def weight_of_replaced_person : ℝ := 65

-- Define the total increase in weight
def total_increase_in_weight : ℝ := num_persons * avg_increase

-- Define the weight of the new person
def weight_of_new_person : ℝ := weight_of_replaced_person + total_increase_in_weight

-- Theorem to prove the weight of the new person is 78.5 kg
theorem new_person_weight : weight_of_new_person = 78.5 := by
  -- proof is omitted
  sorry

end new_person_weight_l30_30026


namespace six_inch_cube_value_is_2700_l30_30032

noncomputable def value_of_six_inch_cube (value_four_inch_cube : ℕ) : ℕ :=
  let volume_four_inch_cube := 4^3
  let volume_six_inch_cube := 6^3
  let scaling_factor := volume_six_inch_cube / volume_four_inch_cube
  value_four_inch_cube * scaling_factor

theorem six_inch_cube_value_is_2700 : value_of_six_inch_cube 800 = 2700 := by
  sorry

end six_inch_cube_value_is_2700_l30_30032


namespace sandra_beignets_16_weeks_l30_30262

-- Define the constants used in the problem
def beignets_per_morning : ℕ := 3
def days_per_week : ℕ := 7
def weeks : ℕ := 16

-- Define the number of beignets Sandra eats in 16 weeks
def beignets_in_16_weeks : ℕ := beignets_per_morning * days_per_week * weeks

-- State the theorem
theorem sandra_beignets_16_weeks : beignets_in_16_weeks = 336 :=
by
  -- Provide a placeholder for the proof
  sorry

end sandra_beignets_16_weeks_l30_30262


namespace original_bill_amount_l30_30030

/-- 
If 8 people decided to split the restaurant bill evenly and each paid $314.15 after rounding
up to the nearest cent, then the original bill amount was $2513.20.
-/
theorem original_bill_amount (n : ℕ) (individual_share : ℝ) (total_amount : ℝ) 
  (h1 : n = 8) (h2 : individual_share = 314.15) 
  (h3 : total_amount = n * individual_share) : 
  total_amount = 2513.20 :=
by
  sorry

end original_bill_amount_l30_30030


namespace sum_of_consecutive_integers_with_product_812_l30_30679

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l30_30679


namespace coal_removal_date_l30_30502

theorem coal_removal_date (m n : ℝ) (h1 : 0 < m) (h2 : 0 < n)
  (h3 : 25 * m + 9 * n = 0.5)
  (h4 : ∃ z : ℝ,  z * (n + m) = 0.5)
  (h5 : ∀ z : ℝ, z = 12 → (16 + z) * m = (9 + z) * n):
  ∃ t : ℝ, t = 28 := 
by 
{
  sorry
}

end coal_removal_date_l30_30502


namespace arithmetic_sequence_sum_l30_30408

variable (a : ℕ → ℤ)

def arithmetic_sequence_condition_1 := a 5 = 3
def arithmetic_sequence_condition_2 := a 6 = -2

theorem arithmetic_sequence_sum :
  arithmetic_sequence_condition_1 a →
  arithmetic_sequence_condition_2 a →
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 3 :=
by
  intros h1 h2
  sorry

end arithmetic_sequence_sum_l30_30408


namespace milk_quality_check_l30_30990

/-
Suppose there is a collection of 850 bags of milk numbered from 001 to 850. 
From this collection, 50 bags are randomly selected for testing by reading numbers 
from a random number table. Starting from the 3rd line and the 1st group of numbers, 
continuing to the right, we need to find the next 4 bag numbers after the sequence 
614, 593, 379, 242.
-/

def random_numbers : List Nat := [
  78226, 85384, 40527, 48987, 60602, 16085, 29971, 61279,
  43021, 92980, 27768, 26916, 27783, 84572, 78483, 39820,
  61459, 39073, 79242, 20372, 21048, 87088, 34600, 74636,
  63171, 58247, 12907, 50303, 28814, 40422, 97895, 61421,
  42372, 53183, 51546, 90385, 12120, 64042, 51320, 22983
]

noncomputable def next_valid_numbers (nums : List Nat) (start_idx : Nat) : List Nat :=
  nums.drop start_idx |>.filter (λ n => n ≤ 850) |>.take 4

theorem milk_quality_check :
  next_valid_numbers random_numbers 18 = [203, 722, 104, 88] :=
sorry

end milk_quality_check_l30_30990


namespace number_of_parallel_lines_l30_30544

theorem number_of_parallel_lines (n : ℕ) (h : (n * (n - 1) / 2) * (8 * 7 / 2) = 784) : n = 8 :=
sorry

end number_of_parallel_lines_l30_30544


namespace letters_into_mailboxes_l30_30795

theorem letters_into_mailboxes (letters : ℕ) (mailboxes : ℕ) (h_letters: letters = 3) (h_mailboxes: mailboxes = 4) :
  (mailboxes ^ letters) = 64 := by
  sorry

end letters_into_mailboxes_l30_30795


namespace scientific_notation_140000000_l30_30591

theorem scientific_notation_140000000 :
  140000000 = 1.4 * 10^8 := 
sorry

end scientific_notation_140000000_l30_30591


namespace sally_cards_l30_30443

theorem sally_cards (x : ℕ) (h1 : 27 + x + 20 = 88) : x = 41 := by
  sorry

end sally_cards_l30_30443


namespace caffeine_over_goal_l30_30295

theorem caffeine_over_goal (cups_per_day : ℕ) (mg_per_cup : ℕ) (caffeine_goal : ℕ) (total_cups : ℕ) :
  total_cups = 3 ->
  cups_per_day = 3 ->
  mg_per_cup = 80 ->
  caffeine_goal = 200 ->
  (cups_per_day * mg_per_cup) - caffeine_goal = 40 := by
  sorry

end caffeine_over_goal_l30_30295


namespace larger_number_is_1590_l30_30127

theorem larger_number_is_1590 (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 7 * S + 15) : L = 1590 :=
by
  sorry

end larger_number_is_1590_l30_30127
