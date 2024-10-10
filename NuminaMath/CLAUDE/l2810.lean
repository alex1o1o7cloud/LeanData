import Mathlib

namespace repeating_digit_equation_solutions_l2810_281019

def repeating_digit (d : ℕ) (n : ℕ) : ℕ :=
  d * ((10^n - 1) / 9)

theorem repeating_digit_equation_solutions :
  ∀ a b c : ℕ,
    (∀ n : ℕ, repeating_digit a n * 10^n + repeating_digit b n + 1 = (repeating_digit c n + 1)^2) →
    ((a = 0 ∧ b = 0 ∧ c = 0) ∨
     (a = 1 ∧ b = 5 ∧ c = 3) ∨
     (a = 4 ∧ b = 8 ∧ c = 6) ∨
     (a = 9 ∧ b = 9 ∧ c = 9)) :=
by sorry

end repeating_digit_equation_solutions_l2810_281019


namespace desk_final_price_l2810_281031

/-- Calculates the final price of an auctioned item given initial price, price increase per bid, and number of bids -/
def final_price (initial_price : ℕ) (price_increase : ℕ) (num_bids : ℕ) : ℕ :=
  initial_price + price_increase * num_bids

/-- Theorem stating the final price of the desk after the bidding war -/
theorem desk_final_price :
  final_price 15 5 10 = 65 := by
  sorry

end desk_final_price_l2810_281031


namespace area_parallelogram_from_diagonals_l2810_281005

/-- A quadrilateral in a plane -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- The area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- The diagonals of a quadrilateral -/
def diagonals (q : Quadrilateral) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- A parallelogram with sides parallel and equal to given line segments -/
def parallelogram_from_diagonals (d : (ℝ × ℝ) × (ℝ × ℝ)) : Quadrilateral := sorry

/-- The theorem stating that the area of the parallelogram formed by the diagonals
    is twice the area of the original quadrilateral -/
theorem area_parallelogram_from_diagonals (q : Quadrilateral) :
  area (parallelogram_from_diagonals (diagonals q)) = 2 * area q := by sorry

end area_parallelogram_from_diagonals_l2810_281005


namespace ophelia_pay_reaches_93_l2810_281015

/-- Ophelia's weekly earnings function -/
def earnings (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 51
  else 51 + 100 * (n - 1)

/-- Average weekly pay after n weeks -/
def average_pay (n : ℕ) : ℚ :=
  if n = 0 then 0 else (earnings n) / n

/-- Theorem: Ophelia's average weekly pay reaches $93 after 7 weeks -/
theorem ophelia_pay_reaches_93 :
  ∃ n : ℕ, n > 0 ∧ average_pay n = 93 ∧ ∀ m : ℕ, m > 0 ∧ m < n → average_pay m < 93 :=
sorry

end ophelia_pay_reaches_93_l2810_281015


namespace g_range_l2810_281013

noncomputable def g (x : ℝ) : ℝ := (Real.arcsin x)^3 - (Real.arccos x)^3

theorem g_range :
  ∀ x : ℝ, x ∈ Set.Icc (-1) 1 →
  (Real.arccos x + Real.arcsin x = π / 2) →
  ∃ y ∈ Set.Icc (-((7 * π^3) / 16)) ((π^3) / 16), g x = y :=
by sorry

end g_range_l2810_281013


namespace unique_m_existence_l2810_281085

theorem unique_m_existence : ∃! m : ℤ,
  50 ≤ m ∧ m ≤ 120 ∧
  m % 7 = 0 ∧
  m % 8 = 5 ∧
  m % 5 = 4 ∧
  m = 189 := by
  sorry

end unique_m_existence_l2810_281085


namespace sandwich_bread_slices_l2810_281016

theorem sandwich_bread_slices 
  (total_sandwiches : ℕ) 
  (bread_packs : ℕ) 
  (slices_per_pack : ℕ) 
  (h1 : total_sandwiches = 8)
  (h2 : bread_packs = 4)
  (h3 : slices_per_pack = 4) :
  (bread_packs * slices_per_pack) / total_sandwiches = 2 := by
  sorry

end sandwich_bread_slices_l2810_281016


namespace unique_apartment_number_l2810_281065

def is_valid_apartment_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (100 * a + 10 * c + b) +
    (100 * b + 10 * a + c) +
    (100 * b + 10 * c + a) +
    (100 * c + 10 * a + b) +
    (100 * c + 10 * b + a) = 2017

theorem unique_apartment_number :
  ∃! n : ℕ, is_valid_apartment_number n ∧ n = 425 :=
sorry

end unique_apartment_number_l2810_281065


namespace pigeonhole_apples_l2810_281081

theorem pigeonhole_apples (n : ℕ) (m : ℕ) (h1 : n = 25) (h2 : m = 3) :
  ∃ (c : Fin m), (n / m : ℚ) ≤ 9 := by
  sorry

end pigeonhole_apples_l2810_281081


namespace tangent_lines_to_C_value_of_m_l2810_281076

-- Define the curve C
def curve_C (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define point P
def point_P : ℝ × ℝ := (3, -1)

-- Define the intersecting line
def line_L (x y : ℝ) : Prop :=
  x + 2*y + 5 = 0

-- Part 1: Tangent lines
theorem tangent_lines_to_C :
  ∃ (k : ℝ), 
    (∀ x y : ℝ, curve_C 1 x y → (x = 3 ∨ 5*x + 12*y - 3 = 0) → 
      ((x - 3)^2 + (y + 1)^2 = k^2)) ∧
    (∀ x y : ℝ, (x = 3 ∨ 5*x + 12*y - 3 = 0) → 
      ((x - 3)^2 + (y + 1)^2 ≤ k^2)) :=
sorry

-- Part 2: Value of m
theorem value_of_m :
  ∃! m : ℝ, 
    (∃ x1 y1 x2 y2 : ℝ,
      curve_C m x1 y1 ∧ curve_C m x2 y2 ∧
      line_L x1 y1 ∧ line_L x2 y2 ∧
      (x1 - x2)^2 + (y1 - y2)^2 = 20) ∧
    m = -20 :=
sorry

end tangent_lines_to_C_value_of_m_l2810_281076


namespace locus_of_points_l2810_281010

/-- The locus of points with a 3:1 distance ratio to a fixed point and line -/
theorem locus_of_points (x y : ℝ) : 
  let F : ℝ × ℝ := (4.5, 0)
  let dist_to_F := Real.sqrt ((x - F.1)^2 + (y - F.2)^2)
  let dist_to_line := |x - 0.5|
  dist_to_F = 3 * dist_to_line → x^2 / 2.25 - y^2 / 18 = 1 := by
  sorry

end locus_of_points_l2810_281010


namespace second_discount_percentage_l2810_281057

theorem second_discount_percentage
  (original_price : ℝ)
  (first_discount_percent : ℝ)
  (final_sale_price : ℝ)
  (h1 : original_price = 495)
  (h2 : first_discount_percent = 15)
  (h3 : final_sale_price = 378.675) :
  ∃ (second_discount_percent : ℝ),
    second_discount_percent = 10 ∧
    final_sale_price = original_price * (1 - first_discount_percent / 100) * (1 - second_discount_percent / 100) :=
by sorry

end second_discount_percentage_l2810_281057


namespace water_transfer_l2810_281084

theorem water_transfer (a b x : ℝ) : 
  a = 13.2 ∧ 
  (13.2 - x = (1/3) * (b + x)) ∧ 
  (b - x = (1/2) * (13.2 + x)) → 
  x = 6 := by
  sorry

end water_transfer_l2810_281084


namespace cos_arctan_equal_x_squared_l2810_281091

theorem cos_arctan_equal_x_squared :
  ∃ (x : ℝ), x > 0 ∧ Real.cos (Real.arctan x) = x → x^2 = (-1 + Real.sqrt 5) / 2 := by
  sorry

end cos_arctan_equal_x_squared_l2810_281091


namespace product_zero_l2810_281060

theorem product_zero (a x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ x₁₀ x₁₁ x₁₂ x₁₃ : ℤ) 
  (h1 : a = (1 + x₁) * (1 + x₂) * (1 + x₃) * (1 + x₄) * (1 + x₅) * (1 + x₆) * (1 + x₇) * 
           (1 + x₈) * (1 + x₉) * (1 + x₁₀) * (1 + x₁₁) * (1 + x₁₂) * (1 + x₁₃))
  (h2 : a = (1 - x₁) * (1 - x₂) * (1 - x₃) * (1 - x₄) * (1 - x₅) * (1 - x₆) * (1 - x₇) * 
           (1 - x₈) * (1 - x₉) * (1 - x₁₀) * (1 - x₁₁) * (1 - x₁₂) * (1 - x₁₃)) :
  a * x₁ * x₂ * x₃ * x₄ * x₅ * x₆ * x₇ * x₈ * x₉ * x₁₀ * x₁₁ * x₁₂ * x₁₃ = 0 := by
  sorry

end product_zero_l2810_281060


namespace race_distances_main_theorem_l2810_281023

/-- Represents a race between three racers over a certain distance -/
structure Race where
  distance : ℝ
  a_beats_b : ℝ
  b_beats_c : ℝ
  a_beats_c : ℝ

/-- The theorem stating the distances of the two races -/
theorem race_distances (race1 race2 : Race) : 
  race1.distance = 150 ∧ race2.distance = 120 :=
  by
    have h1 : race1 = { distance := 150, a_beats_b := 30, b_beats_c := 15, a_beats_c := 42 } := by sorry
    have h2 : race2 = { distance := 120, a_beats_b := 25, b_beats_c := 20, a_beats_c := 40 } := by sorry
    sorry

/-- The main theorem proving the distances of both races -/
theorem main_theorem : ∃ (race1 race2 : Race), 
  race1.a_beats_b = 30 ∧ 
  race1.b_beats_c = 15 ∧ 
  race1.a_beats_c = 42 ∧
  race2.a_beats_b = 25 ∧ 
  race2.b_beats_c = 20 ∧ 
  race2.a_beats_c = 40 ∧
  race1.distance = 150 ∧ 
  race2.distance = 120 :=
by
  sorry

end race_distances_main_theorem_l2810_281023


namespace newspaper_collection_target_l2810_281083

structure Section where
  name : String
  first_week_collection : ℝ

def second_week_increase : ℝ := 0.10
def third_week_increase : ℝ := 0.30

def sections : List Section := [
  ⟨"A", 260⟩,
  ⟨"B", 290⟩,
  ⟨"C", 250⟩,
  ⟨"D", 270⟩,
  ⟨"E", 300⟩,
  ⟨"F", 310⟩,
  ⟨"G", 280⟩,
  ⟨"H", 265⟩
]

def first_week_total : ℝ := (sections.map (·.first_week_collection)).sum

def second_week_total : ℝ :=
  (sections.map (fun s => s.first_week_collection * (1 + second_week_increase))).sum

def third_week_total : ℝ :=
  (sections.map (fun s => s.first_week_collection * (1 + second_week_increase) * (1 + third_week_increase))).sum

def target : ℝ := first_week_total + second_week_total + third_week_total

theorem newspaper_collection_target :
  target = 7854.25 := by sorry

end newspaper_collection_target_l2810_281083


namespace fraction_equality_l2810_281040

theorem fraction_equality (x m : ℝ) (h : x ≠ 0) :
  x / (x^2 - m*x + 1) = 1 →
  x^3 / (x^6 - m^3*x^3 + 1) = 1 / (3*m^2 - 2) :=
by sorry

end fraction_equality_l2810_281040


namespace zeros_of_f_l2810_281024

-- Define the function f(x) = x^2 - 2x - 3
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Theorem stating that 3 and -1 are the zeros of the function f
theorem zeros_of_f : 
  (f 3 = 0 ∧ f (-1) = 0) ∧ 
  ∀ x : ℝ, f x = 0 → x = 3 ∨ x = -1 :=
sorry

end zeros_of_f_l2810_281024


namespace max_value_theorem_max_value_achieved_l2810_281037

theorem max_value_theorem (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  3 * x + 4 * y + 6 * z ≤ Real.sqrt 53 :=
by sorry

theorem max_value_achieved (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  ∃ (x' y' z' : ℝ), 9 * x'^2 + 4 * y'^2 + 25 * z'^2 = 1 ∧ 3 * x' + 4 * y' + 6 * z' = Real.sqrt 53 :=
by sorry

end max_value_theorem_max_value_achieved_l2810_281037


namespace interest_rate_calculation_l2810_281097

/-- Compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Problem statement --/
theorem interest_rate_calculation (principal time : ℕ) (final_amount : ℝ) 
  (h1 : principal = 6000)
  (h2 : time = 2)
  (h3 : final_amount = 7260) :
  ∃ (rate : ℝ), compound_interest principal rate time = final_amount ∧ rate = 0.1 :=
by sorry

end interest_rate_calculation_l2810_281097


namespace spencer_walk_distance_l2810_281017

theorem spencer_walk_distance (total : ℝ) (house_to_library : ℝ) (library_to_post : ℝ)
  (h1 : total = 0.8)
  (h2 : house_to_library = 0.3)
  (h3 : library_to_post = 0.1) :
  total - (house_to_library + library_to_post) = 0.4 := by
  sorry

end spencer_walk_distance_l2810_281017


namespace balloon_min_volume_l2810_281026

/-- Represents the relationship between pressure and volume of a gas in a balloon -/
noncomputable def pressure (k : ℝ) (V : ℝ) : ℝ := k / V

theorem balloon_min_volume (k : ℝ) :
  (pressure k 3 = 8000) →
  (∀ V, V ≥ 0.6 → pressure k V ≤ 40000) ∧
  (∀ ε > 0, ∃ V, 0.6 - ε < V ∧ V < 0.6 ∧ pressure k V > 40000) :=
by sorry

end balloon_min_volume_l2810_281026


namespace line_of_intersection_canonical_equation_l2810_281089

/-- Given two planes in 3D space, this theorem states that their line of intersection 
    can be represented by a specific canonical equation. -/
theorem line_of_intersection_canonical_equation 
  (plane1 : x + 5*y + 2*z = 5) 
  (plane2 : 2*x - 5*y - z = -5) :
  ∃ (t : ℝ), x = 5*t ∧ y = 5*t + 1 ∧ z = -15*t :=
by sorry

end line_of_intersection_canonical_equation_l2810_281089


namespace billys_remaining_crayons_l2810_281055

/-- Given Billy's initial number of crayons and the number eaten by a hippopotamus,
    this theorem proves that the remaining number of crayons is the difference between
    the initial number and the number eaten. -/
theorem billys_remaining_crayons (initial : ℕ) (eaten : ℕ) :
  initial ≥ eaten → initial - eaten = initial - eaten :=
by
  sorry

end billys_remaining_crayons_l2810_281055


namespace no_sum_of_cubes_l2810_281045

theorem no_sum_of_cubes (n : ℕ) : ¬∃ (x y : ℕ), 10^(3*n + 1) = x^3 + y^3 := by
  sorry

end no_sum_of_cubes_l2810_281045


namespace chosen_number_calculation_l2810_281095

theorem chosen_number_calculation : 
  let chosen_number : ℕ := 208
  let divided_result : ℚ := chosen_number / 2
  let final_result : ℚ := divided_result - 100
  final_result = 4 := by
sorry

end chosen_number_calculation_l2810_281095


namespace badminton_team_lineup_count_l2810_281008

theorem badminton_team_lineup_count :
  let total_players : ℕ := 18
  let quadruplets : ℕ := 4
  let starters : ℕ := 8
  let non_quadruplets : ℕ := total_players - quadruplets
  let lineups_without_quadruplets : ℕ := Nat.choose non_quadruplets starters
  let lineups_with_one_quadruplet : ℕ := quadruplets * Nat.choose non_quadruplets (starters - 1)
  lineups_without_quadruplets + lineups_with_one_quadruplet = 16731 :=
by sorry

end badminton_team_lineup_count_l2810_281008


namespace certain_number_problem_l2810_281062

theorem certain_number_problem : ∃ x : ℚ, (5/6 : ℚ) * x = (5/16 : ℚ) * x + 100 ∧ x = 192 := by
  sorry

end certain_number_problem_l2810_281062


namespace combined_tennis_percentage_l2810_281050

theorem combined_tennis_percentage
  (north_total : ℕ)
  (south_total : ℕ)
  (north_tennis_percent : ℚ)
  (south_tennis_percent : ℚ)
  (h1 : north_total = 1800)
  (h2 : south_total = 2700)
  (h3 : north_tennis_percent = 25 / 100)
  (h4 : south_tennis_percent = 35 / 100)
  : (north_total * north_tennis_percent + south_total * south_tennis_percent) / (north_total + south_total) = 31 / 100 :=
by sorry

end combined_tennis_percentage_l2810_281050


namespace bulk_bag_contains_40_oz_l2810_281082

/-- Calculates the number of ounces in a bulk bag of mixed nuts -/
def bulkBagOunces (originalCost : ℚ) (couponValue : ℚ) (costPerServing : ℚ) : ℚ :=
  (originalCost - couponValue) / costPerServing

/-- Theorem stating that the bulk bag contains 40 ounces of mixed nuts -/
theorem bulk_bag_contains_40_oz :
  bulkBagOunces 25 5 (1/2) = 40 := by
  sorry

end bulk_bag_contains_40_oz_l2810_281082


namespace convex_quadrilateral_probability_l2810_281067

/-- The number of points on the circle -/
def num_points : ℕ := 8

/-- The number of chords to be selected -/
def num_selected_chords : ℕ := 4

/-- The total number of possible chords -/
def total_chords : ℕ := num_points.choose 2

/-- The number of ways to select the required number of chords -/
def ways_to_select_chords : ℕ := total_chords.choose num_selected_chords

/-- The number of ways to choose points that form a convex quadrilateral -/
def convex_quadrilaterals : ℕ := num_points.choose 4

/-- The probability of forming a convex quadrilateral -/
def probability : ℚ := convex_quadrilaterals / ways_to_select_chords

theorem convex_quadrilateral_probability : probability = 2 / 585 := by
  sorry

end convex_quadrilateral_probability_l2810_281067


namespace coefficient_of_a_l2810_281078

theorem coefficient_of_a (a b : ℝ) (h1 : a = 2) (h2 : b = 15) : 
  42 * b = 630 := by sorry

end coefficient_of_a_l2810_281078


namespace max_perfect_squares_among_products_l2810_281098

theorem max_perfect_squares_among_products (a b : ℕ) (h : a ≠ b) : 
  let products := {a * (a + 2), a * b, a * (b + 2), (a + 2) * b, (a + 2) * (b + 2), b * (b + 2)}
  (∃ (s : Finset ℕ), s ⊆ products ∧ (∀ x ∈ s, ∃ y : ℕ, x = y * y) ∧ s.card = 2) ∧
  (∀ (s : Finset ℕ), s ⊆ products → (∀ x ∈ s, ∃ y : ℕ, x = y * y) → s.card ≤ 2) :=
by sorry

end max_perfect_squares_among_products_l2810_281098


namespace cistern_leak_time_l2810_281021

theorem cistern_leak_time 
  (fill_time_A : ℝ) 
  (fill_time_both : ℝ) 
  (leak_time_B : ℝ) : 
  fill_time_A = 10 →
  fill_time_both = 29.999999999999993 →
  (1 / fill_time_A - 1 / leak_time_B = 1 / fill_time_both) →
  leak_time_B = 15 := by
sorry

end cistern_leak_time_l2810_281021


namespace graduation_ceremony_chairs_l2810_281068

/-- Calculates the number of chairs needed for a graduation ceremony --/
def chairs_needed (graduates : ℕ) (parents_per_graduate : ℕ) (teachers : ℕ) : ℕ :=
  let parent_chairs := graduates * parents_per_graduate
  let graduate_and_parent_chairs := graduates + parent_chairs
  let administrator_chairs := teachers / 2
  graduate_and_parent_chairs + teachers + administrator_chairs

theorem graduation_ceremony_chairs :
  chairs_needed 50 2 20 = 180 :=
by sorry

end graduation_ceremony_chairs_l2810_281068


namespace three_number_ratio_problem_l2810_281007

theorem three_number_ratio_problem (x y z : ℝ) 
  (h_sum : x + y + z = 120)
  (h_ratio1 : x / y = 3 / 4)
  (h_ratio2 : y / z = 5 / 7)
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0) :
  y = 800 / 21 := by
  sorry

end three_number_ratio_problem_l2810_281007


namespace exists_equivalent_expression_l2810_281072

/-- Define a type for the unknown operations -/
inductive UnknownOp
| add
| sub

/-- Define a function that applies the unknown operation -/
def applyOp (op : UnknownOp) (x y : ℝ) : ℝ :=
  match op with
  | UnknownOp.add => x + y
  | UnknownOp.sub => x - y

/-- Define a function that represents the reversed subtraction -/
def revSub (x y : ℝ) : ℝ := y - x

theorem exists_equivalent_expression :
  ∃ (op1 op2 : UnknownOp) (f1 f2 : ℝ → ℝ → ℝ),
    (f1 = applyOp op1 ∧ f2 = applyOp op2) ∨
    (f1 = applyOp op1 ∧ f2 = revSub) ∨
    (f1 = revSub ∧ f2 = applyOp op2) →
    ∀ (a b : ℝ), ∃ (expr : ℝ), expr = 20 * a - 18 * b :=
by sorry

end exists_equivalent_expression_l2810_281072


namespace power_product_simplification_l2810_281048

theorem power_product_simplification (a : ℝ) : ((-2 * a)^2) * (a^4) = 4 * (a^6) := by
  sorry

end power_product_simplification_l2810_281048


namespace binomial_150_150_l2810_281030

theorem binomial_150_150 : Nat.choose 150 150 = 1 := by
  sorry

end binomial_150_150_l2810_281030


namespace school_cleanup_participants_l2810_281093

/-- The expected number of participants after n years, given an initial number and annual increase rate -/
def expected_participants (initial : ℕ) (rate : ℚ) (years : ℕ) : ℚ :=
  initial * (1 + rate) ^ years

theorem school_cleanup_participants : expected_participants 1000 (60/100) 3 = 4096 := by
  sorry

end school_cleanup_participants_l2810_281093


namespace tan_beta_value_l2810_281025

theorem tan_beta_value (θ β : Real) (h1 : (2 : Real) = 2 * Real.cos θ) 
  (h2 : (-3 : Real) = 2 * Real.sin θ) (h3 : β = θ - 3 * Real.pi / 4) : 
  Real.tan β = -1/5 := by
  sorry

end tan_beta_value_l2810_281025


namespace remaining_integers_count_l2810_281003

def S : Finset Nat := Finset.range 51 \ {0}

theorem remaining_integers_count : 
  (S.filter (fun n => n % 2 ≠ 0 ∧ n % 3 ≠ 0)).card = 17 := by
  sorry

end remaining_integers_count_l2810_281003


namespace three_round_layoffs_result_l2810_281027

def layoff_round (employees : ℕ) : ℕ :=
  (employees * 10) / 100

def remaining_after_layoff (employees : ℕ) : ℕ :=
  employees - layoff_round employees

def total_layoffs (initial_employees : ℕ) : ℕ :=
  let first_round := layoff_round initial_employees
  let after_first := remaining_after_layoff initial_employees
  let second_round := layoff_round after_first
  let after_second := remaining_after_layoff after_first
  let third_round := layoff_round after_second
  first_round + second_round + third_round

theorem three_round_layoffs_result :
  total_layoffs 1000 = 271 :=
sorry

end three_round_layoffs_result_l2810_281027


namespace gum_pack_size_l2810_281053

theorem gum_pack_size (y : ℝ) : 
  (25 - 2 * y) / 40 = 25 / (40 + 4 * y) → y = 2.5 := by
sorry

end gum_pack_size_l2810_281053


namespace four_purchase_options_l2810_281058

/-- Represents the number of different ways to buy masks and alcohol wipes -/
def purchase_options : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 3 * p.1 + 2 * p.2 = 30) (Finset.product (Finset.range 31) (Finset.range 31))).card

/-- Theorem stating that there are exactly 4 ways to purchase masks and alcohol wipes -/
theorem four_purchase_options : purchase_options = 4 := by
  sorry

end four_purchase_options_l2810_281058


namespace high_school_math_team_payment_l2810_281044

theorem high_school_math_team_payment (B : ℕ) : 
  B < 10 → (100 + 10 * B + 3) % 13 = 0 → B = 4 := by
sorry

end high_school_math_team_payment_l2810_281044


namespace prime_between_30_and_40_with_remainder_7_mod_12_l2810_281042

theorem prime_between_30_and_40_with_remainder_7_mod_12 (n : ℕ) : 
  Prime n → 
  30 < n → 
  n < 40 → 
  n % 12 = 7 → 
  n = 31 := by
sorry

end prime_between_30_and_40_with_remainder_7_mod_12_l2810_281042


namespace shooting_statistics_l2810_281090

def scores : List ℕ := [7, 5, 8, 9, 6, 6, 7, 7, 8, 7]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

def mean (l : List ℕ) : ℚ := sorry

def variance (l : List ℕ) : ℚ := sorry

theorem shooting_statistics :
  mode scores = 7 ∧
  median scores = 7 ∧
  mean scores = 7 ∧
  variance scores = 6/5 := by sorry

end shooting_statistics_l2810_281090


namespace runners_meet_time_l2810_281041

def carla_lap_time : ℕ := 5
def jose_lap_time : ℕ := 8
def mary_lap_time : ℕ := 10

theorem runners_meet_time :
  Nat.lcm (Nat.lcm carla_lap_time jose_lap_time) mary_lap_time = 40 := by
  sorry

end runners_meet_time_l2810_281041


namespace min_value_trig_expression_l2810_281014

/-- The minimum value of k*sin^4(x) + cos^4(x) for k ≥ 0 is 0 -/
theorem min_value_trig_expression (k : ℝ) (hk : k ≥ 0) :
  ∃ m : ℝ, m = 0 ∧ ∀ x : ℝ, k * Real.sin x ^ 4 + Real.cos x ^ 4 ≥ m := by
  sorry

end min_value_trig_expression_l2810_281014


namespace water_remaining_l2810_281052

/-- Given 3 gallons of water and using 5/4 gallons, prove that the remaining amount is 7/4 gallons. -/
theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 5/4 → remaining = initial - used → remaining = 7/4 := by
  sorry

end water_remaining_l2810_281052


namespace smallest_superior_discount_l2810_281011

def successive_discount (single_discount : ℝ) (times : ℕ) : ℝ :=
  1 - (1 - single_discount) ^ times

theorem smallest_superior_discount : ∃ (n : ℕ), n = 37 ∧
  (∀ (m : ℕ), m < n →
    (m : ℝ) / 100 ≤ successive_discount (12 / 100) 3 ∨
    (m : ℝ) / 100 ≤ successive_discount (20 / 100) 2 ∨
    (m : ℝ) / 100 ≤ successive_discount (8 / 100) 4) ∧
  (37 : ℝ) / 100 > successive_discount (12 / 100) 3 ∧
  (37 : ℝ) / 100 > successive_discount (20 / 100) 2 ∧
  (37 : ℝ) / 100 > successive_discount (8 / 100) 4 :=
sorry

end smallest_superior_discount_l2810_281011


namespace gcd_of_225_and_135_l2810_281080

theorem gcd_of_225_and_135 : Nat.gcd 225 135 = 45 := by
  sorry

end gcd_of_225_and_135_l2810_281080


namespace isosceles_triangle_perimeter_l2810_281086

/-- An isosceles triangle with two sides of lengths 3 and 6 has a perimeter of 15. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 6 ∧ c = 6 →  -- Two sides are 6, one side is 3
  (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
  a + b + c = 15 :=
by sorry


end isosceles_triangle_perimeter_l2810_281086


namespace company_workforce_company_workforce_proof_l2810_281022

theorem company_workforce (initial_female_percentage : ℚ) 
                          (additional_male_workers : ℕ) 
                          (final_female_percentage : ℚ) : Prop :=
  initial_female_percentage = 60 / 100 →
  additional_male_workers = 22 →
  final_female_percentage = 55 / 100 →
  ∃ (initial_employees final_employees : ℕ),
    (initial_employees : ℚ) * initial_female_percentage = 
      (final_employees : ℚ) * final_female_percentage ∧
    final_employees = initial_employees + additional_male_workers ∧
    final_employees = 264

-- The proof of the theorem
theorem company_workforce_proof : 
  company_workforce (60 / 100) 22 (55 / 100) :=
by
  sorry

end company_workforce_company_workforce_proof_l2810_281022


namespace log_three_five_l2810_281099

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_three_five (a : ℝ) (h : log 5 45 = a) : log 5 3 = (a - 1) / 2 := by
  sorry

end log_three_five_l2810_281099


namespace range_of_a_l2810_281069

def A (a : ℝ) : Set ℝ := {x | (a * x - 1) / (x - a) < 0}

theorem range_of_a : ∀ a : ℝ, 
  (2 ∈ A a ∧ 3 ∉ A a) ↔ 
  (a ∈ Set.Icc (1/3 : ℝ) (1/2 : ℝ) ∪ Set.Ioc (2 : ℝ) (3 : ℝ)) :=
by sorry

end range_of_a_l2810_281069


namespace base_subtraction_proof_l2810_281073

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

theorem base_subtraction_proof :
  let base5_num := [4, 2, 3]  -- 324 in base 5 (least significant digit first)
  let base6_num := [3, 5, 1]  -- 153 in base 6 (least significant digit first)
  toBase10 base5_num 5 - toBase10 base6_num 6 = 20 := by
  sorry

end base_subtraction_proof_l2810_281073


namespace duck_cow_problem_l2810_281009

/-- Proves that in a group of ducks and cows, if the total number of legs is 28 more than twice the number of heads, then the number of cows is 14. -/
theorem duck_cow_problem (ducks cows : ℕ) : 
  (2 * ducks + 4 * cows = 2 * (ducks + cows) + 28) → cows = 14 := by
  sorry


end duck_cow_problem_l2810_281009


namespace close_interval_is_zero_one_l2810_281004

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + x + 2
def g (x : ℝ) : ℝ := 2*x + 1

-- Define what it means for two functions to be "close"
def are_close (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

-- State the theorem
theorem close_interval_is_zero_one :
  are_close f g 0 1 ∧
  ∀ a b, a < 0 ∨ b > 1 → ¬(are_close f g a b) :=
sorry

end close_interval_is_zero_one_l2810_281004


namespace triathlon_bike_speed_l2810_281043

/-- Triathlon problem -/
theorem triathlon_bike_speed 
  (total_time : ℝ) 
  (swim_speed swim_distance : ℝ) 
  (run_speed run_distance : ℝ) 
  (bike_distance : ℝ) :
  total_time = 2.5 →
  swim_speed = 2 →
  swim_distance = 0.25 →
  run_speed = 5 →
  run_distance = 3 →
  bike_distance = 20 →
  ∃ bike_speed : ℝ, 
    (swim_distance / swim_speed + run_distance / run_speed + bike_distance / bike_speed = total_time) ∧
    (abs (bike_speed - 11.27) < 0.01) := by
  sorry

end triathlon_bike_speed_l2810_281043


namespace tucker_bought_three_boxes_l2810_281063

def tissues_per_box : ℕ := 160
def used_tissues : ℕ := 210
def remaining_tissues : ℕ := 270

theorem tucker_bought_three_boxes :
  (used_tissues + remaining_tissues) / tissues_per_box = 3 :=
by sorry

end tucker_bought_three_boxes_l2810_281063


namespace complex_sixth_power_sum_l2810_281020

theorem complex_sixth_power_sum : 
  (((-1 + Complex.I * Real.sqrt 3) / 2) ^ 6 + 
   ((-1 - Complex.I * Real.sqrt 3) / 2) ^ 6) = 2 := by
  sorry

end complex_sixth_power_sum_l2810_281020


namespace red_shirt_pairs_l2810_281064

theorem red_shirt_pairs (green_students : ℕ) (red_students : ℕ) (total_students : ℕ) 
  (total_pairs : ℕ) (green_green_pairs : ℕ) :
  green_students = 63 →
  red_students = 69 →
  total_students = 132 →
  total_pairs = 66 →
  green_green_pairs = 27 →
  ∃ red_red_pairs : ℕ, red_red_pairs = 30 ∧ 
    red_red_pairs = total_pairs - green_green_pairs - (green_students - 2 * green_green_pairs) :=
by sorry

end red_shirt_pairs_l2810_281064


namespace distance_between_points_l2810_281018

def point1 : ℝ × ℝ := (3, 7)
def point2 : ℝ × ℝ := (-5, 2)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 89 := by
  sorry

end distance_between_points_l2810_281018


namespace expression_simplification_l2810_281049

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  ((x + 3) / x - 1) / ((x^2 - 1) / (x^2 + x)) = Real.sqrt 3 := by
  sorry

end expression_simplification_l2810_281049


namespace dividingChordLength_l2810_281038

/-- A hexagon inscribed in a circle with alternating side lengths -/
structure AlternatingHexagon where
  /-- The length of three consecutive sides -/
  shortSide : ℝ
  /-- The length of the other three consecutive sides -/
  longSide : ℝ
  /-- The short sides are indeed shorter than the long sides -/
  shortLessThanLong : shortSide < longSide

/-- The chord dividing the hexagon into two trapezoids -/
def dividingChord (h : AlternatingHexagon) : ℝ := sorry

theorem dividingChordLength (h : AlternatingHexagon) 
  (h_short : h.shortSide = 4)
  (h_long : h.longSide = 6) :
  dividingChord h = 480 / 49 := by
  sorry

end dividingChordLength_l2810_281038


namespace seat_distribution_correct_l2810_281029

/-- Represents the total number of seats on the airplane -/
def total_seats : ℕ := 90

/-- Represents the number of seats in First Class -/
def first_class_seats : ℕ := 36

/-- Represents the proportion of seats in Business Class -/
def business_class_proportion : ℚ := 1/5

/-- Represents the proportion of seats in Premium Economy -/
def premium_economy_proportion : ℚ := 2/5

/-- Theorem stating that the given seat distribution is correct -/
theorem seat_distribution_correct : 
  first_class_seats + 
  (business_class_proportion * total_seats).floor + 
  (premium_economy_proportion * total_seats).floor + 
  (total_seats - first_class_seats - 
   (business_class_proportion * total_seats).floor - 
   (premium_economy_proportion * total_seats).floor) = total_seats :=
by sorry

end seat_distribution_correct_l2810_281029


namespace remaining_money_l2810_281088

def initialAmount : ℚ := 7.10
def spentOnSweets : ℚ := 1.05
def givenToFriend : ℚ := 1.00
def numberOfFriends : ℕ := 2

theorem remaining_money :
  initialAmount - (spentOnSweets + givenToFriend * numberOfFriends) = 4.05 := by
  sorry

end remaining_money_l2810_281088


namespace problem_statement_l2810_281033

theorem problem_statement :
  (∀ n : ℕ, n > 1 → ¬(n ∣ (2^n - 1))) ∧
  (∀ n : ℕ, Nat.Prime n → (n^2 ∣ (2^n + 1)) → n = 3) := by
  sorry

end problem_statement_l2810_281033


namespace one_face_colored_cubes_125_l2810_281001

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  edge_length : ℕ
  num_colors : ℕ

/-- Calculates the number of small cubes with only one face colored -/
def one_face_colored_cubes (c : CutCube) : ℕ :=
  (c.edge_length - 2)^2 * c.num_colors

/-- Theorem: A cube cut into 125 smaller cubes with 6 different colored faces has 54 small cubes with only one face colored -/
theorem one_face_colored_cubes_125 :
  ∀ c : CutCube, c.edge_length = 5 → c.num_colors = 6 → one_face_colored_cubes c = 54 :=
by
  sorry

end one_face_colored_cubes_125_l2810_281001


namespace cube_division_l2810_281032

theorem cube_division (n : ℕ) (h1 : n ≥ 6) (h2 : Even n) :
  ∃ (m : ℕ), m^3 = (3 * n * (n - 2)) / 4 + 2 :=
sorry

end cube_division_l2810_281032


namespace symmetry_line_theorem_l2810_281035

/-- Circle represented by its equation -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- Line represented by its equation -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Define Circle O -/
def circle_O : Circle :=
  { equation := λ x y => x^2 + y^2 = 4 }

/-- Define Circle C -/
def circle_C : Circle :=
  { equation := λ x y => x^2 + y^2 + 4*x - 4*y + 4 = 0 }

/-- Define the line of symmetry -/
def line_of_symmetry : Line :=
  { equation := λ x y => x - y + 2 = 0 }

/-- Function to check if a line is the line of symmetry between two circles -/
def is_line_of_symmetry (l : Line) (c1 c2 : Circle) : Prop :=
  sorry -- Definition of symmetry between circles with respect to a line

/-- Theorem stating that the given line is the line of symmetry between Circle O and Circle C -/
theorem symmetry_line_theorem :
  is_line_of_symmetry line_of_symmetry circle_O circle_C := by
  sorry

end symmetry_line_theorem_l2810_281035


namespace no_valid_license_plate_divisible_by_8_l2810_281012

/-- Represents a 4-digit number of the form aaab -/
structure LicensePlate where
  a : Nat
  b : Nat
  h1 : a < 10
  h2 : b < 10

/-- Checks if a number is prime -/
def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

/-- The theorem to be proved -/
theorem no_valid_license_plate_divisible_by_8 :
  ¬∃ (plate : LicensePlate),
    (∀ (child_age : Nat), child_age ≥ 1 → child_age ≤ 10 → (1000 * plate.a + 100 * plate.a + 10 * plate.a + plate.b) % child_age = 0) ∧
    isPrime (10 * plate.a + plate.b) ∧
    (1000 * plate.a + 100 * plate.a + 10 * plate.a + plate.b) % 8 = 0 :=
by sorry


end no_valid_license_plate_divisible_by_8_l2810_281012


namespace quartic_sum_to_quadratic_sum_l2810_281061

theorem quartic_sum_to_quadratic_sum (x : ℝ) (h : 45 = x^4 + 1/x^4) : 
  x^2 + 1/x^2 = Real.sqrt 47 := by
  sorry

end quartic_sum_to_quadratic_sum_l2810_281061


namespace floor_product_equals_twelve_l2810_281087

theorem floor_product_equals_twelve (x : ℝ) : 
  ⌊x * ⌊x / 2⌋⌋ = 12 ↔ x ≥ 4.9 ∧ x < 5.1 := by sorry

end floor_product_equals_twelve_l2810_281087


namespace probability_red_then_white_l2810_281051

theorem probability_red_then_white (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ)
  (h_total : total_balls = 9)
  (h_red : red_balls = 3)
  (h_white : white_balls = 2)
  : (red_balls : ℚ) / total_balls * (white_balls : ℚ) / total_balls = 2 / 27 := by
  sorry

end probability_red_then_white_l2810_281051


namespace other_root_of_complex_quadratic_l2810_281039

theorem other_root_of_complex_quadratic (z : ℂ) :
  z^2 = -75 + 40*I ∧ (5 + 7*I)^2 = -75 + 40*I →
  (-5 - 7*I)^2 = -75 + 40*I :=
by sorry

end other_root_of_complex_quadratic_l2810_281039


namespace ending_number_of_range_l2810_281047

theorem ending_number_of_range (n : ℕ) (h1 : n = 10) (h2 : ∀ k ∈ Finset.range n, 15 + 5 * k ∣ 5) :
  15 + 5 * (n - 1) = 60 :=
sorry

end ending_number_of_range_l2810_281047


namespace sum_of_odd_and_multiples_of_three_l2810_281071

/-- The number of four-digit odd numbers -/
def A : ℕ := 4500

/-- The number of four-digit multiples of 3 -/
def B : ℕ := 3000

/-- The sum of four-digit odd numbers and four-digit multiples of 3 is 7500 -/
theorem sum_of_odd_and_multiples_of_three : A + B = 7500 := by sorry

end sum_of_odd_and_multiples_of_three_l2810_281071


namespace calculate_expression_l2810_281075

theorem calculate_expression : (-8) * 3 / ((-2)^2) = -6 := by
  sorry

end calculate_expression_l2810_281075


namespace right_triangle_congruence_l2810_281054

-- Define a right-angled triangle
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

-- Define congruence for right-angled triangles
def congruent (t1 t2 : RightTriangle) : Prop :=
  t1.leg1 = t2.leg1 ∧ t1.leg2 = t2.leg2 ∧ t1.hypotenuse = t2.hypotenuse

-- Theorem: Two right-angled triangles with two equal legs are congruent
theorem right_triangle_congruence (t1 t2 : RightTriangle) 
  (h : t1.leg1 = t2.leg1 ∧ t1.leg2 = t2.leg2) : congruent t1 t2 := by
  sorry

end right_triangle_congruence_l2810_281054


namespace smallest_g_is_correct_l2810_281034

/-- The smallest positive integer g such that 3150 * g is a perfect square -/
def smallest_g : ℕ := 14

/-- 3150 * g is a perfect square -/
def is_perfect_square (g : ℕ) : Prop :=
  ∃ n : ℕ, 3150 * g = n^2

theorem smallest_g_is_correct :
  (is_perfect_square smallest_g) ∧
  (∀ g : ℕ, 0 < g ∧ g < smallest_g → ¬(is_perfect_square g)) :=
by sorry

end smallest_g_is_correct_l2810_281034


namespace total_plants_grown_l2810_281059

def eggplants_per_packet : ℕ := 14
def sunflowers_per_packet : ℕ := 10
def tomatoes_per_packet : ℕ := 16
def peas_per_packet : ℕ := 20

def eggplant_packets : ℕ := 4
def sunflower_packets : ℕ := 6
def tomato_packets : ℕ := 5
def pea_packets : ℕ := 7

def spring_growth_rate : ℚ := 7/10
def summer_growth_rate : ℚ := 4/5

theorem total_plants_grown (
  eggplants_per_packet sunflowers_per_packet tomatoes_per_packet peas_per_packet : ℕ)
  (eggplant_packets sunflower_packets tomato_packets pea_packets : ℕ)
  (spring_growth_rate summer_growth_rate : ℚ) :
  ⌊(eggplants_per_packet * eggplant_packets : ℚ) * spring_growth_rate⌋ +
  ⌊(peas_per_packet * pea_packets : ℚ) * spring_growth_rate⌋ +
  ⌊(sunflowers_per_packet * sunflower_packets : ℚ) * summer_growth_rate⌋ +
  ⌊(tomatoes_per_packet * tomato_packets : ℚ) * summer_growth_rate⌋ = 249 :=
by sorry

end total_plants_grown_l2810_281059


namespace arithmetic_progression_x_value_l2810_281079

theorem arithmetic_progression_x_value (x : ℝ) : 
  let a₁ := 2 * x - 4
  let a₂ := 3 * x + 2
  let a₃ := 5 * x - 1
  (a₂ - a₁ = a₃ - a₂) → x = 9 := by
sorry

end arithmetic_progression_x_value_l2810_281079


namespace tom_toy_cost_proof_l2810_281000

def tom_toy_cost (initial_money : ℕ) (game_cost : ℕ) (num_toys : ℕ) : ℕ :=
  (initial_money - game_cost) / num_toys

theorem tom_toy_cost_proof (initial_money : ℕ) (game_cost : ℕ) (num_toys : ℕ) 
  (h1 : initial_money = 57)
  (h2 : game_cost = 49)
  (h3 : num_toys = 2)
  (h4 : initial_money > game_cost) :
  tom_toy_cost initial_money game_cost num_toys = 4 := by
  sorry

end tom_toy_cost_proof_l2810_281000


namespace point_M_properties_segment_MN_length_l2810_281070

def M (m : ℝ) : ℝ × ℝ := (2*m + 1, m + 3)

theorem point_M_properties (m : ℝ) :
  (M m).1 > 0 ∧ (M m).2 > 0 ∧  -- M is in the first quadrant
  (M m).2 = 2 * (M m).1  -- distance to x-axis is twice distance to y-axis
  → m = 1/3 := by sorry

def N : ℝ × ℝ := (2, 1)

theorem segment_MN_length (m : ℝ) :
  (M m).2 = N.2  -- MN is parallel to x-axis
  → |N.1 - (M m).1| = 5 := by sorry

end point_M_properties_segment_MN_length_l2810_281070


namespace sum_of_2010_3_array_remainder_of_sum_l2810_281046

/-- Definition of the sum of a pq-array --/
def pq_array_sum (p q : ℕ) : ℚ :=
  (1 / (1 - 1 / (2 * p))) * (1 / (1 - 1 / q))

/-- Theorem stating the sum of the specific 1/2010,3-array --/
theorem sum_of_2010_3_array :
  pq_array_sum 2010 3 = 6030 / 4019 := by
  sorry

/-- Theorem for the remainder when numerator + denominator is divided by 2010 --/
theorem remainder_of_sum :
  (6030 + 4019) % 2010 = 1009 := by
  sorry

end sum_of_2010_3_array_remainder_of_sum_l2810_281046


namespace quadratic_properties_l2810_281094

-- Define a quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties :
  ∀ (a b c : ℝ),
  (∃ (x_min : ℝ), ∀ (x : ℝ), quadratic a b c x ≥ quadratic a b c x_min ∧ quadratic a b c x_min = 1) →
  quadratic a b c 0 = 3 →
  quadratic a b c 2 = 3 →
  (a = 2 ∧ b = -4 ∧ c = 3) ∧
  (∀ (a_range : ℝ), (∃ (x y : ℝ), 2 * a_range ≤ x ∧ x < y ∧ y ≤ a_range + 1 ∧
    (quadratic 2 (-4) 3 x < quadratic 2 (-4) 3 y ∧ quadratic 2 (-4) 3 y > quadratic 2 (-4) 3 (a_range + 1))) ↔
    (0 < a_range ∧ a_range < 1)) :=
by sorry

end quadratic_properties_l2810_281094


namespace product_remainder_mod_three_l2810_281036

theorem product_remainder_mod_three (a b : ℕ) : 
  a % 3 = 1 → b % 3 = 2 → (a * b) % 3 = 2 := by
  sorry

end product_remainder_mod_three_l2810_281036


namespace todds_initial_gum_l2810_281074

theorem todds_initial_gum (x : ℕ) : x + 16 = 54 → x = 38 := by
  sorry

end todds_initial_gum_l2810_281074


namespace diamond_equation_solution_l2810_281002

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := 4 * a + 2 * b

-- State the theorem
theorem diamond_equation_solution :
  ∃ x : ℚ, diamond 3 (diamond x 7) = 5 ∧ x = -35/8 := by
  sorry

end diamond_equation_solution_l2810_281002


namespace q_polynomial_form_l2810_281028

theorem q_polynomial_form (q : ℝ → ℝ) :
  (∀ x, q x + (2*x^6 + 4*x^4 + 10*x^2) = (5*x^4 + 15*x^3 + 30*x^2 + 10*x + 10)) →
  (∀ x, q x = -2*x^6 + x^4 + 15*x^3 + 20*x^2 + 10*x + 10) := by
sorry

end q_polynomial_form_l2810_281028


namespace smallest_six_digit_divisible_by_3_7_13_l2810_281066

theorem smallest_six_digit_divisible_by_3_7_13 : ∀ n : ℕ,
  100000 ≤ n ∧ n < 1000000 ∧ n % 3 = 0 ∧ n % 7 = 0 ∧ n % 13 = 0 →
  100191 ≤ n :=
by sorry

end smallest_six_digit_divisible_by_3_7_13_l2810_281066


namespace binomial_8_5_l2810_281096

theorem binomial_8_5 : Nat.choose 8 5 = 56 := by
  sorry

end binomial_8_5_l2810_281096


namespace experiment_sequences_l2810_281077

/-- Represents the number of procedures in the experiment -/
def num_procedures : ℕ := 5

/-- Represents the condition that procedure A can only be first or last -/
def a_first_or_last : ℕ := 2

/-- Represents the number of ways to arrange C and D adjacently -/
def cd_adjacent : ℕ := 2

/-- Represents the number of ways to arrange the remaining procedures -/
def remaining_arrangements : ℕ := 3

/-- The total number of possible sequences for the experiment -/
def total_sequences : ℕ := a_first_or_last * cd_adjacent * remaining_arrangements.factorial

theorem experiment_sequences :
  total_sequences = 24 := by sorry

end experiment_sequences_l2810_281077


namespace ones_digit_of_31_power_l2810_281092

theorem ones_digit_of_31_power (n : ℕ) : (31^(15 * 7^7) : ℕ) % 10 = 3 := by
  sorry

end ones_digit_of_31_power_l2810_281092


namespace diamond_two_three_l2810_281006

def diamond (a b : ℝ) : ℝ := a^3 * b^2 - b + 2

theorem diamond_two_three : diamond 2 3 = 71 := by sorry

end diamond_two_three_l2810_281006


namespace ellipse_eccentricity_l2810_281056

/-- The eccentricity of an ellipse with equation x²/m² + y²/9 = 1 (m > 0) and one focus at (4, 0) is 4/5 -/
theorem ellipse_eccentricity (m : ℝ) (h1 : m > 0) : 
  let ellipse := { (x, y) : ℝ × ℝ | x^2 / m^2 + y^2 / 9 = 1 }
  let focus : ℝ × ℝ := (4, 0)
  focus ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 16 ∧ p ∈ ellipse } →
  (∃ (a b c : ℝ), a > b ∧ b > 0 ∧ c > 0 ∧ a^2 = m^2 ∧ b^2 = 9 ∧ c^2 = a^2 - b^2 ∧ c / a = 4 / 5) :=
by sorry

end ellipse_eccentricity_l2810_281056
