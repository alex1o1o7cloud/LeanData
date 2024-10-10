import Mathlib

namespace sum_of_fractions_equals_two_ninths_l2629_262984

theorem sum_of_fractions_equals_two_ninths :
  let sum := (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + (1 / (5 * 6 : ℚ)) +
              (1 / (6 * 7 : ℚ)) + (1 / (7 * 8 : ℚ)) + (1 / (8 * 9 : ℚ))
  sum = 2 / 9 := by
    sorry

end sum_of_fractions_equals_two_ninths_l2629_262984


namespace trees_around_square_theorem_l2629_262939

/-- Represents a rectangle with trees planted along its sides -/
structure TreeRectangle where
  side_ad : ℕ  -- Number of trees along side AD
  side_ab : ℕ  -- Number of trees along side AB

/-- Calculates the number of trees around a square with side length equal to the longer side of the rectangle -/
def trees_around_square (rect : TreeRectangle) : ℕ :=
  4 * (rect.side_ad - 1) + 4

/-- Theorem stating that for a rectangle with 49 trees along AD and 25 along AB,
    the number of trees around the corresponding square is 196 -/
theorem trees_around_square_theorem (rect : TreeRectangle) 
        (h1 : rect.side_ad = 49) (h2 : rect.side_ab = 25) : 
        trees_around_square rect = 196 := by
  sorry

#eval trees_around_square ⟨49, 25⟩

end trees_around_square_theorem_l2629_262939


namespace fourth_root_784_times_cube_root_512_l2629_262942

theorem fourth_root_784_times_cube_root_512 : 
  (784 : ℝ) ^ (1/4) * (512 : ℝ) ^ (1/3) = 16 * Real.sqrt 7 :=
by sorry

end fourth_root_784_times_cube_root_512_l2629_262942


namespace triangle_area_l2629_262948

theorem triangle_area (a c B : ℝ) (h1 : a = 1) (h2 : c = 2) (h3 : B = Real.pi / 3) :
  (1/2) * a * c * Real.sin B = Real.sqrt 3 / 2 := by
  sorry

end triangle_area_l2629_262948


namespace product_of_roots_l2629_262983

theorem product_of_roots (x : ℝ) : (x + 2) * (x - 3) = -10 → 
  ∃ (r₁ r₂ : ℝ), r₁ * r₂ = 4 ∧ (x = r₁ ∨ x = r₂) := by
  sorry

end product_of_roots_l2629_262983


namespace largest_n_binomial_sum_exists_largest_n_binomial_sum_largest_n_is_seven_l2629_262935

theorem largest_n_binomial_sum (n : ℕ) : 
  (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 n) → n ≤ 7 :=
by sorry

theorem exists_largest_n_binomial_sum : 
  ∃ n : ℕ, (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 n) ∧ 
  (∀ m : ℕ, (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 m) → m ≤ n) :=
by sorry

theorem largest_n_is_seven : 
  (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 7) ∧
  (∀ m : ℕ, (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 m) → m ≤ 7) :=
by sorry

end largest_n_binomial_sum_exists_largest_n_binomial_sum_largest_n_is_seven_l2629_262935


namespace probability_of_U_in_SHUXUE_l2629_262978

def pinyin : String := "SHUXUE"

theorem probability_of_U_in_SHUXUE : 
  (pinyin.toList.filter (· = 'U')).length / pinyin.length = 1 / 3 := by
  sorry

end probability_of_U_in_SHUXUE_l2629_262978


namespace kitten_puppy_difference_l2629_262994

theorem kitten_puppy_difference (kittens puppies : ℕ) : 
  kittens = 78 → puppies = 32 → kittens - 2 * puppies = 14 :=
by
  sorry

end kitten_puppy_difference_l2629_262994


namespace water_needed_for_mixture_l2629_262997

/-- Given a mixture of nutrient concentrate and water, calculate the amount of water needed to prepare a larger volume of the same mixture. -/
theorem water_needed_for_mixture (concentrate : ℝ) (initial_water : ℝ) (total_desired : ℝ) : 
  concentrate = 0.05 → 
  initial_water = 0.03 → 
  total_desired = 0.72 → 
  (initial_water / (concentrate + initial_water)) * total_desired = 0.27 := by
sorry

end water_needed_for_mixture_l2629_262997


namespace geometric_sequence_common_ratio_l2629_262937

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- The common ratio
  h : ∀ n, a (n + 1) = q * a n

/-- 
Given a geometric sequence where the third term is equal to twice 
the sum of the first two terms plus 1, and the fourth term is equal 
to twice the sum of the first three terms plus 1, prove that the 
common ratio is 3.
-/
theorem geometric_sequence_common_ratio 
  (seq : GeometricSequence) 
  (h₁ : seq.a 3 = 2 * (seq.a 1 + seq.a 2) + 1)
  (h₂ : seq.a 4 = 2 * (seq.a 1 + seq.a 2 + seq.a 3) + 1) : 
  seq.q = 3 := by
  sorry

end geometric_sequence_common_ratio_l2629_262937


namespace expression_evaluation_l2629_262927

-- Define the ceiling function
def ceiling (x : ℚ) : ℤ := Int.ceil x

-- Define the main expression
def main_expression : ℚ :=
  (ceiling ((21 : ℚ) / 5 - ceiling ((35 : ℚ) / 23))) /
  (ceiling ((35 : ℚ) / 5 + ceiling ((5 * 23 : ℚ) / 35)))

-- Theorem statement
theorem expression_evaluation :
  main_expression = 3 / 11 := by sorry

end expression_evaluation_l2629_262927


namespace quadratic_roots_condition_l2629_262931

/-- The quadratic equation (r-4)x^2 - 2(r-3)x + r = 0 has two distinct roots, both greater than -1,
    if and only if 3.5 < r < 4.5 -/
theorem quadratic_roots_condition (r : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > -1 ∧ x₂ > -1 ∧
    (r - 4) * x₁^2 - 2*(r - 3) * x₁ + r = 0 ∧
    (r - 4) * x₂^2 - 2*(r - 3) * x₂ + r = 0) ↔
  (3.5 < r ∧ r < 4.5) :=
sorry

end quadratic_roots_condition_l2629_262931


namespace base9_multiplication_l2629_262982

/-- Converts a base-9 number to base-10 --/
def base9ToBase10 (n : ℕ) : ℕ :=
  let digits := n.digits 9
  (List.range digits.length).foldl (fun acc i => acc + digits[i]! * 9^i) 0

/-- Converts a base-10 number to base-9 --/
def base10ToBase9 (n : ℕ) : ℕ :=
  n.digits 9 |> List.reverse |> List.foldl (fun acc d => acc * 10 + d) 0

/-- Multiplication in base-9 --/
def multBase9 (a b : ℕ) : ℕ :=
  base10ToBase9 (base9ToBase10 a * base9ToBase10 b)

theorem base9_multiplication :
  multBase9 327 6 = 2226 := by sorry

end base9_multiplication_l2629_262982


namespace algebraic_expression_value_l2629_262903

/-- Given that when x = 1, the value of (1/2)ax³ - 3bx + 4 is 9,
    prove that when x = -1, the value of the expression is -1 -/
theorem algebraic_expression_value (a b : ℝ) :
  (1/2 * a * 1^3 - 3 * b * 1 + 4 = 9) →
  (1/2 * a * (-1)^3 - 3 * b * (-1) + 4 = -1) :=
by sorry

end algebraic_expression_value_l2629_262903


namespace remaining_money_l2629_262952

def octal_to_decimal (n : ℕ) : ℕ := sorry

def john_savings : ℕ := 5555

def airline_cost : ℕ := 1200
def lodging_cost : ℕ := 800
def food_cost : ℕ := 400

def total_expenses : ℕ := airline_cost + lodging_cost + food_cost

theorem remaining_money :
  octal_to_decimal john_savings - total_expenses = 525 := by sorry

end remaining_money_l2629_262952


namespace simplify_product_l2629_262988

theorem simplify_product : 8 * (15 / 4) * (-24 / 25) = -144 / 5 := by
  sorry

end simplify_product_l2629_262988


namespace remainder_of_binary_div_8_l2629_262910

def binary_number : ℕ := 0b1110101101101

theorem remainder_of_binary_div_8 :
  binary_number % 8 = 5 := by sorry

end remainder_of_binary_div_8_l2629_262910


namespace catering_pies_l2629_262934

theorem catering_pies (total_pies : ℕ) (num_teams : ℕ) (first_team_pies : ℕ) (third_team_pies : ℕ) 
  (h1 : total_pies = 750)
  (h2 : num_teams = 3)
  (h3 : first_team_pies = 235)
  (h4 : third_team_pies = 240) :
  total_pies - first_team_pies - third_team_pies = 275 := by
  sorry

end catering_pies_l2629_262934


namespace black_hole_convergence_l2629_262973

/-- Counts the number of even digits in a natural number -/
def countEvenDigits (n : ℕ) : ℕ := sorry

/-- Counts the number of odd digits in a natural number -/
def countOddDigits (n : ℕ) : ℕ := sorry

/-- Counts the total number of digits in a natural number -/
def countTotalDigits (n : ℕ) : ℕ := sorry

/-- Applies the transformation rule to a natural number -/
def transform (n : ℕ) : ℕ :=
  100 * (countEvenDigits n) + 10 * (countOddDigits n) + (countTotalDigits n)

/-- The black hole number -/
def blackHoleNumber : ℕ := 123

/-- Theorem stating that repeated application of the transformation 
    will always result in the black hole number -/
theorem black_hole_convergence (n : ℕ) : 
  ∃ k : ℕ, ∀ m : ℕ, m ≥ k → (transform^[m] n = blackHoleNumber) :=
sorry

end black_hole_convergence_l2629_262973


namespace other_asymptote_equation_l2629_262986

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  /-- The equation of one asymptote -/
  asymptote1 : ℝ → ℝ
  /-- The x-coordinate of the foci -/
  foci_x : ℝ
  /-- Condition that the first asymptote has equation y = 2x + 1 -/
  asymptote1_eq : ∀ x, asymptote1 x = 2 * x + 1
  /-- Condition that the foci have x-coordinate 4 -/
  foci_x_eq : foci_x = 4

/-- The theorem stating the equation of the other asymptote -/
theorem other_asymptote_equation (h : Hyperbola) :
  ∃ (f : ℝ → ℝ), (∀ x, f x = -2 * x + 17) ∧ 
  (∀ x y, y = f x ↔ y = h.asymptote1 x ∨ y = -2 * x + 17) :=
sorry

end other_asymptote_equation_l2629_262986


namespace sin_2alpha_value_l2629_262972

theorem sin_2alpha_value (α : ℝ) (h : Real.sin α + Real.cos α = 1/5) : 
  Real.sin (2 * α) = -24/25 := by
  sorry

end sin_2alpha_value_l2629_262972


namespace ashley_exam_marks_l2629_262995

theorem ashley_exam_marks (marks_secured : ℕ) (percentage : ℚ) (max_marks : ℕ) : 
  marks_secured = 332 → percentage = 83/100 → 
  (marks_secured : ℚ) / (max_marks : ℚ) = percentage →
  max_marks = 400 := by
sorry

end ashley_exam_marks_l2629_262995


namespace function_composition_ratio_l2629_262946

def f (x : ℝ) : ℝ := 3 * x + 4

def g (x : ℝ) : ℝ := 4 * x - 3

theorem function_composition_ratio :
  (f (g (f 3))) / (g (f (g 3))) = 151 / 121 := by
  sorry

end function_composition_ratio_l2629_262946


namespace smallest_b_value_l2629_262900

theorem smallest_b_value (a b : ℝ) : 
  (2 < a ∧ a < b) →
  (2 + a ≤ b) →
  (1 / a + 1 / b ≤ 2) →
  ∀ ε > 0, ∃ b₀ : ℝ, 2 < b₀ ∧ b₀ < 2 + ε ∧
    ∃ a₀ : ℝ, 2 < a₀ ∧ a₀ < b₀ ∧
    (2 + a₀ ≤ b₀) ∧
    (1 / a₀ + 1 / b₀ ≤ 2) :=
by sorry

end smallest_b_value_l2629_262900


namespace power_product_cube_l2629_262977

theorem power_product_cube (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end power_product_cube_l2629_262977


namespace zero_not_in_range_of_f_l2629_262902

noncomputable def f (x : ℝ) : ℤ :=
  if x > 0 then Int.ceil (1 / (x + 1))
  else if x < 0 then Int.ceil (1 / (x - 1))
  else 0  -- This value doesn't matter as we exclude x = 0

theorem zero_not_in_range_of_f :
  ∀ x : ℝ, x ≠ 0 → f x ≠ 0 :=
by sorry

end zero_not_in_range_of_f_l2629_262902


namespace quadratic_equation_solution_l2629_262930

theorem quadratic_equation_solution (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 8 * x + 15 = 0 ↔ x = 3 ∨ x = 5) → k = 1 := by
  sorry

end quadratic_equation_solution_l2629_262930


namespace rope_cutting_l2629_262971

theorem rope_cutting (total_length : ℝ) (piece_length : ℝ) 
  (h1 : total_length = 20) 
  (h2 : piece_length = 3.8) : 
  (∃ (num_pieces : ℕ) (remaining : ℝ), 
    num_pieces = 5 ∧ 
    remaining = 1 ∧ 
    (↑num_pieces : ℝ) * piece_length + remaining = total_length ∧ 
    remaining < piece_length) := by
  sorry

end rope_cutting_l2629_262971


namespace probability_edge_endpoints_is_correct_l2629_262923

structure RegularIcosahedron where
  vertices : Finset (Fin 12)
  edges : Finset (Fin 12 × Fin 12)
  vertex_degree : ∀ v : Fin 12, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 5

def probability_edge_endpoints (I : RegularIcosahedron) : ℚ :=
  5 / 11

theorem probability_edge_endpoints_is_correct (I : RegularIcosahedron) :
  probability_edge_endpoints I = 5 / 11 := by
  sorry

end probability_edge_endpoints_is_correct_l2629_262923


namespace equipment_production_l2629_262918

theorem equipment_production (total : ℕ) (sample_size : ℕ) (sample_A : ℕ) (products_B : ℕ) : 
  total = 4800 → 
  sample_size = 80 → 
  sample_A = 50 → 
  products_B = total - (total * sample_A / sample_size) →
  products_B = 1800 := by
sorry

end equipment_production_l2629_262918


namespace scores_mode_and_median_l2629_262919

def scores : List ℕ := [97, 88, 85, 93, 85]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℕ := sorry

theorem scores_mode_and_median :
  mode scores = 85 ∧ median scores = 88 := by sorry

end scores_mode_and_median_l2629_262919


namespace probability_not_all_same_dice_l2629_262907

def num_sides : ℕ := 8
def num_dice : ℕ := 5

theorem probability_not_all_same_dice :
  1 - (num_sides : ℚ) / (num_sides ^ num_dice) = 4095 / 4096 := by
  sorry

end probability_not_all_same_dice_l2629_262907


namespace sam_not_buying_book_probability_l2629_262979

theorem sam_not_buying_book_probability (p : ℚ) 
  (h : p = 5 / 8) : 1 - p = 3 / 8 := by
  sorry

end sam_not_buying_book_probability_l2629_262979


namespace subject_choice_theorem_l2629_262904

/-- The number of subjects available --/
def num_subjects : ℕ := 7

/-- The number of subjects each student must choose --/
def subjects_to_choose : ℕ := 3

/-- The number of ways Student A can choose subjects --/
def ways_for_A : ℕ := Nat.choose (num_subjects - 1) (subjects_to_choose - 1)

/-- The probability that both Students B and C choose physics --/
def prob_B_and_C_physics : ℚ := 
  (Nat.choose (num_subjects - 1) (subjects_to_choose - 1) ^ 2 : ℚ) / 
  (Nat.choose num_subjects subjects_to_choose ^ 2 : ℚ)

theorem subject_choice_theorem : 
  ways_for_A = 15 ∧ prob_B_and_C_physics = 9 / 49 := by sorry

end subject_choice_theorem_l2629_262904


namespace gcd_12m_18n_min_l2629_262991

/-- For positive integers m and n with gcd(m, n) = 10, the smallest possible value of gcd(12m, 18n) is 60 -/
theorem gcd_12m_18n_min (m n : ℕ+) (h : Nat.gcd m.val n.val = 10) :
  ∃ (k : ℕ+), (∀ (a b : ℕ+), Nat.gcd (12 * a.val) (18 * b.val) ≥ k.val) ∧
              (Nat.gcd (12 * m.val) (18 * n.val) = k.val) ∧
              k = 60 := by
  sorry

end gcd_12m_18n_min_l2629_262991


namespace geometric_sequence_product_l2629_262965

theorem geometric_sequence_product (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∀ n, ∃ r, a (n + 1) = r * a n) →
  (a 1)^2 - 10*(a 1) + 16 = 0 →
  (a 19)^2 - 10*(a 19) + 16 = 0 →
  a 8 * a 10 * a 12 = 64 := by
sorry

end geometric_sequence_product_l2629_262965


namespace binomial_product_l2629_262945

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binomial_product_l2629_262945


namespace touchdown_points_l2629_262956

theorem touchdown_points (total_points : ℕ) (num_touchdowns : ℕ) (points_per_touchdown : ℕ) :
  total_points = 21 →
  num_touchdowns = 3 →
  total_points = num_touchdowns * points_per_touchdown →
  points_per_touchdown = 7 := by
  sorry

end touchdown_points_l2629_262956


namespace tree_survival_probability_l2629_262929

/-- Probability that at least one tree survives after transplantation -/
theorem tree_survival_probability :
  let survival_rate_A : ℚ := 5/6
  let survival_rate_B : ℚ := 4/5
  let num_trees_A : ℕ := 2
  let num_trees_B : ℕ := 2
  -- Probability that at least one tree survives
  1 - (1 - survival_rate_A) ^ num_trees_A * (1 - survival_rate_B) ^ num_trees_B = 899/900 :=
by
  sorry

end tree_survival_probability_l2629_262929


namespace decreasing_implies_a_le_10_l2629_262924

/-- A quadratic function f(x) = x^2 + 2(a-5)x - 6 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-5)*x - 6

/-- The function f is decreasing on the interval (-∞, -5] -/
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x ≤ y → y ≤ -5 → f a x ≥ f a y

theorem decreasing_implies_a_le_10 (a : ℝ) :
  is_decreasing_on_interval a → a ≤ 10 := by sorry

end decreasing_implies_a_le_10_l2629_262924


namespace paper_clips_count_l2629_262921

/-- The number of paper clips in 2 cases -/
def paper_clips_in_two_cases (c b : ℕ) : ℕ := 2 * (c * b) * 600

/-- Theorem stating the number of paper clips in 2 cases -/
theorem paper_clips_count (c b : ℕ) :
  paper_clips_in_two_cases c b = 2 * (c * b) * 600 := by
  sorry

end paper_clips_count_l2629_262921


namespace combined_land_area_l2629_262980

/-- The combined area of two rectangular tracts of land -/
theorem combined_land_area (length1 width1 length2 width2 : ℕ) 
  (h1 : length1 = 300) 
  (h2 : width1 = 500) 
  (h3 : length2 = 250) 
  (h4 : width2 = 630) : 
  length1 * width1 + length2 * width2 = 307500 := by
  sorry

#check combined_land_area

end combined_land_area_l2629_262980


namespace cos_135_degrees_l2629_262932

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_135_degrees_l2629_262932


namespace wire_division_l2629_262981

/-- Given a wire of length 49 cm divided into 7 equal parts, prove that each part is 7 cm long -/
theorem wire_division (wire_length : ℝ) (num_parts : ℕ) (part_length : ℝ) 
  (h1 : wire_length = 49)
  (h2 : num_parts = 7)
  (h3 : part_length * num_parts = wire_length) :
  part_length = 7 := by
  sorry

end wire_division_l2629_262981


namespace right_triangle_hypotenuse_l2629_262985

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right-angled triangle condition
  a^2 + b^2 + c^2 = 2500 →  -- sum of squares condition
  c = 25 * Real.sqrt 2 := by
sorry

end right_triangle_hypotenuse_l2629_262985


namespace seven_bus_routes_l2629_262963

/-- Represents a bus stop in the network -/
structure BusStop :=
  (id : ℕ)

/-- Represents a bus route in the network -/
structure BusRoute :=
  (id : ℕ)
  (stops : Finset BusStop)

/-- Represents the entire bus network -/
structure BusNetwork :=
  (stops : Finset BusStop)
  (routes : Finset BusRoute)

/-- Every stop is reachable from any other stop without transfer -/
def all_stops_reachable (network : BusNetwork) : Prop :=
  ∀ s₁ s₂ : BusStop, s₁ ∈ network.stops → s₂ ∈ network.stops → 
    ∃ r : BusRoute, r ∈ network.routes ∧ s₁ ∈ r.stops ∧ s₂ ∈ r.stops

/-- Each pair of routes intersects at exactly one unique stop -/
def unique_intersection (network : BusNetwork) : Prop :=
  ∀ r₁ r₂ : BusRoute, r₁ ∈ network.routes → r₂ ∈ network.routes → r₁ ≠ r₂ →
    ∃! s : BusStop, s ∈ r₁.stops ∧ s ∈ r₂.stops

/-- Each route has exactly three stops -/
def three_stops_per_route (network : BusNetwork) : Prop :=
  ∀ r : BusRoute, r ∈ network.routes → Finset.card r.stops = 3

/-- There is more than one route -/
def multiple_routes (network : BusNetwork) : Prop :=
  ∃ r₁ r₂ : BusRoute, r₁ ∈ network.routes ∧ r₂ ∈ network.routes ∧ r₁ ≠ r₂

/-- The main theorem: Given the conditions, prove that there are 7 bus routes -/
theorem seven_bus_routes (network : BusNetwork) 
  (h1 : all_stops_reachable network)
  (h2 : unique_intersection network)
  (h3 : three_stops_per_route network)
  (h4 : multiple_routes network) :
  Finset.card network.routes = 7 :=
sorry

end seven_bus_routes_l2629_262963


namespace number_of_boys_l2629_262987

theorem number_of_boys (initial_avg : ℝ) (incorrect_height : ℝ) (correct_height : ℝ) (actual_avg : ℝ) :
  initial_avg = 184 →
  incorrect_height = 166 →
  correct_height = 106 →
  actual_avg = 182 →
  ∃ n : ℕ, n * initial_avg - (incorrect_height - correct_height) = n * actual_avg ∧ n = 30 :=
by sorry

end number_of_boys_l2629_262987


namespace integral_value_l2629_262953

theorem integral_value : ∫ x in (2 : ℝ)..4, (x^3 - 3*x^2 + 5) / x^2 = 5/4 := by
  sorry

end integral_value_l2629_262953


namespace rectangle_area_l2629_262905

/-- A rectangle with perimeter 36 and length three times its width has area 60.75 -/
theorem rectangle_area (width : ℝ) (length : ℝ) : 
  (width + length) * 2 = 36 →
  length = 3 * width →
  width * length = 60.75 := by
sorry

end rectangle_area_l2629_262905


namespace equation_system_solution_l2629_262949

theorem equation_system_solution (a b : ℝ) : 
  (∃ (a' : ℝ), a' * (-1) + 5 * (-1) = 15 ∧ 4 * (-1) - b * (-1) = -2) →
  (∃ (b' : ℝ), a * 5 + 5 * 2 = 15 ∧ 4 * 5 - b' * 2 = -2) →
  (a + 4 * b)^2 = 9 :=
by
  sorry

end equation_system_solution_l2629_262949


namespace fractional_inequality_l2629_262925

theorem fractional_inequality (x : ℝ) : (2*x - 1) / (x + 1) < 0 ↔ -1 < x ∧ x < 1/2 := by
  sorry

end fractional_inequality_l2629_262925


namespace polygon_interior_angles_l2629_262922

theorem polygon_interior_angles (n : ℕ) (h : n > 2) :
  (∃ x : ℝ, x > 0 ∧ x < 180 ∧ 2400 + x = (n - 2) * 180) → n = 16 := by
  sorry

end polygon_interior_angles_l2629_262922


namespace arithmetic_sequence_log_problem_l2629_262969

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

-- Define the theorem
theorem arithmetic_sequence_log_problem 
  (a b : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_arithmetic : arithmetic_sequence (λ n => 
    if n = 0 then log (a^5 * b^4)
    else if n = 1 then log (a^7 * b^9)
    else if n = 2 then log (a^10 * b^13)
    else if n = 9 then log (b^72)
    else 0)) : 
  ∃ n : ℕ, log (b^n) = (λ k => 
    if k = 0 then log (a^5 * b^4)
    else if k = 1 then log (a^7 * b^9)
    else if k = 2 then log (a^10 * b^13)
    else if k = 9 then log (b^72)
    else 0) 9 := by
  sorry

end arithmetic_sequence_log_problem_l2629_262969


namespace arithmetic_sequence_terms_l2629_262938

theorem arithmetic_sequence_terms (a₁ : ℤ) (d : ℤ) (last : ℤ) (n : ℕ) :
  a₁ = -6 →
  d = 4 →
  last ≤ 50 →
  last = a₁ + (n - 1) * d →
  n = 15 :=
by
  sorry

end arithmetic_sequence_terms_l2629_262938


namespace gcd_1729_1323_l2629_262933

theorem gcd_1729_1323 : Nat.gcd 1729 1323 = 7 := by
  sorry

end gcd_1729_1323_l2629_262933


namespace alternative_interest_rate_l2629_262908

theorem alternative_interest_rate 
  (principal : ℝ) 
  (time : ℝ) 
  (chosen_rate : ℝ) 
  (interest_difference : ℝ) : ℝ :=
  let alternative_rate := 
    (principal * chosen_rate * time - interest_difference) / (principal * time)
  
  -- Assumptions
  have h1 : principal = 7000 := by sorry
  have h2 : time = 2 := by sorry
  have h3 : chosen_rate = 0.15 := by sorry
  have h4 : interest_difference = 420 := by sorry

  -- Theorem statement
  alternative_rate * 100

/- Proof
  sorry
-/

end alternative_interest_rate_l2629_262908


namespace rectangle_area_l2629_262961

theorem rectangle_area (y : ℝ) (h : y > 0) : ∃ w l : ℝ,
  w > 0 ∧ l > 0 ∧ l = 3 * w ∧ w^2 + l^2 = y^2 ∧ w * l = (3 * y^2) / 10 :=
by sorry

end rectangle_area_l2629_262961


namespace quadratic_no_real_roots_l2629_262968

theorem quadratic_no_real_roots 
  (p q a b c : ℝ) 
  (hp : p > 0) (hq : q > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hpq : p ≠ q)
  (hgeom : a^2 = p * q)  -- p, a, q form a geometric sequence
  (harith : b + c = p + q)  -- p, b, c, q form an arithmetic sequence
  : ∀ x : ℝ, b * x^2 - 2 * a * x + c ≠ 0 := by
  sorry

end quadratic_no_real_roots_l2629_262968


namespace max_profit_price_l2629_262989

/-- The profit function for the bookstore -/
def profit_function (p : ℝ) : ℝ := 150 * p - 4 * p^2 - 200

/-- The theorem stating the price that maximizes profit -/
theorem max_profit_price :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 30 ∧
  ∀ (q : ℝ), 0 ≤ q ∧ q ≤ 30 → profit_function p ≥ profit_function q ∧
  p = 18.75 := by
  sorry

end max_profit_price_l2629_262989


namespace chandler_apples_per_week_l2629_262996

/-- The number of apples Chandler can eat per week -/
def chandler_apples : ℕ := 23

/-- The number of apples Lucy can eat per week -/
def lucy_apples : ℕ := 19

/-- The number of apples ordered for a month -/
def monthly_order : ℕ := 168

/-- The number of weeks in a month -/
def weeks_per_month : ℕ := 4

/-- Theorem stating that Chandler can eat 23 apples per week -/
theorem chandler_apples_per_week :
  chandler_apples * weeks_per_month + lucy_apples * weeks_per_month = monthly_order :=
by sorry

end chandler_apples_per_week_l2629_262996


namespace cubic_expression_value_l2629_262998

theorem cubic_expression_value : 7^3 - 4 * 7^2 + 4 * 7 - 2 = 173 := by
  sorry

end cubic_expression_value_l2629_262998


namespace evaluate_expression_l2629_262940

theorem evaluate_expression : (8^6 : ℝ) / (4 * 8^3) = 128 := by
  sorry

end evaluate_expression_l2629_262940


namespace sum_of_squares_of_roots_l2629_262993

theorem sum_of_squares_of_roots (a b α β : ℝ) : 
  (∀ x, (x - a) * (x - b) = 1 ↔ x = α ∨ x = β) →
  (∃ x₁ x₂, (x₁ - α) * (x₁ - β) = -1 ∧ (x₂ - α) * (x₂ - β) = -1 ∧ x₁ ≠ x₂) →
  x₁^2 + x₂^2 = a^2 + b^2 :=
sorry

end sum_of_squares_of_roots_l2629_262993


namespace welders_count_l2629_262976

/-- The number of welders initially working on the order -/
def initial_welders : ℕ := 72

/-- The number of days needed to complete the order with all welders -/
def initial_days : ℕ := 5

/-- The number of welders that leave after the first day -/
def leaving_welders : ℕ := 12

/-- The number of additional days needed to complete the order after some welders leave -/
def additional_days : ℕ := 6

/-- The theorem stating that the initial number of welders is 72 -/
theorem welders_count :
  initial_welders = 72 ∧
  (1 : ℚ) / (initial_days * initial_welders) = 
  (1 : ℚ) / (additional_days * (initial_welders - leaving_welders)) :=
by sorry

end welders_count_l2629_262976


namespace cuboid_surface_area_example_l2629_262944

/-- The surface area of a cuboid -/
def cuboid_surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a cuboid with length 12, width 14, and height 7 is 700 -/
theorem cuboid_surface_area_example : cuboid_surface_area 12 14 7 = 700 := by
  sorry

end cuboid_surface_area_example_l2629_262944


namespace zeros_of_f_shifted_l2629_262911

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem zeros_of_f_shifted (x : ℝ) :
  f (x - 1) = 0 ↔ x = 0 ∨ x = 2 :=
by sorry

end zeros_of_f_shifted_l2629_262911


namespace worker_save_fraction_l2629_262943

/-- Represents the worker's monthly savings scenario -/
structure WorkerSavings where
  monthly_pay : ℝ
  save_fraction : ℝ
  (monthly_pay_positive : monthly_pay > 0)
  (save_fraction_valid : 0 ≤ save_fraction ∧ save_fraction ≤ 1)

/-- The total amount saved over a year -/
def yearly_savings (w : WorkerSavings) : ℝ := 12 * w.save_fraction * w.monthly_pay

/-- The amount not saved from monthly pay -/
def monthly_unsaved (w : WorkerSavings) : ℝ := (1 - w.save_fraction) * w.monthly_pay

/-- Theorem stating the fraction of monthly take-home pay saved -/
theorem worker_save_fraction (w : WorkerSavings) 
  (h : yearly_savings w = 5 * monthly_unsaved w) : 
  w.save_fraction = 5 / 17 := by
  sorry

end worker_save_fraction_l2629_262943


namespace extremum_derivative_zero_relation_l2629_262975

/-- A function f has a derivative at x₀ -/
def has_derivative_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ (f' : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → |f x - f x₀ - f' * (x - x₀)| ≤ ε * |x - x₀|

/-- x₀ is an extremum point of f -/
def is_extremum_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - x₀| < δ → f x₀ ≤ f x ∨ f x₀ ≥ f x

/-- The derivative of f at x₀ is 0 -/
def derivative_zero (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ (f' : ℝ), has_derivative_at f x₀ ∧ f' = 0

theorem extremum_derivative_zero_relation (f : ℝ → ℝ) (x₀ : ℝ) :
  (has_derivative_at f x₀ →
    (is_extremum_point f x₀ → derivative_zero f x₀) ∧
    ¬(derivative_zero f x₀ → is_extremum_point f x₀)) :=
sorry

end extremum_derivative_zero_relation_l2629_262975


namespace residue_7_2023_mod_19_l2629_262901

theorem residue_7_2023_mod_19 : 7^2023 % 19 = 3 := by
  sorry

end residue_7_2023_mod_19_l2629_262901


namespace sqrt_sum_of_squares_l2629_262936

theorem sqrt_sum_of_squares : 
  Real.sqrt ((43 * 17)^2 + (43 * 26)^2 + (17 * 26)^2) = 1407 := by
  sorry

end sqrt_sum_of_squares_l2629_262936


namespace calculate_expression_l2629_262959

theorem calculate_expression : 2 * (8 ^ (1/3) - Real.sqrt 2) - (27 ^ (1/3) - Real.sqrt 2) = 1 - Real.sqrt 2 := by
  sorry

end calculate_expression_l2629_262959


namespace parabola_line_intersection_trajectory_l2629_262958

-- Define the parabola Ω
def Ω : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define the circle C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 5)^2 + p.2^2 = 16}

-- Define the line l
def l : Set (ℝ × ℝ) → Prop := λ L => ∃ (m b : ℝ), L = {p : ℝ × ℝ | p.1 = m * p.2 + b} ∨ L = {p : ℝ × ℝ | p.1 = 1} ∨ L = {p : ℝ × ℝ | p.1 = 9}

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the theorem
theorem parabola_line_intersection_trajectory
  (L : Set (ℝ × ℝ))
  (A B M Q : ℝ × ℝ)
  (hΩ : Ω.Nonempty)
  (hC : C.Nonempty)
  (hl : l L)
  (hAB : A ∈ L ∧ B ∈ L ∧ A ∈ Ω ∧ B ∈ Ω ∧ A ≠ B)
  (hM : M ∈ L ∧ M ∈ C)
  (hMmid : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hOAOB : (A.1 * B.1 + A.2 * B.2) = 0)
  (hQ : Q ∈ L ∧ (Q.1 - O.1) * (B.1 - A.1) + (Q.2 - O.2) * (B.2 - A.2) = 0) :
  Q.1^2 - 4 * Q.1 + Q.2^2 = 0 :=
sorry

end parabola_line_intersection_trajectory_l2629_262958


namespace sin_translation_equivalence_l2629_262916

theorem sin_translation_equivalence :
  ∀ x : ℝ, 2 * Real.sin (3 * x + π / 6) = 2 * Real.sin (3 * (x + π / 18)) :=
by sorry

end sin_translation_equivalence_l2629_262916


namespace qiqi_problem_solving_l2629_262974

/-- Represents the number of problems completed in a given time -/
structure ProblemRate where
  problems : ℕ
  minutes : ℕ

/-- Calculates the number of problems that can be completed in a given time,
    given a known problem rate -/
def calculateProblems (rate : ProblemRate) (time : ℕ) : ℕ :=
  (rate.problems * time) / rate.minutes

theorem qiqi_problem_solving :
  let initialRate : ProblemRate := ⟨15, 5⟩
  calculateProblems initialRate 8 = 24 := by
  sorry

end qiqi_problem_solving_l2629_262974


namespace deck_size_problem_l2629_262913

theorem deck_size_problem (r b : ℕ) : 
  -- Initial probability of selecting a red card
  (r : ℚ) / (r + b) = 2 / 5 →
  -- Probability after adding 6 black cards
  (r : ℚ) / (r + b + 6) = 1 / 3 →
  -- Total number of cards initially
  r + b = 5 := by
sorry

end deck_size_problem_l2629_262913


namespace equal_projections_imply_relation_l2629_262966

-- Define the points A, B, and C
def A (a : ℝ) : ℝ × ℝ := (a, 1)
def B (b : ℝ) : ℝ × ℝ := (2, b)
def C : ℝ × ℝ := (3, 4)

-- Define vectors OA, OB, and OC
def OA (a : ℝ) : ℝ × ℝ := A a
def OB (b : ℝ) : ℝ × ℝ := B b
def OC : ℝ × ℝ := C

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem equal_projections_imply_relation (a b : ℝ) :
  dot_product (OA a) OC = dot_product (OB b) OC →
  3 * a - 4 * b = 2 := by
  sorry


end equal_projections_imply_relation_l2629_262966


namespace quadratic_root_difference_l2629_262941

theorem quadratic_root_difference : ∃ (x₁ x₂ : ℝ),
  (5 + 3 * Real.sqrt 2) * x₁^2 - (1 - Real.sqrt 2) * x₁ - 1 = 0 ∧
  (5 + 3 * Real.sqrt 2) * x₂^2 - (1 - Real.sqrt 2) * x₂ - 1 = 0 ∧
  x₁ ≠ x₂ ∧
  max x₁ x₂ - min x₁ x₂ = 2 * Real.sqrt 2 + 1 :=
by sorry

end quadratic_root_difference_l2629_262941


namespace truck_speed_problem_l2629_262920

/-- 
Proves that given two trucks 1025 km apart, with Driver A starting at 90 km/h 
and Driver B starting 1 hour later, if Driver A has driven 145 km farther than 
Driver B when they meet, then Driver B's average speed is 485/6 km/h.
-/
theorem truck_speed_problem (distance : ℝ) (speed_A : ℝ) (extra_distance : ℝ) 
  (h1 : distance = 1025)
  (h2 : speed_A = 90)
  (h3 : extra_distance = 145) : 
  ∃ (speed_B : ℝ) (time : ℝ), 
    speed_B = 485 / 6 ∧ 
    time > 0 ∧
    speed_A * (time + 1) = speed_B * time + extra_distance ∧
    speed_A * (time + 1) + speed_B * time = distance :=
by sorry


end truck_speed_problem_l2629_262920


namespace smallest_m_is_13_l2629_262962

/-- The set of complex numbers with real part between 1/2 and √2/2 -/
def T : Set ℂ :=
  {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ Real.sqrt 2 / 2}

/-- Property that for all n ≥ m, there exists z in T such that z^n = 1 -/
def HasNthRoot (m : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ m → ∃ z : ℂ, z ∈ T ∧ z^n = 1

/-- 13 is the smallest positive integer satisfying the HasNthRoot property -/
theorem smallest_m_is_13 :
  HasNthRoot 13 ∧ ∀ m : ℕ, m > 0 → m < 13 → ¬HasNthRoot m :=
sorry

end smallest_m_is_13_l2629_262962


namespace nurse_lice_check_l2629_262909

/-- The number of Kindergarteners the nurse needs to check -/
def kindergarteners_to_check : ℕ := by sorry

theorem nurse_lice_check :
  let first_graders : ℕ := 19
  let second_graders : ℕ := 20
  let third_graders : ℕ := 25
  let minutes_per_check : ℕ := 2
  let total_hours : ℕ := 3
  let total_minutes : ℕ := total_hours * 60
  
  kindergarteners_to_check = 
    (total_minutes - 
      (first_graders + second_graders + third_graders) * minutes_per_check) / 
    minutes_per_check :=
by sorry

end nurse_lice_check_l2629_262909


namespace snack_machine_cost_l2629_262951

/-- The total cost for students buying candy bars and chips -/
def total_cost (num_students : ℕ) (candy_price chip_price : ℚ) (candy_per_student chip_per_student : ℕ) : ℚ :=
  num_students * (candy_price * candy_per_student + chip_price * chip_per_student)

/-- Theorem: The total cost for 5 students to each get 1 candy bar at $2 and 2 bags of chips at $0.50 per bag is $15 -/
theorem snack_machine_cost : total_cost 5 2 (1/2) 1 2 = 15 := by
  sorry

end snack_machine_cost_l2629_262951


namespace polygon_sum_l2629_262964

/-- Given a polygon JKLMNO with specific properties, prove that MN + NO = 14.5 -/
theorem polygon_sum (area_JKLMNO : ℝ) (JK KL NO : ℝ) :
  area_JKLMNO = 68 ∧ JK = 10 ∧ KL = 11 ∧ NO = 7 →
  ∃ (MN : ℝ), MN + NO = 14.5 := by
  sorry

end polygon_sum_l2629_262964


namespace monotonic_quadratic_function_l2629_262950

/-- The function f(x) = x^2 - 2mx + 3 is monotonic on the interval [1, 3] if and only if m ≤ 1 or m ≥ 3 -/
theorem monotonic_quadratic_function (m : ℝ) :
  (∀ x ∈ Set.Icc 1 3, Monotone (fun x => x^2 - 2*m*x + 3)) ↔ (m ≤ 1 ∨ m ≥ 3) := by
  sorry

end monotonic_quadratic_function_l2629_262950


namespace root_product_l2629_262928

theorem root_product (x₁ x₂ : ℝ) (h₁ : x₁ * Real.log x₁ = 2006) (h₂ : x₂ * Real.exp x₂ = 2006) : 
  x₁ * x₂ = 2006 := by
sorry

end root_product_l2629_262928


namespace baker_remaining_cakes_l2629_262954

/-- The number of cakes the baker still has after selling some -/
def remaining_cakes (cakes_made : ℕ) (cakes_sold : ℕ) : ℕ :=
  cakes_made - cakes_sold

/-- Theorem stating that the baker has 139 cakes remaining -/
theorem baker_remaining_cakes :
  remaining_cakes 149 10 = 139 := by
  sorry

end baker_remaining_cakes_l2629_262954


namespace additional_workers_needed_l2629_262957

/-- Represents the problem of determining the number of additional workers needed to complete a construction project on time. -/
theorem additional_workers_needed
  (total_days : ℕ)
  (initial_workers : ℕ)
  (days_passed : ℕ)
  (work_completed_percentage : ℚ)
  (h1 : total_days = 50)
  (h2 : initial_workers = 20)
  (h3 : days_passed = 25)
  (h4 : work_completed_percentage = 2/5)
  : ∃ (additional_workers : ℕ),
    (initial_workers + additional_workers) * (total_days - days_passed) =
    (1 - work_completed_percentage) * (initial_workers * total_days) ∧
    additional_workers = 4 := by
  sorry

end additional_workers_needed_l2629_262957


namespace inequality_system_solution_l2629_262999

theorem inequality_system_solution (x : ℝ) :
  x + 3 ≥ 2 ∧ 2 * (x + 4) > 4 * x + 2 → -1 ≤ x ∧ x < 3 := by
  sorry

end inequality_system_solution_l2629_262999


namespace rectangle_ratio_l2629_262992

theorem rectangle_ratio (w : ℝ) (h1 : w > 0) (h2 : 2 * w + 2 * 12 = 36) :
  w / 12 = 1 / 2 := by
  sorry

end rectangle_ratio_l2629_262992


namespace urn_operations_theorem_l2629_262960

/-- Represents the state of the urn with white and black marbles -/
structure UrnState where
  white : ℕ
  black : ℕ

/-- Represents the three possible operations on the urn -/
inductive Operation
  | removeBlackAddWhite
  | removeWhiteAddBoth
  | removeBothAddBoth

/-- Applies a single operation to the urn state -/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.removeBlackAddWhite => 
      ⟨state.white + 3, state.black - 2⟩
  | Operation.removeWhiteAddBoth => 
      ⟨state.white - 2, state.black + 1⟩
  | Operation.removeBothAddBoth => 
      ⟨state.white - 2, state.black⟩

/-- Represents a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to the initial state -/
def applySequence (init : UrnState) (seq : OperationSequence) : UrnState :=
  seq.foldl applyOperation init

/-- The theorem to be proved -/
theorem urn_operations_theorem : 
  ∃ (seq : OperationSequence), 
    applySequence ⟨150, 50⟩ seq = ⟨148, 2⟩ :=
sorry

end urn_operations_theorem_l2629_262960


namespace touching_spheres_bounds_l2629_262990

/-- Represents a tetrahedron -/
structure Tetrahedron where
  faces : Fin 4 → Real
  volume : Real

/-- Represents a sphere touching all face planes of a tetrahedron -/
structure TouchingSphere where
  radius : Real
  center : Fin 3 → Real

/-- The number of spheres touching all face planes of a tetrahedron -/
def num_touching_spheres (t : Tetrahedron) : ℕ :=
  sorry

/-- Theorem stating the bounds on the number of touching spheres -/
theorem touching_spheres_bounds (t : Tetrahedron) :
  5 ≤ num_touching_spheres t ∧ num_touching_spheres t ≤ 8 :=
sorry

end touching_spheres_bounds_l2629_262990


namespace six_balls_three_boxes_l2629_262912

/-- The number of ways to place distinguishable balls into distinguishable boxes -/
def place_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 729 ways to place 6 distinguishable balls into 3 distinguishable boxes -/
theorem six_balls_three_boxes : place_balls 6 3 = 729 := by
  sorry

end six_balls_three_boxes_l2629_262912


namespace max_product_sum_2000_l2629_262915

theorem max_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ x * y = 1000000 ∧ 
  ∀ (a b : ℤ), a + b = 2000 → a * b ≤ 1000000 := by
  sorry

end max_product_sum_2000_l2629_262915


namespace paint_usage_fraction_l2629_262947

theorem paint_usage_fraction (total_paint : ℝ) (paint_used_total : ℝ) :
  total_paint = 360 →
  paint_used_total = 168 →
  let paint_used_first_week := total_paint / 3
  let paint_remaining := total_paint - paint_used_first_week
  let paint_used_second_week := paint_used_total - paint_used_first_week
  paint_used_second_week / paint_remaining = 1 / 5 := by
sorry

end paint_usage_fraction_l2629_262947


namespace car_stopping_distance_l2629_262970

/-- The distance function representing the car's motion during emergency braking -/
def s (t : ℝ) : ℝ := 30 * t - 5 * t^2

/-- The maximum distance traveled by the car before stopping -/
def max_distance : ℝ := 45

/-- Theorem stating that the maximum value of s(t) is 45 -/
theorem car_stopping_distance :
  ∃ t₀ : ℝ, ∀ t : ℝ, s t ≤ s t₀ ∧ s t₀ = max_distance :=
sorry

end car_stopping_distance_l2629_262970


namespace blocks_used_for_tower_l2629_262917

/-- Given Randy's block usage, prove the number of blocks used for the tower -/
theorem blocks_used_for_tower 
  (total_blocks : ℕ) 
  (blocks_for_house : ℕ) 
  (blocks_for_tower : ℕ) 
  (h1 : total_blocks = 95) 
  (h2 : blocks_for_house = 20) 
  (h3 : blocks_for_tower = blocks_for_house + 30) : 
  blocks_for_tower = 50 := by
  sorry

end blocks_used_for_tower_l2629_262917


namespace comprehensive_office_increases_profit_building_comprehensive_offices_increases_profit_l2629_262955

/-- Represents a technology company -/
structure TechCompany where
  name : String
  profit : ℝ
  employeeRetention : ℝ
  productivity : ℝ
  workLifeIntegration : ℝ

/-- Represents an office environment -/
structure OfficeEnvironment where
  hasWorkSpaces : Bool
  hasLeisureSpaces : Bool
  hasLivingSpaces : Bool

/-- Function to determine if an office environment is comprehensive -/
def isComprehensiveOffice (office : OfficeEnvironment) : Bool :=
  office.hasWorkSpaces ∧ office.hasLeisureSpaces ∧ office.hasLivingSpaces

/-- Function to calculate the impact of office environment on company metrics -/
def officeImpact (company : TechCompany) (office : OfficeEnvironment) : TechCompany :=
  if isComprehensiveOffice office then
    { company with
      employeeRetention := company.employeeRetention * 1.1
      productivity := company.productivity * 1.15
      workLifeIntegration := company.workLifeIntegration * 1.2
    }
  else
    company

/-- Theorem stating that comprehensive offices increase profit -/
theorem comprehensive_office_increases_profit (company : TechCompany) (office : OfficeEnvironment) :
  isComprehensiveOffice office →
  (officeImpact company office).profit > company.profit :=
by sorry

/-- Main theorem proving that building comprehensive offices increases profit through specific factors -/
theorem building_comprehensive_offices_increases_profit (company : TechCompany) (office : OfficeEnvironment) :
  isComprehensiveOffice office →
  (∃ newCompany : TechCompany,
    newCompany = officeImpact company office ∧
    newCompany.profit > company.profit ∧
    newCompany.employeeRetention > company.employeeRetention ∧
    newCompany.productivity > company.productivity ∧
    newCompany.workLifeIntegration > company.workLifeIntegration) :=
by sorry

end comprehensive_office_increases_profit_building_comprehensive_offices_increases_profit_l2629_262955


namespace area_triangle_QCA_l2629_262967

/-- The area of triangle QCA given the coordinates of points Q, A, and C -/
theorem area_triangle_QCA (p : ℝ) : 
  let Q : ℝ × ℝ := (0, 15)
  let A : ℝ × ℝ := (3, 15)
  let C : ℝ × ℝ := (0, p)
  let area := (1/2) * (A.1 - Q.1) * (Q.2 - C.2)
  area = 45/2 - 3*p/2 := by
sorry

end area_triangle_QCA_l2629_262967


namespace logarithm_inequality_l2629_262906

theorem logarithm_inequality (a b c : ℝ) (ha : a ≥ 2) (hb : b ≥ 2) (hc : c ≥ 2) :
  Real.log a^2 / Real.log (b + c) + Real.log b^2 / Real.log (a + c) + Real.log c^2 / Real.log (a + b) ≥ 3 := by
  sorry

end logarithm_inequality_l2629_262906


namespace canoe_downstream_speed_l2629_262914

/-- Given a canoe that rows upstream at 6 km/hr and a stream with a speed of 2 km/hr,
    this theorem proves that the speed of the canoe when rowing downstream is 10 km/hr. -/
theorem canoe_downstream_speed
  (upstream_speed : ℝ)
  (stream_speed : ℝ)
  (h1 : upstream_speed = 6)
  (h2 : stream_speed = 2) :
  upstream_speed + 2 * stream_speed = 10 :=
by sorry

end canoe_downstream_speed_l2629_262914


namespace arithmetic_simplification_l2629_262926

theorem arithmetic_simplification :
  (5 + 7 + 3) / 3 - 2 / 3 = 13 / 3 := by sorry

end arithmetic_simplification_l2629_262926
