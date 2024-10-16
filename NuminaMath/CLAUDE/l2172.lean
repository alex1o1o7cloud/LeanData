import Mathlib

namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2172_217237

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  d^2 / 2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2172_217237


namespace NUMINAMATH_CALUDE_count_even_numbers_between_300_and_600_l2172_217287

theorem count_even_numbers_between_300_and_600 :
  (Finset.filter (fun n => n % 2 = 0 ∧ 300 < n ∧ n < 600) (Finset.range 600)).card = 149 := by
  sorry

end NUMINAMATH_CALUDE_count_even_numbers_between_300_and_600_l2172_217287


namespace NUMINAMATH_CALUDE_ninety_degrees_to_radians_l2172_217295

theorem ninety_degrees_to_radians :
  let degrees_to_radians (d : ℝ) : ℝ := d * (π / 180)
  degrees_to_radians 90 = π / 2 := by sorry

end NUMINAMATH_CALUDE_ninety_degrees_to_radians_l2172_217295


namespace NUMINAMATH_CALUDE_max_remainder_when_divided_by_25_l2172_217277

theorem max_remainder_when_divided_by_25 (A B C : ℕ) : 
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A = 25 * B + C →
  C ≤ 24 :=
by sorry

end NUMINAMATH_CALUDE_max_remainder_when_divided_by_25_l2172_217277


namespace NUMINAMATH_CALUDE_max_area_between_lines_l2172_217238

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 16

-- Define the area function
def area (x₀ : ℝ) : ℝ :=
  2 * (-2 * x₀^2 + 32 - 4 * x₀)

-- State the theorem
theorem max_area_between_lines :
  ∃ (x₀ : ℝ), x₀ ∈ Set.Icc (-3) 1 ∧ 
  (∀ (x : ℝ), x ∈ Set.Icc (-3) 1 → area x ≤ area x₀) ∧
  area x₀ = 68 := by
  sorry

end NUMINAMATH_CALUDE_max_area_between_lines_l2172_217238


namespace NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l2172_217218

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem quadratic_inequality_negation :
  (¬ ∃ x : ℝ, x^2 - 2*x - 3 < 0) ↔ (∀ x : ℝ, x^2 - 2*x - 3 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l2172_217218


namespace NUMINAMATH_CALUDE_rounding_effect_on_expression_l2172_217263

theorem rounding_effect_on_expression (a b c a' b' c' : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ha' : a' ≥ a) (hb' : b' ≤ b) (hc' : c' ≤ c) : 
  2 * (a' / b') + 2 * c' > 2 * (a / b) + 2 * c :=
sorry

end NUMINAMATH_CALUDE_rounding_effect_on_expression_l2172_217263


namespace NUMINAMATH_CALUDE_deepak_current_age_l2172_217241

-- Define the ratio of Rahul to Deepak's age
def age_ratio : ℚ := 4 / 3

-- Define Rahul's age after 4 years
def rahul_future_age : ℕ := 32

-- Define the number of years in the future for Rahul's age
def years_in_future : ℕ := 4

-- Theorem to prove Deepak's current age
theorem deepak_current_age :
  ∃ (rahul_age deepak_age : ℕ),
    (rahul_age : ℚ) / deepak_age = age_ratio ∧
    rahul_age + years_in_future = rahul_future_age ∧
    deepak_age = 21 := by
  sorry

end NUMINAMATH_CALUDE_deepak_current_age_l2172_217241


namespace NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_square_l2172_217299

theorem consecutive_integers_product_plus_one_is_square (n : ℤ) : 
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n * (n + 3) + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_square_l2172_217299


namespace NUMINAMATH_CALUDE_greatest_number_l2172_217228

theorem greatest_number : ∀ (a b c : ℝ), 
  a = 43.23 ∧ b = 2/5 ∧ c = 21.23 →
  a > b ∧ a > c :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_l2172_217228


namespace NUMINAMATH_CALUDE_component_qualification_l2172_217290

def lower_limit : ℝ := 20 - 0.05
def upper_limit : ℝ := 20 + 0.02

def is_qualified (diameter : ℝ) : Prop :=
  lower_limit ≤ diameter ∧ diameter ≤ upper_limit

theorem component_qualification :
  is_qualified 19.96 ∧
  ¬is_qualified 19.50 ∧
  ¬is_qualified 20.2 ∧
  ¬is_qualified 20.05 := by
  sorry

end NUMINAMATH_CALUDE_component_qualification_l2172_217290


namespace NUMINAMATH_CALUDE_fraction_value_l2172_217221

theorem fraction_value (x y : ℝ) (h : 1 / x - 1 / y = 3) :
  (2 * x + 3 * x * y - 2 * y) / (x - 2 * x * y - y) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2172_217221


namespace NUMINAMATH_CALUDE_smallest_batch_size_l2172_217262

theorem smallest_batch_size (N : ℕ) (h1 : N > 70) (h2 : (21 * N) % 70 = 0) :
  N ≥ 80 ∧ ∀ m : ℕ, m > 70 ∧ (21 * m) % 70 = 0 → m ≥ N := by
  sorry

end NUMINAMATH_CALUDE_smallest_batch_size_l2172_217262


namespace NUMINAMATH_CALUDE_solution_equation_l2172_217291

theorem solution_equation (p q : ℝ) (h1 : p ≠ q) (h2 : p ≠ 0) (h3 : q ≠ 0) :
  ∃ x : ℝ, (x + p)^2 - (x + q)^2 = 4*(p-q)^2 ∧ x = 2*p - 2*q :=
by sorry

end NUMINAMATH_CALUDE_solution_equation_l2172_217291


namespace NUMINAMATH_CALUDE_valid_integers_count_l2172_217286

/-- The number of digits in the integers we're counting -/
def num_digits : ℕ := 8

/-- The number of choices for the first digit (2-9) -/
def first_digit_choices : ℕ := 8

/-- The number of choices for each subsequent digit (0-9) -/
def other_digit_choices : ℕ := 10

/-- The number of different 8-digit positive integers where the first digit cannot be 0 or 1 -/
def count_valid_integers : ℕ := first_digit_choices * (other_digit_choices ^ (num_digits - 1))

theorem valid_integers_count :
  count_valid_integers = 80000000 := by sorry

end NUMINAMATH_CALUDE_valid_integers_count_l2172_217286


namespace NUMINAMATH_CALUDE_largest_fraction_l2172_217214

def fraction_set : Set ℚ := {1/2, 1/3, 1/4, 1/5, 1/10}

theorem largest_fraction :
  ∀ x ∈ fraction_set, (1/2 : ℚ) ≥ x :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_l2172_217214


namespace NUMINAMATH_CALUDE_gift_splitting_l2172_217239

theorem gift_splitting (initial_cost : ℝ) (dropout_count : ℕ) (extra_cost : ℝ) : 
  initial_cost = 120 ∧ 
  dropout_count = 4 ∧ 
  extra_cost = 8 →
  ∃ (n : ℕ), 
    n > dropout_count ∧
    initial_cost / (n - dropout_count : ℝ) = initial_cost / n + extra_cost ∧
    n = 10 := by
  sorry

end NUMINAMATH_CALUDE_gift_splitting_l2172_217239


namespace NUMINAMATH_CALUDE_angle_cosine_relation_l2172_217249

/-- Given a point Q in 3D space with positive coordinates, and angles α, β, γ between OQ and the x, y, z axes respectively, prove that if cos α = 2/5 and cos β = 1/4, then cos γ = √(311)/20 -/
theorem angle_cosine_relation (Q : ℝ × ℝ × ℝ) (α β γ : ℝ) 
  (h_pos : Q.1 > 0 ∧ Q.2.1 > 0 ∧ Q.2.2 > 0)
  (h_α : α = Real.arccos (Q.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2)))
  (h_β : β = Real.arccos (Q.2.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2)))
  (h_γ : γ = Real.arccos (Q.2.2 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2)))
  (h_cos_α : Real.cos α = 2/5)
  (h_cos_β : Real.cos β = 1/4) :
  Real.cos γ = Real.sqrt 311 / 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_cosine_relation_l2172_217249


namespace NUMINAMATH_CALUDE_age_difference_proof_l2172_217244

-- Define variables for ages
variable (A B C : ℕ)

-- Define the condition that C is 10 years younger than A
def age_difference : Prop := C = A - 10

-- Define the difference in total ages
def total_age_difference : ℕ := (A + B) - (B + C)

-- Theorem to prove
theorem age_difference_proof (h : age_difference A C) : total_age_difference A B C = 10 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2172_217244


namespace NUMINAMATH_CALUDE_reflection_sum_l2172_217229

/-- Given a line y = mx + b, if the reflection of point (2,3) across this line is (6,7), then m + b = 8 -/
theorem reflection_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    -- The midpoint of (2,3) and (6,7) lies on the line y = mx + b
    y = m * x + b ∧ 
    x = (2 + 6) / 2 ∧ 
    y = (3 + 7) / 2 ∧
    -- The line y = mx + b is perpendicular to the line connecting (2,3) and (6,7)
    m * ((7 - 3) / (6 - 2)) = -1) →
  m + b = 8 := by
  sorry

end NUMINAMATH_CALUDE_reflection_sum_l2172_217229


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l2172_217296

/-- A triangle with two sides of length 3 and 6 can have a third side of length 6 -/
theorem triangle_third_side_length : ∃ (a b c : ℝ), 
  a = 3 ∧ b = 6 ∧ c = 6 ∧ 
  a + b > c ∧ b + c > a ∧ a + c > b ∧
  a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l2172_217296


namespace NUMINAMATH_CALUDE_quadratic_solution_set_theorem_l2172_217254

/-- Given a quadratic function f(x) = ax² + bx + c, 
    this is the type of its solution set when f(x) > 0 -/
def QuadraticSolutionSet (a b c : ℝ) := Set ℝ

/-- The condition that the solution set of ax² + bx + c > 0 
    is the open interval (3, 6) -/
def SolutionSetCondition (a b c : ℝ) : Prop :=
  QuadraticSolutionSet a b c = {x : ℝ | 3 < x ∧ x < 6}

theorem quadratic_solution_set_theorem 
  (a b c : ℝ) (h : SolutionSetCondition a b c) :
  QuadraticSolutionSet c b a = {x : ℝ | x < 1/6 ∨ x > 1/3} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_set_theorem_l2172_217254


namespace NUMINAMATH_CALUDE_cindy_added_pens_l2172_217220

/-- Proves the number of pens Cindy added given the initial conditions and final result --/
theorem cindy_added_pens (initial_pens : ℕ) (mike_gives : ℕ) (sharon_receives : ℕ) (final_pens : ℕ)
  (h1 : initial_pens = 7)
  (h2 : mike_gives = 22)
  (h3 : sharon_receives = 19)
  (h4 : final_pens = 39) :
  final_pens = initial_pens + mike_gives - sharon_receives + 29 := by
  sorry

#check cindy_added_pens

end NUMINAMATH_CALUDE_cindy_added_pens_l2172_217220


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l2172_217294

theorem integer_pairs_satisfying_equation :
  {(x, y) : ℤ × ℤ | x^2 = y^2 + 2*y + 13} =
  {(4, -3), (4, 1), (-4, 1), (-4, -3)} := by
  sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l2172_217294


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l2172_217251

/-- The number of second-year students in the chess tournament --/
def n : ℕ := 7

/-- The total number of participants in the tournament --/
def total_participants : ℕ := n + 2

/-- The total number of games played in the tournament --/
def total_games : ℕ := (total_participants * (total_participants - 1)) / 2

/-- The total points scored in the tournament --/
def total_points : ℕ := total_games

/-- The points scored by the two freshmen --/
def freshman_points : ℕ := 8

/-- The points scored by all second-year students --/
def secondyear_points : ℕ := total_points - freshman_points

/-- The points scored by each second-year student --/
def points_per_secondyear : ℕ := secondyear_points / n

theorem chess_tournament_participants :
  n > 0 ∧
  total_participants = n + 2 ∧
  total_games = (total_participants * (total_participants - 1)) / 2 ∧
  total_points = total_games ∧
  freshman_points = 8 ∧
  secondyear_points = total_points - freshman_points ∧
  points_per_secondyear = secondyear_points / n ∧
  points_per_secondyear * n = secondyear_points ∧
  (∀ m : ℕ, m ≠ n → (m > 0 → 
    (m + 2) * (m + 1) / 2 - 8 ≠ ((m + 2) * (m + 1) / 2 - 8) / m * m)) :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l2172_217251


namespace NUMINAMATH_CALUDE_order_of_cube_roots_l2172_217268

theorem order_of_cube_roots : ∀ (a b c : ℝ),
  a = 2^(4/3) →
  b = 3^(2/3) →
  c = 2.5^(1/3) →
  c < b ∧ b < a :=
by sorry

end NUMINAMATH_CALUDE_order_of_cube_roots_l2172_217268


namespace NUMINAMATH_CALUDE_ba_atomic_weight_l2172_217206

/-- The atomic weight of Bromine (Br) -/
def atomic_weight_Br : ℝ := 79.9

/-- The molecular weight of the compound BaBr₂ -/
def molecular_weight_compound : ℝ := 297

/-- The atomic weight of Barium (Ba) -/
def atomic_weight_Ba : ℝ := molecular_weight_compound - 2 * atomic_weight_Br

theorem ba_atomic_weight :
  atomic_weight_Ba = 137.2 := by sorry

end NUMINAMATH_CALUDE_ba_atomic_weight_l2172_217206


namespace NUMINAMATH_CALUDE_leahs_coins_value_l2172_217215

/-- Represents the value of a coin in cents -/
def coinValue (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | _ => 0

/-- Calculates the total value of coins given their quantities -/
def totalValue (pennies nickels dimes : ℕ) : ℕ :=
  pennies * coinValue "penny" + nickels * coinValue "nickel" + dimes * coinValue "dime"

theorem leahs_coins_value :
  ∀ (pennies nickels dimes : ℕ),
    pennies + nickels + dimes = 17 →
    nickels + 2 = pennies →
    totalValue pennies nickels dimes = 68 :=
by sorry

end NUMINAMATH_CALUDE_leahs_coins_value_l2172_217215


namespace NUMINAMATH_CALUDE_total_votes_l2172_217255

/-- Proves that the total number of votes is 290 given the specified conditions -/
theorem total_votes (votes_against : ℕ) (votes_in_favor : ℕ) (total_votes : ℕ) : 
  votes_in_favor = votes_against + 58 →
  votes_against = (40 * total_votes) / 100 →
  total_votes = votes_in_favor + votes_against →
  total_votes = 290 := by
sorry

end NUMINAMATH_CALUDE_total_votes_l2172_217255


namespace NUMINAMATH_CALUDE_team_a_games_l2172_217236

theorem team_a_games (a : ℕ) (h1 : 3 * a = 4 * (a - (a / 4)))
  (h2 : 2 * (a + 16) = 3 * ((a + 16) - ((a + 16) / 3)))
  (h3 : (a + 16) - ((a + 16) / 3) = a - (a / 4) + 8)
  (h4 : ((a + 16) / 3) = (a / 4) + 8) : a = 192 := by
  sorry

end NUMINAMATH_CALUDE_team_a_games_l2172_217236


namespace NUMINAMATH_CALUDE_intersection_when_p_zero_union_equals_B_implies_p_range_l2172_217200

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set B with parameter p
def B (p : ℝ) : Set ℝ := {x | |x - p| > 1}

-- Theorem for part (1)
theorem intersection_when_p_zero :
  A ∩ B 0 = {x | 1 < x ∧ x < 3} := by sorry

-- Theorem for part (2)
theorem union_equals_B_implies_p_range (p : ℝ) :
  A ∪ B p = B p → p ≤ -2 ∨ p ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_when_p_zero_union_equals_B_implies_p_range_l2172_217200


namespace NUMINAMATH_CALUDE_triangle_problem_l2172_217242

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- Vector in 2D space -/
structure Vector2D where
  x : Real
  y : Real

/-- Dot product of two 2D vectors -/
def dot_product (v w : Vector2D) : Real :=
  v.x * w.x + v.y * w.y

variable (ABC : Triangle)

/-- Vector m as defined in the problem -/
def m : Vector2D :=
  { x := Real.cos (ABC.A - ABC.B),
    y := Real.sin (ABC.A - ABC.B) }

/-- Vector n as defined in the problem -/
def n : Vector2D :=
  { x := Real.cos ABC.B,
    y := -Real.sin ABC.B }

/-- Main theorem capturing the problem statement and its solution -/
theorem triangle_problem (h1 : dot_product (m ABC) (n ABC) = -3/5)
                         (h2 : ABC.a = 4 * Real.sqrt 2)
                         (h3 : ABC.b = 5) :
  Real.sin ABC.A = 4/5 ∧
  ABC.B = π/4 ∧
  -(ABC.c * Real.cos ABC.B) = -Real.sqrt 2 / 2 :=
sorry

end

end NUMINAMATH_CALUDE_triangle_problem_l2172_217242


namespace NUMINAMATH_CALUDE_yoki_cans_count_l2172_217274

/-- Given a scenario where:
  - The total number of cans collected is 85
  - LaDonna picked up 25 cans
  - Prikya picked up twice as many cans as LaDonna
  - Yoki picked up the rest of the cans
This theorem proves that Yoki picked up 10 cans. -/
theorem yoki_cans_count (total : ℕ) (ladonna : ℕ) (prikya : ℕ) (yoki : ℕ) 
  (h1 : total = 85)
  (h2 : ladonna = 25)
  (h3 : prikya = 2 * ladonna)
  (h4 : total = ladonna + prikya + yoki) :
  yoki = 10 := by
  sorry

end NUMINAMATH_CALUDE_yoki_cans_count_l2172_217274


namespace NUMINAMATH_CALUDE_salt_production_average_l2172_217260

/-- The salt production problem --/
theorem salt_production_average (initial_production : ℕ) (monthly_increase : ℕ) (months : ℕ) (days_in_year : ℕ) :
  let total_production := initial_production + (monthly_increase * (months * (months - 1)) / 2)
  (total_production : ℚ) / days_in_year = 121.1 := by
  sorry

#check salt_production_average 3000 100 12 365

end NUMINAMATH_CALUDE_salt_production_average_l2172_217260


namespace NUMINAMATH_CALUDE_closet_probability_l2172_217283

def shirts : ℕ := 5
def shorts : ℕ := 7
def socks : ℕ := 8
def total_articles : ℕ := shirts + shorts + socks
def articles_picked : ℕ := 4

theorem closet_probability : 
  (Nat.choose shirts 2 * Nat.choose shorts 1 * Nat.choose socks 1) / 
  Nat.choose total_articles articles_picked = 112 / 969 := by
  sorry

end NUMINAMATH_CALUDE_closet_probability_l2172_217283


namespace NUMINAMATH_CALUDE_negation_of_existence_is_forall_not_l2172_217267

theorem negation_of_existence_is_forall_not :
  (¬ ∃ x : ℚ, x^2 - 2 = 0) ↔ (∀ x : ℚ, x^2 - 2 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_forall_not_l2172_217267


namespace NUMINAMATH_CALUDE_quadratic_specific_value_l2172_217261

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_specific_value (a b c : ℝ) :
  (∃ (f : ℝ → ℝ), f = quadratic a b c) →
  (∀ x, quadratic a b c x ≥ -4) →
  (quadratic a b c (-5) = -4) →
  (quadratic a b c 0 = 6) →
  (quadratic a b c (-3) = -2.4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_specific_value_l2172_217261


namespace NUMINAMATH_CALUDE_power_of_power_l2172_217265

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2172_217265


namespace NUMINAMATH_CALUDE_point_B_coordinates_l2172_217256

/-- Given point A (2, 4), vector a⃗ = (3, 4), and AB⃗ = 2a⃗, prove that the coordinates of point B are (8, 12). -/
theorem point_B_coordinates (A B : ℝ × ℝ) (a : ℝ × ℝ) :
  A = (2, 4) →
  a = (3, 4) →
  B.1 - A.1 = 2 * a.1 →
  B.2 - A.2 = 2 * a.2 →
  B = (8, 12) := by
sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l2172_217256


namespace NUMINAMATH_CALUDE_spinner_probability_l2172_217234

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_D = 1/6 → p_A + p_B + p_C + p_D = 1 → p_C = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l2172_217234


namespace NUMINAMATH_CALUDE_max_projection_area_of_special_tetrahedron_l2172_217293

/-- Represents a tetrahedron with specific properties -/
structure Tetrahedron where
  side_length : ℝ
  dihedral_angle : ℝ

/-- The area of the projection of the tetrahedron onto a plane -/
noncomputable def projection_area (t : Tetrahedron) (rotation_angle : ℝ) : ℝ :=
  sorry

/-- The maximum area of the projection over all rotation angles -/
noncomputable def max_projection_area (t : Tetrahedron) : ℝ :=
  sorry

theorem max_projection_area_of_special_tetrahedron :
  let t : Tetrahedron := { side_length := 1, dihedral_angle := π/3 }
  max_projection_area t = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_projection_area_of_special_tetrahedron_l2172_217293


namespace NUMINAMATH_CALUDE_white_spotted_mushrooms_count_l2172_217252

/-- The number of white-spotted mushrooms gathered by Bill and Ted -/
def white_spotted_mushrooms : ℕ :=
  let bill_red := 12
  let bill_brown := 6
  let ted_blue := 6
  let red_with_spots := (2 * bill_red) / 3
  let brown_with_spots := bill_brown
  let blue_with_spots := ted_blue / 2
  red_with_spots + brown_with_spots + blue_with_spots

/-- Theorem stating that the total number of white-spotted mushrooms is 17 -/
theorem white_spotted_mushrooms_count : white_spotted_mushrooms = 17 := by
  sorry

end NUMINAMATH_CALUDE_white_spotted_mushrooms_count_l2172_217252


namespace NUMINAMATH_CALUDE_august_mail_total_l2172_217224

/-- The number of pieces of mail Vivian sent in a given month -/
def mail_sent (month : String) : ℕ :=
  match month with
  | "April" => 5
  | "May" => 10
  | "June" => 20
  | "July" => 40
  | _ => 0

/-- The number of business days in August -/
def august_business_days : ℕ := 23

/-- The number of holidays in August -/
def august_holidays : ℕ := 8

/-- The amount of mail sent on a business day in August -/
def august_business_day_mail : ℕ := 2 * mail_sent "July"

/-- The amount of mail sent on a holiday in August -/
def august_holiday_mail : ℕ := mail_sent "July" / 2

theorem august_mail_total :
  august_business_days * august_business_day_mail +
  august_holidays * august_holiday_mail = 2000 := by
  sorry

end NUMINAMATH_CALUDE_august_mail_total_l2172_217224


namespace NUMINAMATH_CALUDE_zit_difference_l2172_217210

def swanson_avg : ℕ := 5
def swanson_kids : ℕ := 25
def jones_avg : ℕ := 6
def jones_kids : ℕ := 32

theorem zit_difference : 
  jones_avg * jones_kids - swanson_avg * swanson_kids = 67 := by
  sorry

end NUMINAMATH_CALUDE_zit_difference_l2172_217210


namespace NUMINAMATH_CALUDE_constant_sequence_is_ap_and_gp_l2172_217248

def constant_sequence : ℕ → ℝ := λ n => 7

def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem constant_sequence_is_ap_and_gp :
  is_arithmetic_progression constant_sequence ∧
  is_geometric_progression constant_sequence := by
  sorry

#check constant_sequence_is_ap_and_gp

end NUMINAMATH_CALUDE_constant_sequence_is_ap_and_gp_l2172_217248


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2172_217227

theorem quadratic_one_solution (k : ℝ) : 
  (k > 0) → (∃! x, 4 * x^2 + k * x + 4 = 0) ↔ k = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2172_217227


namespace NUMINAMATH_CALUDE_tailor_buttons_count_l2172_217276

/-- The number of green buttons purchased by the tailor -/
def green_buttons : ℕ := 90

/-- The number of yellow buttons purchased by the tailor -/
def yellow_buttons : ℕ := green_buttons + 10

/-- The number of blue buttons purchased by the tailor -/
def blue_buttons : ℕ := green_buttons - 5

/-- The total number of buttons purchased by the tailor -/
def total_buttons : ℕ := green_buttons + yellow_buttons + blue_buttons

theorem tailor_buttons_count : total_buttons = 275 := by
  sorry

end NUMINAMATH_CALUDE_tailor_buttons_count_l2172_217276


namespace NUMINAMATH_CALUDE_total_cost_after_discounts_l2172_217232

/-- Calculate the total cost of items after applying discounts --/
theorem total_cost_after_discounts :
  let board_game_cost : ℚ := 2
  let action_figure_cost : ℚ := 7
  let action_figure_count : ℕ := 4
  let puzzle_cost : ℚ := 6
  let deck_cost : ℚ := 3.5
  let toy_car_cost : ℚ := 4
  let toy_car_count : ℕ := 2
  let action_figure_discount : ℚ := 0.15
  let puzzle_toy_car_discount : ℚ := 0.10
  let deck_discount : ℚ := 0.05

  let total_cost : ℚ := 
    board_game_cost +
    (action_figure_cost * action_figure_count) * (1 - action_figure_discount) +
    puzzle_cost * (1 - puzzle_toy_car_discount) +
    deck_cost * (1 - deck_discount) +
    (toy_car_cost * toy_car_count) * (1 - puzzle_toy_car_discount)

  total_cost = 41.73 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_after_discounts_l2172_217232


namespace NUMINAMATH_CALUDE_book_shelf_problem_l2172_217209

theorem book_shelf_problem (paperbacks hardbacks : ℕ) 
  (h1 : paperbacks = 2)
  (h2 : hardbacks = 6)
  (h3 : Nat.choose paperbacks 1 * Nat.choose hardbacks 2 + 
        Nat.choose paperbacks 2 * Nat.choose hardbacks 1 = 36) :
  paperbacks + hardbacks = 8 := by
sorry

end NUMINAMATH_CALUDE_book_shelf_problem_l2172_217209


namespace NUMINAMATH_CALUDE_pencil_sales_problem_l2172_217212

/-- The number of pencils initially sold for a rupee -/
def initial_pencils : ℝ := 11

/-- The number of pencils sold for a rupee to achieve a 20% gain -/
def gain_pencils : ℝ := 8.25

/-- The loss percentage when selling the initial number of pencils -/
def loss_percentage : ℝ := 10

/-- The gain percentage when selling 8.25 pencils -/
def gain_percentage : ℝ := 20

theorem pencil_sales_problem :
  (1 = (1 - loss_percentage / 100) * initial_pencils * (1 / gain_pencils)) ∧
  (1 = (1 + gain_percentage / 100) * 1) ∧
  initial_pencils = 11 := by sorry

end NUMINAMATH_CALUDE_pencil_sales_problem_l2172_217212


namespace NUMINAMATH_CALUDE_optimal_production_plan_l2172_217284

/-- Represents a production plan with quantities of products A and B. -/
structure ProductionPlan where
  productA : ℕ
  productB : ℕ

/-- Represents the available raw materials and profit data. -/
structure FactoryData where
  rawMaterialA : ℝ
  rawMaterialB : ℝ
  totalProducts : ℕ
  materialAForProductA : ℝ
  materialBForProductA : ℝ
  materialAForProductB : ℝ
  materialBForProductB : ℝ
  profitProductA : ℕ
  profitProductB : ℕ

/-- Checks if a production plan is valid given the factory data. -/
def isValidPlan (plan : ProductionPlan) (data : FactoryData) : Prop :=
  plan.productA + plan.productB = data.totalProducts ∧
  plan.productA * data.materialAForProductA + plan.productB * data.materialAForProductB ≤ data.rawMaterialA ∧
  plan.productA * data.materialBForProductA + plan.productB * data.materialBForProductB ≤ data.rawMaterialB

/-- Calculates the profit for a given production plan. -/
def calculateProfit (plan : ProductionPlan) (data : FactoryData) : ℕ :=
  plan.productA * data.profitProductA + plan.productB * data.profitProductB

/-- The main theorem to prove. -/
theorem optimal_production_plan (data : FactoryData)
  (h_data : data.rawMaterialA = 66 ∧ data.rawMaterialB = 66.4 ∧ data.totalProducts = 90 ∧
            data.materialAForProductA = 0.5 ∧ data.materialBForProductA = 0.8 ∧
            data.materialAForProductB = 1.2 ∧ data.materialBForProductB = 0.6 ∧
            data.profitProductA = 30 ∧ data.profitProductB = 20) :
  ∃ (optimalPlan : ProductionPlan),
    isValidPlan optimalPlan data ∧
    calculateProfit optimalPlan data = 2420 ∧
    ∀ (plan : ProductionPlan), isValidPlan plan data → calculateProfit plan data ≤ 2420 :=
  sorry

end NUMINAMATH_CALUDE_optimal_production_plan_l2172_217284


namespace NUMINAMATH_CALUDE_max_m_inequality_l2172_217223

theorem max_m_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 / a + 1 / b = 1 / 4) :
  (∀ m : ℝ, 2 * a + b ≥ 4 * m) → (∃ m : ℝ, m = 9 ∧ 2 * a + b = 4 * m) :=
by sorry

end NUMINAMATH_CALUDE_max_m_inequality_l2172_217223


namespace NUMINAMATH_CALUDE_prob_three_correct_is_five_twelfths_l2172_217202

-- Define the probability of A and B guessing correctly
def prob_A_correct : ℚ := 3/4
def prob_B_correct : ℚ := 2/3

-- Define the function to calculate the probability of exactly three correct guesses
def prob_three_correct : ℚ :=
  let p_A := prob_A_correct
  let p_B := prob_B_correct
  let q_A := 1 - p_A
  let q_B := 1 - p_B
  
  -- Calculate the probability of each scenario
  let scenario1 := p_A * p_A * p_A * p_B * q_B * q_B * q_B
  let scenario2 := p_A * p_A * p_A * p_B * q_B * p_B * q_B
  let scenario3 := p_A * p_A * p_A * p_B * p_B * q_B * q_B
  let scenario4 := p_A * p_A * p_A * q_B * p_B * p_B * q_B
  
  -- Sum up all scenarios
  scenario1 + scenario2 + scenario3 + scenario4

-- Theorem statement
theorem prob_three_correct_is_five_twelfths :
  prob_three_correct = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_correct_is_five_twelfths_l2172_217202


namespace NUMINAMATH_CALUDE_minimum_width_proof_l2172_217272

/-- Represents the width of the rectangular fence -/
def width : ℝ → ℝ := λ w => w

/-- Represents the length of the rectangular fence -/
def length : ℝ → ℝ := λ w => w + 20

/-- Represents the area of the rectangular fence -/
def area : ℝ → ℝ := λ w => width w * length w

/-- Represents the perimeter of the rectangular fence -/
def perimeter : ℝ → ℝ := λ w => 2 * (width w + length w)

/-- The minimum width of the rectangular fence that satisfies the given conditions -/
def min_width : ℝ := 10

theorem minimum_width_proof :
  (∀ w : ℝ, w > 0 → area w ≥ 200 → perimeter min_width ≤ perimeter w) ∧
  area min_width ≥ 200 := by sorry

end NUMINAMATH_CALUDE_minimum_width_proof_l2172_217272


namespace NUMINAMATH_CALUDE_monochromatic_triangle_exists_l2172_217280

/-- A coloring of the edges of a complete graph using three colors. -/
def ThreeColoring (n : ℕ) := Fin n → Fin n → Fin 3

/-- A complete graph with n vertices. -/
def CompleteGraph (n : ℕ) := Fin n

/-- A triangle in a graph is a set of three distinct vertices. -/
def Triangle (n : ℕ) := { t : Fin n × Fin n × Fin n // t.1 ≠ t.2.1 ∧ t.1 ≠ t.2.2 ∧ t.2.1 ≠ t.2.2 }

/-- A triangle is monochromatic if all its edges have the same color. -/
def IsMonochromatic (n : ℕ) (coloring : ThreeColoring n) (t : Triangle n) : Prop :=
  coloring t.val.1 t.val.2.1 = coloring t.val.1 t.val.2.2 ∧
  coloring t.val.1 t.val.2.1 = coloring t.val.2.1 t.val.2.2

/-- The main theorem: any 3-coloring of K₁₇ contains a monochromatic triangle. -/
theorem monochromatic_triangle_exists :
  ∀ (coloring : ThreeColoring 17),
  ∃ (t : Triangle 17), IsMonochromatic 17 coloring t :=
sorry


end NUMINAMATH_CALUDE_monochromatic_triangle_exists_l2172_217280


namespace NUMINAMATH_CALUDE_ninth_square_difference_l2172_217246

/-- The number of tiles in the n-th square of the sequence -/
def tiles_in_square (n : ℕ) : ℕ := (2 * n) ^ 2

/-- The difference in tiles between the n-th and (n-1)-th squares -/
def tile_difference (n : ℕ) : ℕ := tiles_in_square n - tiles_in_square (n - 1)

theorem ninth_square_difference : tile_difference 9 = 68 := by
  sorry

end NUMINAMATH_CALUDE_ninth_square_difference_l2172_217246


namespace NUMINAMATH_CALUDE_solve_for_b_l2172_217269

theorem solve_for_b (b : ℚ) (h : 2 * b + b / 4 = 5 / 2) : b = 10 / 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l2172_217269


namespace NUMINAMATH_CALUDE_mn_length_is_8_l2172_217288

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points on a line parallel to the x-axis -/
def distanceOnXParallelLine (p1 p2 : Point) : ℝ :=
  |p1.x - p2.x|

theorem mn_length_is_8 (x : ℝ) :
  let m : Point := ⟨x + 5, x - 4⟩
  let n : Point := ⟨-1, -2⟩
  m.y = n.y → distanceOnXParallelLine m n = 8 := by
  sorry

end NUMINAMATH_CALUDE_mn_length_is_8_l2172_217288


namespace NUMINAMATH_CALUDE_smallest_n_and_y_over_x_l2172_217278

theorem smallest_n_and_y_over_x :
  ∃ (n : ℕ+) (x y : ℝ), 
    x > 0 ∧ y > 0 ∧
    (Complex.I : ℂ)^2 = -1 ∧
    (x + 2*y*Complex.I)^(n:ℕ) = (x - 2*y*Complex.I)^(n:ℕ) ∧
    (∀ (m : ℕ+), m < n → ¬∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (a + 2*b*Complex.I)^(m:ℕ) = (a - 2*b*Complex.I)^(m:ℕ)) ∧
    n = 3 ∧
    y / x = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_and_y_over_x_l2172_217278


namespace NUMINAMATH_CALUDE_circle_equation_m_range_l2172_217253

theorem circle_equation_m_range (m : ℝ) :
  (∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), x^2 + y^2 - x + y + m = 0 ↔ (x - h)^2 + (y - k)^2 = r^2) →
  m < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_m_range_l2172_217253


namespace NUMINAMATH_CALUDE_fitness_center_membership_ratio_l2172_217205

theorem fitness_center_membership_ratio :
  ∀ (f m : ℕ) (f_avg m_avg total_avg : ℚ),
    f_avg = 45 →
    m_avg = 20 →
    total_avg = 28 →
    (f_avg * f + m_avg * m) / (f + m) = total_avg →
    (f : ℚ) / m = 8 / 17 := by
  sorry

end NUMINAMATH_CALUDE_fitness_center_membership_ratio_l2172_217205


namespace NUMINAMATH_CALUDE_triangle_area_l2172_217275

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 1 ∧ 
  t.b = Real.sqrt 3 ∧ 
  t.A + t.C = 2 * t.B

-- Theorem statement
theorem triangle_area (t : Triangle) 
  (h : triangle_conditions t) : 
  (1/2 : Real) * t.a * t.c * Real.sin t.B = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_l2172_217275


namespace NUMINAMATH_CALUDE_parabola_focus_l2172_217292

/-- A parabola is defined by the equation x = -1/4 * y^2 -/
def parabola (x y : ℝ) : Prop := x = -1/4 * y^2

/-- The focus of a parabola is a point (f, 0) where f is a real number -/
def is_focus (f : ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, parabola x y → 
    ((x - f)^2 + y^2 = (x - (-f))^2) ∧ 
    (∀ g : ℝ, g ≠ f → ∃ x y : ℝ, parabola x y ∧ (x - g)^2 + y^2 ≠ (x - (-g))^2)

/-- The focus of the parabola x = -1/4 * y^2 is at the point (-1, 0) -/
theorem parabola_focus : is_focus (-1) parabola := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l2172_217292


namespace NUMINAMATH_CALUDE_calculate_a10_l2172_217233

/-- A sequence satisfying the given property -/
def special_sequence (a : ℕ+ → ℤ) : Prop :=
  ∀ (p q : ℕ+), a (p + q) = a p + a q

/-- The theorem to prove -/
theorem calculate_a10 (a : ℕ+ → ℤ) 
  (h1 : special_sequence a) 
  (h2 : a 2 = -6) : 
  a 10 = -30 := by
sorry

end NUMINAMATH_CALUDE_calculate_a10_l2172_217233


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2172_217285

/-- A geometric sequence with positive terms -/
structure PositiveGeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h_positive : ∀ n, a n > 0
  h_geometric : ∀ n, a (n + 1) = q * a n

/-- The property that 2a_1, (1/2)a_3, and a_2 form an arithmetic sequence -/
def ArithmeticProperty (s : PositiveGeometricSequence) : Prop :=
  2 * s.a 1 + s.a 2 = 2 * ((1/2) * s.a 3)

theorem geometric_sequence_property (s : PositiveGeometricSequence) 
  (h_arith : ArithmeticProperty s) : s.q = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2172_217285


namespace NUMINAMATH_CALUDE_hexagon_area_l2172_217231

/-- Right triangle with legs 3 and 4, hypotenuse 5 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  a_eq : a = 3
  b_eq : b = 4
  c_eq : c = 5

/-- Square with side length 3 -/
def square1_area : ℝ := 9

/-- Square with side length 4 -/
def square2_area : ℝ := 16

/-- Rectangle with sides 5 and 6 -/
def rectangle_area : ℝ := 30

/-- Area of the triangle formed by extending one side of the first square -/
def extended_triangle_area : ℝ := 4.5

/-- Theorem: The area of the hexagon DEFGHI is 52.5 -/
theorem hexagon_area (t : RightTriangle) : 
  square1_area + square2_area + rectangle_area + extended_triangle_area = 52.5 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_l2172_217231


namespace NUMINAMATH_CALUDE_max_regions_circle_rectangle_triangle_l2172_217243

/-- Represents a shape in the plane -/
inductive Shape
  | Circle
  | Rectangle
  | Triangle

/-- The number of regions created by intersecting shapes in the plane -/
def num_regions (shapes : List Shape) : ℕ :=
  sorry

/-- The maximum number of regions created by intersecting a circle, rectangle, and triangle -/
theorem max_regions_circle_rectangle_triangle :
  num_regions [Shape.Circle, Shape.Rectangle, Shape.Triangle] = 21 :=
by sorry

end NUMINAMATH_CALUDE_max_regions_circle_rectangle_triangle_l2172_217243


namespace NUMINAMATH_CALUDE_hiker_distance_at_blast_l2172_217207

/-- The time in seconds for which the timer is set -/
def timer_duration : ℝ := 45

/-- The speed of the hiker in yards per second -/
def hiker_speed : ℝ := 6

/-- The speed of sound in feet per second -/
def sound_speed : ℝ := 1200

/-- The distance the hiker has traveled at time t -/
def hiker_distance (t : ℝ) : ℝ := hiker_speed * t * 3

/-- The distance the sound has traveled at time t (t ≥ timer_duration) -/
def sound_distance (t : ℝ) : ℝ := sound_speed * (t - timer_duration)

/-- The time at which the hiker hears the blast -/
noncomputable def blast_time : ℝ := 
  (sound_speed * timer_duration) / (sound_speed - hiker_speed * 3)

/-- The theorem stating that the hiker's distance when they hear the blast is approximately 275 yards -/
theorem hiker_distance_at_blast : 
  ∃ ε > 0, abs (hiker_distance blast_time / 3 - 275) < ε :=
sorry

end NUMINAMATH_CALUDE_hiker_distance_at_blast_l2172_217207


namespace NUMINAMATH_CALUDE_solve_system_l2172_217279

theorem solve_system (x y : ℝ) 
  (eq1 : 3 * x - y = 7)
  (eq2 : x + 3 * y = 7) :
  x = 2.8 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l2172_217279


namespace NUMINAMATH_CALUDE_canal_bottom_width_l2172_217247

/-- Given a trapezoidal canal cross-section with the following properties:
  - Top width: 12 meters
  - Depth: 84 meters
  - Area: 840 square meters
  Prove that the bottom width is 8 meters. -/
theorem canal_bottom_width (top_width : ℝ) (depth : ℝ) (area : ℝ) (bottom_width : ℝ) :
  top_width = 12 →
  depth = 84 →
  area = 840 →
  area = (1/2) * (top_width + bottom_width) * depth →
  bottom_width = 8 := by
sorry

end NUMINAMATH_CALUDE_canal_bottom_width_l2172_217247


namespace NUMINAMATH_CALUDE_extremum_implies_f_2_eq_18_l2172_217281

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_implies_f_2_eq_18 (a b : ℝ) :
  f' a b 1 = 0 →  -- f has a critical point at x = 1
  f a b 1 = 10 →  -- The value of f at x = 1 is 10
  f a b 2 = 18    -- Then f(2) = 18
:= by sorry

end NUMINAMATH_CALUDE_extremum_implies_f_2_eq_18_l2172_217281


namespace NUMINAMATH_CALUDE_triangle_ratio_bound_l2172_217235

theorem triangle_ratio_bound (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + b^2 + c^2) / (a*b + b*c + c*a) ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_triangle_ratio_bound_l2172_217235


namespace NUMINAMATH_CALUDE_value_of_y_minus_x_l2172_217217

theorem value_of_y_minus_x (x y : ℚ) 
  (h1 : x + y = 8) 
  (h2 : y - 3 * x = 7) : 
  y - x = 7.5 := by
sorry

end NUMINAMATH_CALUDE_value_of_y_minus_x_l2172_217217


namespace NUMINAMATH_CALUDE_smallest_factorial_divisible_by_2016_smallest_factorial_divisible_by_2016_power_10_l2172_217245

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem smallest_factorial_divisible_by_2016 :
  ∀ n : ℕ, n < 8 → ¬(2016 ∣ factorial n) ∧ (2016 ∣ factorial 8) :=
sorry

theorem smallest_factorial_divisible_by_2016_power_10 :
  ∀ n : ℕ, n < 63 → ¬(2016^10 ∣ factorial n) ∧ (2016^10 ∣ factorial 63) :=
sorry

end NUMINAMATH_CALUDE_smallest_factorial_divisible_by_2016_smallest_factorial_divisible_by_2016_power_10_l2172_217245


namespace NUMINAMATH_CALUDE_ticket_price_increase_l2172_217270

theorem ticket_price_increase (original_price : ℝ) (increase_percentage : ℝ) : 
  original_price = 85 → 
  increase_percentage = 20 → 
  original_price * (1 + increase_percentage / 100) = 102 := by
sorry

end NUMINAMATH_CALUDE_ticket_price_increase_l2172_217270


namespace NUMINAMATH_CALUDE_job_completion_time_l2172_217204

theorem job_completion_time (days : ℝ) (fraction_completed : ℝ) (h1 : fraction_completed = 5 / 8) (h2 : days = 10) :
  (days / fraction_completed) = 16 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2172_217204


namespace NUMINAMATH_CALUDE_subtract_negative_three_l2172_217264

theorem subtract_negative_three : 0 - (-3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_three_l2172_217264


namespace NUMINAMATH_CALUDE_find_y_when_x_is_12_l2172_217208

-- Define the inverse proportionality constant
def k : ℝ := 675

-- Define the relationship between x and y
def inverse_proportional (x y : ℝ) : Prop := x * y = k

-- State the theorem
theorem find_y_when_x_is_12 (x y : ℝ) 
  (h1 : inverse_proportional x y) 
  (h2 : x + y = 60) 
  (h3 : x = 3 * y) :
  x = 12 → y = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_find_y_when_x_is_12_l2172_217208


namespace NUMINAMATH_CALUDE_max_integer_k_l2172_217211

theorem max_integer_k (x y k : ℝ) : 
  x - 4*y = k - 1 →
  2*x + y = k →
  x - y ≤ 0 →
  ∀ m : ℤ, m ≤ k → m ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_max_integer_k_l2172_217211


namespace NUMINAMATH_CALUDE_green_ball_probability_l2172_217298

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- Calculates the probability of selecting a green ball from a container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- Theorem: The probability of selecting a green ball is 53/96 -/
theorem green_ball_probability :
  let containers : List Container := [
    ⟨10, 5⟩,  -- Container I
    ⟨3, 5⟩,   -- Container II
    ⟨2, 6⟩,   -- Container III
    ⟨4, 4⟩    -- Container IV
  ]
  let totalContainers : ℕ := containers.length
  let containerProbability : ℚ := 1 / totalContainers
  let totalProbability : ℚ := (containers.map greenProbability).sum * containerProbability
  totalProbability = 53 / 96 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l2172_217298


namespace NUMINAMATH_CALUDE_unique_solution_system_l2172_217282

theorem unique_solution_system (x y z : ℝ) : 
  y^3 - 6*x^2 + 12*x - 8 = 0 ∧
  z^3 - 6*y^2 + 12*y - 8 = 0 ∧
  x^3 - 6*z^2 + 12*z - 8 = 0 →
  x = 2 ∧ y = 2 ∧ z = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l2172_217282


namespace NUMINAMATH_CALUDE_cake_calories_l2172_217222

/-- Proves that given a cake with 8 slices and a pan of 6 brownies where each brownie has 375 calories,
    if the cake has 526 more calories than the pan of brownies, then each slice of the cake has 347 calories. -/
theorem cake_calories (cake_slices : ℕ) (brownie_count : ℕ) (brownie_calories : ℕ) (extra_calories : ℕ) :
  cake_slices = 8 →
  brownie_count = 6 →
  brownie_calories = 375 →
  extra_calories = 526 →
  (cake_slices * (brownie_count * brownie_calories + extra_calories) / cake_slices : ℚ) = 347 := by
  sorry

end NUMINAMATH_CALUDE_cake_calories_l2172_217222


namespace NUMINAMATH_CALUDE_power_function_domain_and_oddness_l2172_217266

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem power_function_domain_and_oddness (a : ℤ) :
  a ∈ ({-1, 1, 3} : Set ℤ) →
  (∀ x : ℝ, ∃ y : ℝ, y = x^a) ∧ is_odd_function (λ x : ℝ ↦ x^a) ↔
  a ∈ ({1, 3} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_power_function_domain_and_oddness_l2172_217266


namespace NUMINAMATH_CALUDE_complex_number_problem_l2172_217216

theorem complex_number_problem (z₁ z₂ z : ℂ) : 
  z₁ = 1 - 2*I →
  z₂ = 4 + 3*I →
  Complex.abs z = 2 →
  Complex.im z = Complex.re (3*z₁ - z₂) →
  Complex.re z < 0 ∧ Complex.im z < 0 →
  z = -Real.sqrt 2 - I * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2172_217216


namespace NUMINAMATH_CALUDE_chicken_surprise_servings_l2172_217203

/-- Calculates the number of servings for Chicken Surprise recipe -/
theorem chicken_surprise_servings 
  (chicken_pounds : ℝ) 
  (stuffing_ounces : ℝ) 
  (serving_size_ounces : ℝ) : 
  chicken_pounds = 4.5 ∧ 
  stuffing_ounces = 24 ∧ 
  serving_size_ounces = 8 → 
  (chicken_pounds * 16 + stuffing_ounces) / serving_size_ounces = 12 := by
sorry


end NUMINAMATH_CALUDE_chicken_surprise_servings_l2172_217203


namespace NUMINAMATH_CALUDE_exist_mutual_wins_l2172_217259

/-- Represents a football tournament --/
structure Tournament :=
  (num_teams : Nat)
  (scores_round1 : Fin num_teams → Nat)
  (scores_round2 : Fin num_teams → Nat)

/-- Properties of the tournament --/
def TournamentProperties (t : Tournament) : Prop :=
  t.num_teams = 20 ∧
  (∀ i j, i ≠ j → t.scores_round1 i ≠ t.scores_round1 j) ∧
  (∃ s, ∀ i, t.scores_round2 i = s)

/-- Theorem stating the existence of two teams that each won one game against the other --/
theorem exist_mutual_wins (t : Tournament) (h : TournamentProperties t) :
  ∃ i j, i ≠ j ∧ 
    t.scores_round2 i - t.scores_round1 i = 2 ∧
    t.scores_round2 j - t.scores_round1 j = 2 :=
by sorry

end NUMINAMATH_CALUDE_exist_mutual_wins_l2172_217259


namespace NUMINAMATH_CALUDE_count_k_eq_1006_l2172_217219

/-- The number of positive integers k such that (k/2013)(a+b) = lcm(a,b) has a solution in positive integers (a,b) -/
def count_k : ℕ := sorry

/-- The equation (k/2013)(a+b) = lcm(a,b) has a solution in positive integers (a,b) -/
def has_solution (k : ℕ+) : Prop :=
  ∃ (a b : ℕ+), (k : ℚ) / 2013 * (a + b) = Nat.lcm a b

theorem count_k_eq_1006 : count_k = 1006 := by sorry

end NUMINAMATH_CALUDE_count_k_eq_1006_l2172_217219


namespace NUMINAMATH_CALUDE_line_through_circle_center_l2172_217201

/-- The value of 'a' when the line 3x + y + a = 0 passes through the center of the circle x^2 + y^2 + 2x - 4y = 0 -/
theorem line_through_circle_center (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 4*y = 0 ∧ 3*x + y + a = 0 ∧ 
   ∀ x' y' : ℝ, x'^2 + y'^2 + 2*x' - 4*y' = 0 → (x - x')^2 + (y - y')^2 ≤ (x' - x)^2 + (y' - y)^2) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l2172_217201


namespace NUMINAMATH_CALUDE_cubic_factorization_l2172_217297

theorem cubic_factorization (x : ℝ) : x^3 - 4*x = x*(x+2)*(x-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2172_217297


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2172_217258

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 2| + |x + 3| ≥ 4} = {x : ℝ | x ≤ -5/2} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2172_217258


namespace NUMINAMATH_CALUDE_expression_equality_l2172_217257

theorem expression_equality : 
  Real.sqrt 8 + |1 - Real.sqrt 2| - (1 / 2)⁻¹ + (π - Real.sqrt 3)^0 = 3 * Real.sqrt 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2172_217257


namespace NUMINAMATH_CALUDE_reusable_bags_estimate_conditional_probability_second_spender_l2172_217273

/-- Represents the survey data for each age group -/
structure AgeGroupData :=
  (spent_more : Nat)  -- Number of people who spent ≥ $188
  (spent_less : Nat)  -- Number of people who spent < $188

/-- Represents the survey results -/
def survey_data : List AgeGroupData := [
  ⟨8, 2⟩,   -- [20,30)
  ⟨15, 3⟩,  -- [30,40)
  ⟨23, 5⟩,  -- [40,50)
  ⟨15, 9⟩,  -- [50,60)
  ⟨9, 11⟩   -- [60,70]
]

/-- Total number of surveyed customers -/
def total_surveyed : Nat := 100

/-- Expected number of shoppers on the event day -/
def expected_shoppers : Nat := 5000

/-- Theorem for the number of reusable shopping bags to prepare -/
theorem reusable_bags_estimate :
  (expected_shoppers * (survey_data.map (·.spent_more)).sum / total_surveyed : Nat) = 3500 := by
  sorry

/-- Theorem for the conditional probability -/
theorem conditional_probability_second_spender :
  let total_spent_more := (survey_data.map (·.spent_more)).sum
  let total_spent_less := (survey_data.map (·.spent_less)).sum
  (total_spent_more : Rat) / (total_surveyed - 1) = 70 / 99 := by
  sorry

end NUMINAMATH_CALUDE_reusable_bags_estimate_conditional_probability_second_spender_l2172_217273


namespace NUMINAMATH_CALUDE_road_construction_l2172_217213

theorem road_construction (total_length : ℚ) : 
  total_length > 0 → 
  (1 : ℚ) - (1 / 10) * total_length / total_length - (1 / 5) * total_length / total_length = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_road_construction_l2172_217213


namespace NUMINAMATH_CALUDE_divisibility_of_P_and_Q_l2172_217226

/-- Given that there exists a natural number n such that 1997 divides 111...1 (n ones),
    prove that 1997 divides both P and Q. -/
theorem divisibility_of_P_and_Q (n : ℕ) (h : ∃ k : ℕ, (10^n - 1) / 9 = 1997 * k) :
  ∃ (p q : ℕ), P = 1997 * p ∧ Q = 1997 * q :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_P_and_Q_l2172_217226


namespace NUMINAMATH_CALUDE_identical_solutions_quadratic_linear_l2172_217289

theorem identical_solutions_quadratic_linear (k : ℝ) :
  (∃! x y : ℝ, y = x^2 ∧ y = 4*x + k ∧ 
   ∀ x' y' : ℝ, y' = x'^2 ∧ y' = 4*x' + k → x' = x ∧ y' = y) ↔ k = -4 :=
by sorry

end NUMINAMATH_CALUDE_identical_solutions_quadratic_linear_l2172_217289


namespace NUMINAMATH_CALUDE_height_difference_l2172_217225

theorem height_difference (amy_height helen_height angela_height : ℕ) : 
  helen_height = amy_height + 3 →
  amy_height = 150 →
  angela_height = 157 →
  angela_height - helen_height = 4 := by
sorry

end NUMINAMATH_CALUDE_height_difference_l2172_217225


namespace NUMINAMATH_CALUDE_line_intercept_form_l2172_217240

/-- A line passing through a point with a given direction vector has a specific intercept form -/
theorem line_intercept_form (P : ℝ × ℝ) (v : ℝ × ℝ) :
  P = (2, 3) →
  v = (2, -6) →
  ∃ (f : ℝ × ℝ → ℝ), f = (λ (x, y) => x / 3 + y / 9) ∧
    (∀ (Q : ℝ × ℝ), (∃ t : ℝ, Q = (P.1 + t * v.1, P.2 + t * v.2)) ↔ f Q = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_intercept_form_l2172_217240


namespace NUMINAMATH_CALUDE_expression_square_l2172_217230

theorem expression_square (x y : ℕ) 
  (h : (1 : ℚ) / x + 1 / y + 1 / (x * y) = 1 / (x + 4) + 1 / (y - 4) + 1 / ((x + 4) * (y - 4))) : 
  ∃ n : ℕ, x * y + 4 = n^2 := by
sorry

end NUMINAMATH_CALUDE_expression_square_l2172_217230


namespace NUMINAMATH_CALUDE_fifth_month_sales_l2172_217271

def sales_1 : ℕ := 4000
def sales_2 : ℕ := 6524
def sales_3 : ℕ := 5689
def sales_4 : ℕ := 7230
def sales_6 : ℕ := 12557
def average_sale : ℕ := 7000
def num_months : ℕ := 6

theorem fifth_month_sales :
  ∃ (sales_5 : ℕ),
    (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / num_months = average_sale ∧
    sales_5 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sales_l2172_217271


namespace NUMINAMATH_CALUDE_largest_difference_l2172_217250

def A : ℕ := 3 * 2023^2024
def B : ℕ := 2023^2024
def C : ℕ := 2022 * 2023^2023
def D : ℕ := 3 * 2023^2023
def E : ℕ := 2023^2023
def F : ℕ := 2023^2022

theorem largest_difference : 
  (A - B > B - C) ∧ 
  (A - B > C - D) ∧ 
  (A - B > D - E) ∧ 
  (A - B > E - F) := by
  sorry

end NUMINAMATH_CALUDE_largest_difference_l2172_217250
