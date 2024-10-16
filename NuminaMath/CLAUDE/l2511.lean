import Mathlib

namespace NUMINAMATH_CALUDE_class_average_calculation_l2511_251164

theorem class_average_calculation (total_students : ℕ) 
  (group1_students : ℕ) (group1_average : ℚ)
  (group2_students : ℕ) (group2_average : ℚ) :
  total_students = 30 →
  group1_students = 24 →
  group2_students = 6 →
  group1_average = 85 / 100 →
  group2_average = 92 / 100 →
  (group1_students * group1_average + group2_students * group2_average) / total_students = 864 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_class_average_calculation_l2511_251164


namespace NUMINAMATH_CALUDE_ramsey_theorem_l2511_251152

/-- A coloring of edges in a complete graph with 6 vertices -/
def Coloring := Fin 6 → Fin 6 → Bool

/-- A triangle in the graph -/
def Triangle := Fin 3 → Fin 6

/-- Check if a triangle is monochromatic (all edges same color) -/
def isMonochromatic (c : Coloring) (t : Triangle) : Bool :=
  (c (t 0) (t 1) = c (t 1) (t 2)) ∧ (c (t 1) (t 2) = c (t 2) (t 0))

/-- The main theorem -/
theorem ramsey_theorem : 
  ∀ (c : Coloring), ∃ (t : Triangle), isMonochromatic c t :=
sorry


end NUMINAMATH_CALUDE_ramsey_theorem_l2511_251152


namespace NUMINAMATH_CALUDE_stamp_collection_l2511_251124

theorem stamp_collection (aj kj cj bj : ℕ) : 
  kj = aj / 2 →
  cj = 2 * kj + 5 →
  bj = 3 * aj - 3 →
  aj + kj + cj + bj = 1472 →
  aj = 267 := by
sorry

end NUMINAMATH_CALUDE_stamp_collection_l2511_251124


namespace NUMINAMATH_CALUDE_min_overs_for_max_wickets_l2511_251122

/-- Represents the maximum number of wickets a bowler can take in one over -/
def max_wickets_per_over : ℕ := 3

/-- Represents the maximum number of wickets a bowler can take in the innings -/
def max_wickets_in_innings : ℕ := 10

/-- Theorem stating the minimum number of overs required to potentially take the maximum wickets -/
theorem min_overs_for_max_wickets :
  ∃ (overs : ℕ), overs * max_wickets_per_over ≥ max_wickets_in_innings ∧
  ∀ (n : ℕ), n < overs → n * max_wickets_per_over < max_wickets_in_innings :=
by sorry

end NUMINAMATH_CALUDE_min_overs_for_max_wickets_l2511_251122


namespace NUMINAMATH_CALUDE_triangle_ABC_angle_proof_l2511_251176

def triangle_ABC_angle (A B C : ℝ × ℝ) : Prop :=
  let BA : ℝ × ℝ := (Real.sqrt 3, 1)
  let BC : ℝ × ℝ := (0, 1)
  let AB : ℝ × ℝ := (-BA.1, -BA.2)
  let angle := Real.arccos (AB.1 * BC.1 + AB.2 * BC.2) / 
               (Real.sqrt (AB.1^2 + AB.2^2) * Real.sqrt (BC.1^2 + BC.2^2))
  angle = 2 * Real.pi / 3

theorem triangle_ABC_angle_proof (A B C : ℝ × ℝ) : 
  triangle_ABC_angle A B C := by sorry

end NUMINAMATH_CALUDE_triangle_ABC_angle_proof_l2511_251176


namespace NUMINAMATH_CALUDE_ant_walk_probability_l2511_251149

/-- The probability of returning to the starting vertex after n moves on a square,
    given the probability of moving clockwise and counter-clockwise. -/
def return_probability (n : ℕ) (p_cw : ℚ) (p_ccw : ℚ) : ℚ :=
  sorry

/-- The number of ways to choose k items from n items. -/
def binomial_coefficient (n k : ℕ) : ℕ :=
  sorry

theorem ant_walk_probability :
  let n : ℕ := 6
  let p_cw : ℚ := 2/3
  let p_ccw : ℚ := 1/3
  return_probability n p_cw p_ccw = 160/729 :=
sorry

end NUMINAMATH_CALUDE_ant_walk_probability_l2511_251149


namespace NUMINAMATH_CALUDE_percentage_equality_l2511_251154

theorem percentage_equality (x y : ℝ) (h1 : 2 * x = 0.5 * y) (h2 : x = 16) : y = 64 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l2511_251154


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l2511_251188

/-- For a quadratic equation ax^2 + bx + c = 0, if one root is three times the other,
    then the coefficients a, b, and c satisfy the relationship 3b^2 = 16ac. -/
theorem quadratic_root_relation (a b c : ℝ) (h₁ : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 3 * x) →
  3 * b^2 = 16 * a * c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l2511_251188


namespace NUMINAMATH_CALUDE_inequality_proof_l2511_251141

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) :
  Real.sqrt (a * b / (c + a * b)) + 
  Real.sqrt (b * c / (a + b * c)) + 
  Real.sqrt (c * a / (b + c * a)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2511_251141


namespace NUMINAMATH_CALUDE_input_statement_is_INPUT_l2511_251106

-- Define the possible statement types
inductive Statement
  | PRINT
  | INPUT
  | IF
  | WHILE

-- Define the function of each statement
def statementFunction (s : Statement) : String :=
  match s with
  | Statement.PRINT => "output"
  | Statement.INPUT => "input"
  | Statement.IF => "conditional execution"
  | Statement.WHILE => "looping"

-- Theorem to prove
theorem input_statement_is_INPUT :
  ∃ s : Statement, statementFunction s = "input" ∧ s = Statement.INPUT :=
  sorry

end NUMINAMATH_CALUDE_input_statement_is_INPUT_l2511_251106


namespace NUMINAMATH_CALUDE_green_marbles_count_l2511_251111

/-- The number of marbles in a jar with blue, red, yellow, and green marbles -/
def total_marbles : ℕ := 164

/-- The number of yellow marbles in the jar -/
def yellow_marbles : ℕ := 14

/-- The number of blue marbles in the jar -/
def blue_marbles : ℕ := total_marbles / 2

/-- The number of red marbles in the jar -/
def red_marbles : ℕ := total_marbles / 4

/-- The number of green marbles in the jar -/
def green_marbles : ℕ := total_marbles - (blue_marbles + red_marbles + yellow_marbles)

/-- Theorem stating that the number of green marbles is 27 -/
theorem green_marbles_count : green_marbles = 27 := by
  sorry

end NUMINAMATH_CALUDE_green_marbles_count_l2511_251111


namespace NUMINAMATH_CALUDE_system_solutions_l2511_251195

/-- The system of equations -/
def system (p x y : ℝ) : Prop :=
  p * (x^2 - y^2) = (p^2 - 1) * x * y ∧ |x - 1| + |y| = 1

/-- The system has at least three different real solutions -/
def has_three_solutions (p : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ), 
    system p x₁ y₁ ∧ 
    system p x₂ y₂ ∧ 
    system p x₃ y₃ ∧ 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ 
    (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧ 
    (x₂ ≠ x₃ ∨ y₂ ≠ y₃)

/-- The main theorem -/
theorem system_solutions :
  ∀ p : ℝ, has_three_solutions p ↔ p = 1 ∨ p = -1 :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l2511_251195


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l2511_251159

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeeds where
  swimmer : ℝ  -- Speed of the swimmer in still water
  stream : ℝ   -- Speed of the stream

/-- Calculates the effective speed of the swimmer. -/
def effectiveSpeed (s : SwimmerSpeeds) (downstream : Bool) : ℝ :=
  if downstream then s.swimmer + s.stream else s.swimmer - s.stream

/-- Theorem stating that given the conditions, the swimmer's speed in still water is 5.17 km/h. -/
theorem swimmer_speed_in_still_water :
  ∃ (s : SwimmerSpeeds),
    (effectiveSpeed s true * 5 = 36) ∧
    (effectiveSpeed s false * 7 = 22) ∧
    (s.swimmer = 5.17) := by
  sorry

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l2511_251159


namespace NUMINAMATH_CALUDE_city_distance_proof_l2511_251196

def is_valid_gcd (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 13

theorem city_distance_proof :
  ∃ (S : ℕ), S > 0 ∧
  (∀ (x : ℕ), x ≤ S → is_valid_gcd (Nat.gcd x (S - x))) ∧
  (∀ (T : ℕ), T > 0 →
    (∀ (y : ℕ), y ≤ T → is_valid_gcd (Nat.gcd y (T - y))) →
    S ≤ T) ∧
  S = 39 :=
sorry

end NUMINAMATH_CALUDE_city_distance_proof_l2511_251196


namespace NUMINAMATH_CALUDE_parabola_intersection_distance_squared_l2511_251162

-- Define the parabolas and their properties
def Parabola : Type := ℝ × ℝ → Prop

-- Define the focus and directrix of a parabola
structure ParabolaProperties where
  focus : ℝ × ℝ
  directrix : ℝ → ℝ

-- Define the two parabolas
def P₁ : Parabola := sorry
def P₂ : Parabola := sorry

-- Define the properties of the two parabolas
def P₁_props : ParabolaProperties := sorry
def P₂_props : ParabolaProperties := sorry

-- Define the condition that the foci and directrices are parallel
def parallel_condition (P₁_props P₂_props : ParabolaProperties) : Prop := sorry

-- Define the condition that each focus lies on the other parabola
def focus_on_parabola (P : Parabola) (F : ℝ × ℝ) : Prop := sorry

-- Define the distance between foci
def foci_distance (P₁_props P₂_props : ParabolaProperties) : ℝ := sorry

-- Define the intersection points of the parabolas
def intersection_points (P₁ P₂ : Parabola) : Set (ℝ × ℝ) := sorry

-- Main theorem
theorem parabola_intersection_distance_squared 
  (P₁ P₂ : Parabola) 
  (P₁_props P₂_props : ParabolaProperties) 
  (h₁ : parallel_condition P₁_props P₂_props)
  (h₂ : focus_on_parabola P₂ P₁_props.focus)
  (h₃ : focus_on_parabola P₁ P₂_props.focus)
  (h₄ : foci_distance P₁_props P₂_props = 1)
  (h₅ : ∃ A B, A ∈ intersection_points P₁ P₂ ∧ B ∈ intersection_points P₁ P₂ ∧ A ≠ B) :
  ∃ A B, A ∈ intersection_points P₁ P₂ ∧ B ∈ intersection_points P₁ P₂ ∧ 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 15/4 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_distance_squared_l2511_251162


namespace NUMINAMATH_CALUDE_sin_plus_two_cos_l2511_251145

/-- Given a point P(-3, 4) on the terminal side of angle α, prove that sin α + 2cos α = -2/5 -/
theorem sin_plus_two_cos (α : Real) (P : ℝ × ℝ) (h : P = (-3, 4)) : 
  Real.sin α + 2 * Real.cos α = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_two_cos_l2511_251145


namespace NUMINAMATH_CALUDE_smallest_factor_for_square_l2511_251119

theorem smallest_factor_for_square (a : ℕ) : 
  3150 = 2 * 3^2 * 5^2 * 7 → 
  (∀ k : ℕ, k > 0 ∧ k < 14 → ¬ ∃ m : ℕ, 3150 * k = m^2) ∧
  (∃ m : ℕ, 3150 * 14 = m^2) ∧
  (14 > 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_factor_for_square_l2511_251119


namespace NUMINAMATH_CALUDE_equation_solution_l2511_251175

theorem equation_solution : ∃ (x : ℝ), x ≠ 0 ∧ x ≠ -3 ∧ (2 / x + x / (x + 3) = 1) ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2511_251175


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2511_251177

theorem inequality_solution_set (x : ℝ) : 
  (x - 2) / (3 - x) ≤ 1 ↔ x > 3 ∨ x ≤ 5/2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2511_251177


namespace NUMINAMATH_CALUDE_inequality_proof_l2511_251166

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a + b) / 2 > Real.sqrt (a * b) ∧ Real.sqrt (a * b) > (2 * a * b) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2511_251166


namespace NUMINAMATH_CALUDE_miju_handshakes_l2511_251100

/-- Calculate the total number of handshakes in a group where each person shakes hands with every other person exactly once. -/
def totalHandshakes (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- The problem statement -/
theorem miju_handshakes :
  totalHandshakes 12 = 66 := by
  sorry

end NUMINAMATH_CALUDE_miju_handshakes_l2511_251100


namespace NUMINAMATH_CALUDE_sum_of_digits_7_pow_2023_l2511_251129

theorem sum_of_digits_7_pow_2023 :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ 7^2023 ≡ 10 * a + b [ZMOD 100] ∧ a + b = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_7_pow_2023_l2511_251129


namespace NUMINAMATH_CALUDE_susan_scores_arithmetic_mean_l2511_251155

def susan_scores : List ℝ := [87, 90, 95, 98, 100]

theorem susan_scores_arithmetic_mean :
  (susan_scores.sum / susan_scores.length : ℝ) = 94 := by
  sorry

end NUMINAMATH_CALUDE_susan_scores_arithmetic_mean_l2511_251155


namespace NUMINAMATH_CALUDE_mean_height_is_70_l2511_251117

def heights : List ℕ := [58, 59, 60, 62, 63, 65, 65, 68, 70, 71, 71, 72, 76, 76, 78, 79, 79]

theorem mean_height_is_70 : 
  (List.sum heights) / (heights.length : ℚ) = 70 := by
  sorry

end NUMINAMATH_CALUDE_mean_height_is_70_l2511_251117


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l2511_251133

theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a = 4 * b →   -- ratio of angles is 4:1
  b = 18 :=     -- smaller angle is 18 degrees
by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l2511_251133


namespace NUMINAMATH_CALUDE_union_of_sets_l2511_251138

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {m : ℕ | m = 1 ∨ m = 4 ∨ m = 7}

theorem union_of_sets (h : A ∩ B = {1, 4}) : A ∪ B = {1, 2, 3, 4, 7} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l2511_251138


namespace NUMINAMATH_CALUDE_min_cubes_for_specific_box_l2511_251163

/-- Calculates the minimum number of cubes required to build a box -/
def min_cubes_for_box (length width height cube_volume : ℕ) : ℕ :=
  (length * width * height + cube_volume - 1) / cube_volume

theorem min_cubes_for_specific_box :
  min_cubes_for_box 7 18 3 9 = 42 := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_for_specific_box_l2511_251163


namespace NUMINAMATH_CALUDE_weekend_newspaper_delivery_l2511_251193

/-- The total number of newspapers delivered on the weekend -/
def total_newspapers (saturday_papers sunday_papers : ℕ) : ℕ :=
  saturday_papers + sunday_papers

/-- Theorem: The total number of newspapers delivered on the weekend is 110 -/
theorem weekend_newspaper_delivery : total_newspapers 45 65 = 110 := by
  sorry

end NUMINAMATH_CALUDE_weekend_newspaper_delivery_l2511_251193


namespace NUMINAMATH_CALUDE_vase_capacity_l2511_251102

/-- The number of flowers each vase can hold -/
def flowers_per_vase (carnations : ℕ) (roses : ℕ) (vases : ℕ) : ℕ :=
  (carnations + roses) / vases

/-- Proof that each vase can hold 6 flowers -/
theorem vase_capacity : flowers_per_vase 7 47 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_vase_capacity_l2511_251102


namespace NUMINAMATH_CALUDE_identity_proof_l2511_251197

theorem identity_proof (a b : ℝ) : a^4 + b^4 + (a+b)^4 = 2*(a^2 + a*b + b^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_identity_proof_l2511_251197


namespace NUMINAMATH_CALUDE_problem_p5_l2511_251137

theorem problem_p5 (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h1 : a^2 + b^2 = 1)
  (h2 : c^2 + d^2 = 1)
  (h3 : a * d - b * c = 1/7) :
  a * c + b * d = 4 * Real.sqrt 3 / 7 := by
sorry

end NUMINAMATH_CALUDE_problem_p5_l2511_251137


namespace NUMINAMATH_CALUDE_mans_upstream_speed_l2511_251173

/-- Given a man's downstream speed and the stream speed, calculate the man's upstream speed. -/
theorem mans_upstream_speed
  (downstream_speed : ℝ)
  (stream_speed : ℝ)
  (h1 : downstream_speed = 11)
  (h2 : stream_speed = 1.5) :
  downstream_speed - 2 * stream_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_mans_upstream_speed_l2511_251173


namespace NUMINAMATH_CALUDE_sum_of_three_squares_mod_8_l2511_251198

theorem sum_of_three_squares_mod_8 (a b c : ℤ) : (a^2 + b^2 + c^2) % 8 ≠ 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_mod_8_l2511_251198


namespace NUMINAMATH_CALUDE_people_on_stairs_l2511_251128

-- Define the number of people and steps
def num_people : ℕ := 3
def num_steps : ℕ := 7

-- Define a function to calculate the number of arrangements
def arrange_people (people : ℕ) (steps : ℕ) : ℕ :=
  -- Number of ways with each person on a different step
  (steps.choose people) * (people.factorial) +
  -- Number of ways with one step having 2 people and another having 1 person
  (people.choose 2) * (steps.choose 2) * 2

-- State the theorem
theorem people_on_stairs :
  arrange_people num_people num_steps = 336 := by
  sorry

end NUMINAMATH_CALUDE_people_on_stairs_l2511_251128


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2511_251140

theorem polynomial_divisibility (n : ℤ) : 
  ∃ k : ℤ, (n + 7)^2 - n^2 = 7 * k := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2511_251140


namespace NUMINAMATH_CALUDE_solve_inequality_range_of_a_l2511_251171

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + |x - 1| - 5

-- Theorem for part 1
theorem solve_inequality (a : ℝ) (ha : a ≠ 0) (h2 : f 2 a = 0) :
  (a = 4 → ∀ x, f x a ≤ 10 ↔ -10/3 ≤ x ∧ x ≤ 20/3) ∧
  (a = -4 → ∀ x, f x a ≤ 10 ↔ -6 ≤ x ∧ x ≤ 4) :=
sorry

-- Theorem for part 2
theorem range_of_a (a : ℝ) (ha : a < 0) 
  (h_triangle : ∃ x₁ x₂ x₃, x₁ < x₂ ∧ x₂ < x₃ ∧ f x₁ a = 0 ∧ f x₂ a < 0 ∧ f x₃ a = 0) :
  -3 ≤ a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_solve_inequality_range_of_a_l2511_251171


namespace NUMINAMATH_CALUDE_polly_tweets_theorem_l2511_251107

/-- Polly's tweeting behavior -/
structure PollyTweets where
  happy_rate : ℕ
  hungry_rate : ℕ
  mirror_rate : ℕ
  happy_duration : ℕ
  hungry_duration : ℕ
  mirror_duration : ℕ

/-- Calculate the total number of tweets -/
def total_tweets (p : PollyTweets) : ℕ :=
  p.happy_rate * p.happy_duration +
  p.hungry_rate * p.hungry_duration +
  p.mirror_rate * p.mirror_duration

/-- Theorem: Polly's total tweets equal 1340 -/
theorem polly_tweets_theorem (p : PollyTweets) 
  (h1 : p.happy_rate = 18)
  (h2 : p.hungry_rate = 4)
  (h3 : p.mirror_rate = 45)
  (h4 : p.happy_duration = 20)
  (h5 : p.hungry_duration = 20)
  (h6 : p.mirror_duration = 20) :
  total_tweets p = 1340 := by
  sorry

end NUMINAMATH_CALUDE_polly_tweets_theorem_l2511_251107


namespace NUMINAMATH_CALUDE_complex_magnitude_l2511_251146

theorem complex_magnitude (z : ℂ) (h : Complex.I * z + 2 = z - 2 * Complex.I) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2511_251146


namespace NUMINAMATH_CALUDE_revenue_in_scientific_notation_l2511_251121

/-- Represents the value of 1 billion in scientific notation -/
def billion : ℝ := 10^9

/-- The box office revenue in billions -/
def revenue : ℝ := 53.96

theorem revenue_in_scientific_notation :
  revenue * billion = 5.396 * 10^10 := by sorry

end NUMINAMATH_CALUDE_revenue_in_scientific_notation_l2511_251121


namespace NUMINAMATH_CALUDE_mrs_kaplan_pizza_slices_l2511_251169

theorem mrs_kaplan_pizza_slices :
  ∀ (bobby_pizzas : ℕ) (slices_per_pizza : ℕ) (kaplan_fraction : ℚ),
    bobby_pizzas = 2 →
    slices_per_pizza = 6 →
    kaplan_fraction = 1 / 4 →
    (↑bobby_pizzas * ↑slices_per_pizza : ℚ) * kaplan_fraction = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_mrs_kaplan_pizza_slices_l2511_251169


namespace NUMINAMATH_CALUDE_golden_hyperbola_eccentricity_l2511_251123

theorem golden_hyperbola_eccentricity :
  ∀ e : ℝ, e > 1 → e^2 - e = 1 → e = (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_hyperbola_eccentricity_l2511_251123


namespace NUMINAMATH_CALUDE_count_valid_a_l2511_251113

theorem count_valid_a : ∃! n : ℕ, n > 0 ∧ 
  (∃ a_set : Finset ℕ, 
    (∀ a ∈ a_set, a > 0 ∧ 3 ∣ a ∧ a ∣ 18 ∧ a ∣ 27) ∧
    (∀ a : ℕ, a > 0 → 3 ∣ a → a ∣ 18 → a ∣ 27 → a ∈ a_set) ∧
    Finset.card a_set = n) :=
by sorry

end NUMINAMATH_CALUDE_count_valid_a_l2511_251113


namespace NUMINAMATH_CALUDE_greatest_x_with_lcm_l2511_251101

theorem greatest_x_with_lcm (x : ℕ) : 
  (∃ (lcm : ℕ), lcm = Nat.lcm x (Nat.lcm 15 21) ∧ lcm = 105) →
  x ≤ 105 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_x_with_lcm_l2511_251101


namespace NUMINAMATH_CALUDE_rugby_league_matches_l2511_251172

/-- The number of matches played in a rugby league -/
def total_matches (n : ℕ) (k : ℕ) : ℕ :=
  k * (n.choose 2)

/-- Theorem: In a league with 10 teams, where each team plays against every other team exactly 4 times, the total number of matches played is 180. -/
theorem rugby_league_matches :
  total_matches 10 4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_rugby_league_matches_l2511_251172


namespace NUMINAMATH_CALUDE_root_product_sum_l2511_251181

-- Define the polynomial
def f (x : ℝ) : ℝ := 5 * x^3 - 10 * x^2 + 17 * x - 7

-- Define the roots
def p : ℝ := sorry
def q : ℝ := sorry
def r : ℝ := sorry

-- State the theorem
theorem root_product_sum :
  f p = 0 ∧ f q = 0 ∧ f r = 0 →
  p * q + p * r + q * r = 17 / 5 := by
  sorry

end NUMINAMATH_CALUDE_root_product_sum_l2511_251181


namespace NUMINAMATH_CALUDE_triangle_median_angle_equivalence_l2511_251179

/-- In a triangle ABC, prove that (1/a + 1/b = 1/t_a) ⟺ (C = 2π/3) -/
theorem triangle_median_angle_equivalence 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (t_a : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : A > 0 ∧ B > 0 ∧ C > 0)
  (h_sum_angles : A + B + C = π)
  (h_side_a : a = 2 * (Real.sin A))
  (h_side_b : b = 2 * (Real.sin B))
  (h_side_c : c = 2 * (Real.sin C))
  (h_median : t_a = (2 * b * c) / (b + c) * Real.cos (A / 2)) :
  (1 / a + 1 / b = 1 / t_a) ↔ (C = 2 * π / 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_median_angle_equivalence_l2511_251179


namespace NUMINAMATH_CALUDE_angle_1120_in_first_quadrant_l2511_251134

/-- An angle is in the first quadrant if its equivalent angle in [0, 360) is between 0 and 90 degrees. -/
def in_first_quadrant (angle : ℝ) : Prop :=
  0 ≤ (angle % 360) ∧ (angle % 360) < 90

/-- Theorem stating that 1120 degrees is in the first quadrant -/
theorem angle_1120_in_first_quadrant : in_first_quadrant 1120 := by
  sorry

end NUMINAMATH_CALUDE_angle_1120_in_first_quadrant_l2511_251134


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l2511_251147

theorem divisibility_equivalence (r : ℕ) (k : ℕ) :
  (∃ (m n : ℕ), m > 1 ∧ m % 2 = 1 ∧ k ∣ m^(2^r) - 1 ∧ m ∣ n^((m^(2^r) - 1)/k) + 1) ↔
  (2^(r+1) ∣ k) := by
sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l2511_251147


namespace NUMINAMATH_CALUDE_count_no_adjacent_same_digits_eq_597880_l2511_251191

/-- Counts the number of integers from 0 to 999999 with no two adjacent digits being the same. -/
def count_no_adjacent_same_digits : ℕ :=
  10 + (9^2 + 9^3 + 9^4 + 9^5 + 9^6)

/-- Theorem stating that the count of integers from 0 to 999999 with no two adjacent digits 
    being the same is equal to 597880. -/
theorem count_no_adjacent_same_digits_eq_597880 : 
  count_no_adjacent_same_digits = 597880 := by
  sorry

end NUMINAMATH_CALUDE_count_no_adjacent_same_digits_eq_597880_l2511_251191


namespace NUMINAMATH_CALUDE_book_selection_ways_l2511_251130

-- Define the number of books on the shelf
def n : ℕ := 10

-- Define the number of books to be selected
def k : ℕ := 5

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Theorem statement
theorem book_selection_ways : combination n k = 252 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_ways_l2511_251130


namespace NUMINAMATH_CALUDE_probability_skew_edges_probability_skew_edges_proof_l2511_251157

/-- A cube with edge length 1 -/
structure Cube where
  edge_length : ℝ
  edge_length_eq_one : edge_length = 1

/-- The number of edges in a cube -/
def num_edges : ℕ := 12

/-- The number of edges remaining after choosing one edge -/
def remaining_edges : ℕ := 11

/-- The number of edges parallel to a chosen edge -/
def parallel_edges : ℕ := 3

/-- The number of edges perpendicular to a chosen edge -/
def perpendicular_edges : ℕ := 4

/-- The number of edges skew to a chosen edge -/
def skew_edges : ℕ := 4

/-- The probability that two randomly chosen edges of a cube lie on skew lines -/
theorem probability_skew_edges (c : Cube) : ℚ :=
  4 / 11

/-- Proof that the probability of two randomly chosen edges of a cube lying on skew lines is 4/11 -/
theorem probability_skew_edges_proof (c : Cube) :
  probability_skew_edges c = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_skew_edges_probability_skew_edges_proof_l2511_251157


namespace NUMINAMATH_CALUDE_surface_area_of_cut_solid_l2511_251148

/-- A right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  baseSideLength : ℝ

/-- Midpoints of edges in the prism -/
structure Midpoints where
  L : ℝ × ℝ × ℝ
  M : ℝ × ℝ × ℝ
  N : ℝ × ℝ × ℝ

/-- The solid formed by cutting the prism through midpoints -/
def CutSolid (p : RightPrism) (m : Midpoints) : ℝ × ℝ × ℝ × ℝ := sorry

/-- Calculate the surface area of the cut solid -/
def surfaceArea (solid : ℝ × ℝ × ℝ × ℝ) : ℝ := sorry

/-- Main theorem: The surface area of the cut solid -/
theorem surface_area_of_cut_solid (p : RightPrism) (m : Midpoints) :
  p.height = 20 ∧ p.baseSideLength = 10 →
  surfaceArea (CutSolid p m) = 50 + (25 * Real.sqrt 3) / 4 + (5 * Real.sqrt 118.75) / 2 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_cut_solid_l2511_251148


namespace NUMINAMATH_CALUDE_finite_state_machine_cannot_generate_sqrt_two_l2511_251168

/-- Represents a finite state machine -/
structure FiniteStateMachine where
  states : Finset ℕ
  initialState : ℕ
  transition : ℕ → ℕ
  output : ℕ → ℕ

/-- Represents an infinite sequence of natural numbers -/
def InfiniteSequence := ℕ → ℕ

/-- The decimal representation of √2 -/
noncomputable def sqrtTwoDecimal : InfiniteSequence :=
  sorry

/-- A sequence is eventually periodic if there exist n and p such that
    for all k ≥ n, f(k) = f(k+p) -/
def EventuallyPeriodic (f : InfiniteSequence) : Prop :=
  ∃ n p : ℕ, p > 0 ∧ ∀ k ≥ n, f k = f (k + p)

/-- The main theorem: No finite state machine can generate the decimal representation of √2 -/
theorem finite_state_machine_cannot_generate_sqrt_two :
  ∀ (fsm : FiniteStateMachine),
  ¬∃ (f : InfiniteSequence),
    (∀ n, f n = fsm.output (fsm.transition^[n] fsm.initialState)) ∧
    (f = sqrtTwoDecimal) :=
  sorry

end NUMINAMATH_CALUDE_finite_state_machine_cannot_generate_sqrt_two_l2511_251168


namespace NUMINAMATH_CALUDE_linear_regression_approximation_l2511_251167

/-- Linear regression problem -/
theorem linear_regression_approximation 
  (b : ℝ) -- Slope of the regression line
  (x_mean y_mean : ℝ) -- Mean values of x and y
  (h1 : y_mean = b * x_mean + 0.2) -- Regression line passes through (x_mean, y_mean)
  (h2 : x_mean = 4) -- Given mean of x
  (h3 : y_mean = 5) -- Given mean of y
  : b * 2 + 0.2 = 2.6 := by
  sorry

end NUMINAMATH_CALUDE_linear_regression_approximation_l2511_251167


namespace NUMINAMATH_CALUDE_combinations_equal_sixty_l2511_251132

/-- The number of paint colors available -/
def num_colors : ℕ := 5

/-- The number of painting methods available -/
def num_methods : ℕ := 4

/-- The number of pattern options available -/
def num_patterns : ℕ := 3

/-- The total number of unique combinations of color, method, and pattern -/
def total_combinations : ℕ := num_colors * num_methods * num_patterns

/-- Theorem stating that the total number of combinations is 60 -/
theorem combinations_equal_sixty : total_combinations = 60 := by
  sorry

end NUMINAMATH_CALUDE_combinations_equal_sixty_l2511_251132


namespace NUMINAMATH_CALUDE_max_value_abc_l2511_251158

theorem max_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b * c * (a + b + c + a * b)) / ((a + b)^3 * (b + c)^3) ≤ 1/16 := by
  sorry

end NUMINAMATH_CALUDE_max_value_abc_l2511_251158


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2511_251142

theorem arithmetic_mean_problem (x y : ℚ) :
  (((3 * x + 12) + (2 * y + 18) + 5 * x + 6 * y + (3 * x + y + 16)) / 5 = 60) →
  (x = 2 * y) →
  (x = 254 / 15 ∧ y = 127 / 15) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2511_251142


namespace NUMINAMATH_CALUDE_all_three_sports_count_l2511_251187

/-- Represents a sports club with members playing various sports -/
structure SportsClub where
  total_members : ℕ
  badminton_players : ℕ
  tennis_players : ℕ
  basketball_players : ℕ
  no_sport_players : ℕ
  badminton_tennis_players : ℕ
  badminton_basketball_players : ℕ
  tennis_basketball_players : ℕ

/-- Calculates the number of members playing all three sports -/
def all_three_sports (club : SportsClub) : ℕ :=
  club.total_members - club.no_sport_players -
    (club.badminton_players + club.tennis_players + club.basketball_players -
     club.badminton_tennis_players - club.badminton_basketball_players - club.tennis_basketball_players)

/-- Theorem stating the number of members playing all three sports -/
theorem all_three_sports_count (club : SportsClub)
    (h1 : club.total_members = 60)
    (h2 : club.badminton_players = 25)
    (h3 : club.tennis_players = 30)
    (h4 : club.basketball_players = 15)
    (h5 : club.no_sport_players = 10)
    (h6 : club.badminton_tennis_players = 15)
    (h7 : club.badminton_basketball_players = 10)
    (h8 : club.tennis_basketball_players = 5) :
    all_three_sports club = 10 := by
  sorry


end NUMINAMATH_CALUDE_all_three_sports_count_l2511_251187


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_powers_divided_by_five_l2511_251156

theorem units_digit_of_sum_of_powers_divided_by_five :
  ∃ n : ℕ, (2^2023 + 3^2023) / 5 ≡ n [ZMOD 10] ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_powers_divided_by_five_l2511_251156


namespace NUMINAMATH_CALUDE_berry_count_l2511_251189

theorem berry_count (total : ℕ) (raspberries blackberries blueberries : ℕ) : 
  total = 42 →
  raspberries = total / 2 →
  blackberries = total / 3 →
  total = raspberries + blackberries + blueberries →
  blueberries = 7 := by
sorry

end NUMINAMATH_CALUDE_berry_count_l2511_251189


namespace NUMINAMATH_CALUDE_anti_terrorism_drill_mode_l2511_251112

/-- The mode of a binomial distribution with parameters n and p -/
def binomial_mode (n : ℕ) (p : ℝ) : Set ℕ :=
  {k : ℕ | k ≤ n ∧ (k.pred : ℝ) < n * p ∧ n * p ≤ k}

theorem anti_terrorism_drill_mode :
  binomial_mode 99 0.8 = {79, 80} := by
  sorry

end NUMINAMATH_CALUDE_anti_terrorism_drill_mode_l2511_251112


namespace NUMINAMATH_CALUDE_max_three_digit_with_remainders_l2511_251116

theorem max_three_digit_with_remainders :
  ∀ N : ℕ,
  (100 ≤ N ∧ N ≤ 999) →
  (N % 3 = 1) →
  (N % 7 = 3) →
  (N % 11 = 8) →
  (∀ M : ℕ, (100 ≤ M ∧ M ≤ 999) → (M % 3 = 1) → (M % 7 = 3) → (M % 11 = 8) → M ≤ N) →
  N = 976 := by
sorry

end NUMINAMATH_CALUDE_max_three_digit_with_remainders_l2511_251116


namespace NUMINAMATH_CALUDE_star_three_neg_five_l2511_251160

-- Define the new operation "*"
def star (a b : ℚ) : ℚ := a * b + a - b

-- Theorem statement
theorem star_three_neg_five : star 3 (-5) = -7 := by sorry

end NUMINAMATH_CALUDE_star_three_neg_five_l2511_251160


namespace NUMINAMATH_CALUDE_min_value_and_angle_l2511_251183

theorem min_value_and_angle (A : Real) : 
  let f := fun A => 2 * Real.sin (A / 2) - Real.cos (A / 2)
  ∃ (min_value : Real) (min_angle : Real),
    (∀ A, f A ≥ min_value) ∧
    (f min_angle = min_value) ∧
    (min_value = -1) ∧
    (min_angle = 270 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_angle_l2511_251183


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2511_251170

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧
  ∃ y : ℝ, -49 ≤ y ∧ y ≤ 49 ∧ Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2511_251170


namespace NUMINAMATH_CALUDE_equation_solution_l2511_251108

theorem equation_solution (x : ℚ) :
  x ≠ 2/3 →
  ((3*x + 2) / (3*x^2 + 4*x - 4) = 3*x / (3*x - 2)) ↔ (x = 1/3 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2511_251108


namespace NUMINAMATH_CALUDE_problem_solution_l2511_251165

theorem problem_solution (x y z : ℝ) 
  (sum_condition : x + y + z = 150)
  (equal_condition : x - 5 = y + 3 ∧ y + 3 = z^2) :
  y = 71 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2511_251165


namespace NUMINAMATH_CALUDE_number_difference_proof_l2511_251118

theorem number_difference_proof (x : ℝ) (h : x - (3/4) * x = 100) : (1/4) * x = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_proof_l2511_251118


namespace NUMINAMATH_CALUDE_dessert_menus_count_l2511_251151

/-- Represents the different types of desserts --/
inductive Dessert
| Cake
| Pie
| IceCream
| Pudding

/-- Represents the days of the week --/
inductive Day
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- A function that returns the number of possible desserts for a given day --/
def possibleDesserts (day : Day) (prevDessert : Option Dessert) : ℕ :=
  match day with
  | Day.Wednesday => 1  -- Must be ice cream
  | Day.Friday => 1     -- Must be pie
  | _ => match prevDessert with
         | none => 4    -- First day, all desserts possible
         | some _ => 3  -- Excluding previous day's dessert

/-- The theorem stating the number of possible dessert menus --/
theorem dessert_menus_count :
  (List.prod (List.map (λ d => possibleDesserts d none) [Day.Sunday, Day.Monday, Day.Tuesday, Day.Wednesday, Day.Thursday, Day.Friday, Day.Saturday])) = 324 :=
sorry

end NUMINAMATH_CALUDE_dessert_menus_count_l2511_251151


namespace NUMINAMATH_CALUDE_negative_five_squared_opposite_l2511_251185

-- Define opposite numbers
def are_opposite (a b : ℤ) : Prop := a = -b

-- Theorem statement
theorem negative_five_squared_opposite : are_opposite (-5^2) ((-5)^2) := by
  sorry

end NUMINAMATH_CALUDE_negative_five_squared_opposite_l2511_251185


namespace NUMINAMATH_CALUDE_point_P_coordinates_l2511_251110

/-- The coordinates of a point satisfying the given conditions -/
def find_point_P : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 
    let (x, y) := p
    -- P lies on the line x = 11
    x = 11 ∧
    -- Q is on the circle and is the midpoint of AP
    ∃ (q : ℝ × ℝ),
      -- Q is on the circle
      q.1^2 + q.2^2 = 25 ∧
      -- Q is on line AP
      (q.2 - 0) / (q.1 - (-5)) = (y - 0) / (x - (-5)) ∧
      -- Q is midpoint of AP
      q.1 = (x + (-5)) / 2 ∧
      q.2 = (y + 0) / 2
  }

/-- The theorem stating that P has coordinates (11, 8) or (11, -8) -/
theorem point_P_coordinates : find_point_P = {(11, 8), (11, -8)} := by
  sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l2511_251110


namespace NUMINAMATH_CALUDE_max_garden_area_l2511_251135

/-- Represents a rectangular garden with given constraints -/
structure Garden where
  width : ℝ
  length : ℝ
  fence_length : ℝ
  fence_constraint : length + 2 * width = fence_length
  size_constraint : length ≥ 2 * width

/-- The area of a garden -/
def Garden.area (g : Garden) : ℝ := g.length * g.width

/-- The maximum area of a garden given the constraints -/
theorem max_garden_area :
  ∃ (g : Garden), g.fence_length = 480 ∧ 
    (∀ (h : Garden), h.fence_length = 480 → g.area ≥ h.area) ∧
    g.area = 28800 := by
  sorry

end NUMINAMATH_CALUDE_max_garden_area_l2511_251135


namespace NUMINAMATH_CALUDE_patricia_hair_donation_l2511_251174

/-- Patricia's hair donation problem -/
theorem patricia_hair_donation :
  ∀ (current_length donation_length remaining_length growth_needed : ℕ),
    donation_length = 23 →
    remaining_length = 12 →
    growth_needed = 21 →
    current_length + growth_needed = donation_length + remaining_length →
    current_length = 14 :=
by sorry

end NUMINAMATH_CALUDE_patricia_hair_donation_l2511_251174


namespace NUMINAMATH_CALUDE_water_tank_capacity_l2511_251114

theorem water_tank_capacity : ∀ (c : ℝ), c > 0 →
  (1 / 3 : ℝ) * c + 5 = (1 / 2 : ℝ) * c → c = 30 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l2511_251114


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l2511_251125

theorem gcd_lcm_product (a b : Nat) (h1 : a = 180) (h2 : b = 250) :
  (Nat.gcd a b) * (Nat.lcm a b) = 45000 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l2511_251125


namespace NUMINAMATH_CALUDE_simplified_form_of_T_l2511_251126

theorem simplified_form_of_T (x : ℝ) : 
  (x - 2)^4 - 4*(x - 2)^3 + 6*(x - 2)^2 - 4*(x - 2) + 1 = (x - 3)^4 := by
sorry

end NUMINAMATH_CALUDE_simplified_form_of_T_l2511_251126


namespace NUMINAMATH_CALUDE_expression_simplification_l2511_251178

theorem expression_simplification (x : ℝ) : 
  (x - 1)^5 + 5*(x - 1)^4 + 10*(x - 1)^3 + 10*(x - 1)^2 + 5*(x - 1) = x^5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2511_251178


namespace NUMINAMATH_CALUDE_inequality_proof_l2511_251131

theorem inequality_proof (a b c d e f : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f) : 
  (((a * b * c) / (a + b + d)) ^ (1/3 : ℝ)) + (((d * e * f) / (c + e + f)) ^ (1/3 : ℝ)) < 
  ((a + b + d) * (c + e + f)) ^ (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2511_251131


namespace NUMINAMATH_CALUDE_greatest_perimeter_of_special_triangle_l2511_251103

theorem greatest_perimeter_of_special_triangle :
  ∀ a b c : ℕ,
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (b = 4 * a ∨ a = 4 * b ∨ c = 4 * a ∨ a = 4 * c ∨ b = 4 * c ∨ c = 4 * b) →
  (a = 12 ∨ b = 12 ∨ c = 12) →
  (a + b > c ∧ b + c > a ∧ a + c > b) →
  a + b + c ≤ 27 :=
by sorry

end NUMINAMATH_CALUDE_greatest_perimeter_of_special_triangle_l2511_251103


namespace NUMINAMATH_CALUDE_even_monotone_decreasing_inequality_l2511_251153

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def monotone_decreasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f y < f x

-- State the theorem
theorem even_monotone_decreasing_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_mono : monotone_decreasing_on_pos f) : 
  f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end NUMINAMATH_CALUDE_even_monotone_decreasing_inequality_l2511_251153


namespace NUMINAMATH_CALUDE_counterexample_exists_l2511_251109

theorem counterexample_exists : ∃ (x y : ℝ), x > y ∧ x^2 ≤ y^2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2511_251109


namespace NUMINAMATH_CALUDE_rectangle_area_l2511_251143

theorem rectangle_area (width : ℝ) (length : ℝ) (perimeter : ℝ) (area : ℝ) :
  length = 2 * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 120 →
  area = length * width →
  area = 800 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2511_251143


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2511_251136

theorem quadratic_roots_sum_product (k p : ℝ) : 
  (∃ α β : ℝ, 3 * α^2 - k * α + p = 0 ∧ 3 * β^2 - k * β + p = 0) →
  (∃ α β : ℝ, α + β = 9 ∧ α * β = 10) →
  k + p = 57 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2511_251136


namespace NUMINAMATH_CALUDE_exists_convex_quadrilateral_geometric_progression_l2511_251105

/-- A convex quadrilateral with sides a₁, a₂, a₃, a₄ and diagonals d₁, d₂ -/
structure ConvexQuadrilateral where
  a₁ : ℝ
  a₂ : ℝ
  a₃ : ℝ
  a₄ : ℝ
  d₁ : ℝ
  d₂ : ℝ
  a₁_pos : a₁ > 0
  a₂_pos : a₂ > 0
  a₃_pos : a₃ > 0
  a₄_pos : a₄ > 0
  d₁_pos : d₁ > 0
  d₂_pos : d₂ > 0
  convex : a₁ + a₂ + a₃ > a₄ ∧
           a₁ + a₂ + a₄ > a₃ ∧
           a₁ + a₃ + a₄ > a₂ ∧
           a₂ + a₃ + a₄ > a₁

/-- Predicate to check if a sequence forms a geometric progression -/
def IsGeometricProgression (seq : List ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ i : Fin (seq.length - 1), seq[i.val + 1] = seq[i.val] * r

/-- Theorem stating the existence of a convex quadrilateral with sides and diagonals
    forming a geometric progression -/
theorem exists_convex_quadrilateral_geometric_progression :
  ∃ q : ConvexQuadrilateral, IsGeometricProgression [q.a₁, q.a₂, q.a₃, q.a₄, q.d₁, q.d₂] :=
sorry

end NUMINAMATH_CALUDE_exists_convex_quadrilateral_geometric_progression_l2511_251105


namespace NUMINAMATH_CALUDE_average_of_six_numbers_l2511_251144

theorem average_of_six_numbers 
  (total_average : ℝ) 
  (second_pair_average : ℝ) 
  (third_pair_average : ℝ) 
  (h1 : total_average = 3.9) 
  (h2 : second_pair_average = 3.85) 
  (h3 : third_pair_average = 4.45) : 
  ∃ first_pair_average : ℝ, first_pair_average = 3.4 ∧ 
  6 * total_average = 2 * first_pair_average + 2 * second_pair_average + 2 * third_pair_average :=
sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l2511_251144


namespace NUMINAMATH_CALUDE_rectangle_ratio_extension_l2511_251139

theorem rectangle_ratio_extension (x : ℝ) :
  (2*x > 0) →
  (5*x > 0) →
  ((2*x + 9) / (5*x + 9) = 3/7) →
  ((2*x + 18) / (5*x + 18) = 5/11) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_ratio_extension_l2511_251139


namespace NUMINAMATH_CALUDE_david_average_marks_l2511_251184

def david_marks : List ℝ := [96, 95, 82, 87, 92]
def num_subjects : ℕ := 5

theorem david_average_marks :
  (david_marks.sum / num_subjects : ℝ) = 90.4 := by
  sorry

end NUMINAMATH_CALUDE_david_average_marks_l2511_251184


namespace NUMINAMATH_CALUDE_second_player_winning_strategy_l2511_251180

/-- Represents a position on the 10x10 board -/
def Position := Fin 10 × Fin 10

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents a mark on the board -/
inductive Mark
| X
| O

/-- Represents the game state -/
structure GameState where
  board : Position → Option Mark
  currentPlayer : Player

/-- Checks if a position is winning -/
def isWinningPosition (board : Position → Option Mark) : Bool :=
  sorry

/-- Applies a move to the game state -/
def applyMove (state : GameState) (pos : Position) : GameState :=
  sorry

/-- Represents a strategy for a player -/
def Strategy := GameState → Position

/-- Checks if a strategy is winning for a player -/
def isWinningStrategy (player : Player) (strategy : Strategy) : Prop :=
  sorry

/-- The main theorem stating that the second player has a winning strategy -/
theorem second_player_winning_strategy :
  ∃ (strategy : Strategy), isWinningStrategy Player.Second strategy :=
sorry

end NUMINAMATH_CALUDE_second_player_winning_strategy_l2511_251180


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l2511_251190

theorem quadratic_equation_m_value :
  ∀ m : ℝ,
  (∀ x : ℝ, (m + 1) * x^(m * (m - 2) - 1) + 2 * m * x - 1 = 0 → ∃ a b c : ℝ, a ≠ 0 ∧ (m + 1) * x^(m * (m - 2) - 1) + 2 * m * x - 1 = a * x^2 + b * x + c) →
  m = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l2511_251190


namespace NUMINAMATH_CALUDE_correct_number_of_ways_l2511_251127

/-- The number of ways to choose k elements from a set of n elements -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The total number of people in the group -/
def total_people : ℕ := 12

/-- The number of people to be chosen -/
def chosen_people : ℕ := 5

/-- The number of special people (A, B, C) -/
def special_people : ℕ := 3

/-- The maximum number of special people that can be chosen -/
def max_special_chosen : ℕ := 2

/-- The number of ways to choose 5 people from 12, with at most 2 from a specific group of 3 -/
def ways_to_choose : ℕ := 
  binomial (total_people - special_people) chosen_people + 
  binomial special_people 1 * binomial (total_people - special_people) (chosen_people - 1) +
  binomial special_people 2 * binomial (total_people - special_people) (chosen_people - 2)

theorem correct_number_of_ways : ways_to_choose = 756 := by sorry

end NUMINAMATH_CALUDE_correct_number_of_ways_l2511_251127


namespace NUMINAMATH_CALUDE_cubic_expansion_coefficient_l2511_251150

theorem cubic_expansion_coefficient (a a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) →
  a₂ = 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_expansion_coefficient_l2511_251150


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l2511_251192

def U : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}
def N : Set Int := {0, -3, -4}

theorem complement_intersection_problem : (Mᶜ ∩ N) = {-3, -4} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l2511_251192


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2511_251186

theorem inequality_equivalence (b c : ℝ) : 
  (∀ x : ℝ, |2*x - 3| < 5 ↔ -x^2 + b*x + c > 0) → b + c = 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2511_251186


namespace NUMINAMATH_CALUDE_intersection_area_zero_l2511_251104

-- Define the triangle vertices
def P : ℝ × ℝ := (3, -2)
def Q : ℝ × ℝ := (5, 4)
def R : ℝ × ℝ := (1, 1)

-- Define the reflection function across y = 0
def reflect (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Define the triangle and its reflection
def triangle : Set (ℝ × ℝ) := {p | ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧ 
  p = (a * P.1 + b * Q.1 + c * R.1, a * P.2 + b * Q.2 + c * R.2)}

def reflectedTriangle : Set (ℝ × ℝ) := {p | ∃ q ∈ triangle, p = reflect q}

-- State the theorem
theorem intersection_area_zero : 
  MeasureTheory.volume (triangle ∩ reflectedTriangle) = 0 := by sorry

end NUMINAMATH_CALUDE_intersection_area_zero_l2511_251104


namespace NUMINAMATH_CALUDE_digit_equation_solution_l2511_251194

theorem digit_equation_solution : ∃! (Θ : ℕ), Θ > 0 ∧ Θ < 10 ∧ (476 : ℚ) / Θ = 50 + 4 * Θ :=
by sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l2511_251194


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2511_251115

-- Define the sets A and B
def A : Set ℝ := {-1, 0, 2}
def B (a : ℝ) : Set ℝ := {2^a}

-- State the theorem
theorem subset_implies_a_equals_one (a : ℝ) : B a ⊆ A → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2511_251115


namespace NUMINAMATH_CALUDE_john_tax_payment_l2511_251161

/-- Calculates the total tax payment given earnings, deductions, and tax rates -/
def calculate_tax (earnings deductions : ℕ) (low_rate high_rate : ℚ) : ℚ :=
  let taxable_income := earnings - deductions
  let low_bracket := min taxable_income 20000
  let high_bracket := taxable_income - low_bracket
  (low_bracket : ℚ) * low_rate + (high_bracket : ℚ) * high_rate

/-- Theorem stating that John's tax payment is $12,000 -/
theorem john_tax_payment :
  calculate_tax 100000 30000 (1/10) (1/5) = 12000 := by
  sorry

end NUMINAMATH_CALUDE_john_tax_payment_l2511_251161


namespace NUMINAMATH_CALUDE_f_decreasing_l2511_251199

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 12*x + b

-- State the theorem
theorem f_decreasing (b : ℝ) : 
  ∀ x ∈ Set.Icc (-2) 2, 
    ∀ y ∈ Set.Icc (-2) 2, 
      x < y → f b x > f b y :=
by
  sorry


end NUMINAMATH_CALUDE_f_decreasing_l2511_251199


namespace NUMINAMATH_CALUDE_fifteen_points_max_planes_l2511_251120

/-- The maximum number of planes determined by n points in space,
    where no four points are coplanar. -/
def maxPlanes (n : ℕ) : ℕ := Nat.choose n 3

/-- The condition that no four points are coplanar is implicitly assumed
    in the definition of maxPlanes. -/
theorem fifteen_points_max_planes :
  maxPlanes 15 = 455 := by sorry

end NUMINAMATH_CALUDE_fifteen_points_max_planes_l2511_251120


namespace NUMINAMATH_CALUDE_positive_test_probability_l2511_251182

/-- Probability of a positive test result given the disease prevalence and test characteristics -/
theorem positive_test_probability
  (P_A : ℝ)
  (P_B_given_A : ℝ)
  (P_B_given_not_A : ℝ)
  (h1 : P_A = 0.01)
  (h2 : P_B_given_A = 0.99)
  (h3 : P_B_given_not_A = 0.1)
  (h4 : ∀ (P_A P_B_given_A P_B_given_not_A : ℝ),
    P_A ≥ 0 ∧ P_A ≤ 1 →
    P_B_given_A ≥ 0 ∧ P_B_given_A ≤ 1 →
    P_B_given_not_A ≥ 0 ∧ P_B_given_not_A ≤ 1 →
    P_B_given_A * P_A + P_B_given_not_A * (1 - P_A) ≥ 0 ∧
    P_B_given_A * P_A + P_B_given_not_A * (1 - P_A) ≤ 1) :
  P_B_given_A * P_A + P_B_given_not_A * (1 - P_A) = 0.1089 := by
  sorry


end NUMINAMATH_CALUDE_positive_test_probability_l2511_251182
