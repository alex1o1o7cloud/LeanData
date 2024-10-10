import Mathlib

namespace value_of_expression_l3696_369686

theorem value_of_expression (a : ℝ) (h : a^2 - 2*a - 2 = 3) : 3*a*(a-2) = 15 := by
  sorry

end value_of_expression_l3696_369686


namespace set_A_theorem_l3696_369614

def A (a : ℝ) := {x : ℝ | 2 * x + a > 0}

theorem set_A_theorem (a : ℝ) :
  (1 ∉ A a) → (2 ∈ A a) → -4 < a ∧ a ≤ -2 := by
  sorry

end set_A_theorem_l3696_369614


namespace intersection_range_l3696_369654

-- Define the semicircle
def semicircle (x y : ℝ) : Prop := x^2 + y^2 = 9 ∧ y ≥ 0

-- Define the line
def line (k x y : ℝ) : Prop := y = k*(x-3) + 4

-- Define the condition for two distinct solutions
def has_two_distinct_solutions (k : ℝ) : Prop :=
  ∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    semicircle x₁ y₁ ∧ semicircle x₂ y₂ ∧ 
    line k x₁ y₁ ∧ line k x₂ y₂

-- Theorem statement
theorem intersection_range :
  ∀ k : ℝ, has_two_distinct_solutions k ↔ 7/24 < k ∧ k ≤ 2/3 :=
sorry

end intersection_range_l3696_369654


namespace abc_mod_9_l3696_369647

theorem abc_mod_9 (a b c : ℕ) (ha : a < 9) (hb : b < 9) (hc : c < 9)
  (h1 : (a + 3*b + 2*c) % 9 = 0)
  (h2 : (2*a + 2*b + 3*c) % 9 = 3)
  (h3 : (3*a + b + 2*c) % 9 = 6) :
  (a * b * c) % 9 = 0 := by
sorry

end abc_mod_9_l3696_369647


namespace farm_legs_count_l3696_369639

/-- The number of legs for a given animal type -/
def legs_per_animal (animal : String) : ℕ :=
  match animal with
  | "cow" => 4
  | "duck" => 2
  | _ => 0

/-- The total number of animals in the farm -/
def total_animals : ℕ := 15

/-- The number of cows in the farm -/
def num_cows : ℕ := 6

/-- The number of ducks in the farm -/
def num_ducks : ℕ := total_animals - num_cows

theorem farm_legs_count : 
  legs_per_animal "cow" * num_cows + legs_per_animal "duck" * num_ducks = 42 := by
sorry

end farm_legs_count_l3696_369639


namespace largest_prime_factor_of_9689_l3696_369643

theorem largest_prime_factor_of_9689 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 9689 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 9689 → q ≤ p :=
by sorry

end largest_prime_factor_of_9689_l3696_369643


namespace friend_riding_area_l3696_369616

/-- Given a rectangular riding area of width 2 and length 3, 
    prove that another area 4 times larger is 24 square blocks. -/
theorem friend_riding_area (width : ℕ) (length : ℕ) (multiplier : ℕ) : 
  width = 2 → length = 3 → multiplier = 4 → 
  (width * length * multiplier : ℕ) = 24 := by
  sorry

end friend_riding_area_l3696_369616


namespace circle_trajectory_and_line_l3696_369682

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 2
def C₂ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 2

-- Define the trajectory of M
def M_trajectory (x y : ℝ) : Prop := x^2 / 2 - y^2 / 14 = 1 ∧ x ≥ Real.sqrt 2

-- Define the line l
def line_l (x y : ℝ) : Prop := y = 14 * x - 27

-- Define the point P
def P : ℝ × ℝ := (2, 1)

-- Theorem statement
theorem circle_trajectory_and_line :
  ∃ (M : Set (ℝ × ℝ)),
    (∀ (x y : ℝ), (x, y) ∈ M → ∃ (r : ℝ), r > 0 ∧
      (∀ (x₁ y₁ : ℝ), C₁ x₁ y₁ → ((x - x₁)^2 + (y - y₁)^2 = (r + Real.sqrt 2)^2)) ∧
      (∀ (x₂ y₂ : ℝ), C₂ x₂ y₂ → ((x - x₂)^2 + (y - y₂)^2 = (r - Real.sqrt 2)^2))) ∧
    (∀ (x y : ℝ), (x, y) ∈ M ↔ M_trajectory x y) ∧
    (∃ (A B : ℝ × ℝ), A ∈ M ∧ B ∈ M ∧ A ≠ B ∧
      ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = P ∧
      (∀ (x y : ℝ), line_l x y ↔ (y - A.2) / (x - A.1) = (B.2 - A.2) / (B.1 - A.1) ∧ x ≠ A.1)) :=
by sorry

end circle_trajectory_and_line_l3696_369682


namespace steven_apple_count_l3696_369601

/-- The number of apples Jake has -/
def jake_apples : ℕ := 11

/-- The difference between Jake's and Steven's apple count -/
def apple_difference : ℕ := 3

/-- Proves that Steven has 8 apples given the conditions -/
theorem steven_apple_count : ∃ (steven_apples : ℕ), steven_apples = jake_apples - apple_difference :=
  sorry

end steven_apple_count_l3696_369601


namespace grade12_population_l3696_369603

/-- Represents the number of students in each grade (10, 11, 12) -/
structure GradePopulation where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- The ratio of students in grades 10, 11, and 12 -/
def gradeRatio : GradePopulation := ⟨10, 8, 7⟩

/-- The number of students sampled -/
def sampleSize : ℕ := 200

/-- The sampling probability for each student -/
def samplingProbability : ℚ := 1/5

theorem grade12_population (pop : GradePopulation) :
  pop.grade10 / gradeRatio.grade10 = pop.grade11 / gradeRatio.grade11 ∧
  pop.grade11 / gradeRatio.grade11 = pop.grade12 / gradeRatio.grade12 ∧
  pop.grade10 + pop.grade11 + pop.grade12 = sampleSize / samplingProbability →
  pop.grade12 = 280 := by
sorry

end grade12_population_l3696_369603


namespace both_correct_calculation_l3696_369663

/-- Represents a class test scenario -/
structure ClassTest where
  total : ℕ
  correct1 : ℕ
  correct2 : ℕ
  absent : ℕ

/-- Calculates the number of students who answered both questions correctly -/
def bothCorrect (test : ClassTest) : ℕ :=
  test.correct1 + test.correct2 - (test.total - test.absent)

/-- Theorem stating the number of students who answered both questions correctly -/
theorem both_correct_calculation (test : ClassTest) 
  (h1 : test.total = 25)
  (h2 : test.correct1 = 22)
  (h3 : test.correct2 = 20)
  (h4 : test.absent = 3) :
  bothCorrect test = 17 := by
  sorry

end both_correct_calculation_l3696_369663


namespace f_properties_l3696_369650

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 2^x

-- Theorem statement
theorem f_properties :
  f (f 4) = 1/2 ∧ ∀ x, f x ≤ 1 :=
by sorry

end f_properties_l3696_369650


namespace sufficient_not_necessary_l3696_369632

/-- The complex number i -/
def i : ℂ := Complex.I

/-- Predicate for the condition (a + bi)^2 = 2i -/
def condition (a b : ℝ) : Prop := (Complex.mk a b)^2 = 2*i

/-- Statement: a=b=1 is sufficient but not necessary for (a + bi)^2 = 2i -/
theorem sufficient_not_necessary :
  (∀ a b : ℝ, a = 1 ∧ b = 1 → condition a b) ∧
  (∃ a b : ℝ, condition a b ∧ (a ≠ 1 ∨ b ≠ 1)) :=
sorry

end sufficient_not_necessary_l3696_369632


namespace red_balls_count_l3696_369645

/-- Given a bag with 2400 balls of red, green, and blue colors,
    where the ratio of red:green:blue is 15:13:17,
    prove that the number of red balls is 795. -/
theorem red_balls_count (total : ℕ) (red green blue : ℕ) :
  total = 2400 →
  red + green + blue = 45 →
  red = 15 →
  green = 13 →
  blue = 17 →
  red * (total / (red + green + blue)) = 795 := by
  sorry


end red_balls_count_l3696_369645


namespace hyperbola_line_intersection_specific_intersection_l3696_369646

/-- The hyperbola C: x²/a² - y² = 1 (a > 0) -/
def C (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 = 1 ∧ a > 0

/-- The line l: x + y = 1 -/
def l (x y : ℝ) : Prop := x + y = 1

/-- P is the intersection point of line l and the y-axis -/
def P : ℝ × ℝ := (0, 1)

/-- A and B are distinct intersection points of C and l -/
def intersectionPoints (a : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), A ≠ B ∧ C a A.1 A.2 ∧ l A.1 A.2 ∧ C a B.1 B.2 ∧ l B.1 B.2

/-- PA = (5/12)PB -/
def vectorRelation (A B : ℝ × ℝ) : Prop :=
  (A.1 - P.1, A.2 - P.2) = (5/12 * (B.1 - P.1), 5/12 * (B.2 - P.2))

theorem hyperbola_line_intersection (a : ℝ) :
  intersectionPoints a ↔ (0 < a ∧ a < Real.sqrt 2 ∧ a ≠ 1) :=
sorry

theorem specific_intersection (a : ℝ) (A B : ℝ × ℝ) :
  C a A.1 A.2 ∧ l A.1 A.2 ∧ C a B.1 B.2 ∧ l B.1 B.2 ∧ vectorRelation A B →
  a = 17/13 :=
sorry

end hyperbola_line_intersection_specific_intersection_l3696_369646


namespace bc_values_l3696_369660

theorem bc_values (a b c : ℝ) 
  (sum_eq : a + b + c = 100)
  (prod_sum_eq : a * b + b * c + c * a = 20)
  (mixed_prod_eq : (a + b) * (a + c) = 24) :
  b * c = -176 ∨ b * c = 224 :=
by sorry

end bc_values_l3696_369660


namespace square_equality_solutions_l3696_369691

theorem square_equality_solutions (x : ℝ) : 
  (x + 1)^2 = (2*x - 1)^2 ↔ x = 0 ∨ x = 2 := by
sorry

end square_equality_solutions_l3696_369691


namespace special_sequence_2016th_term_l3696_369624

/-- A sequence with specific properties -/
def special_sequence (a : ℕ → ℝ) : Prop :=
  a 4 = 1 ∧ 
  a 11 = 9 ∧ 
  ∀ n : ℕ, a n + a (n + 1) + a (n + 2) = 15

/-- The 2016th term of the special sequence is 5 -/
theorem special_sequence_2016th_term (a : ℕ → ℝ) 
  (h : special_sequence a) : a 2016 = 5 := by
  sorry

end special_sequence_2016th_term_l3696_369624


namespace quadratic_triple_root_relation_l3696_369613

/-- For a quadratic equation px^2 + qx + r = 0, if one root is triple the other, 
    then 3q^2 = 16pr -/
theorem quadratic_triple_root_relation (p q r : ℝ) (x₁ x₂ : ℝ) : 
  (p * x₁^2 + q * x₁ + r = 0) →
  (p * x₂^2 + q * x₂ + r = 0) →
  (x₂ = 3 * x₁) →
  (3 * q^2 = 16 * p * r) := by
  sorry

end quadratic_triple_root_relation_l3696_369613


namespace age_sum_problem_l3696_369672

theorem age_sum_problem (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 128 → a + b + c = 18 := by
  sorry

end age_sum_problem_l3696_369672


namespace tangent_line_passes_through_fixed_point_l3696_369628

/-- The parabola Γ -/
def Γ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 = 4 * p.2}

/-- The point P from which tangents are drawn -/
def P (m : ℝ) : ℝ × ℝ := (m, -4)

/-- The fixed point through which AB always passes -/
def fixedPoint : ℝ × ℝ := (0, 4)

/-- Theorem stating that AB always passes through the fixed point -/
theorem tangent_line_passes_through_fixed_point (m : ℝ) :
  ∀ A B : ℝ × ℝ,
  A ∈ Γ → B ∈ Γ →
  (∃ t : ℝ, A = (1 - t) • P m + t • B) →
  (∃ s : ℝ, B = (1 - s) • P m + s • A) →
  ∃ r : ℝ, fixedPoint = (1 - r) • A + r • B :=
sorry

end tangent_line_passes_through_fixed_point_l3696_369628


namespace max_xy_given_constraint_l3696_369673

theorem max_xy_given_constraint (x y : ℝ) (h : 2 * x + y = 1) : 
  ∃ (max : ℝ), max = (1/8 : ℝ) ∧ ∀ (x' y' : ℝ), 2 * x' + y' = 1 → x' * y' ≤ max :=
by sorry

end max_xy_given_constraint_l3696_369673


namespace manny_cookie_pies_l3696_369622

theorem manny_cookie_pies :
  ∀ (num_pies : ℕ) (num_classmates : ℕ) (num_teacher : ℕ) (slices_per_pie : ℕ) (slices_left : ℕ),
    num_classmates = 24 →
    num_teacher = 1 →
    slices_per_pie = 10 →
    slices_left = 4 →
    (num_pies * slices_per_pie = num_classmates + num_teacher + 1 + slices_left) →
    num_pies = 3 :=
by
  sorry

#check manny_cookie_pies

end manny_cookie_pies_l3696_369622


namespace alternate_tree_planting_l3696_369687

/-- The number of ways to arrange n items from a set of m items, where order matters -/
def arrangements (m n : ℕ) : ℕ := sorry

/-- The number of ways to plant w willow trees and p poplar trees alternately in a row -/
def alternate_tree_arrangements (w p : ℕ) : ℕ :=
  2 * arrangements w w * arrangements p p

theorem alternate_tree_planting :
  alternate_tree_arrangements 4 4 = 1152 := by sorry

end alternate_tree_planting_l3696_369687


namespace sunglasses_cap_probability_l3696_369692

theorem sunglasses_cap_probability 
  (total_sunglasses : ℕ) 
  (total_caps : ℕ) 
  (total_hats : ℕ) 
  (prob_cap_and_sunglasses : ℚ) 
  (h1 : total_sunglasses = 80) 
  (h2 : total_caps = 60) 
  (h3 : total_hats = 40) 
  (h4 : prob_cap_and_sunglasses = 1/3) :
  (total_caps * prob_cap_and_sunglasses) / total_sunglasses = 1/4 := by
  sorry

end sunglasses_cap_probability_l3696_369692


namespace num_biology_books_is_15_l3696_369623

def num_chemistry_books : ℕ := 8
def total_ways_to_pick : ℕ := 2940

-- Function to calculate combinations
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem to prove
theorem num_biology_books_is_15 : 
  ∃ (B : ℕ), choose B 2 * choose num_chemistry_books 2 = total_ways_to_pick ∧ B = 15 :=
by sorry

end num_biology_books_is_15_l3696_369623


namespace max_distance_squared_l3696_369602

theorem max_distance_squared (x y : ℝ) : 
  (x + 2)^2 + (y - 5)^2 = 9 → 
  ∃ (max : ℝ), max = 64 ∧ ∀ (x' y' : ℝ), (x' + 2)^2 + (y' - 5)^2 = 9 → (x' - 1)^2 + (y' - 1)^2 ≤ max := by
sorry

end max_distance_squared_l3696_369602


namespace hide_and_seek_players_l3696_369689

-- Define variables for each person
variable (Andrew Boris Vasya Gena Denis : Prop)

-- Define the conditions
axiom condition1 : Andrew → (Boris ∧ ¬Vasya)
axiom condition2 : Boris → (Gena ∨ Denis)
axiom condition3 : ¬Vasya → (¬Boris ∧ ¬Denis)
axiom condition4 : ¬Andrew → (Boris ∧ ¬Gena)

-- Theorem to prove
theorem hide_and_seek_players :
  (Boris ∧ Vasya ∧ Denis ∧ ¬Andrew ∧ ¬Gena) :=
sorry

end hide_and_seek_players_l3696_369689


namespace vector_expression_equality_l3696_369698

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

theorem vector_expression_equality (a b : V) :
  (1 / 2 : ℝ) • ((2 : ℝ) • a - (4 : ℝ) • b) + (2 : ℝ) • b = a := by sorry

end vector_expression_equality_l3696_369698


namespace min_square_value_l3696_369683

theorem min_square_value (a b : ℕ+) 
  (h1 : ∃ m : ℕ+, (15 * a + 16 * b : ℕ) = m ^ 2)
  (h2 : ∃ n : ℕ+, (16 * a - 15 * b : ℕ) = n ^ 2) :
  min (15 * a + 16 * b) (16 * a - 15 * b) ≥ 481 ^ 2 := by
sorry

end min_square_value_l3696_369683


namespace distinct_prime_factors_of_90_l3696_369688

theorem distinct_prime_factors_of_90 : Nat.card (Nat.factors 90).toFinset = 3 := by
  sorry

end distinct_prime_factors_of_90_l3696_369688


namespace total_amount_calculation_l3696_369675

/-- The total amount Kanul had -/
def T : ℝ := sorry

/-- Theorem stating the relationship between the total amount and the expenses -/
theorem total_amount_calculation :
  T = 3000 + 2000 + 0.1 * T ∧ T = 5000 / 0.9 := by sorry

end total_amount_calculation_l3696_369675


namespace chord_equation_of_ellipse_l3696_369608

/-- The equation of a line that forms a chord of the ellipse x^2/2 + y^2 = 1,
    bisected by the point (1/2, 1/2) -/
theorem chord_equation_of_ellipse (x y : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 / 2 + y₁^2 = 1) ∧
    (x₂^2 / 2 + y₂^2 = 1) ∧
    ((x₁ + x₂) / 2 = 1/2) ∧
    ((y₁ + y₂) / 2 = 1/2) ∧
    (y - y₁) = ((y₂ - y₁) / (x₂ - x₁)) * (x - x₁)) →
  2*x + 4*y - 3 = 0 :=
by sorry

end chord_equation_of_ellipse_l3696_369608


namespace snowflake_area_ratio_l3696_369699

/-- Represents the snowflake shape after n iterations --/
def Snowflake (n : ℕ) : Type := Unit

/-- The area of the snowflake shape after n iterations --/
def area (s : Snowflake n) : ℚ := sorry

/-- The initial equilateral triangle --/
def initial_triangle : Snowflake 0 := sorry

/-- The snowflake shape after one iteration --/
def first_iteration : Snowflake 1 := sorry

/-- The snowflake shape after two iterations --/
def second_iteration : Snowflake 2 := sorry

theorem snowflake_area_ratio :
  area second_iteration / area initial_triangle = 40 / 27 := by sorry

end snowflake_area_ratio_l3696_369699


namespace tan_sixty_degrees_l3696_369690

theorem tan_sixty_degrees : Real.tan (60 * π / 180) = Real.sqrt 3 := by
  sorry

end tan_sixty_degrees_l3696_369690


namespace equation_one_integral_root_l3696_369679

theorem equation_one_integral_root :
  ∃! (x : ℤ), x - 12 / (x - 3) = 5 - 12 / (x - 3) :=
by sorry

end equation_one_integral_root_l3696_369679


namespace meaningful_fraction_l3696_369651

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 := by sorry

end meaningful_fraction_l3696_369651


namespace negation_of_all_squares_positive_l3696_369610

theorem negation_of_all_squares_positive :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, ¬(x^2 > 0)) :=
by sorry

end negation_of_all_squares_positive_l3696_369610


namespace scientific_notation_6500_l3696_369664

theorem scientific_notation_6500 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 6500 = a * (10 : ℝ) ^ n ∧ a = 6.5 ∧ n = 3 := by
  sorry

end scientific_notation_6500_l3696_369664


namespace line_through_point_representation_l3696_369676

/-- A line in a 2D plane --/
structure Line where
  slope : Option ℝ
  yIntercept : ℝ

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a line passes through a point --/
def Line.passesThrough (l : Line) (p : Point) : Prop :=
  match l.slope with
  | some k => p.y = k * p.x + l.yIntercept
  | none => p.x = l.yIntercept

/-- The statement to be proven false --/
theorem line_through_point_representation (b : ℝ) :
  ∃ (k : ℝ), ∀ (l : Line), l.passesThrough ⟨0, b⟩ → 
  ∃ (k' : ℝ), l.slope = some k' ∧ l.yIntercept = b :=
sorry

end line_through_point_representation_l3696_369676


namespace functional_equation_solution_l3696_369684

/-- A complex-valued function satisfying the given functional equation is constant and equal to 1. -/
theorem functional_equation_solution (f : ℂ → ℂ) : 
  (∀ z : ℂ, f z + z * f (1 - z) = 1 + z) → 
  (∀ z : ℂ, f z = 1) := by
sorry

end functional_equation_solution_l3696_369684


namespace inequality_for_increasing_function_l3696_369671

/-- An increasing function on the real line. -/
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

/-- Theorem: Given an increasing function f on ℝ and real numbers a and b
    such that a + b ≤ 0, the inequality f(a) + f(b) ≤ f(-a) + f(-b) holds. -/
theorem inequality_for_increasing_function
  (f : ℝ → ℝ) (hf : IncreasingFunction f) (a b : ℝ) (hab : a + b ≤ 0) :
  f a + f b ≤ f (-a) + f (-b) := by
  sorry

end inequality_for_increasing_function_l3696_369671


namespace part1_part2_l3696_369648

-- Define the function f
def f (a x : ℝ) : ℝ := 2 * a * x^2 - (a^2 + 4) * x + 2 * a

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x, f a x > 0 ↔ -4 < x ∧ x < -1/4) → (a = -8 ∨ a = -1/2) :=
sorry

-- Part 2
theorem part2 (a : ℝ) (h : a > 0) :
  (∀ x, f a x ≤ 0 ↔ 
    ((0 < a ∧ a < 2 → a/2 ≤ x ∧ x ≤ 2/a) ∧
     (a > 2 → 2/a ≤ x ∧ x ≤ a/2) ∧
     (a = 2 → x = 1))) :=
sorry

end part1_part2_l3696_369648


namespace expression_simplification_l3696_369670

theorem expression_simplification (x y : ℚ) (hx : x = 3) (hy : y = -1/2) :
  x * (x - 4 * y) + (2 * x + y) * (2 * x - y) - (2 * x - y)^2 = 17/2 := by
  sorry

end expression_simplification_l3696_369670


namespace technician_permanent_percentage_l3696_369685

def factory_workforce (total_workers : ℝ) : Prop :=
  let technicians := 0.8 * total_workers
  let non_technicians := 0.2 * total_workers
  let permanent_non_technicians := 0.2 * non_technicians
  let temporary_workers := 0.68 * total_workers
  ∃ (permanent_technicians : ℝ),
    permanent_technicians + permanent_non_technicians = total_workers - temporary_workers ∧
    permanent_technicians / technicians = 0.35

theorem technician_permanent_percentage :
  ∀ (total_workers : ℝ), total_workers > 0 → factory_workforce total_workers :=
by sorry

end technician_permanent_percentage_l3696_369685


namespace derivative_of_x_exp_x_l3696_369694

theorem derivative_of_x_exp_x :
  let f : ℝ → ℝ := λ x ↦ x * Real.exp x
  deriv f = λ x ↦ (1 + x) * Real.exp x := by
  sorry

end derivative_of_x_exp_x_l3696_369694


namespace inequality_proof_l3696_369661

theorem inequality_proof (x y z : ℝ) :
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
by sorry

end inequality_proof_l3696_369661


namespace quarter_difference_zero_l3696_369625

/-- Represents a coin collection with nickels, dimes, and quarters. -/
structure CoinCollection where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- The total number of coins in the collection. -/
def CoinCollection.total (c : CoinCollection) : ℕ :=
  c.nickels + c.dimes + c.quarters

/-- The total value of the collection in cents. -/
def CoinCollection.value (c : CoinCollection) : ℕ :=
  5 * c.nickels + 10 * c.dimes + 25 * c.quarters

/-- Predicate for a valid coin collection according to the problem conditions. -/
def isValidCollection (c : CoinCollection) : Prop :=
  c.total = 150 ∧ c.value = 2000

/-- The theorem to be proved. -/
theorem quarter_difference_zero :
  ∀ c₁ c₂ : CoinCollection, isValidCollection c₁ → isValidCollection c₂ →
  c₁.quarters = c₂.quarters :=
sorry

end quarter_difference_zero_l3696_369625


namespace stratified_sample_sum_l3696_369612

def total_population : Nat := 100
def sample_size : Nat := 20
def stratum1_size : Nat := 10
def stratum2_size : Nat := 20

theorem stratified_sample_sum :
  let stratum1_sample := sample_size * stratum1_size / total_population
  let stratum2_sample := sample_size * stratum2_size / total_population
  stratum1_sample + stratum2_sample = 6 := by
  sorry

end stratified_sample_sum_l3696_369612


namespace inequality_proof_l3696_369626

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (hln : Real.log a * Real.log b > 0) :
  a^(b - 1) < b^(a - 1) := by
  sorry

end inequality_proof_l3696_369626


namespace factorization_and_simplification_l3696_369674

theorem factorization_and_simplification (x : ℝ) (h : x^2 ≠ 3 ∧ x^2 ≠ -1) :
  (12 * x^6 + 36 * x^4 - 9) / (3 * x^4 - 9 * x^2 - 9) =
  (4 * x^4 * (x^2 + 3) - 3) / ((x^2 - 3) * (x^2 + 1)) :=
by sorry

end factorization_and_simplification_l3696_369674


namespace largest_time_for_85_degrees_l3696_369621

/-- The temperature function in Denver, CO on a specific day -/
def temperature (t : ℝ) : ℝ := -t^2 + 10*t + 60

/-- The largest non-negative real solution to the equation temperature(t) = 85 is 15 -/
theorem largest_time_for_85_degrees :
  (∃ (t : ℝ), t ≥ 0 ∧ temperature t = 85) →
  (∀ (t : ℝ), t ≥ 0 ∧ temperature t = 85 → t ≤ 15) ∧
  (temperature 15 = 85) := by
sorry

end largest_time_for_85_degrees_l3696_369621


namespace smallest_surface_area_is_cube_l3696_369693

-- Define a rectangular parallelepiped
structure Parallelepiped where
  x : ℝ
  y : ℝ
  z : ℝ
  x_pos : 0 < x
  y_pos : 0 < y
  z_pos : 0 < z

-- Define the volume of a parallelepiped
def volume (p : Parallelepiped) : ℝ := p.x * p.y * p.z

-- Define the surface area of a parallelepiped
def surfaceArea (p : Parallelepiped) : ℝ := 2 * (p.x * p.y + p.x * p.z + p.y * p.z)

-- State the theorem
theorem smallest_surface_area_is_cube (V : ℝ) (hV : 0 < V) :
  ∃ (p : Parallelepiped), volume p = V ∧
    ∀ (q : Parallelepiped), volume q = V → surfaceArea p ≤ surfaceArea q ∧
      (surfaceArea p = surfaceArea q → p.x = p.y ∧ p.y = p.z) :=
by sorry

end smallest_surface_area_is_cube_l3696_369693


namespace third_set_candy_count_l3696_369658

/-- Represents the number of candies of a specific type in a set -/
structure CandyCount where
  hard : ℕ
  chocolate : ℕ
  gummy : ℕ

/-- Represents the total candy distribution across three sets -/
structure CandyDistribution where
  set1 : CandyCount
  set2 : CandyCount
  set3 : CandyCount

/-- The conditions of the candy distribution problem -/
def validDistribution (d : CandyDistribution) : Prop :=
  -- Total number of each type is equal across all sets
  d.set1.hard + d.set2.hard + d.set3.hard = 
  d.set1.chocolate + d.set2.chocolate + d.set3.chocolate ∧
  d.set1.hard + d.set2.hard + d.set3.hard = 
  d.set1.gummy + d.set2.gummy + d.set3.gummy ∧
  -- First set conditions
  d.set1.chocolate = d.set1.gummy ∧
  d.set1.hard = d.set1.chocolate + 7 ∧
  -- Second set conditions
  d.set2.hard = d.set2.chocolate ∧
  d.set2.gummy = d.set2.hard - 15 ∧
  -- Third set condition
  d.set3.hard = 0

/-- The main theorem stating that any valid distribution has 29 candies in the third set -/
theorem third_set_candy_count (d : CandyDistribution) : 
  validDistribution d → d.set3.chocolate + d.set3.gummy = 29 := by
  sorry


end third_set_candy_count_l3696_369658


namespace sum_greater_than_three_l3696_369644

theorem sum_greater_than_three (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_ineq : a * b + b * c + c * a > a + b + c) : 
  a + b + c > 3 := by
  sorry

end sum_greater_than_three_l3696_369644


namespace unique_two_digit_reverse_ratio_l3696_369652

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem unique_two_digit_reverse_ratio :
  ∃! n : ℕ, is_two_digit n ∧ (n : ℚ) / (reverse_digits n : ℚ) = 7 / 4 :=
by
  -- The proof would go here
  sorry

end unique_two_digit_reverse_ratio_l3696_369652


namespace M_mod_1000_l3696_369620

/-- The number of characters in the string -/
def n : ℕ := 15

/-- The number of A's in the string -/
def a : ℕ := 3

/-- The number of B's in the string -/
def b : ℕ := 5

/-- The number of C's in the string -/
def c : ℕ := 4

/-- The number of D's in the string -/
def d : ℕ := 3

/-- The length of the first section where A's are not allowed -/
def first_section : ℕ := 3

/-- The length of the middle section where B's are not allowed -/
def middle_section : ℕ := 5

/-- The length of the last section where C's are not allowed -/
def last_section : ℕ := 7

/-- The function that calculates the number of permutations -/
def M : ℕ := sorry

theorem M_mod_1000 : M % 1000 = 60 := by sorry

end M_mod_1000_l3696_369620


namespace original_number_problem_l3696_369662

theorem original_number_problem (x : ℝ) : 3 * (2 * x + 5) = 111 → x = 16 := by
  sorry

end original_number_problem_l3696_369662


namespace min_a_squared_plus_b_squared_l3696_369609

theorem min_a_squared_plus_b_squared : ∀ a b : ℝ,
  (∀ x : ℝ, x^2 + a*x + b - 3 = 0 → x = 2) →
  a^2 + b^2 ≥ 4 :=
by sorry

end min_a_squared_plus_b_squared_l3696_369609


namespace max_cars_theorem_l3696_369695

/-- Represents the maximum number of cars that can pass a point on a highway in 30 minutes -/
def M : ℕ := 6000

/-- Theorem stating the maximum number of cars and its relation to M/10 -/
theorem max_cars_theorem :
  (∀ (car_length : ℝ) (time : ℝ),
    car_length = 5 ∧ 
    time = 30 ∧ 
    (∀ (speed : ℝ) (distance : ℝ),
      distance = car_length * (speed / 10))) →
  M = 6000 ∧ M / 10 = 600 := by
  sorry

#check max_cars_theorem

end max_cars_theorem_l3696_369695


namespace find_A_l3696_369619

theorem find_A (A B C D : ℝ) 
  (diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (eq1 : 2 * B + B = 12)
  (eq2 : C - B = 5)
  (eq3 : D + C = 12)
  (eq4 : A - D = 5) :
  A = 8 := by
sorry

end find_A_l3696_369619


namespace unique_solution_set_l3696_369611

def A (m : ℝ) : Set ℝ := {x : ℝ | m * x^2 + 2 * x + 3 = 0}

def M : Set ℝ := {m : ℝ | ∃! x : ℝ, m * x^2 + 2 * x + 3 = 0}

theorem unique_solution_set : M = {0, 1/3} := by sorry

end unique_solution_set_l3696_369611


namespace max_a_for_three_integer_solutions_l3696_369678

theorem max_a_for_three_integer_solutions : 
  ∃ (a : ℝ), 
    (∀ x : ℤ, (-1/3 : ℝ) * (x : ℝ) > 2/3 - (x : ℝ) ∧ 
               (1/2 : ℝ) * (x : ℝ) - 1 < (1/2 : ℝ) * (a - 2)) →
    (∃! (x₁ x₂ x₃ : ℤ), 
      ((-1/3 : ℝ) * (x₁ : ℝ) > 2/3 - (x₁ : ℝ) ∧ 
       (1/2 : ℝ) * (x₁ : ℝ) - 1 < (1/2 : ℝ) * (a - 2)) ∧
      ((-1/3 : ℝ) * (x₂ : ℝ) > 2/3 - (x₂ : ℝ) ∧ 
       (1/2 : ℝ) * (x₂ : ℝ) - 1 < (1/2 : ℝ) * (a - 2)) ∧
      ((-1/3 : ℝ) * (x₃ : ℝ) > 2/3 - (x₃ : ℝ) ∧ 
       (1/2 : ℝ) * (x₃ : ℝ) - 1 < (1/2 : ℝ) * (a - 2)) ∧
      x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) →
    a = 5 ∧ ∀ b > a, 
      ¬(∃! (x₁ x₂ x₃ : ℤ), 
        ((-1/3 : ℝ) * (x₁ : ℝ) > 2/3 - (x₁ : ℝ) ∧ 
         (1/2 : ℝ) * (x₁ : ℝ) - 1 < (1/2 : ℝ) * (b - 2)) ∧
        ((-1/3 : ℝ) * (x₂ : ℝ) > 2/3 - (x₂ : ℝ) ∧ 
         (1/2 : ℝ) * (x₂ : ℝ) - 1 < (1/2 : ℝ) * (b - 2)) ∧
        ((-1/3 : ℝ) * (x₃ : ℝ) > 2/3 - (x₃ : ℝ) ∧ 
         (1/2 : ℝ) * (x₃ : ℝ) - 1 < (1/2 : ℝ) * (b - 2)) ∧
        x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) :=
by sorry

end max_a_for_three_integer_solutions_l3696_369678


namespace chef_potato_usage_l3696_369630

/-- The number of potatoes used for lunch -/
def lunch_potatoes : ℕ := 5

/-- The number of potatoes used for dinner -/
def dinner_potatoes : ℕ := 2

/-- The total number of potatoes used -/
def total_potatoes : ℕ := lunch_potatoes + dinner_potatoes

theorem chef_potato_usage : total_potatoes = 7 := by
  sorry

end chef_potato_usage_l3696_369630


namespace sum_of_constants_l3696_369659

theorem sum_of_constants (a b : ℝ) : 
  (∀ x y : ℝ, y = a + b / x) →
  (2 = a + b / (-2)) →
  (7 = a + b / (-4)) →
  a + b = 32 := by
sorry

end sum_of_constants_l3696_369659


namespace inconsistent_equations_l3696_369604

theorem inconsistent_equations : ¬∃ (x y S : ℝ), (x + y = S) ∧ (x + 3*y = 1) ∧ (x + 2*y = 10) := by
  sorry

end inconsistent_equations_l3696_369604


namespace odd_coefficient_probability_l3696_369697

/-- The number of terms in the expansion of (1+x)^11 -/
def n : ℕ := 12

/-- The number of terms with odd coefficients in the expansion of (1+x)^11 -/
def k : ℕ := 8

/-- The probability of selecting a term with an odd coefficient from the expansion of (1+x)^11 -/
def p : ℚ := k / n

theorem odd_coefficient_probability : p = 2/3 := by
  sorry

end odd_coefficient_probability_l3696_369697


namespace stamp_cost_theorem_l3696_369649

/-- The total cost of stamps in cents -/
def total_cost (type_a_cost type_b_cost type_c_cost : ℕ) 
               (type_a_quantity type_b_quantity type_c_quantity : ℕ) : ℕ :=
  type_a_cost * type_a_quantity + 
  type_b_cost * type_b_quantity + 
  type_c_cost * type_c_quantity

/-- Theorem: The total cost of stamps is 594 cents -/
theorem stamp_cost_theorem : 
  total_cost 34 52 73 4 6 2 = 594 := by
  sorry

end stamp_cost_theorem_l3696_369649


namespace system_solution_l3696_369636

theorem system_solution : ∃ (x y : ℝ), 
  x^2 + y * Real.sqrt (x * y) = 105 ∧
  y^2 + x * Real.sqrt (x * y) = 70 ∧
  x = 9 ∧ y = 4 := by
  sorry

end system_solution_l3696_369636


namespace quadrilateral_diagonal_theorem_l3696_369680

/-- Represents a quadrilateral with side lengths and diagonal lengths. -/
structure Quadrilateral where
  a : ℝ  -- Length of side AB
  b : ℝ  -- Length of side BC
  c : ℝ  -- Length of side CD
  d : ℝ  -- Length of side DA
  m : ℝ  -- Length of diagonal AC
  n : ℝ  -- Length of diagonal BD
  A : ℝ  -- Angle at vertex A
  C : ℝ  -- Angle at vertex C

/-- Theorem stating the relationship between side lengths, diagonal lengths, and angles in a quadrilateral. -/
theorem quadrilateral_diagonal_theorem (q : Quadrilateral) :
  q.m^2 * q.n^2 = q.a^2 * q.c^2 + q.b^2 * q.d^2 - 2 * q.a * q.b * q.c * q.d * Real.cos (q.A + q.C) := by
  sorry

end quadrilateral_diagonal_theorem_l3696_369680


namespace total_messages_equals_680_l3696_369667

/-- The total number of messages sent by Alina and Lucia over three days -/
def total_messages (lucia_day1 : ℕ) (alina_difference : ℕ) : ℕ :=
  let alina_day1 := lucia_day1 - alina_difference
  let lucia_day2 := lucia_day1 / 3
  let alina_day2 := 2 * alina_day1
  let lucia_day3 := lucia_day1
  let alina_day3 := alina_day1
  lucia_day1 + lucia_day2 + lucia_day3 + alina_day1 + alina_day2 + alina_day3

/-- Theorem stating that the total number of messages sent over three days is 680 -/
theorem total_messages_equals_680 :
  total_messages 120 20 = 680 := by
  sorry

end total_messages_equals_680_l3696_369667


namespace complex_equation_sum_l3696_369666

theorem complex_equation_sum (a b : ℝ) :
  (a - 2 * Complex.I) * Complex.I = b - Complex.I →
  a + b = 1 := by
  sorry

end complex_equation_sum_l3696_369666


namespace fruit_basket_count_l3696_369677

/-- Represents the number of apples available -/
def num_apples : ℕ := 6

/-- Represents the number of oranges available -/
def num_oranges : ℕ := 8

/-- Represents the minimum number of apples required in each basket -/
def min_apples : ℕ := 2

/-- Calculates the number of possible fruit baskets -/
def num_fruit_baskets : ℕ :=
  (num_apples - min_apples + 1) * (num_oranges + 1)

/-- Theorem stating the number of possible fruit baskets -/
theorem fruit_basket_count :
  num_fruit_baskets = 45 := by sorry

end fruit_basket_count_l3696_369677


namespace complex_sum_product_theorem_l3696_369607

theorem complex_sum_product_theorem (x y z : ℂ) 
  (hx : x = Complex.mk x.re x.im)
  (hy : y = Complex.mk y.re y.im)
  (hz : z = Complex.mk z.re z.im)
  (h_magnitude : Complex.abs x = Complex.abs y ∧ Complex.abs y = Complex.abs z)
  (h_sum : x + y + z = Complex.mk (-Real.sqrt 3 / 2) (-Real.sqrt 5))
  (h_product : x * y * z = Complex.mk (Real.sqrt 3) (Real.sqrt 5)) :
  (x.re * x.im + y.re * y.im + z.re * z.im)^2 = 15/1 := by
  sorry

end complex_sum_product_theorem_l3696_369607


namespace and_false_necessary_not_sufficient_for_or_false_l3696_369618

theorem and_false_necessary_not_sufficient_for_or_false (p q : Prop) :
  (¬(p ∧ q) → ¬(p ∨ q)) ∧ ¬(¬(p ∧ q) ↔ ¬(p ∨ q)) := by
  sorry

end and_false_necessary_not_sufficient_for_or_false_l3696_369618


namespace brown_gumdrops_after_replacement_l3696_369637

/-- Theorem about the number of brown gumdrops after replacement in a jar --/
theorem brown_gumdrops_after_replacement (total : ℕ) (green blue brown red yellow : ℕ) :
  total = 200 →
  green = 40 →
  blue = 50 →
  brown = 60 →
  red = 20 →
  yellow = 30 →
  (brown + (red / 3 : ℕ)) = 67 := by
  sorry

#check brown_gumdrops_after_replacement

end brown_gumdrops_after_replacement_l3696_369637


namespace largest_class_size_l3696_369640

theorem largest_class_size (total_students : ℕ) (num_classes : ℕ) (diff : ℕ) : 
  total_students = 140 → num_classes = 5 → diff = 2 →
  ∃ x : ℕ, x = 32 ∧ 
    (x + (x - diff) + (x - 2*diff) + (x - 3*diff) + (x - 4*diff) = total_students) :=
by sorry

end largest_class_size_l3696_369640


namespace twentieth_digit_sum_one_thirteenth_one_eleventh_l3696_369606

/-- The decimal representation of a rational number -/
def decimalRepresentation (q : ℚ) : ℕ → ℕ := sorry

/-- The sum of decimal representations of two rational numbers -/
def sumDecimalRepresentations (q₁ q₂ : ℚ) : ℕ → ℕ := sorry

/-- The nth digit after the decimal point in a decimal representation -/
def nthDigitAfterDecimal (f : ℕ → ℕ) (n : ℕ) : ℕ := sorry

theorem twentieth_digit_sum_one_thirteenth_one_eleventh :
  nthDigitAfterDecimal (sumDecimalRepresentations (1/13) (1/11)) 20 = 6 := by sorry

end twentieth_digit_sum_one_thirteenth_one_eleventh_l3696_369606


namespace negation_of_proposition_l3696_369669

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 + 3 ≥ 2*x) ↔ (∃ x : ℝ, x^2 + 3 < 2*x) := by sorry

end negation_of_proposition_l3696_369669


namespace minimum_seats_for_adjacent_seating_l3696_369633

/-- Represents a seating arrangement in a row of seats. -/
structure SeatingArrangement where
  total_seats : ℕ
  occupied_seats : ℕ
  max_gap : ℕ

/-- Checks if a seating arrangement is valid. -/
def is_valid_arrangement (s : SeatingArrangement) : Prop :=
  s.occupied_seats ≤ s.total_seats ∧ 
  s.max_gap ≤ 2

/-- Checks if adding one more person would force them to sit next to someone. -/
def forces_adjacent_seating (s : SeatingArrangement) : Prop :=
  s.max_gap ≤ 1

/-- The main theorem to prove. -/
theorem minimum_seats_for_adjacent_seating :
  ∃ (s : SeatingArrangement),
    s.total_seats = 150 ∧
    s.occupied_seats = 30 ∧
    is_valid_arrangement s ∧
    forces_adjacent_seating s ∧
    (∀ (s' : SeatingArrangement),
      s'.total_seats = 150 →
      s'.occupied_seats < 30 →
      is_valid_arrangement s' →
      ¬forces_adjacent_seating s') :=
sorry

end minimum_seats_for_adjacent_seating_l3696_369633


namespace polynomial_coefficient_B_l3696_369634

theorem polynomial_coefficient_B (A C D : ℤ) : 
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (r₁ : ℤ) + r₂ + r₃ + r₄ + r₅ + r₆ = 10 →
    ∀ (z : ℂ), z^6 - 10*z^5 + A*z^4 + (-108)*z^3 + C*z^2 + D*z + 16 = 
      (z - r₁) * (z - r₂) * (z - r₃) * (z - r₄) * (z - r₅) * (z - r₆) :=
by sorry

end polynomial_coefficient_B_l3696_369634


namespace min_values_theorem_l3696_369631

theorem min_values_theorem :
  (∀ x > 1, x + 4 / (x - 1) ≥ 5) ∧
  (∀ a b, a > 0 → b > 0 → a + b = a * b → 9 * a + b ≥ 16) := by
sorry

end min_values_theorem_l3696_369631


namespace a_values_l3696_369657

def P : Set ℝ := {x | x^2 = 1}
def Q (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem a_values (a : ℝ) : Q a ⊆ P → a = 0 ∨ a = 1 ∨ a = -1 := by
  sorry

end a_values_l3696_369657


namespace horner_method_v2_l3696_369655

def horner_polynomial (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

def horner_v0 : ℝ := 1

def horner_v1 (x : ℝ) : ℝ := horner_v0 * x + 5

def horner_v2 (x : ℝ) : ℝ := horner_v1 x * x + 10

theorem horner_method_v2 :
  horner_v2 2 = 24 :=
by sorry

end horner_method_v2_l3696_369655


namespace largest_coeff_x5_implies_n10_l3696_369653

theorem largest_coeff_x5_implies_n10 (n : ℕ+) :
  (∀ k : ℕ, k ≠ 5 → Nat.choose n 5 ≥ Nat.choose n k) →
  n = 10 := by
sorry

end largest_coeff_x5_implies_n10_l3696_369653


namespace count_monomials_l3696_369605

/-- A function that determines if an algebraic expression is a monomial -/
def isMonomial (expr : String) : Bool :=
  match expr with
  | "(m+n)/2" => false
  | "2x^2y" => true
  | "1/x" => false
  | "-5" => true
  | "a" => true
  | _ => false

/-- The set of given algebraic expressions -/
def expressions : List String := ["(m+n)/2", "2x^2y", "1/x", "-5", "a"]

/-- Theorem stating that the number of monomials in the given set of expressions is 3 -/
theorem count_monomials :
  (expressions.filter isMonomial).length = 3 := by sorry

end count_monomials_l3696_369605


namespace negation_of_all_linear_functions_are_monotonic_l3696_369656

-- Define the type of functions from real numbers to real numbers
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be linear
def IsLinear (f : RealFunction) : Prop := ∀ x y : ℝ, ∀ c : ℝ, f (c * x + y) = c * f x + f y

-- Define what it means for a function to be monotonic
def IsMonotonic (f : RealFunction) : Prop := ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- State the theorem
theorem negation_of_all_linear_functions_are_monotonic :
  (¬ ∀ f : RealFunction, IsLinear f → IsMonotonic f) ↔
  (∃ f : RealFunction, IsLinear f ∧ ¬IsMonotonic f) :=
sorry

end negation_of_all_linear_functions_are_monotonic_l3696_369656


namespace andrew_winning_strategy_l3696_369641

/-- Represents the state of the game with two heaps of pebbles -/
structure GameState where
  a : ℕ
  b : ℕ

/-- Predicate to check if a number is of the form 2^x + 1 -/
def isPowerOfTwoPlusOne (n : ℕ) : Prop :=
  ∃ x : ℕ, n = 2^x + 1

/-- Predicate to check if Andrew has a winning strategy -/
def andrewWins (state : GameState) : Prop :=
  state.a = 1 ∨ state.b = 1 ∨
  isPowerOfTwoPlusOne (state.a + state.b) ∨
  (isPowerOfTwoPlusOne state.a ∧ state.b < state.a) ∨
  (isPowerOfTwoPlusOne state.b ∧ state.a < state.b)

/-- The main theorem stating the winning condition for Andrew -/
theorem andrew_winning_strategy (state : GameState) :
  andrewWins state ↔ ∃ (strategy : GameState → ℕ → GameState),
    ∀ (move : ℕ), andrewWins (strategy state move) :=
  sorry

end andrew_winning_strategy_l3696_369641


namespace mia_wall_paint_area_l3696_369638

/-- The area to be painted on Mia's wall --/
def areaToBePainted (wallHeight wallLength unPaintedWidth unPaintedHeight : ℝ) : ℝ :=
  wallHeight * wallLength - unPaintedWidth * unPaintedHeight

/-- Theorem stating the area Mia needs to paint --/
theorem mia_wall_paint_area :
  areaToBePainted 10 15 3 5 = 135 := by
  sorry

end mia_wall_paint_area_l3696_369638


namespace dream_sequence_sum_l3696_369635

/-- A sequence is a "dream sequence" if it satisfies the given equation -/
def isDreamSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, 1 / a (n + 1) - 2 / a n = 0

theorem dream_sequence_sum (b : ℕ → ℝ) :
  (∀ n, b n > 0) →  -- b is a positive sequence
  isDreamSequence (λ n => 1 / b n) →  -- 1/b_n is a dream sequence
  b 1 + b 2 + b 3 = 2 →  -- sum of first three terms is 2
  b 6 + b 7 + b 8 = 64 :=  -- sum of 6th, 7th, and 8th terms is 64
by
  sorry

end dream_sequence_sum_l3696_369635


namespace sum_abcd_equals_negative_28_over_3_l3696_369627

theorem sum_abcd_equals_negative_28_over_3 
  (a b c d : ℚ) 
  (h : a + 3 = b + 7 ∧ a + 3 = c + 5 ∧ a + 3 = d + 9 ∧ a + 3 = a + b + c + d + 13) : 
  a + b + c + d = -28/3 := by
sorry

end sum_abcd_equals_negative_28_over_3_l3696_369627


namespace grid_value_theorem_l3696_369617

/-- Represents a 7x2 grid of rational numbers -/
def Grid := Fin 7 → Fin 2 → ℚ

/-- The main column forms an arithmetic sequence -/
def is_main_column_arithmetic (g : Grid) : Prop :=
  ∃ d : ℚ, ∀ i : Fin 6, g (i + 1) 0 - g i 0 = d

/-- The first two rows form arithmetic sequences -/
def are_first_two_rows_arithmetic (g : Grid) : Prop :=
  ∃ d₁ d₂ : ℚ, (g 0 1 - g 0 0 = d₁) ∧ (g 1 1 - g 1 0 = d₂)

/-- The grid satisfies the given conditions -/
def satisfies_conditions (g : Grid) : Prop :=
  (g 0 0 = -9) ∧ (g 3 0 = 56) ∧ (g 6 1 = 16) ∧
  is_main_column_arithmetic g ∧
  are_first_two_rows_arithmetic g

theorem grid_value_theorem (g : Grid) (h : satisfies_conditions g) : g 4 1 = -851/3 := by
  sorry

end grid_value_theorem_l3696_369617


namespace z_in_terms_of_x_l3696_369668

theorem z_in_terms_of_x (p : ℝ) (x z : ℝ) 
  (hx : x = 2 + 3^p) 
  (hz : z = 2 + 3^(-p)) : 
  z = (2*x - 3) / (x - 2) := by
  sorry

end z_in_terms_of_x_l3696_369668


namespace first_day_over_200_l3696_369681

def paperclips (n : ℕ) : ℕ := 5 * 3^(n - 1)

theorem first_day_over_200 :
  ∀ k : ℕ, k < 5 → paperclips k ≤ 200 ∧ paperclips 5 > 200 :=
by sorry

end first_day_over_200_l3696_369681


namespace inequality_solution_range_l3696_369665

theorem inequality_solution_range (k : ℝ) : 
  (k ≠ 0 ∧ k^2 * 1^2 - 6*k*1 + 8 ≥ 0) →
  k ∈ (Set.Ioi 4 : Set ℝ) ∪ (Set.Icc 0 2 : Set ℝ) ∪ (Set.Iio 0 : Set ℝ) :=
sorry

end inequality_solution_range_l3696_369665


namespace total_graduation_messages_l3696_369629

def number_of_students : ℕ := 40

theorem total_graduation_messages :
  (number_of_students * (number_of_students - 1)) / 2 = 1560 :=
by sorry

end total_graduation_messages_l3696_369629


namespace special_numbers_characterization_l3696_369615

/-- A function that returns true if a natural number has all distinct digits -/
def has_distinct_digits (n : ℕ) : Bool :=
  sorry

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  sorry

/-- A function that returns the product of digits of a natural number -/
def product_of_digits (n : ℕ) : ℕ :=
  sorry

/-- The set of numbers that satisfy the conditions -/
def special_numbers : Finset ℕ :=
  {123, 132, 213, 231, 312, 321}

theorem special_numbers_characterization :
  ∀ n : ℕ, n ∈ special_numbers ↔
    n > 9 ∧
    has_distinct_digits n ∧
    sum_of_digits n = product_of_digits n :=
by sorry

end special_numbers_characterization_l3696_369615


namespace fractional_equation_solution_l3696_369600

theorem fractional_equation_solution :
  ∃! x : ℝ, x ≠ 1 ∧ x ≠ -1 ∧ (1 / (x - 1) + 1 = 2 / (x^2 - 1)) ∧ x = -2 := by
  sorry

end fractional_equation_solution_l3696_369600


namespace function_minimum_l3696_369696

def f (x : ℝ) : ℝ := x^2 - 8*x + 5

theorem function_minimum :
  ∃ (x_min : ℝ), 
    (∀ x, f x ≥ f x_min) ∧ 
    x_min = 4 ∧ 
    f x_min = -11 := by
  sorry

end function_minimum_l3696_369696


namespace negative_sqrt_geq_a_plus_sqrt_neg_two_l3696_369642

theorem negative_sqrt_geq_a_plus_sqrt_neg_two (a : ℝ) (h : a > 0) :
  -Real.sqrt a ≥ a + Real.sqrt (-2) :=
by sorry

end negative_sqrt_geq_a_plus_sqrt_neg_two_l3696_369642
