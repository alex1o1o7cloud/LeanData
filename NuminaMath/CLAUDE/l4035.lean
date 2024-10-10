import Mathlib

namespace mode_is_highest_rectangle_middle_l4035_403533

/-- Represents a frequency distribution histogram --/
structure FrequencyHistogram where
  -- Add necessary fields here

/-- The mode of a frequency distribution --/
def mode (h : FrequencyHistogram) : ℝ :=
  sorry

/-- The middle position of the highest rectangle in a frequency histogram --/
def highestRectangleMiddle (h : FrequencyHistogram) : ℝ :=
  sorry

/-- Theorem stating that the mode corresponds to the middle of the highest rectangle --/
theorem mode_is_highest_rectangle_middle (h : FrequencyHistogram) :
  mode h = highestRectangleMiddle h :=
sorry

end mode_is_highest_rectangle_middle_l4035_403533


namespace parabola_vertex_l4035_403505

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop := y^2 - 8*x + 6*y + 17 = 0

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, -3)

/-- Theorem: The vertex of the parabola y^2 - 8x + 6y + 17 = 0 is at the point (1, -3) -/
theorem parabola_vertex : 
  ∀ x y : ℝ, parabola_eq x y → (x, y) = vertex ∨ ∃ t : ℝ, parabola_eq (x + t) y :=
sorry

end parabola_vertex_l4035_403505


namespace sqrt_11_diamond_sqrt_11_l4035_403523

-- Define the ¤ operation
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem sqrt_11_diamond_sqrt_11 : diamond (Real.sqrt 11) (Real.sqrt 11) = 44 := by sorry

end sqrt_11_diamond_sqrt_11_l4035_403523


namespace tangent_line_property_l4035_403582

/-- Given a differentiable function f : ℝ → ℝ with a tangent line y = (1/2)x + 2
    at the point (1, f(1)), prove that f(1) + f'(1) = 3 -/
theorem tangent_line_property (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_tangent : (fun x ↦ (1/2 : ℝ) * x + 2) = fun x ↦ f 1 + deriv f 1 * (x - 1)) :
  f 1 + deriv f 1 = 3 := by
  sorry

end tangent_line_property_l4035_403582


namespace modes_of_test_scores_l4035_403552

/-- Represents a frequency distribution of test scores -/
def FrequencyDistribution := List (Nat × Nat)

/-- Finds the modes (most frequent scores) in a frequency distribution -/
def findModes (scores : FrequencyDistribution) : List Nat :=
  sorry

/-- The actual frequency distribution of the test scores -/
def testScores : FrequencyDistribution := [
  (62, 1), (65, 2), (70, 1), (74, 2), (78, 1),
  (81, 2), (86, 1), (89, 1), (92, 1), (97, 3),
  (101, 4), (104, 4), (110, 3)
]

theorem modes_of_test_scores :
  findModes testScores = [101, 104] :=
sorry

end modes_of_test_scores_l4035_403552


namespace cameron_work_time_l4035_403556

theorem cameron_work_time (cameron_alone : ℝ) 
  (h1 : cameron_alone > 0)
  (h2 : 9 / cameron_alone + 1 / 2 = 1)
  (h3 : (1 / cameron_alone + 1 / 7) * 7 = 1) : 
  cameron_alone = 18 := by
sorry

end cameron_work_time_l4035_403556


namespace cycle_selling_price_l4035_403576

/-- Calculates the selling price of an item given its cost price and gain percent. -/
def sellingPrice (costPrice : ℕ) (gainPercent : ℕ) : ℕ :=
  costPrice + (costPrice * gainPercent) / 100

/-- Theorem stating that the selling price of a cycle with cost price 900 and gain percent 30 is 1170. -/
theorem cycle_selling_price :
  sellingPrice 900 30 = 1170 := by
  sorry

end cycle_selling_price_l4035_403576


namespace davis_class_groups_l4035_403501

/-- The number of groups in Miss Davis's class -/
def number_of_groups (sticks_per_group : ℕ) (initial_sticks : ℕ) (remaining_sticks : ℕ) : ℕ :=
  (initial_sticks - remaining_sticks) / sticks_per_group

/-- Theorem stating the number of groups in Miss Davis's class -/
theorem davis_class_groups :
  number_of_groups 15 170 20 = 10 := by
  sorry

end davis_class_groups_l4035_403501


namespace salvadore_earnings_l4035_403520

/-- Given that Salvadore earned S dollars, Santo earned half of that, and their combined earnings were $2934, prove that Salvadore earned $1956. -/
theorem salvadore_earnings (S : ℝ) 
  (h1 : S + S / 2 = 2934) : S = 1956 := by
  sorry

end salvadore_earnings_l4035_403520


namespace four_inch_cube_painted_faces_l4035_403565

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Counts the number of 1-inch cubes with at least two painted faces in a painted n×n×n cube -/
def count_painted_cubes (c : Cube n) : ℕ :=
  sorry

/-- Theorem: In a 4x4x4 painted cube, there are 56 1-inch cubes with at least two painted faces -/
theorem four_inch_cube_painted_faces :
  ∃ (c : Cube 4), count_painted_cubes c = 56 := by
  sorry

end four_inch_cube_painted_faces_l4035_403565


namespace block_distance_is_200_l4035_403564

/-- The distance of one time around the block -/
def block_distance : ℝ := sorry

/-- The number of times Johnny runs around the block -/
def johnny_laps : ℕ := 4

/-- The number of times Mickey runs around the block -/
def mickey_laps : ℕ := johnny_laps / 2

/-- The average distance run by Johnny and Mickey -/
def average_distance : ℝ := 600

theorem block_distance_is_200 :
  block_distance = 200 :=
by
  sorry

end block_distance_is_200_l4035_403564


namespace cube_equation_solution_l4035_403569

theorem cube_equation_solution (a x : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * x * 45 * 35) : x = 35 := by
  sorry

end cube_equation_solution_l4035_403569


namespace head_circumference_ratio_l4035_403528

theorem head_circumference_ratio :
  let jack_circumference : ℝ := 12
  let charlie_circumference : ℝ := 9 + (jack_circumference / 2)
  let bill_circumference : ℝ := 10
  bill_circumference / charlie_circumference = 2 / 3 := by
sorry

end head_circumference_ratio_l4035_403528


namespace recipe_fraction_is_two_thirds_l4035_403550

/-- Represents the amount of an ingredient required for a recipe --/
structure RecipeIngredient where
  amount : ℚ
  deriving Repr

/-- Represents the amount of an ingredient available --/
structure AvailableIngredient where
  amount : ℚ
  deriving Repr

/-- Calculates the fraction of the recipe that can be made for a single ingredient --/
def ingredientFraction (required : RecipeIngredient) (available : AvailableIngredient) : ℚ :=
  available.amount / required.amount

/-- Finds the maximum fraction of the recipe that can be made given all ingredients --/
def maxRecipeFraction (sugar : RecipeIngredient × AvailableIngredient) 
                      (milk : RecipeIngredient × AvailableIngredient)
                      (flour : RecipeIngredient × AvailableIngredient) : ℚ :=
  min (ingredientFraction sugar.1 sugar.2)
      (min (ingredientFraction milk.1 milk.2)
           (ingredientFraction flour.1 flour.2))

theorem recipe_fraction_is_two_thirds :
  let sugar_required := RecipeIngredient.mk (3/4)
  let sugar_available := AvailableIngredient.mk (2/4)
  let milk_required := RecipeIngredient.mk (2/3)
  let milk_available := AvailableIngredient.mk (1/2)
  let flour_required := RecipeIngredient.mk (3/8)
  let flour_available := AvailableIngredient.mk (1/4)
  maxRecipeFraction (sugar_required, sugar_available)
                    (milk_required, milk_available)
                    (flour_required, flour_available) = 2/3 := by
  sorry

#eval maxRecipeFraction (RecipeIngredient.mk (3/4), AvailableIngredient.mk (2/4))
                        (RecipeIngredient.mk (2/3), AvailableIngredient.mk (1/2))
                        (RecipeIngredient.mk (3/8), AvailableIngredient.mk (1/4))

end recipe_fraction_is_two_thirds_l4035_403550


namespace p_sufficient_not_necessary_for_q_l4035_403560

-- Define the propositions
def p (m : ℝ) : Prop := ∀ x : ℝ, |x| + |x - 1| > m
def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (-(5 - 2*m)^x) > (-(5 - 2*m)^y)

-- State the theorem
theorem p_sufficient_not_necessary_for_q :
  (∃ m : ℝ, p m ∧ q m) ∧ (∃ m : ℝ, q m ∧ ¬(p m)) :=
sorry

end p_sufficient_not_necessary_for_q_l4035_403560


namespace sticker_distribution_l4035_403522

/-- The number of ways to partition n identical objects into at most k parts -/
def partition_count (n k : ℕ) : ℕ := sorry

/-- The theorem stating that there are 30 ways to partition 10 identical objects into at most 5 parts -/
theorem sticker_distribution : partition_count 10 5 = 30 := by sorry

end sticker_distribution_l4035_403522


namespace inequality_solution_set_l4035_403517

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x + x^2

theorem inequality_solution_set (x : ℝ) :
  (x ∈ Set.Ioo (Real.exp (-1)) (Real.exp 1)) ↔
  (f (Real.log x) + f (Real.log (1/x)) < 2 * f 1) :=
by sorry

end inequality_solution_set_l4035_403517


namespace abs_cube_complex_l4035_403566

/-- The absolute value of (3 + √7i)^3 is equal to 64, where i is the imaginary unit. -/
theorem abs_cube_complex : Complex.abs ((3 + Complex.I * Real.sqrt 7) ^ 3) = 64 := by
  sorry

end abs_cube_complex_l4035_403566


namespace inscribed_squares_ratio_l4035_403515

/-- A right triangle with sides 5, 12, and 13 (hypotenuse) -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  hypotenuse : c = 13
  right_angle : a^2 + b^2 = c^2
  side_a : a = 5
  side_b : b = 12

/-- First inscribed square with side length x -/
def first_square (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x < t.a ∧ x < t.b ∧ x / t.a = x / t.b

/-- Second inscribed square with side length y -/
def second_square (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y < t.c ∧ (t.a - y) / y = (t.b - y) / y

/-- The main theorem -/
theorem inscribed_squares_ratio (t : RightTriangle) 
  (x y : ℝ) (h1 : first_square t x) (h2 : second_square t y) : 
  x / y = 78 / 102 := by
  sorry

end inscribed_squares_ratio_l4035_403515


namespace equation_solution_l4035_403595

theorem equation_solution : ∃! y : ℚ, (1 / 6 : ℚ) + 6 / y = 14 / y + (1 / 14 : ℚ) ∧ y = 84 := by
  sorry

end equation_solution_l4035_403595


namespace triangle_properties_l4035_403577

/-- Given a triangle ABC with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C) 
  (h2 : t.a + t.c = 6) 
  (h3 : (1/2) * t.a * t.c * Real.sin t.B = Real.sqrt 3) : 
  t.B = π/3 ∧ t.a + t.b + t.c = 6 + 2 * Real.sqrt 6 := by
  sorry

end triangle_properties_l4035_403577


namespace sqrt_meaningful_range_l4035_403573

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 6 - 3*x) → x ≤ 2 := by
sorry

end sqrt_meaningful_range_l4035_403573


namespace arithmetic_sequence_fifth_term_l4035_403555

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 2 + a 8 = 12) :
  a 5 = 6 := by
sorry

end arithmetic_sequence_fifth_term_l4035_403555


namespace ratio_problem_l4035_403581

theorem ratio_problem (x : ℝ) : 0.75 / x = 5 / 7 → x = 1.05 := by
  sorry

end ratio_problem_l4035_403581


namespace carrie_text_messages_l4035_403519

/-- The number of text messages Carrie sends to her brother on Saturday -/
def saturday_messages : ℕ := 5

/-- The number of text messages Carrie sends to her brother on Sunday -/
def sunday_messages : ℕ := 5

/-- The number of text messages Carrie sends to her brother on each weekday -/
def weekday_messages : ℕ := 2

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weeks we're considering -/
def total_weeks : ℕ := 4

/-- The total number of text messages Carrie sends to her brother over the given period -/
def total_messages : ℕ := 
  total_weeks * (saturday_messages + sunday_messages + weekdays_per_week * weekday_messages)

theorem carrie_text_messages : total_messages = 80 := by
  sorry

end carrie_text_messages_l4035_403519


namespace min_subset_size_for_acute_triangle_l4035_403557

def is_acute_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

theorem min_subset_size_for_acute_triangle :
  ∃ (k : ℕ), k = 29 ∧
  (∀ (S : Finset ℕ), S ⊆ Finset.range 2004 → S.card ≥ k →
    ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ is_acute_triangle a b c) ∧
  (∀ (k' : ℕ), k' < k →
    ∃ (S : Finset ℕ), S ⊆ Finset.range 2004 ∧ S.card = k' ∧
      ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → c ≠ a → ¬is_acute_triangle a b c) :=
by sorry

end min_subset_size_for_acute_triangle_l4035_403557


namespace sphere_ratio_theorem_l4035_403504

/-- Given two spheres with radii r₁ and r₂ where r₁ : r₂ = 1 : 3, 
    prove that their surface areas are in the ratio 1:9 
    and their volumes are in the ratio 1:27 -/
theorem sphere_ratio_theorem (r₁ r₂ : ℝ) (h : r₁ / r₂ = 1 / 3) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 9 ∧
  ((4 / 3) * Real.pi * r₁^3) / ((4 / 3) * Real.pi * r₂^3) = 1 / 27 := by
  sorry

end sphere_ratio_theorem_l4035_403504


namespace perfect_square_trinomial_l4035_403518

/-- A trinomial of the form ax² + bx + c is a perfect square if there exist real numbers p and q
    such that ax² + bx + c = (px + q)² for all x. -/
def IsPerfectSquare (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial (k : ℝ) :
  IsPerfectSquare 4 k 9 → k = 12 ∨ k = -12 := by
  sorry

end perfect_square_trinomial_l4035_403518


namespace quadratic_real_roots_iff_k_eq_one_l4035_403585

/-- 
A quadratic equation ax^2 + bx + c = 0 has real roots if and only if its discriminant b^2 - 4ac is non-negative.
-/
def has_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4*a*c ≥ 0

/--
Given the quadratic equation kx^2 - 3x + 2 = 0, where k is a non-negative integer,
the equation has real roots if and only if k = 1.
-/
theorem quadratic_real_roots_iff_k_eq_one :
  ∀ k : ℕ, has_real_roots k (-3) 2 ↔ k = 1 :=
by sorry

end quadratic_real_roots_iff_k_eq_one_l4035_403585


namespace solve_linear_equation_l4035_403532

theorem solve_linear_equation (y : ℚ) (h : -3 * y - 8 = 10 * y + 5) : y = -1 := by
  sorry

end solve_linear_equation_l4035_403532


namespace parabola_properties_l4035_403525

/-- Parabola properties -/
theorem parabola_properties (a : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^2 - (a + 1) * x
  (f 2 = 0) →
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = -4 ∧ f x₁ = f x₂ → a = -1/5) ∧
  (∀ (x₁ x₂ : ℝ), x₁ > x₂ ∧ x₂ ≥ -2 ∧ f x₁ < f x₂ → -1/5 ≤ a ∧ a < 0) ∧
  (∃ (x : ℝ), x = 1 ∧ ∀ (y : ℝ), f (x + y) = f (x - y)) := by
  sorry

end parabola_properties_l4035_403525


namespace sum_of_g_and_h_l4035_403544

theorem sum_of_g_and_h (a b c d e f g h : ℝ) 
  (avg_abc : (a + b + c) / 3 = 103 / 3)
  (avg_def : (d + e + f) / 3 = 375 / 6)
  (avg_all : (a + b + c + d + e + f + g + h) / 8 = 23 / 2) :
  g + h = -198.5 := by sorry

end sum_of_g_and_h_l4035_403544


namespace octal_1072_equals_base5_4240_l4035_403587

def octal_to_decimal (n : List Nat) : Nat :=
  n.enum.foldr (λ (i, d) acc => acc + d * (8 ^ i)) 0

def decimal_to_base5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem octal_1072_equals_base5_4240 :
  decimal_to_base5 (octal_to_decimal [2, 7, 0, 1]) = [4, 2, 4, 0] := by
  sorry

end octal_1072_equals_base5_4240_l4035_403587


namespace infinite_k_sin_k_greater_than_C_l4035_403561

theorem infinite_k_sin_k_greater_than_C :
  ∀ C : ℝ, ∃ S : Set ℤ, (Set.Infinite S) ∧ (∀ k ∈ S, (k : ℝ) * Real.sin k > C) := by
  sorry

end infinite_k_sin_k_greater_than_C_l4035_403561


namespace cuboids_painted_equals_five_l4035_403500

/-- The number of outer faces on each cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The total number of faces painted -/
def total_faces_painted : ℕ := 30

/-- The number of cuboids painted -/
def num_cuboids : ℕ := total_faces_painted / faces_per_cuboid

theorem cuboids_painted_equals_five :
  num_cuboids = 5 :=
by sorry

end cuboids_painted_equals_five_l4035_403500


namespace quadratic_roots_sum_of_squares_l4035_403575

theorem quadratic_roots_sum_of_squares (a b s p : ℝ) : 
  a^2 + b^2 = 15 → 
  s = a + b → 
  p = a * b → 
  (∀ x, x^2 - s*x + p = 0 ↔ x = a ∨ x = b) → 
  15 = s^2 - 2*p := by
sorry

end quadratic_roots_sum_of_squares_l4035_403575


namespace books_read_l4035_403578

theorem books_read (total_books : ℕ) (books_left : ℕ) (h : total_books = 19 ∧ books_left = 15) : 
  total_books - books_left = 4 := by
  sorry

end books_read_l4035_403578


namespace students_in_both_competitions_l4035_403568

/-- The number of students who participated in both Go and Chess competitions -/
def both_competitions (total : ℕ) (go : ℕ) (chess : ℕ) : ℕ :=
  go + chess - total

/-- Theorem stating the number of students in both competitions -/
theorem students_in_both_competitions :
  both_competitions 32 18 23 = 9 := by
  sorry

end students_in_both_competitions_l4035_403568


namespace certain_number_proof_l4035_403508

theorem certain_number_proof : ∃ x : ℝ, (213 * 16 = 3408 ∧ 16 * x = 340.8) → x = 21.3 := by
  sorry

end certain_number_proof_l4035_403508


namespace vertex_of_quadratic_l4035_403551

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x - 1)^2 + 2

-- State the theorem
theorem vertex_of_quadratic :
  ∃ (a h k : ℝ), (∀ x, f x = a * (x - h)^2 + k) ∧ (f h = k) ∧ (∀ x, f x ≤ k) :=
by
  -- The proof would go here
  sorry

end vertex_of_quadratic_l4035_403551


namespace triangle_properties_l4035_403586

/-- Given a triangle ABC with the following properties:
  BC = √5
  AC = 3
  sin C = 2 sin A
  Prove that:
  1. AB = 2√5
  2. sin(A - π/4) = -√10/10
-/
theorem triangle_properties (A B C : ℝ) (h1 : BC = Real.sqrt 5) (h2 : AC = 3)
    (h3 : Real.sin C = 2 * Real.sin A) :
  AB = 2 * Real.sqrt 5 ∧ Real.sin (A - π/4) = -(Real.sqrt 10)/10 := by
  sorry

end triangle_properties_l4035_403586


namespace tonys_initial_money_l4035_403583

/-- Represents the problem of calculating Tony's initial amount of money --/
theorem tonys_initial_money 
  (bucket_capacity : ℝ)
  (sandbox_depth sandbox_width sandbox_length : ℝ)
  (sand_density : ℝ)
  (water_per_break : ℝ)
  (trips_per_break : ℕ)
  (bottle_capacity : ℝ)
  (bottle_cost : ℝ)
  (change : ℝ)
  (h1 : bucket_capacity = 2)
  (h2 : sandbox_depth = 2)
  (h3 : sandbox_width = 4)
  (h4 : sandbox_length = 5)
  (h5 : sand_density = 3)
  (h6 : water_per_break = 3)
  (h7 : trips_per_break = 4)
  (h8 : bottle_capacity = 15)
  (h9 : bottle_cost = 2)
  (h10 : change = 4) :
  ∃ (initial_money : ℝ), initial_money = 10 := by
sorry

end tonys_initial_money_l4035_403583


namespace currency_conversion_area_conversion_l4035_403521

-- Define the currency units
def yuan : ℝ := 1
def jiao : ℝ := 0.1
def fen : ℝ := 0.01

-- Define the area units
def hectare : ℝ := 10000
def square_meter : ℝ := 1

-- Theorem for currency conversion
theorem currency_conversion :
  6.89 * yuan = 6 * yuan + 8 * jiao + 9 * fen := by sorry

-- Theorem for area conversion
theorem area_conversion :
  2 * hectare + 60 * square_meter = 20060 * square_meter := by sorry

end currency_conversion_area_conversion_l4035_403521


namespace variance_best_for_stability_l4035_403548

/-- Represents a statistical measure -/
inductive StatMeasure
  | Mode
  | Variance
  | Mean
  | Frequency

/-- Represents an athlete's performance data -/
structure AthleteData where
  results : List Float
  len : Nat
  h_len : len = 10

/-- Assesses the stability of performance based on a statistical measure -/
def assessStability (measure : StatMeasure) (data : AthleteData) : Bool :=
  sorry

/-- Theorem stating that variance is the most suitable measure for assessing stability -/
theorem variance_best_for_stability (data : AthleteData) :
  ∀ (m : StatMeasure), m ≠ StatMeasure.Variance →
    assessStability StatMeasure.Variance data = true ∧
    assessStability m data = false :=
  sorry

end variance_best_for_stability_l4035_403548


namespace work_time_proof_l4035_403599

theorem work_time_proof (a b c h : ℝ) : 
  (1 / a + 1 / b + 1 / c = 1 / (a - 6)) →
  (1 / a + 1 / b + 1 / c = 1 / (b - 1)) →
  (1 / a + 1 / b + 1 / c = 2 / c) →
  (1 / a + 1 / b = 1 / h) →
  (a > 0) → (b > 0) → (c > 0) → (h > 0) →
  h = 4/3 := by
sorry

end work_time_proof_l4035_403599


namespace f_g_intersection_l4035_403541

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * Real.log x

/-- The function g(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 - (a + 1) * x

/-- Theorem stating that f and g have exactly one intersection point when a ≥ 0 -/
theorem f_g_intersection (a : ℝ) (h : a ≥ 0) :
  ∃! x : ℝ, x > 0 ∧ f a x = g a x :=
sorry

end f_g_intersection_l4035_403541


namespace min_a_for_decreasing_h_range_a_for_p_greater_q_l4035_403524

noncomputable section

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := x + 4 * a / x - 1
def g (a : ℝ) (x : ℝ) : ℝ := a * Real.log x
def h (a : ℝ) (x : ℝ) : ℝ := f a x - g a x
def p (x : ℝ) : ℝ := (2 - x^3) * Real.exp x
def q (a : ℝ) (x : ℝ) : ℝ := g a x / x + 2

-- Part I: Minimum value of a for h to be decreasing on [1,3]
theorem min_a_for_decreasing_h : 
  (∀ x ∈ Set.Icc 1 3, ∀ y ∈ Set.Icc 1 3, x ≤ y → h (9/7) x ≥ h (9/7) y) ∧
  (∀ a < 9/7, ∃ x ∈ Set.Icc 1 3, ∃ y ∈ Set.Icc 1 3, x < y ∧ h a x < h a y) :=
sorry

-- Part II: Range of a for p(x₁) > q(x₂) to hold for any x₁, x₂ ∈ (0,1)
theorem range_a_for_p_greater_q :
  (∀ a ≥ 0, ∀ x₁ ∈ Set.Ioo 0 1, ∀ x₂ ∈ Set.Ioo 0 1, p x₁ > q a x₂) ∧
  (∀ a < 0, ∃ x₁ ∈ Set.Ioo 0 1, ∃ x₂ ∈ Set.Ioo 0 1, p x₁ ≤ q a x₂) :=
sorry

end min_a_for_decreasing_h_range_a_for_p_greater_q_l4035_403524


namespace quadratic_roots_sum_l4035_403502

theorem quadratic_roots_sum (x₁ x₂ : ℝ) : 
  x₁^2 + x₁ - 2023 = 0 → 
  x₂^2 + x₂ - 2023 = 0 → 
  x₁^2 + 2*x₁ + x₂ = 2022 := by
sorry

end quadratic_roots_sum_l4035_403502


namespace expand_square_root_two_l4035_403513

theorem expand_square_root_two (a b : ℚ) : (1 + Real.sqrt 2)^5 = a + b * Real.sqrt 2 → a + b = 70 := by
  sorry

end expand_square_root_two_l4035_403513


namespace diana_remaining_paint_l4035_403538

/-- The amount of paint required for one statue in gallons -/
def paint_per_statue : ℚ := 1/8

/-- The number of statues Diana can paint with the remaining paint -/
def statues_to_paint : ℕ := 7

/-- The total amount of paint Diana has remaining in gallons -/
def remaining_paint : ℚ := paint_per_statue * statues_to_paint

theorem diana_remaining_paint : remaining_paint = 7/8 := by
  sorry

end diana_remaining_paint_l4035_403538


namespace correct_decision_probability_l4035_403567

-- Define the probability of a consultant giving a correct opinion
def p_correct : ℝ := 0.8

-- Define the number of consultants
def n_consultants : ℕ := 3

-- Define the probability of making a correct decision
def p_correct_decision : ℝ :=
  (Nat.choose n_consultants 2) * p_correct^2 * (1 - p_correct) +
  (Nat.choose n_consultants 3) * p_correct^3

-- Theorem statement
theorem correct_decision_probability :
  p_correct_decision = 0.896 := by sorry

end correct_decision_probability_l4035_403567


namespace joseph_kyle_distance_difference_l4035_403589

theorem joseph_kyle_distance_difference : 
  let joseph_speed : ℝ := 50
  let joseph_time : ℝ := 2.5
  let kyle_speed : ℝ := 62
  let kyle_time : ℝ := 2
  let joseph_distance := joseph_speed * joseph_time
  let kyle_distance := kyle_speed * kyle_time
  joseph_distance - kyle_distance = 1 := by
sorry

end joseph_kyle_distance_difference_l4035_403589


namespace rectangular_plot_length_l4035_403571

/-- Given a rectangular plot with the following properties:
  - The length is 10 meters more than the breadth
  - The cost of fencing is 26.50 per meter
  - The total cost of fencing is 5300
  Prove that the length of the plot is 55 meters. -/
theorem rectangular_plot_length (breadth : ℝ) (length : ℝ) : 
  length = breadth + 10 →
  26.50 * (2 * (length + breadth)) = 5300 →
  length = 55 := by sorry

end rectangular_plot_length_l4035_403571


namespace inequality_implication_l4035_403536

theorem inequality_implication (a b : ℝ) : a < b → -a + 3 > -b + 3 := by
  sorry

end inequality_implication_l4035_403536


namespace unique_four_digit_number_l4035_403540

theorem unique_four_digit_number : ∃! n : ℕ, 
  (1000 ≤ n ∧ n ≤ 9999) ∧ 
  (∃ a : ℕ, n = a^2) ∧
  (∃ b : ℕ, n % 1000 = b^3) ∧
  (∃ c : ℕ, n % 100 = c^4) ∧
  n = 9216 :=
by sorry

end unique_four_digit_number_l4035_403540


namespace range_of_m_l4035_403592

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = x * y)
  (h_inequality : ∃ x y, x > 0 ∧ y > 0 ∧ x + y = x * y ∧ x + 4 * y < m^2 + 8 * m) :
  m < -9 ∨ m > 1 :=
by sorry

end range_of_m_l4035_403592


namespace min_distance_line_parabola_l4035_403542

noncomputable def line (x : ℝ) : ℝ := (15/8) * x - 8

noncomputable def parabola (x : ℝ) : ℝ := x^2

theorem min_distance_line_parabola :
  ∃ (x₁ x₂ : ℝ),
    (∀ y₁ y₂ : ℝ,
      y₁ = line x₁ ∧ y₂ = parabola x₂ →
      (x₂ - x₁)^2 + (y₂ - y₁)^2 ≥ (1823/544)^2) :=
by sorry

end min_distance_line_parabola_l4035_403542


namespace inequality_preservation_l4035_403545

theorem inequality_preservation (a b c : ℝ) (h : a > b) : a + c > b + c := by
  sorry

end inequality_preservation_l4035_403545


namespace intersection_condition_l4035_403562

theorem intersection_condition (m : ℝ) : 
  let A := {x : ℝ | x^2 - 3*x + 2 = 0}
  let C := {x : ℝ | x^2 - m*x + 2 = 0}
  (A ∩ C = C) → (m = 3 ∨ -2*Real.sqrt 2 < m ∧ m < 2*Real.sqrt 2) :=
by sorry

end intersection_condition_l4035_403562


namespace jackies_tree_climb_l4035_403510

theorem jackies_tree_climb (h : ℝ) : 
  h > 0 →                             -- Height is positive
  (h + h/2 + h/2 + (h + 200)) / 4 = 800 →  -- Average height condition
  h = 1000 := by
sorry

end jackies_tree_climb_l4035_403510


namespace correct_passengers_off_l4035_403507

/-- Calculates the number of passengers who got off the bus at other stops -/
def passengers_who_got_off (initial : ℕ) (first_stop : ℕ) (other_stops : ℕ) (final : ℕ) : ℕ :=
  initial + first_stop - (final - other_stops)

theorem correct_passengers_off : passengers_who_got_off 50 16 5 49 = 22 := by
  sorry

end correct_passengers_off_l4035_403507


namespace simple_interest_principal_l4035_403554

def simple_interest_rate : ℚ := 8 / 100
def simple_interest_time : ℕ := 5
def compound_principal : ℕ := 8000
def compound_interest_rate : ℚ := 15 / 100
def compound_interest_time : ℕ := 2

def compound_interest (P : ℕ) (r : ℚ) (t : ℕ) : ℚ :=
  P * ((1 + r) ^ t - 1)

theorem simple_interest_principal :
  ∃ (P : ℕ), 
    (P : ℚ) * simple_interest_rate * simple_interest_time = 
    (1 / 2) * compound_interest compound_principal compound_interest_rate compound_interest_time ∧
    P = 3225 := by
  sorry

end simple_interest_principal_l4035_403554


namespace combined_mean_of_two_sets_l4035_403553

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℚ) 
                                  (set2_count : ℕ) (set2_mean : ℚ) :
  set1_count = 7 →
  set1_mean = 15 →
  set2_count = 8 →
  set2_mean = 30 →
  let total_count := set1_count + set2_count
  let total_sum := set1_count * set1_mean + set2_count * set2_mean
  (total_sum / total_count : ℚ) = 23 := by
  sorry

end combined_mean_of_two_sets_l4035_403553


namespace sculpture_cost_in_cny_l4035_403570

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℚ := 8

/-- Exchange rate from US dollars to Chinese yuan -/
def usd_to_cny : ℚ := 5

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℚ := 160

/-- Converts Namibian dollars to Chinese yuan -/
def nad_to_cny (nad : ℚ) : ℚ :=
  nad * (usd_to_cny / usd_to_nad)

theorem sculpture_cost_in_cny :
  nad_to_cny sculpture_cost_nad = 100 := by
  sorry

end sculpture_cost_in_cny_l4035_403570


namespace wilsonTotalIsCorrect_l4035_403547

/-- Calculates the total amount Wilson pays at a fast-food restaurant -/
def wilsonTotal : ℝ :=
  let hamburgerPrice := 5
  let hamburgerCount := 2
  let colaPrice := 2
  let colaCount := 3
  let friesPrice := 3
  let sundaePrice := 4
  let nuggetPrice := 1.5
  let nuggetCount := 4
  let saladPrice := 6.25
  let couponDiscount := 4
  let loyaltyDiscount := 0.1
  let freeNuggetCount := 1

  let initialTotal := hamburgerPrice * hamburgerCount + colaPrice * colaCount + 
                      friesPrice + sundaePrice + nuggetPrice * nuggetCount + saladPrice
  let promotionDiscount := nuggetPrice * freeNuggetCount
  let afterPromotionTotal := initialTotal - promotionDiscount
  let afterCouponTotal := afterPromotionTotal - couponDiscount
  let finalTotal := afterCouponTotal * (1 - loyaltyDiscount)

  finalTotal

theorem wilsonTotalIsCorrect : wilsonTotal = 26.77 := by sorry

end wilsonTotalIsCorrect_l4035_403547


namespace pages_to_read_on_day_three_l4035_403559

theorem pages_to_read_on_day_three 
  (total_pages : ℕ) 
  (pages_day_one : ℕ) 
  (pages_day_two : ℕ) 
  (h1 : total_pages = 100)
  (h2 : pages_day_one = 35)
  (h3 : pages_day_two = pages_day_one - 5) :
  total_pages - (pages_day_one + pages_day_two) = 35 := by
  sorry

end pages_to_read_on_day_three_l4035_403559


namespace exact_sequence_2007_l4035_403503

/-- An exact sequence of integers. -/
def ExactSequence (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n > m → a n ^ 2 - a m ^ 2 = a (n - m) * a (n + m)

/-- The 2007th term of the exact sequence with given initial conditions. -/
theorem exact_sequence_2007 (a : ℕ → ℤ) 
    (h_exact : ExactSequence a) 
    (h_init1 : a 1 = 1) 
    (h_init2 : a 2 = 0) : 
  a 2007 = -1 := by
  sorry

end exact_sequence_2007_l4035_403503


namespace quadratic_equation_coefficients_l4035_403543

theorem quadratic_equation_coefficients :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 4 * x - 1
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c ∧ a = 3 ∧ b = -4 ∧ c = -1) :=
by
  sorry

end quadratic_equation_coefficients_l4035_403543


namespace trip_cost_is_1050_l4035_403537

-- Define the distances and costs
def distance_AB : ℝ := 4000
def distance_BC : ℝ := 3000
def bus_rate : ℝ := 0.15
def plane_rate : ℝ := 0.12
def plane_booking_fee : ℝ := 120

-- Define the total trip cost function
def total_trip_cost : ℝ :=
  (distance_AB * plane_rate + plane_booking_fee) + (distance_BC * bus_rate)

-- Theorem statement
theorem trip_cost_is_1050 : total_trip_cost = 1050 := by
  sorry

end trip_cost_is_1050_l4035_403537


namespace parabola_focus_l4035_403539

/-- A parabola is defined by its coefficients a, b, and c in the equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The focus of a parabola is a point (h, k) -/
structure Focus where
  h : ℝ
  k : ℝ

/-- Given a parabola y = -2x^2 - 4x + 1, its focus is at (-1, 23/8) -/
theorem parabola_focus (p : Parabola) (f : Focus) :
  p.a = -2 ∧ p.b = -4 ∧ p.c = 1 →
  f.h = -1 ∧ f.k = 23/8 :=
by sorry

end parabola_focus_l4035_403539


namespace zoo_total_revenue_l4035_403534

def monday_children : Nat := 7
def monday_adults : Nat := 5
def tuesday_children : Nat := 4
def tuesday_adults : Nat := 2
def child_ticket_cost : Nat := 3
def adult_ticket_cost : Nat := 4

theorem zoo_total_revenue : 
  (monday_children + tuesday_children) * child_ticket_cost + 
  (monday_adults + tuesday_adults) * adult_ticket_cost = 61 := by
  sorry

#eval (monday_children + tuesday_children) * child_ticket_cost + 
      (monday_adults + tuesday_adults) * adult_ticket_cost

end zoo_total_revenue_l4035_403534


namespace absolute_value_inequality_l4035_403531

theorem absolute_value_inequality (m : ℝ) : 
  (∀ x : ℝ, |x - 2| + |x + 4| ≥ m^2 - 5*m) ↔ -1 ≤ m ∧ m ≤ 6 := by
sorry

end absolute_value_inequality_l4035_403531


namespace min_blue_eyes_and_lunch_box_l4035_403516

theorem min_blue_eyes_and_lunch_box 
  (total_students : ℕ) 
  (blue_eyes : ℕ) 
  (lunch_box : ℕ) 
  (h1 : total_students = 35) 
  (h2 : blue_eyes = 15) 
  (h3 : lunch_box = 25) :
  ∃ (overlap : ℕ), 
    overlap ≥ 5 ∧ 
    overlap ≤ blue_eyes ∧ 
    overlap ≤ lunch_box ∧ 
    (∀ (x : ℕ), x < overlap → 
      x + (total_students - lunch_box) < blue_eyes ∨ 
      x + (total_students - blue_eyes) < lunch_box) :=
by sorry

end min_blue_eyes_and_lunch_box_l4035_403516


namespace selling_price_loss_l4035_403530

/-- Represents the ratio of selling price to cost price -/
def price_ratio : ℚ := 2 / 5

/-- The loss percentage when selling price is less than cost price -/
def loss_percent (r : ℚ) : ℚ := (1 - r) * 100

theorem selling_price_loss :
  price_ratio = 2 / 5 →
  loss_percent price_ratio = 60 := by
sorry

end selling_price_loss_l4035_403530


namespace last_digit_of_large_power_l4035_403593

theorem last_digit_of_large_power : ∃ (n1 n2 n3 : ℕ), 
  n1 = 99^9 ∧ 
  n2 = 999^n1 ∧ 
  n3 = 9999^n2 ∧ 
  99999^n3 % 10 = 9 := by
  sorry

end last_digit_of_large_power_l4035_403593


namespace complex_fraction_equality_l4035_403598

theorem complex_fraction_equality : (1 : ℂ) / (3 * I + 1) = (1 : ℂ) / 10 + (3 : ℂ) * I / 10 := by
  sorry

end complex_fraction_equality_l4035_403598


namespace triangle_on_hyperbola_l4035_403526

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y = 1 / x

-- Define a point on the hyperbola
structure PointOnHyperbola where
  x : ℝ
  y : ℝ
  on_hyperbola : hyperbola x y

-- Define parallel lines
def parallel (p1 p2 p3 p4 : PointOnHyperbola) : Prop :=
  (p2.y - p1.y) / (p2.x - p1.x) = (p4.y - p3.y) / (p4.x - p3.x)

-- Define the theorem
theorem triangle_on_hyperbola
  (A B C A₁ B₁ C₁ : PointOnHyperbola)
  (h1 : parallel A B A₁ B₁)
  (h2 : parallel B C B₁ C₁) :
  parallel A C₁ A₁ C := by
  sorry

end triangle_on_hyperbola_l4035_403526


namespace extended_parallelepiped_volume_l4035_403529

/-- The volume of a set described by a rectangular parallelepiped extended by unit radius cylinders and spheres -/
theorem extended_parallelepiped_volume :
  let l : ℝ := 2  -- length
  let w : ℝ := 3  -- width
  let h : ℝ := 6  -- height
  let r : ℝ := 1  -- radius of extension

  -- Volume of the original parallelepiped
  let v_box := l * w * h

  -- Volume of outward projecting parallelepipeds
  let v_out := 2 * (r * w * h + r * l * h + r * l * w)

  -- Volume of quarter-cylinders along edges
  let edge_length := 2 * (l + w + h)
  let v_cyl := (π * r^2 / 4) * edge_length

  -- Volume of eighth-spheres at vertices
  let v_sph := 8 * ((4 / 3) * π * r^3 / 8)

  -- Total volume
  let v_total := v_box + v_out + v_cyl + v_sph

  v_total = (324 + 70 * π) / 3 :=
by sorry

end extended_parallelepiped_volume_l4035_403529


namespace ping_pong_practice_time_l4035_403511

theorem ping_pong_practice_time 
  (total_students : ℕ) 
  (practicing_simultaneously : ℕ) 
  (total_time : ℕ) 
  (h1 : total_students = 5)
  (h2 : practicing_simultaneously = 2)
  (h3 : total_time = 90) :
  (total_time * practicing_simultaneously) / total_students = 36 :=
by sorry

end ping_pong_practice_time_l4035_403511


namespace fewer_threes_for_hundred_l4035_403512

-- Define a type for arithmetic expressions
inductive Expr
  | num : Int → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

-- Function to evaluate an expression
def eval : Expr → Int
  | Expr.num n => n
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

-- Function to count the number of threes in an expression
def countThrees : Expr → Nat
  | Expr.num 3 => 1
  | Expr.num _ => 0
  | Expr.add e1 e2 => countThrees e1 + countThrees e2
  | Expr.sub e1 e2 => countThrees e1 + countThrees e2
  | Expr.mul e1 e2 => countThrees e1 + countThrees e2
  | Expr.div e1 e2 => countThrees e1 + countThrees e2

-- Theorem: There exists an expression using fewer than ten threes that evaluates to 100
theorem fewer_threes_for_hundred : ∃ e : Expr, eval e = 100 ∧ countThrees e < 10 := by
  sorry


end fewer_threes_for_hundred_l4035_403512


namespace square_root_of_1024_l4035_403580

theorem square_root_of_1024 (y : ℝ) (h1 : y > 0) (h2 : y^2 = 1024) : y = 32 := by
  sorry

end square_root_of_1024_l4035_403580


namespace cookie_ratio_l4035_403563

/-- Given a total of 14 bags, 28 cookies, and 2 bags of cookies,
    prove that the ratio of cookies in each bag to the total number of cookies is 1:2 -/
theorem cookie_ratio (total_bags : ℕ) (total_cookies : ℕ) (cookie_bags : ℕ)
  (h1 : total_bags = 14)
  (h2 : total_cookies = 28)
  (h3 : cookie_bags = 2) :
  (total_cookies / cookie_bags) / total_cookies = 1 / 2 :=
by sorry

end cookie_ratio_l4035_403563


namespace more_girls_than_boys_l4035_403558

theorem more_girls_than_boys (total_students : ℕ) (boy_ratio girl_ratio : ℕ) 
  (h_total : total_students = 49)
  (h_ratio : boy_ratio = 3 ∧ girl_ratio = 4) : 
  let y := total_students / (boy_ratio + girl_ratio)
  let num_boys := boy_ratio * y
  let num_girls := girl_ratio * y
  num_girls - num_boys = 7 := by
  sorry

end more_girls_than_boys_l4035_403558


namespace dance_group_size_l4035_403594

theorem dance_group_size (calligraphy_group : ℕ) (dance_group : ℕ) : 
  calligraphy_group = 28 → 
  calligraphy_group = 2 * dance_group + 6 → 
  dance_group = 11 := by
sorry

end dance_group_size_l4035_403594


namespace factorization_of_x_squared_minus_x_l4035_403590

theorem factorization_of_x_squared_minus_x (x : ℝ) : x^2 - x = x * (x - 1) := by
  sorry

end factorization_of_x_squared_minus_x_l4035_403590


namespace triangle_angle_not_all_greater_than_60_l4035_403588

theorem triangle_angle_not_all_greater_than_60 :
  ¬ ∀ (a b c : ℝ), 
    (a > 0) → (b > 0) → (c > 0) → 
    (a + b + c = 180) → 
    (a > 60 ∧ b > 60 ∧ c > 60) :=
by sorry

end triangle_angle_not_all_greater_than_60_l4035_403588


namespace square_circle_area_fraction_l4035_403572

theorem square_circle_area_fraction (r : ℝ) (h : r > 0) :
  let square_area := (2 * r)^2
  let circle_area := π * r^2
  let outside_area := square_area - circle_area
  outside_area / square_area = 1 - π / 4 := by
  sorry

end square_circle_area_fraction_l4035_403572


namespace unique_x_l4035_403514

theorem unique_x : ∃! x : ℕ, x > 0 ∧ ∃ k : ℕ, x = 9 * k ∧ x^2 < 200 ∧ x < 25 := by
  sorry

end unique_x_l4035_403514


namespace max_value_theorem_equality_conditions_l4035_403596

theorem max_value_theorem (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  (a * b * c * d) ^ (1/4) + ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ^ (1/2) ≤ 1 :=
sorry

theorem equality_conditions (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  (a * b * c * d) ^ (1/4) + ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ^ (1/2) = 1 ↔ 
  ((a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨ (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1)) :=
sorry

end max_value_theorem_equality_conditions_l4035_403596


namespace imaginary_product_condition_l4035_403579

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the theorem
theorem imaginary_product_condition (a : ℝ) : 
  (((1 : ℂ) + i) * ((1 : ℂ) + a * i)).re = 0 → a = 1 := by
  sorry

-- Note: We use .re to get the real part of the complex number, 
-- which should be 0 for a purely imaginary number.

end imaginary_product_condition_l4035_403579


namespace fraction_subtraction_l4035_403546

theorem fraction_subtraction : ((5 / 2) / (7 / 12)) - 4 / 9 = 242 / 63 := by sorry

end fraction_subtraction_l4035_403546


namespace problem_1_problem_2_l4035_403597

-- Problem 1
theorem problem_1 (x y : ℝ) : (x + y)^2 + x * (x - 2*y) = 2*x^2 + y^2 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h : x ≠ 2 ∧ x ≠ 0) : 
  (x^2 - 6*x + 9) / (x - 2) / (x + 2 - (3*x - 4) / (x - 2)) = (x - 3) / x := by
  sorry

end problem_1_problem_2_l4035_403597


namespace fraction_inequality_l4035_403591

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < 0) :
  c / a > c / b := by
  sorry

end fraction_inequality_l4035_403591


namespace common_off_days_count_l4035_403506

/-- Charlie's work cycle in days -/
def charlie_cycle : ℕ := 6

/-- Dana's work cycle in days -/
def dana_cycle : ℕ := 7

/-- Total number of days -/
def total_days : ℕ := 1500

/-- Function to calculate the number of common off days -/
def common_off_days (charlie_cycle dana_cycle total_days : ℕ) : ℕ :=
  2 * (total_days / (charlie_cycle.lcm dana_cycle))

/-- Theorem stating that Charlie and Dana have 70 common off days -/
theorem common_off_days_count : 
  common_off_days charlie_cycle dana_cycle total_days = 70 := by
  sorry

end common_off_days_count_l4035_403506


namespace original_photo_dimensions_l4035_403535

/-- Represents the dimensions of a rectangular photo frame --/
structure PhotoFrame where
  width : ℕ
  height : ℕ

/-- Calculates the number of squares needed for a frame --/
def squares_for_frame (frame : PhotoFrame) : ℕ :=
  2 * (frame.width + frame.height)

/-- Theorem stating the dimensions of the original photo --/
theorem original_photo_dimensions 
  (original_squares : ℕ) 
  (cut_squares : ℕ) 
  (h1 : original_squares = 1812)
  (h2 : cut_squares = 2018) :
  ∃ (frame : PhotoFrame), 
    squares_for_frame frame = original_squares ∧ 
    frame.width = 803 ∧ 
    frame.height = 101 ∧
    cut_squares - original_squares = 2 * frame.height :=
sorry


end original_photo_dimensions_l4035_403535


namespace greatest_3digit_base8_divisible_by_7_l4035_403584

/-- Converts a base 8 number to base 10 -/
def base8ToBase10 (n : Nat) : Nat :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

/-- Checks if a number is a valid 3-digit base 8 number -/
def isValid3DigitBase8 (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 777

theorem greatest_3digit_base8_divisible_by_7 :
  ∃ (n : Nat), isValid3DigitBase8 n ∧
               base8ToBase10 n % 7 = 0 ∧
               ∀ (m : Nat), isValid3DigitBase8 m ∧ base8ToBase10 m % 7 = 0 → m ≤ n :=
by sorry

end greatest_3digit_base8_divisible_by_7_l4035_403584


namespace rectangle_area_diagonal_l4035_403574

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 4) (h2 : l^2 + w^2 = d^2) :
  l * w = (20 / 41) * d^2 := by
  sorry

end rectangle_area_diagonal_l4035_403574


namespace cosine_angle_between_vectors_l4035_403509

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (5, 12)

theorem cosine_angle_between_vectors :
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  dot_product / (magnitude_a * magnitude_b) = 63 / 65 := by
sorry

end cosine_angle_between_vectors_l4035_403509


namespace unfair_die_expected_value_l4035_403549

/-- An unfair 8-sided die with specific probability distribution -/
structure UnfairDie where
  /-- The probability of rolling an 8 -/
  p_eight : ℝ
  /-- The probability of rolling any number from 1 to 7 -/
  p_others : ℝ
  /-- The die has 8 sides -/
  sides : Nat
  sides_eq : sides = 8
  /-- The probability of rolling an 8 is 3/8 -/
  p_eight_eq : p_eight = 3/8
  /-- The sum of all probabilities is 1 -/
  prob_sum : p_eight + 7 * p_others = 1

/-- The expected value of rolling the unfair die -/
def expected_value (d : UnfairDie) : ℝ :=
  d.p_others * (1 + 2 + 3 + 4 + 5 + 6 + 7) + d.p_eight * 8

/-- Theorem: The expected value of rolling this unfair 8-sided die is 5.5 -/
theorem unfair_die_expected_value (d : UnfairDie) : expected_value d = 5.5 := by
  sorry

end unfair_die_expected_value_l4035_403549


namespace rectangular_prism_parallel_edges_l4035_403527

/-- A rectangular prism with different dimensions for length, width, and height. -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : length > 0
  width_pos : width > 0
  height_pos : height > 0
  different_dimensions : length ≠ width ∧ width ≠ height ∧ length ≠ height

/-- The number of pairs of parallel edges in a rectangular prism. -/
def parallel_edge_pairs (prism : RectangularPrism) : ℕ := 12

/-- Theorem stating that a rectangular prism has exactly 12 pairs of parallel edges. -/
theorem rectangular_prism_parallel_edges (prism : RectangularPrism) :
  parallel_edge_pairs prism = 12 := by
  sorry

end rectangular_prism_parallel_edges_l4035_403527
