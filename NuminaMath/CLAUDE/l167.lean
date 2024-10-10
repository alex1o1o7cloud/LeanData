import Mathlib

namespace tank_cost_l167_16742

theorem tank_cost (buy_price sell_price : ℚ) (num_sold : ℕ) (profit_percentage : ℚ) :
  buy_price = 0.25 →
  sell_price = 0.75 →
  num_sold = 110 →
  profit_percentage = 0.55 →
  (sell_price - buy_price) * num_sold = profit_percentage * 100 :=
by sorry

end tank_cost_l167_16742


namespace prime_fourth_powers_sum_l167_16723

theorem prime_fourth_powers_sum (p q r s : ℕ) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s →
  p ≤ q ∧ q ≤ r →
  p^4 + q^4 + r^4 + 119 = s^2 →
  p = 2 ∧ q = 3 ∧ r = 5 ∧ s = 29 := by
  sorry

end prime_fourth_powers_sum_l167_16723


namespace dog_food_weight_l167_16792

theorem dog_food_weight (initial_amount : ℝ) (second_bag_weight : ℝ) (final_amount : ℝ) 
  (h1 : initial_amount = 15)
  (h2 : second_bag_weight = 10)
  (h3 : final_amount = 40) :
  ∃ (first_bag_weight : ℝ), 
    initial_amount + first_bag_weight + second_bag_weight = final_amount ∧ 
    first_bag_weight = 15 := by
  sorry

end dog_food_weight_l167_16792


namespace infinite_product_l167_16717

open Set Filter

-- Define the concept of a function being infinite at a point
def IsInfiniteAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ K > 0, ∃ δ > 0, ∀ x, 0 < |x - x₀| ∧ |x - x₀| < δ → |f x| > K

-- Define the theorem
theorem infinite_product (f g : ℝ → ℝ) (x₀ M : ℝ) (hM : M > 0)
    (hg : ∀ x, |x - x₀| > 0 → |g x| ≥ M)
    (hf : IsInfiniteAt f x₀) :
    IsInfiniteAt (fun x ↦ f x * g x) x₀ := by
  sorry


end infinite_product_l167_16717


namespace fourth_person_height_l167_16793

/-- Given four people with heights in increasing order, prove the height of the fourth person. -/
theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℝ) : 
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →  -- Heights in increasing order
  h₂ - h₁ = 2 →  -- Difference between 1st and 2nd
  h₃ - h₂ = 2 →  -- Difference between 2nd and 3rd
  h₄ - h₃ = 6 →  -- Difference between 3rd and 4th
  (h₁ + h₂ + h₃ + h₄) / 4 = 79 →  -- Average height
  h₄ = 85 :=
by sorry

end fourth_person_height_l167_16793


namespace quadratic_always_real_roots_triangle_perimeter_l167_16712

-- Define the quadratic equation
def quadratic_equation (x k : ℝ) : Prop :=
  x^2 - (k + 2) * x + 2 * k = 0

-- Theorem 1: The equation always has real roots
theorem quadratic_always_real_roots (k : ℝ) :
  ∃ x : ℝ, quadratic_equation x k := by sorry

-- Define a right triangle with hypotenuse 3 and other sides as roots of the equation
def right_triangle_from_equation (k : ℝ) : Prop :=
  ∃ b c : ℝ,
    quadratic_equation b k ∧
    quadratic_equation c k ∧
    b^2 + c^2 = 3^2

-- Theorem 2: The perimeter of the triangle is 5 + √5
theorem triangle_perimeter (k : ℝ) :
  right_triangle_from_equation k →
  ∃ b c : ℝ, b + c + 3 = 5 + Real.sqrt 5 := by sorry

end quadratic_always_real_roots_triangle_perimeter_l167_16712


namespace total_digits_first_2500_even_integers_l167_16729

/-- The number of digits in a positive integer -/
def numDigits (n : ℕ) : ℕ := sorry

/-- The sum of digits for all even numbers from 2 to n -/
def sumDigitsEven (n : ℕ) : ℕ := sorry

/-- The 2500th positive even integer -/
def nthEvenInteger : ℕ := 5000

theorem total_digits_first_2500_even_integers :
  sumDigitsEven nthEvenInteger = 9448 := by sorry

end total_digits_first_2500_even_integers_l167_16729


namespace isosceles_right_triangle_congruence_l167_16745

/-- Given two congruent isosceles right triangles sharing a common base,
    if one leg of one triangle is 12, then the corresponding leg of the other triangle is 6√2 -/
theorem isosceles_right_triangle_congruence (a b c d : ℝ) :
  a = b ∧                    -- Triangle 1 is isosceles
  c = d ∧                    -- Triangle 2 is isosceles
  a^2 + a^2 = b^2 ∧          -- Triangle 1 is right-angled (Pythagorean theorem)
  c^2 + c^2 = d^2 ∧          -- Triangle 2 is right-angled (Pythagorean theorem)
  b = d ∧                    -- Triangles share a common base
  a = 12                     -- Given leg length in Triangle 1
  → c = 6 * Real.sqrt 2      -- To prove: corresponding leg in Triangle 2
:= by sorry

end isosceles_right_triangle_congruence_l167_16745


namespace tech_students_count_l167_16772

/-- Number of students in subject elective courses -/
def subject_students : ℕ → ℕ := fun m ↦ m

/-- Number of students in physical education and arts elective courses -/
def pe_arts_students : ℕ → ℕ := fun m ↦ m + 9

/-- Number of students in technology elective courses -/
def tech_students : ℕ → ℕ := fun m ↦ (pe_arts_students m) / 3 + 5

theorem tech_students_count (m : ℕ) : 
  tech_students m = m / 3 + 8 := by sorry

end tech_students_count_l167_16772


namespace sum_of_exponents_is_eight_l167_16757

/-- Represents the exponents of variables in a simplified cube root expression -/
structure SimplifiedCubeRootExponents where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Simplifies the cube root of 40a^6b^7c^14 and returns the exponents of variables outside the radical -/
def simplify_cube_root : SimplifiedCubeRootExponents := {
  a := 2,
  b := 2,
  c := 4
}

/-- The sum of exponents outside the radical after simplifying ∛(40a^6b^7c^14) is 8 -/
theorem sum_of_exponents_is_eight :
  (simplify_cube_root.a + simplify_cube_root.b + simplify_cube_root.c) = 8 := by
  sorry

end sum_of_exponents_is_eight_l167_16757


namespace intersection_of_A_and_B_l167_16759

def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := {2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {3} := by
  sorry

end intersection_of_A_and_B_l167_16759


namespace marks_trees_l167_16776

theorem marks_trees (initial_trees : ℕ) (new_trees_per_existing : ℕ) : 
  initial_trees = 93 → new_trees_per_existing = 8 → 
  initial_trees + initial_trees * new_trees_per_existing = 837 := by
sorry

end marks_trees_l167_16776


namespace one_true_statement_proves_normal_one_false_statement_proves_normal_one_statement_proves_normal_l167_16711

-- Define the types of people on the island
inductive PersonType
  | Knight
  | Liar
  | Normal

-- Define a statement as either true or false
inductive Statement
  | True
  | False

-- Define a function to determine if a person can make a given statement
def canMakeStatement (person : PersonType) (statement : Statement) : Prop :=
  match person, statement with
  | PersonType.Knight, Statement.True => True
  | PersonType.Knight, Statement.False => False
  | PersonType.Liar, Statement.True => False
  | PersonType.Liar, Statement.False => True
  | PersonType.Normal, _ => True

-- Theorem: One true statement is sufficient to prove one is a normal person
theorem one_true_statement_proves_normal (person : PersonType) :
  canMakeStatement person Statement.True → person = PersonType.Normal :=
sorry

-- Theorem: One false statement is sufficient to prove one is a normal person
theorem one_false_statement_proves_normal (person : PersonType) :
  canMakeStatement person Statement.False → person = PersonType.Normal :=
sorry

-- Main theorem: Either one true or one false statement is sufficient to prove one is a normal person
theorem one_statement_proves_normal (person : PersonType) :
  (canMakeStatement person Statement.True ∨ canMakeStatement person Statement.False) →
  person = PersonType.Normal :=
sorry

end one_true_statement_proves_normal_one_false_statement_proves_normal_one_statement_proves_normal_l167_16711


namespace k_value_proof_l167_16740

theorem k_value_proof (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 5)) → k = 5 := by
  sorry

end k_value_proof_l167_16740


namespace michaels_subtraction_l167_16735

theorem michaels_subtraction (a b : ℕ) (h1 : a = 40) (h2 : b = 39) :
  a^2 - b^2 = 79 := by
  sorry

end michaels_subtraction_l167_16735


namespace modulus_of_z_is_5_l167_16790

-- Define the complex number z
def z : ℂ := (2 - Complex.I) ^ 2

-- Theorem stating that the modulus of z is 5
theorem modulus_of_z_is_5 : Complex.abs z = 5 := by
  sorry

end modulus_of_z_is_5_l167_16790


namespace sequence_properties_l167_16778

-- Define the sequence type
def Sequence := ℕ → ℝ

-- Define arithmetic progression
def is_arithmetic_progression (a : Sequence) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define geometric progression
def is_geometric_progression (a : Sequence) :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem sequence_properties (a : Sequence) 
  (h1 : a 4 + a 7 = 2) 
  (h2 : a 5 * a 6 = -8) :
  (is_arithmetic_progression a → a 1 * a 10 = -728) ∧
  (is_geometric_progression a → a 1 + a 10 = -7) := by
  sorry

end sequence_properties_l167_16778


namespace birthday_celebration_friends_l167_16762

/-- The number of friends attending Paolo and Sevilla's birthday celebration -/
def num_friends : ℕ := sorry

/-- The total bill amount -/
def total_bill : ℕ := sorry

theorem birthday_celebration_friends :
  (total_bill = 12 * (num_friends + 2)) ∧
  (total_bill = 16 * num_friends) →
  num_friends = 6 := by sorry

end birthday_celebration_friends_l167_16762


namespace line_through_point_with_equal_intercepts_l167_16728

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Function to check if a point is on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts
def hasEqualIntercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = l.c / l.b

-- Theorem statement
theorem line_through_point_with_equal_intercepts :
  ∃ (l : Line2D), pointOnLine ⟨1, 1⟩ l ∧ hasEqualIntercepts l ∧
    ((l.a = 1 ∧ l.b = 1 ∧ l.c = -2) ∨ (l.a = -1 ∧ l.b = 1 ∧ l.c = 0)) :=
sorry

end line_through_point_with_equal_intercepts_l167_16728


namespace right_triangle_345_l167_16751

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_345 :
  is_right_triangle 3 4 5 ∧
  ¬is_right_triangle 1 2 3 ∧
  ¬is_right_triangle 2 3 4 ∧
  ¬is_right_triangle 4 5 6 :=
by sorry

end right_triangle_345_l167_16751


namespace trigonometric_equation_solution_l167_16743

open Real

theorem trigonometric_equation_solution :
  ∀ x : ℝ, 2 * sin (2 * x) - cos (π / 2 + 3 * x) - cos (3 * x) * arccos (5 * x) * cos (π / 2 - 5 * x) = 0 ↔
  (∃ k : ℤ, x = k * π) ∨ (∃ n : ℤ, x = π / 15 + 2 * n * π / 5) ∨ (∃ n : ℤ, x = -π / 15 + 2 * n * π / 5) :=
by sorry

end trigonometric_equation_solution_l167_16743


namespace tom_bikes_11860_miles_l167_16706

/-- The number of miles Tom bikes in a year -/
def total_miles : ℕ :=
  let miles_per_day_first_period : ℕ := 30
  let days_first_period : ℕ := 183
  let miles_per_day_second_period : ℕ := 35
  let days_in_year : ℕ := 365
  let days_second_period : ℕ := days_in_year - days_first_period
  miles_per_day_first_period * days_first_period + miles_per_day_second_period * days_second_period

/-- Theorem stating that Tom bikes 11860 miles in a year -/
theorem tom_bikes_11860_miles : total_miles = 11860 := by
  sorry

end tom_bikes_11860_miles_l167_16706


namespace class_size_proof_l167_16774

theorem class_size_proof (original_average : ℝ) (new_students : ℕ) (new_students_average : ℝ) (average_decrease : ℝ) :
  original_average = 40 →
  new_students = 8 →
  new_students_average = 32 →
  average_decrease = 4 →
  ∃ (original_size : ℕ), 
    (original_size * original_average + new_students * new_students_average) / (original_size + new_students) = original_average - average_decrease ∧
    original_size = 8 :=
by sorry

end class_size_proof_l167_16774


namespace odd_sum_probability_l167_16733

theorem odd_sum_probability (n : Nat) (h : n = 16) :
  let grid_size := 4
  let total_arrangements := n.factorial
  let valid_arrangements := (grid_size.choose 2) * (n / 2).factorial * (n / 2).factorial
  (valid_arrangements : ℚ) / total_arrangements = 1 / 2150 := by
  sorry

end odd_sum_probability_l167_16733


namespace fraction_meaningfulness_l167_16708

theorem fraction_meaningfulness (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x + 2)) ↔ x ≠ -2 := by
  sorry

end fraction_meaningfulness_l167_16708


namespace inequality_holds_iff_l167_16754

theorem inequality_holds_iff (m : ℝ) :
  (∀ x : ℝ, (x^2 - m*x - 2) / (x^2 - 3*x + 4) > -1) ↔ -7 < m ∧ m < 1 := by
  sorry

end inequality_holds_iff_l167_16754


namespace zero_in_interval_l167_16720

def f (x : ℝ) := x^3 + 3*x - 1

theorem zero_in_interval :
  (f 0 < 0) →
  (f 0.5 > 0) →
  (f 0.25 < 0) →
  ∃ x : ℝ, x ∈ Set.Ioo 0.25 0.5 ∧ f x = 0 :=
by
  sorry


end zero_in_interval_l167_16720


namespace element_in_set_l167_16750

def U : Set ℕ := {1, 2, 3, 4, 5}

theorem element_in_set (M : Set ℕ) (h : Set.compl M = {1, 3}) : 2 ∈ M := by
  sorry

end element_in_set_l167_16750


namespace blueberry_muffin_percentage_l167_16752

/-- The number of cartons of blueberries Mason has -/
def num_cartons : ℕ := 8

/-- The number of blueberries in each carton -/
def blueberries_per_carton : ℕ := 300

/-- The number of blueberries used per muffin -/
def blueberries_per_muffin : ℕ := 18

/-- The number of blueberries left after making blueberry muffins -/
def blueberries_left : ℕ := 54

/-- The number of cinnamon muffins made -/
def cinnamon_muffins : ℕ := 80

/-- The number of chocolate muffins made -/
def chocolate_muffins : ℕ := 40

/-- The number of cranberry muffins made -/
def cranberry_muffins : ℕ := 50

/-- The number of lemon muffins made -/
def lemon_muffins : ℕ := 30

/-- Theorem stating that the percentage of blueberry muffins is approximately 39.39% -/
theorem blueberry_muffin_percentage :
  let total_blueberries := num_cartons * blueberries_per_carton
  let used_blueberries := total_blueberries - blueberries_left
  let blueberry_muffins := used_blueberries / blueberries_per_muffin
  let total_muffins := blueberry_muffins + cinnamon_muffins + chocolate_muffins + cranberry_muffins + lemon_muffins
  let percentage := (blueberry_muffins : ℚ) / (total_muffins : ℚ) * 100
  abs (percentage - 39.39) < 0.01 :=
by sorry

end blueberry_muffin_percentage_l167_16752


namespace number_problem_l167_16766

theorem number_problem : ∃ x : ℝ, (x / 3) * 12 = 9 ∧ x = 2.25 := by
  sorry

end number_problem_l167_16766


namespace mary_bought_24_cards_l167_16770

/-- The number of baseball cards Mary bought -/
def cards_bought (initial_cards promised_cards remaining_cards : ℝ) : ℝ :=
  remaining_cards - (initial_cards - promised_cards)

/-- Theorem: Mary bought 24.0 baseball cards -/
theorem mary_bought_24_cards :
  cards_bought 18.0 26.0 32.0 = 24.0 := by
  sorry

end mary_bought_24_cards_l167_16770


namespace monotonicity_a_eq_zero_monotonicity_a_pos_monotonicity_a_neg_l167_16741

noncomputable section

-- Define the function f(x) = x^2 * e^(ax)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 * Real.exp (a * x)

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := (2 * x + a * x^2) * Real.exp (a * x)

-- Theorem for monotonicity when a = 0
theorem monotonicity_a_eq_zero :
  ∀ x : ℝ, x < 0 → (∀ y : ℝ, y < x → f 0 y > f 0 x) ∧
            x > 0 → (∀ y : ℝ, y > x → f 0 y > f 0 x) :=
sorry

-- Theorem for monotonicity when a > 0
theorem monotonicity_a_pos :
  ∀ a : ℝ, a > 0 → 
  ∀ x : ℝ, (x < -2/a → (∀ y : ℝ, y < x → f a y < f a x)) ∧
           (x > 0 → (∀ y : ℝ, y > x → f a y > f a x)) ∧
           (-2/a < x ∧ x < 0 → (∀ y : ℝ, -2/a < y ∧ y < x → f a y > f a x)) :=
sorry

-- Theorem for monotonicity when a < 0
theorem monotonicity_a_neg :
  ∀ a : ℝ, a < 0 → 
  ∀ x : ℝ, (x < 0 → (∀ y : ℝ, y < x → f a y > f a x)) ∧
           (x > -2/a → (∀ y : ℝ, y > x → f a y < f a x)) ∧
           (0 < x ∧ x < -2/a → (∀ y : ℝ, x < y ∧ y < -2/a → f a y > f a x)) :=
sorry

end

end monotonicity_a_eq_zero_monotonicity_a_pos_monotonicity_a_neg_l167_16741


namespace diagonal_division_l167_16769

/-- A regular polygon with 2018 vertices, labeled clockwise from 1 to 2018 -/
structure RegularPolygon2018 where
  vertices : Fin 2018

/-- The number of vertices between two given vertices in a clockwise direction -/
def verticesBetween (a b : Fin 2018) : ℕ :=
  if b.val ≥ a.val then
    b.val - a.val + 1
  else
    (2018 - a.val) + b.val + 1

/-- The result of drawing diagonals in the polygon -/
def diagonalResult (p : RegularPolygon2018) : Prop :=
  let polygon1 := verticesBetween 18 1018
  let polygon2 := verticesBetween 1018 2000
  let polygon3 := verticesBetween 2000 18 + 1  -- Adding 1 for vertex 1018
  polygon1 = 1001 ∧ polygon2 = 983 ∧ polygon3 = 38

theorem diagonal_division (p : RegularPolygon2018) : diagonalResult p := by
  sorry

end diagonal_division_l167_16769


namespace max_min_f_on_interval_l167_16731

noncomputable def f (x : ℝ) : ℝ := x - Real.sqrt 2 * Real.sin x

theorem max_min_f_on_interval :
  let a := 0
  let b := Real.pi
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc a b ∧ x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    f x_max = Real.pi ∧
    f x_min = Real.pi / 4 - 1 :=
sorry

end max_min_f_on_interval_l167_16731


namespace monochromatic_solution_exists_l167_16794

def Color := Bool

def NumberSet : Set Nat := {1, 2, 3, 4, 5}

def Coloring := Nat → Color

theorem monochromatic_solution_exists (c : Coloring) : 
  ∃ (x y z : Nat), x ∈ NumberSet ∧ y ∈ NumberSet ∧ z ∈ NumberSet ∧ 
  x + y = z ∧ c x = c y ∧ c y = c z :=
sorry

end monochromatic_solution_exists_l167_16794


namespace jenny_house_improvements_l167_16782

/-- Represents the problem of calculating the maximum value of improvements Jenny can make to her house. -/
theorem jenny_house_improvements
  (tax_rate : ℝ)
  (initial_house_value : ℝ)
  (rail_project_increase : ℝ)
  (max_affordable_tax : ℝ)
  (h1 : tax_rate = 0.02)
  (h2 : initial_house_value = 400000)
  (h3 : rail_project_increase = 0.25)
  (h4 : max_affordable_tax = 15000) :
  let new_house_value := initial_house_value * (1 + rail_project_increase)
  let max_house_value := max_affordable_tax / tax_rate
  max_house_value - new_house_value = 250000 :=
by sorry

end jenny_house_improvements_l167_16782


namespace inequality_proof_l167_16749

theorem inequality_proof (x : ℝ) : 
  -2 < (x^2 - 10*x + 21) / (x^2 - 6*x + 10) ∧ 
  (x^2 - 10*x + 21) / (x^2 - 6*x + 10) < 3 ↔ 
  3/2 < x ∧ x < 3 :=
by sorry

end inequality_proof_l167_16749


namespace inverse_of_A_cubed_l167_16784

theorem inverse_of_A_cubed (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A⁻¹ = !![1, 4; -2, -7] →
  (A^3)⁻¹ = !![41, 144; -72, -247] := by
sorry

end inverse_of_A_cubed_l167_16784


namespace sequence_2017th_term_l167_16775

theorem sequence_2017th_term (a : ℕ+ → ℚ) 
  (h1 : a 1 = 0)
  (h2 : ∀ n : ℕ+, n ≥ 2 → (1 / (1 - a n) - 1 / (1 - a (n - 1)) = 1)) :
  a 2017 = 2016 / 2017 := by
  sorry

end sequence_2017th_term_l167_16775


namespace specific_rental_cost_l167_16737

/-- Calculates the total cost of a car rental given the daily rate, per-mile rate, number of days, and miles driven. -/
def carRentalCost (dailyRate perMileRate : ℚ) (days miles : ℕ) : ℚ :=
  dailyRate * days + perMileRate * miles

/-- Theorem stating that for the given rental conditions, the total cost is $162.5 -/
theorem specific_rental_cost :
  carRentalCost 25 0.25 3 350 = 162.5 := by
  sorry

end specific_rental_cost_l167_16737


namespace polaroid_photo_length_l167_16753

/-- The circumference of a rectangle given its length and width -/
def rectangleCircumference (length width : ℝ) : ℝ :=
  2 * (length + width)

/-- Theorem: A rectangular Polaroid photo with circumference 40 cm and width 8 cm has a length of 12 cm -/
theorem polaroid_photo_length (circumference width : ℝ) 
    (h_circumference : circumference = 40)
    (h_width : width = 8)
    (h_rect : rectangleCircumference length width = circumference) :
    length = 12 := by
  sorry


end polaroid_photo_length_l167_16753


namespace reasoning_is_analogical_l167_16761

/-- A type representing different reasoning methods -/
inductive ReasoningMethod
  | Inductive
  | Analogical
  | Deductive
  | None

/-- A circle with radius R -/
structure Circle (R : ℝ) where
  radius : R > 0

/-- A rectangle inscribed in a circle -/
structure InscribedRectangle (R : ℝ) extends Circle R where
  width : ℝ
  height : ℝ
  inscribed : width^2 + height^2 ≤ 4 * R^2

/-- A sphere with radius R -/
structure Sphere (R : ℝ) where
  radius : R > 0

/-- A rectangular solid inscribed in a sphere -/
structure InscribedRectangularSolid (R : ℝ) extends Sphere R where
  length : ℝ
  width : ℝ
  height : ℝ
  inscribed : length^2 + width^2 + height^2 ≤ 4 * R^2

/-- Theorem about maximum area rectangle in a circle -/
axiom max_area_square_in_circle (R : ℝ) :
  ∀ (rect : InscribedRectangle R), rect.width * rect.height ≤ 2 * R^2

/-- The reasoning method used to deduce the theorem about cubes in spheres -/
def reasoning_method : ReasoningMethod := by sorry

/-- The main theorem stating that the reasoning method is analogical -/
theorem reasoning_is_analogical :
  reasoning_method = ReasoningMethod.Analogical := by sorry

end reasoning_is_analogical_l167_16761


namespace power_of_x_in_product_l167_16744

theorem power_of_x_in_product (x y z : ℕ) (hx : Prime x) (hy : Prime y) (hz : Prime z) 
  (hdiff : x ≠ y ∧ y ≠ z ∧ x ≠ z) :
  ∃ (a b c : ℕ), (a + 1) * (b + 1) * (c + 1) = 12 ∧ a = 1 := by
  sorry

end power_of_x_in_product_l167_16744


namespace pole_top_distance_difference_l167_16703

theorem pole_top_distance_difference 
  (h₁ : ℝ) (h₂ : ℝ) (d : ℝ)
  (height_pole1 : h₁ = 6)
  (height_pole2 : h₂ = 11)
  (distance_between_feet : d = 12) :
  Real.sqrt ((h₂ - h₁)^2 + d^2) - d = Real.sqrt 13 :=
by sorry

end pole_top_distance_difference_l167_16703


namespace eggs_taken_l167_16773

theorem eggs_taken (initial_eggs : ℕ) (remaining_eggs : ℕ) (h1 : initial_eggs = 47) (h2 : remaining_eggs = 42) :
  initial_eggs - remaining_eggs = 5 := by
  sorry

end eggs_taken_l167_16773


namespace collinear_vectors_x_value_l167_16707

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, -4)
  collinear a b → x = -2 :=
by
  sorry

end collinear_vectors_x_value_l167_16707


namespace coupon_value_l167_16700

/-- Calculates the value of each coupon given the original price, discount percentage, number of bottles, and total cost after discounts and coupons. -/
theorem coupon_value (original_price discount_percent bottles total_cost : ℝ) : 
  original_price = 15 →
  discount_percent = 20 →
  bottles = 3 →
  total_cost = 30 →
  (original_price * (1 - discount_percent / 100) * bottles - total_cost) / bottles = 2 := by
  sorry

end coupon_value_l167_16700


namespace max_cola_bottles_30_yuan_l167_16721

/-- Calculates the maximum number of cola bottles that can be consumed given an initial amount of money, the cost per bottle, and the exchange rate of empty bottles for full bottles. -/
def max_cola_bottles (initial_money : ℕ) (cost_per_bottle : ℕ) (exchange_rate : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given 30 yuan, a cola cost of 2 yuan per bottle, and the ability to exchange 2 empty bottles for 1 full bottle, the maximum number of cola bottles that can be consumed is 29. -/
theorem max_cola_bottles_30_yuan :
  max_cola_bottles 30 2 2 = 29 :=
sorry

end max_cola_bottles_30_yuan_l167_16721


namespace total_hamburger_combinations_l167_16786

/-- The number of condiments available. -/
def num_condiments : ℕ := 10

/-- The number of choices for each condiment (include or not include). -/
def choices_per_condiment : ℕ := 2

/-- The number of choices for meat patties. -/
def meat_patty_choices : ℕ := 3

/-- The number of choices for bun types. -/
def bun_choices : ℕ := 3

/-- Theorem stating the total number of different hamburger combinations. -/
theorem total_hamburger_combinations :
  (choices_per_condiment ^ num_condiments) * meat_patty_choices * bun_choices = 9216 := by
  sorry


end total_hamburger_combinations_l167_16786


namespace subtraction_multiplication_addition_l167_16718

theorem subtraction_multiplication_addition (x : ℤ) : 
  423 - x = 421 → (x * 423) + 421 = 1267 := by
  sorry

end subtraction_multiplication_addition_l167_16718


namespace units_digit_sum_factorials_500_l167_16787

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_500 :
  units_digit (sum_factorials 500) = 3 := by
  sorry

end units_digit_sum_factorials_500_l167_16787


namespace eat_cereal_together_l167_16722

/-- The time taken for two people to eat a certain amount of cereal together -/
def time_to_eat_together (rate1 rate2 amount : ℚ) : ℚ :=
  amount / (rate1 + rate2)

/-- Theorem: Mr. Fat and Mr. Thin will take 96 minutes to eat 4 pounds of cereal together -/
theorem eat_cereal_together : 
  let fat_rate : ℚ := 1 / 40
  let thin_rate : ℚ := 1 / 15
  let amount : ℚ := 4
  time_to_eat_together fat_rate thin_rate amount = 96 := by
  sorry

#eval time_to_eat_together (1 / 40) (1 / 15) 4

end eat_cereal_together_l167_16722


namespace centroid_trace_area_centroid_trace_area_diameter_30_l167_16771

/-- The area of the region bounded by the curve traced by the centroid of a triangle
    inscribed in a circle, where the base of the triangle is a diameter of the circle. -/
theorem centroid_trace_area (r : ℝ) (h : r > 0) : 
  (π * (r / 3)^2) = (25 * π / 9) * r^2 := by
  sorry

/-- The specific case where the diameter of the circle is 30 -/
theorem centroid_trace_area_diameter_30 : 
  (π * 5^2) = 25 * π := by
  sorry

end centroid_trace_area_centroid_trace_area_diameter_30_l167_16771


namespace number_of_digits_in_N_l167_16765

theorem number_of_digits_in_N : ∃ (N : ℕ), 
  N = 2^12 * 5^8 ∧ (Nat.log 10 N + 1 = 10) := by sorry

end number_of_digits_in_N_l167_16765


namespace motorcycle_meeting_distance_l167_16710

/-- The distance traveled by a constant speed motorcyclist when meeting an accelerating motorcyclist on a circular track -/
theorem motorcycle_meeting_distance (v : ℝ) (a : ℝ) : 
  v > 0 → a > 0 →
  v * (1 / v) = 1 →
  (1/2) * a * (1 / v)^2 = 1 →
  ∃ (T : ℝ), T > 0 ∧ v * T + (1/2) * a * T^2 = 1 →
  v * T = (-1 + Real.sqrt 5) / 2 := by
sorry

end motorcycle_meeting_distance_l167_16710


namespace coin_division_problem_l167_16798

theorem coin_division_problem : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n % 8 = 6) ∧ 
  (n % 7 = 5) ∧ 
  (n % 9 = 0) ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 8 = 6 ∧ m % 7 = 5 → m ≥ n) :=
by sorry

end coin_division_problem_l167_16798


namespace total_logs_cut_l167_16713

/-- The number of logs produced by cutting different types of trees -/
theorem total_logs_cut (pine_logs maple_logs walnut_logs oak_logs birch_logs : ℕ)
  (pine_trees maple_trees walnut_trees oak_trees birch_trees : ℕ)
  (h1 : pine_logs = 80)
  (h2 : maple_logs = 60)
  (h3 : walnut_logs = 100)
  (h4 : oak_logs = 90)
  (h5 : birch_logs = 55)
  (h6 : pine_trees = 8)
  (h7 : maple_trees = 3)
  (h8 : walnut_trees = 4)
  (h9 : oak_trees = 7)
  (h10 : birch_trees = 5) :
  pine_logs * pine_trees + maple_logs * maple_trees + walnut_logs * walnut_trees +
  oak_logs * oak_trees + birch_logs * birch_trees = 2125 := by
  sorry

end total_logs_cut_l167_16713


namespace unit_price_sum_l167_16779

theorem unit_price_sum (x y z : ℝ) 
  (eq1 : 3 * x + 7 * y + z = 24)
  (eq2 : 4 * x + 10 * y + z = 33) : 
  x + y + z = 6 := by
sorry

end unit_price_sum_l167_16779


namespace range_of_a_l167_16758

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 > 0) → a ∈ Set.Ioo (-1) 3 := by
  sorry

end range_of_a_l167_16758


namespace celine_collected_ten_erasers_l167_16756

/-- The number of erasers collected by each person -/
structure EraserCollection where
  gabriel : ℕ
  celine : ℕ
  julian : ℕ
  erica : ℕ
  david : ℕ

/-- The conditions of the eraser collection problem -/
def valid_collection (ec : EraserCollection) : Prop :=
  ec.celine = 2 * ec.gabriel ∧
  ec.julian = 2 * ec.celine ∧
  ec.erica = 3 * ec.julian ∧
  ec.david = 5 * ec.erica ∧
  ec.gabriel ≥ 1 ∧ ec.celine ≥ 1 ∧ ec.julian ≥ 1 ∧ ec.erica ≥ 1 ∧ ec.david ≥ 1 ∧
  ec.gabriel + ec.celine + ec.julian + ec.erica + ec.david = 380

/-- The theorem stating that Celine collected 10 erasers -/
theorem celine_collected_ten_erasers (ec : EraserCollection) 
  (h : valid_collection ec) : ec.celine = 10 := by
  sorry

end celine_collected_ten_erasers_l167_16756


namespace amoeba_count_after_week_l167_16709

/-- The number of amoebas after n days, given an initial population of 1 and a tripling rate each day -/
def amoeba_count (n : ℕ) : ℕ := 3^n

/-- Theorem: After 7 days, the number of amoebas is 2187 -/
theorem amoeba_count_after_week : amoeba_count 7 = 2187 := by
  sorry

end amoeba_count_after_week_l167_16709


namespace hot_dogs_per_pack_l167_16715

theorem hot_dogs_per_pack (total_hot_dogs : ℕ) (buns_per_pack : ℕ) (hot_dogs_per_pack : ℕ) : 
  total_hot_dogs = 36 →
  buns_per_pack = 9 →
  total_hot_dogs % buns_per_pack = 0 →
  total_hot_dogs % hot_dogs_per_pack = 0 →
  total_hot_dogs / buns_per_pack = total_hot_dogs / hot_dogs_per_pack →
  hot_dogs_per_pack = 9 := by
  sorry

end hot_dogs_per_pack_l167_16715


namespace count_possible_denominators_l167_16788

def is_valid_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def repeating_decimal_to_fraction (c d : ℕ) : ℚ :=
  (10 * c + d : ℚ) / 99

theorem count_possible_denominators :
  ∃ (S : Finset ℕ),
    (∀ c d : ℕ, is_valid_digit c → is_valid_digit d →
      (c ≠ 8 ∨ d ≠ 8) → (c ≠ 0 ∨ d ≠ 0) →
      (repeating_decimal_to_fraction c d).den ∈ S) ∧
    S.card = 5 :=
sorry

end count_possible_denominators_l167_16788


namespace cookie_sharing_proof_l167_16755

/-- The number of people sharing cookies baked by Beth --/
def number_of_people : ℕ :=
  let batches : ℕ := 4
  let dozens_per_batch : ℕ := 2
  let cookies_per_dozen : ℕ := 12
  let cookies_per_person : ℕ := 6
  let total_cookies : ℕ := batches * dozens_per_batch * cookies_per_dozen
  total_cookies / cookies_per_person

/-- Proof that the number of people sharing the cookies is 16 --/
theorem cookie_sharing_proof : number_of_people = 16 := by
  sorry

end cookie_sharing_proof_l167_16755


namespace share_calculation_l167_16785

theorem share_calculation (total : ℝ) (a b c : ℝ) : 
  total = 300 →
  a + b + c = total →
  a = (2/3) * (b + c) →
  b = (6/9) * (a + c) →
  a = 120 := by sorry

end share_calculation_l167_16785


namespace ninth_root_of_unity_l167_16799

theorem ninth_root_of_unity (y : ℂ) : 
  y = Complex.exp (2 * Real.pi * I / 9) → y^9 = 1 := by
  sorry

end ninth_root_of_unity_l167_16799


namespace a_range_when_A_B_disjoint_l167_16739

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | |x - 1| ≤ a ∧ a > 0}
def B : Set ℝ := {x : ℝ | x^2 - 6*x - 7 > 0}

-- State the theorem
theorem a_range_when_A_B_disjoint :
  ∀ a : ℝ, (A a ∩ B = ∅) ↔ (0 < a ∧ a ≤ 6) :=
by sorry

end a_range_when_A_B_disjoint_l167_16739


namespace angle_of_inclination_cosine_l167_16716

theorem angle_of_inclination_cosine (θ : Real) :
  (∃ (m : Real), m = 2 ∧ θ = Real.arctan m) →
  Real.cos θ = Real.sqrt 5 / 5 := by
  sorry

end angle_of_inclination_cosine_l167_16716


namespace gcd_765432_654321_l167_16705

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l167_16705


namespace goldfish_to_pretzel_ratio_l167_16704

/-- Given the number of pretzels, suckers, kids, and items per baggie, 
    prove that the ratio of goldfish to pretzels is 4:1 -/
theorem goldfish_to_pretzel_ratio 
  (pretzels : ℕ) 
  (suckers : ℕ) 
  (kids : ℕ) 
  (items_per_baggie : ℕ) 
  (h1 : pretzels = 64) 
  (h2 : suckers = 32) 
  (h3 : kids = 16) 
  (h4 : items_per_baggie = 22) : 
  (kids * items_per_baggie - pretzels - suckers) / pretzels = 4 := by
  sorry

end goldfish_to_pretzel_ratio_l167_16704


namespace num_outfits_l167_16764

/-- Number of shirts available -/
def num_shirts : ℕ := 8

/-- Number of ties available -/
def num_ties : ℕ := 5

/-- Number of pairs of pants available -/
def num_pants : ℕ := 4

/-- Number of jackets available -/
def num_jackets : ℕ := 2

/-- Number of tie options (including no tie) -/
def tie_options : ℕ := num_ties + 1

/-- Number of jacket options (including no jacket) -/
def jacket_options : ℕ := num_jackets + 1

/-- Theorem stating the number of distinct outfits -/
theorem num_outfits : num_shirts * num_pants * tie_options * jacket_options = 576 := by
  sorry

end num_outfits_l167_16764


namespace a_all_positive_l167_16725

/-- Sequence a_n defined recursively -/
def a (α : ℝ) : ℕ → ℝ
  | 0 => α
  | n + 1 => 2 * a α n - n^2

/-- Theorem stating the condition for all terms of a_n to be positive -/
theorem a_all_positive (α : ℝ) : (∀ n : ℕ, a α n > 0) ↔ α ≥ 3 := by
  sorry

end a_all_positive_l167_16725


namespace banana_distribution_exists_l167_16734

-- Define the number of bananas and boxes
def total_bananas : ℕ := 40
def num_boxes : ℕ := 8

-- Define a valid distribution
def is_valid_distribution (dist : List ℕ) : Prop :=
  dist.length = num_boxes ∧
  dist.sum = total_bananas ∧
  dist.Nodup

-- Theorem statement
theorem banana_distribution_exists : 
  ∃ (dist : List ℕ), is_valid_distribution dist :=
sorry

end banana_distribution_exists_l167_16734


namespace condition_relationship_l167_16781

theorem condition_relationship (x : ℝ) :
  (∀ x, -1 < x ∧ x < 3 → x < 3) ∧ 
  ¬(∀ x, x < 3 → -1 < x ∧ x < 3) :=
by sorry

end condition_relationship_l167_16781


namespace exists_polygon_with_area_16_l167_16747

/-- A polygon represented by a list of points in 2D space -/
def Polygon := List (Real × Real)

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (p : Polygon) : Real := sorry

/-- Check if a polygon can be formed from given line segments -/
def canFormPolygon (segments : List Real) (p : Polygon) : Prop := sorry

/-- The main theorem stating that a polygon with area 16 can be formed from 12 segments of length 2 -/
theorem exists_polygon_with_area_16 :
  ∃ (p : Polygon), 
    polygonArea p = 16 ∧ 
    canFormPolygon (List.replicate 12 2) p :=
sorry

end exists_polygon_with_area_16_l167_16747


namespace grain_milling_problem_l167_16795

theorem grain_milling_problem (grain_weight : ℚ) : 
  (grain_weight * (1 - 1/10) = 100) → grain_weight = 1000/9 := by
  sorry

end grain_milling_problem_l167_16795


namespace sequence_equation_proof_l167_16719

/-- Given a sequence of equations, prove the value of (b+1)/a^2 -/
theorem sequence_equation_proof (a b : ℕ) (h : ∀ (n : ℕ), 32 ≤ n → n ≤ 32016 → 
  ∃ (m : ℕ), n + m / n = (n - 32 + 3) * (3 + m / n)) 
  (h_last : 32016 + a / b = 2016 * 3 * (a / b)) : 
  (b + 1) / (a^2 : ℚ) = 2016 := by
  sorry

end sequence_equation_proof_l167_16719


namespace scale_division_theorem_l167_16760

/-- Represents the length of a scale in inches -/
def scale_length : ℕ := 6 * 12 + 8

/-- Represents the number of parts the scale is divided into -/
def num_parts : ℕ := 4

/-- Represents the length of each part in inches -/
def part_length : ℕ := scale_length / num_parts

/-- Proves that each part of the scale is 20 inches (1 foot 8 inches) long -/
theorem scale_division_theorem : part_length = 20 := by
  sorry

end scale_division_theorem_l167_16760


namespace triangle_property_l167_16732

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given conditions
  2 * b * Real.cos B = c * Real.cos A + a * Real.cos C →
  b = 2 →
  a + c = 4 →
  -- Conclusions
  Real.sin B = Real.sqrt 3 / 2 ∧ 
  (1/2 : ℝ) * a * c * Real.sin B = Real.sqrt 3 := by
sorry

end triangle_property_l167_16732


namespace apple_eating_contest_l167_16791

def classroom (n : ℕ) (total_apples : ℕ) (aaron_apples : ℕ) (zeb_apples : ℕ) : Prop :=
  n = 8 ∧
  total_apples > 20 ∧
  ∀ student, student ≠ aaron_apples → aaron_apples ≥ student ∧
  ∀ student, student ≠ zeb_apples → student ≥ zeb_apples

theorem apple_eating_contest (n : ℕ) (total_apples : ℕ) (aaron_apples : ℕ) (zeb_apples : ℕ) 
  (h : classroom n total_apples aaron_apples zeb_apples) : 
  aaron_apples - zeb_apples = 6 := by
  sorry

end apple_eating_contest_l167_16791


namespace expand_expression_l167_16736

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end expand_expression_l167_16736


namespace divisibility_implies_equality_l167_16726

/-- For natural numbers a, b, and n, if a - k^n is divisible by b - k for all natural k ≠ b, 
    then a = b^n -/
theorem divisibility_implies_equality (a b n : ℕ) 
  (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) : 
  a = b^n := by
sorry

end divisibility_implies_equality_l167_16726


namespace weak_to_strong_ratio_l167_16789

/-- Represents the amount of coffee used for different strengths --/
structure CoffeeUsage where
  weak_per_cup : ℕ
  strong_per_cup : ℕ
  cups_each : ℕ
  total_tablespoons : ℕ

/-- Theorem stating the ratio of weak to strong coffee usage --/
theorem weak_to_strong_ratio (c : CoffeeUsage) 
  (h1 : c.weak_per_cup = 1)
  (h2 : c.strong_per_cup = 2)
  (h3 : c.cups_each = 12)
  (h4 : c.total_tablespoons = 36) :
  (c.weak_per_cup * c.cups_each) / (c.strong_per_cup * c.cups_each) = 1 / 2 := by
  sorry

#check weak_to_strong_ratio

end weak_to_strong_ratio_l167_16789


namespace dennis_floor_number_l167_16701

theorem dennis_floor_number :
  ∀ (frank_floor charlie_floor bob_floor dennis_floor : ℕ),
    frank_floor = 16 →
    charlie_floor = frank_floor / 4 →
    charlie_floor = bob_floor + 1 →
    dennis_floor = charlie_floor + 2 →
    dennis_floor = 6 := by
  sorry

end dennis_floor_number_l167_16701


namespace counterexample_exists_l167_16730

theorem counterexample_exists : ∃ (a b c : ℝ), 
  (a^2 + b^2) / (b^2 + c^2) = a / c ∧ a / b ≠ b / c := by
  sorry

end counterexample_exists_l167_16730


namespace quadratic_one_root_l167_16768

theorem quadratic_one_root (m : ℝ) : 
  (∃! x : ℝ, x^2 + 6*m*x + m = 0) ↔ m = 1/9 ∧ m > 0 :=
sorry

end quadratic_one_root_l167_16768


namespace set_operations_and_conditions_l167_16783

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 8}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Define the theorem
theorem set_operations_and_conditions :
  -- Part 1
  (A 0 ∩ B = {x | 5 < x ∧ x ≤ 8}) ∧
  (A 0 ∪ Bᶜ = {x | -1 ≤ x ∧ x ≤ 8}) ∧
  -- Part 2
  (∀ a : ℝ, A a ∪ B = B ↔ a < -9 ∨ a > 5) :=
by sorry

end set_operations_and_conditions_l167_16783


namespace driver_distance_theorem_l167_16777

/-- Calculates the total distance traveled by a driver given their speed and driving durations. -/
def total_distance_traveled (speed : ℝ) (first_duration second_duration : ℝ) : ℝ :=
  speed * (first_duration + second_duration)

/-- Theorem stating that a driver traveling at 60 mph for 4 hours and 9 hours will cover 780 miles. -/
theorem driver_distance_theorem :
  let speed := 60
  let first_duration := 4
  let second_duration := 9
  total_distance_traveled speed first_duration second_duration = 780 := by
  sorry

#check driver_distance_theorem

end driver_distance_theorem_l167_16777


namespace sum_of_fractions_l167_16714

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end sum_of_fractions_l167_16714


namespace max_value_sqrt_sum_l167_16796

/-- Given 2x + 3y + 5z = 29, the maximum value of √(2x+1) + √(3y+4) + √(5z+6) is 2√30 -/
theorem max_value_sqrt_sum (x y z : ℝ) (h : 2*x + 3*y + 5*z = 29) :
  (∀ a b c : ℝ, 2*a + 3*b + 5*c = 29 →
    Real.sqrt (2*a + 1) + Real.sqrt (3*b + 4) + Real.sqrt (5*c + 6) ≤
    Real.sqrt (2*x + 1) + Real.sqrt (3*y + 4) + Real.sqrt (5*z + 6)) →
  Real.sqrt (2*x + 1) + Real.sqrt (3*y + 4) + Real.sqrt (5*z + 6) = 2 * Real.sqrt 30 :=
by sorry

end max_value_sqrt_sum_l167_16796


namespace ellipse_axis_ratio_l167_16748

theorem ellipse_axis_ratio (k : ℝ) : 
  (∀ x y : ℝ, x^2 + k*y^2 = 1 → ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x^2/a^2 + y^2/b^2 = 1) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x^2/a^2 + y^2/b^2 = 1 ∧ a^2 = 1/k ∧ b^2 = 1) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*a = 2*b) →
  k = 1/4 :=
sorry

end ellipse_axis_ratio_l167_16748


namespace symmetric_line_ratio_l167_16767

-- Define the triangle ABC and points M, N
structure Triangle :=
  (A B C M N : ℝ × ℝ)

-- Define the property of AM and AN being symmetric with respect to angle bisector of A
def isSymmetric (t : Triangle) : Prop :=
  -- This is a placeholder for the actual geometric condition
  sorry

-- Define the lengths of sides and segments
def length (p q : ℝ × ℝ) : ℝ :=
  sorry

-- State the theorem
theorem symmetric_line_ratio (t : Triangle) :
  isSymmetric t →
  (length t.B t.M * length t.B t.N) / (length t.C t.M * length t.C t.N) =
  (length t.A t.C)^2 / (length t.A t.B)^2 :=
by
  sorry

end symmetric_line_ratio_l167_16767


namespace no_valid_arrangement_with_odd_sums_l167_16702

def Grid := Matrix (Fin 4) (Fin 4) Nat

def validArrangement (g : Grid) : Prop :=
  (∀ i j, g i j ∈ Finset.range 17) ∧
  (∀ i j k l, i ≠ k ∨ j ≠ l → g i j ≠ g k l)

def rowSum (g : Grid) (i : Fin 4) : Nat :=
  (Finset.range 4).sum (λ j => g i j)

def colSum (g : Grid) (j : Fin 4) : Nat :=
  (Finset.range 4).sum (λ i => g i j)

def mainDiagSum (g : Grid) : Nat :=
  (Finset.range 4).sum (λ i => g i i)

def antiDiagSum (g : Grid) : Nat :=
  (Finset.range 4).sum (λ i => g i (3 - i))

def allSumsOdd (g : Grid) : Prop :=
  (∀ i, Odd (rowSum g i)) ∧
  (∀ j, Odd (colSum g j)) ∧
  Odd (mainDiagSum g) ∧
  Odd (antiDiagSum g)

theorem no_valid_arrangement_with_odd_sums :
  ¬∃ g : Grid, validArrangement g ∧ allSumsOdd g :=
sorry

end no_valid_arrangement_with_odd_sums_l167_16702


namespace least_integer_in_ratio_l167_16797

theorem least_integer_in_ratio (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  b = (3 * a) / 2 →
  c = (5 * a) / 2 →
  a + b + c = 60 →
  a = 12 :=
sorry

end least_integer_in_ratio_l167_16797


namespace trigonometric_equation_solution_l167_16727

theorem trigonometric_equation_solution :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) /
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
  x = 110 := by
sorry

end trigonometric_equation_solution_l167_16727


namespace crows_left_on_branch_l167_16780

/-- The number of crows remaining on a tree branch after some birds flew away -/
def remaining_crows (initial_parrots initial_total initial_crows remaining_parrots : ℕ) : ℕ :=
  initial_crows - (initial_parrots - remaining_parrots)

/-- Theorem stating the number of crows remaining on the branch -/
theorem crows_left_on_branch :
  ∀ (initial_parrots initial_total initial_crows remaining_parrots : ℕ),
    initial_parrots = 7 →
    initial_total = 13 →
    initial_crows = initial_total - initial_parrots →
    remaining_parrots = 2 →
    remaining_crows initial_parrots initial_total initial_crows remaining_parrots = 1 := by
  sorry

#eval remaining_crows 7 13 6 2

end crows_left_on_branch_l167_16780


namespace scale_division_l167_16724

/-- Proves that dividing a scale of 80 inches into 5 equal parts results in parts of 16 inches each -/
theorem scale_division (scale_length : ℕ) (num_parts : ℕ) (part_length : ℕ) :
  scale_length = 80 ∧ num_parts = 5 → part_length = scale_length / num_parts → part_length = 16 := by
  sorry

end scale_division_l167_16724


namespace cubic_root_sum_l167_16763

theorem cubic_root_sum (a b c : ℝ) : 
  (15 * a^3 - 30 * a^2 + 20 * a - 2 = 0) →
  (15 * b^3 - 30 * b^2 + 20 * b - 2 = 0) →
  (15 * c^3 - 30 * c^2 + 20 * c - 2 = 0) →
  (0 < a ∧ a < 1) →
  (0 < b ∧ b < 1) →
  (0 < c ∧ c < 1) →
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 1) :=
by sorry

end cubic_root_sum_l167_16763


namespace mrs_hilt_travel_l167_16746

/-- Calculates the total miles traveled given the number of books read and miles per book -/
def total_miles (books_read : ℕ) (miles_per_book : ℕ) : ℕ :=
  books_read * miles_per_book

/-- Proves that Mrs. Hilt traveled 6750 miles to Japan -/
theorem mrs_hilt_travel : total_miles 15 450 = 6750 := by
  sorry

end mrs_hilt_travel_l167_16746


namespace marks_additional_height_l167_16738

/-- Proves that Mark is 3 inches tall in addition to his height in feet given the conditions -/
theorem marks_additional_height :
  -- Define constants
  let feet_to_inches : ℕ := 12
  let marks_feet : ℕ := 5
  let mikes_feet : ℕ := 6
  let mikes_additional_inches : ℕ := 1
  let height_difference : ℕ := 10

  -- Calculate Mike's height in inches
  let mikes_height : ℕ := mikes_feet * feet_to_inches + mikes_additional_inches

  -- Calculate Mark's height in inches
  let marks_height : ℕ := mikes_height - height_difference

  -- Calculate Mark's additional inches
  let marks_additional_inches : ℕ := marks_height - (marks_feet * feet_to_inches)

  -- Theorem statement
  marks_additional_inches = 3 := by
  sorry

end marks_additional_height_l167_16738
