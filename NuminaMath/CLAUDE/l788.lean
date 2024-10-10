import Mathlib

namespace tangent_product_l788_78894

theorem tangent_product (A B : ℝ) (h1 : A + B = 5 * Real.pi / 4) 
  (h2 : ∀ k : ℤ, A + B ≠ k * Real.pi + Real.pi / 2) : 
  (1 + Real.tan A) * (1 + Real.tan B) = 2 := by
  sorry

end tangent_product_l788_78894


namespace coke_cost_l788_78857

def cheeseburger_cost : ℚ := 3.65
def milkshake_cost : ℚ := 2
def fries_cost : ℚ := 4
def cookie_cost : ℚ := 0.5
def tax : ℚ := 0.2
def toby_initial : ℚ := 15
def toby_change : ℚ := 7

theorem coke_cost (coke_price : ℚ) : coke_price = 1 := by
  sorry

end coke_cost_l788_78857


namespace range_of_c_l788_78835

theorem range_of_c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + b = a * b) (h2 : a + b + c = a * b * c) :
  1 < c ∧ c ≤ 4/3 := by
sorry

end range_of_c_l788_78835


namespace jake_and_sister_weight_l788_78830

theorem jake_and_sister_weight (jake_weight : ℕ) (h : jake_weight = 188) :
  ∃ (sister_weight : ℕ),
    jake_weight - 8 = 2 * sister_weight ∧
    jake_weight + sister_weight = 278 := by
  sorry

end jake_and_sister_weight_l788_78830


namespace geometric_sequence_ratio_l788_78826

/-- Given a geometric sequence with first term a₁ and common ratio q,
    if the sum of the first three terms equals 3a₁, then q = 1 or q = -2 -/
theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) :
  a₁ ≠ 0 →
  a₁ + a₁ * q + a₁ * q^2 = 3 * a₁ →
  q = 1 ∨ q = -2 := by
  sorry

end geometric_sequence_ratio_l788_78826


namespace point_M_coordinates_l788_78898

-- Define the coordinates of points M and N
def M (m : ℝ) : ℝ × ℝ := (4*m + 4, 3*m - 6)
def N : ℝ × ℝ := (-8, 12)

-- Define the condition for MN being parallel to x-axis
def parallel_to_x_axis (M N : ℝ × ℝ) : Prop := M.2 = N.2

-- Theorem statement
theorem point_M_coordinates :
  ∃ m : ℝ, parallel_to_x_axis (M m) N ∧ M m = (28, 12) := by
  sorry

end point_M_coordinates_l788_78898


namespace area_difference_l788_78836

/-- Given a square BDEF with side length 4 + 2√2, surrounded by a rectangle with sides 2s and s
    (where s is the side length of BDEF), and with a regular octagon inscribed in BDEF,
    the total area of the shape composed of the rectangle and square minus the area of
    the inscribed regular octagon is 56 + 24√2 square units. -/
theorem area_difference (s : ℝ) (h : s = 4 + 2 * Real.sqrt 2) : 
  2 * s^2 + s^2 - (2 * (1 + Real.sqrt 2) * (2 * Real.sqrt 2)^2) = 56 + 24 * Real.sqrt 2 := by
  sorry

end area_difference_l788_78836


namespace inscribed_sphere_pyramid_volume_l788_78883

/-- A pyramid with an inscribed sphere -/
structure InscribedSpherePyramid where
  /-- The volume of the pyramid -/
  volume : ℝ
  /-- The radius of the inscribed sphere -/
  radius : ℝ
  /-- The total surface area of the pyramid -/
  surface_area : ℝ
  /-- The radius is positive -/
  radius_pos : radius > 0
  /-- The surface area is positive -/
  surface_area_pos : surface_area > 0

/-- 
Theorem: The volume of a pyramid with an inscribed sphere is equal to 
one-third of the product of the radius of the sphere and the total surface area of the pyramid.
-/
theorem inscribed_sphere_pyramid_volume 
  (p : InscribedSpherePyramid) : p.volume = (1 / 3) * p.surface_area * p.radius := by
  sorry

end inscribed_sphere_pyramid_volume_l788_78883


namespace organizing_related_to_excellent_scores_expectation_X_l788_78807

-- Define the total number of students
def total_students : ℕ := 100

-- Define the number of students with excellent and poor math scores
def excellent_scores : ℕ := 40
def poor_scores : ℕ := 60

-- Define the number of students not organizing regularly
def not_organizing_excellent : ℕ := 8  -- 20% of 40
def not_organizing_poor : ℕ := 32

-- Define the chi-square statistic
def chi_square (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99% certainty
def critical_value : ℚ := 6635 / 1000

-- Theorem for the relationship between organizing regularly and excellent math scores
theorem organizing_related_to_excellent_scores :
  chi_square not_organizing_excellent (excellent_scores - not_organizing_excellent)
              not_organizing_poor (poor_scores - not_organizing_poor) > critical_value := by
  sorry

-- Define the probability distribution of X
def prob_X (x : ℕ) : ℚ :=
  match x with
  | 0 => 28 / 45
  | 1 => 16 / 45
  | 2 => 1 / 45
  | _ => 0

-- Theorem for the expectation of X
theorem expectation_X :
  (0 : ℚ) * prob_X 0 + 1 * prob_X 1 + 2 * prob_X 2 = 2 / 5 := by
  sorry

end organizing_related_to_excellent_scores_expectation_X_l788_78807


namespace tea_milk_mixture_l788_78843

/-- Represents a cup with a certain amount of liquid -/
structure Cup where
  tea : ℚ
  milk : ℚ

/-- The problem setup and solution -/
theorem tea_milk_mixture : 
  let initial_cup1 : Cup := { tea := 6, milk := 0 }
  let initial_cup2 : Cup := { tea := 0, milk := 6 }
  let cup_size : ℚ := 12

  -- Transfer 1/3 of tea from Cup 1 to Cup 2
  let transfer1_amount : ℚ := initial_cup1.tea / 3
  let after_transfer1_cup1 : Cup := { tea := initial_cup1.tea - transfer1_amount, milk := initial_cup1.milk }
  let after_transfer1_cup2 : Cup := { tea := initial_cup2.tea + transfer1_amount, milk := initial_cup2.milk }

  -- Transfer 1/4 of mixture from Cup 2 back to Cup 1
  let total_liquid_cup2 : ℚ := after_transfer1_cup2.tea + after_transfer1_cup2.milk
  let transfer2_amount : ℚ := total_liquid_cup2 / 4
  let tea_ratio_cup2 : ℚ := after_transfer1_cup2.tea / total_liquid_cup2
  let milk_ratio_cup2 : ℚ := after_transfer1_cup2.milk / total_liquid_cup2
  let final_cup1 : Cup := {
    tea := after_transfer1_cup1.tea + transfer2_amount * tea_ratio_cup2,
    milk := after_transfer1_cup1.milk + transfer2_amount * milk_ratio_cup2
  }

  -- The fraction of milk in Cup 1 at the end
  let milk_fraction : ℚ := final_cup1.milk / (final_cup1.tea + final_cup1.milk)

  milk_fraction = 1/4 := by sorry

end tea_milk_mixture_l788_78843


namespace fraction_simplification_l788_78821

theorem fraction_simplification (a x b : ℝ) (hb : b > 0) :
  (Real.sqrt b * (Real.sqrt (a^2 + x^2) - (x^2 - a^2) / Real.sqrt (a^2 + x^2))) / (b * (a^2 + x^2)) =
  (2 * a^2 * Real.sqrt b) / (b * (a^2 + x^2)^(3/2)) :=
by sorry

end fraction_simplification_l788_78821


namespace solution_set_part1_range_of_m_part2_l788_78819

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 1| + |x - 2| - m

-- Part 1: Solution set for f(x) > 0 when m = 5
theorem solution_set_part1 : 
  {x : ℝ | f x 5 > 0} = {x : ℝ | x < -2 ∨ x > 3} := by sorry

-- Part 2: Range of m for which f(x) ≥ 2 has solution set ℝ
theorem range_of_m_part2 : 
  {m : ℝ | ∀ x, f x m ≥ 2} = {m : ℝ | m ≤ 1} := by sorry

end solution_set_part1_range_of_m_part2_l788_78819


namespace parabola_point_l788_78851

/-- Given a point P (x₀, y₀) on the parabola y = 3x² with derivative 6 at x₀, prove P = (1, 3) -/
theorem parabola_point (x₀ y₀ : ℝ) : 
  y₀ = 3 * x₀^2 →                   -- Point P lies on the parabola
  (6 : ℝ) = 6 * x₀ →                -- Derivative at x₀ is 6
  (x₀, y₀) = (1, 3) :=              -- Conclusion: P = (1, 3)
by sorry

end parabola_point_l788_78851


namespace permutation_reachable_l788_78859

/-- A transformation step on a tuple of natural numbers -/
def transform (a : Fin 2015 → ℕ) (k l : Fin 2015) (h : Even (a k)) : Fin 2015 → ℕ :=
  fun i => if i = k then a k / 2
           else if i = l then a l + a k / 2
           else a i

/-- The set of all permutations of (1, 2, ..., 2015) -/
def permutations : Set (Fin 2015 → ℕ) :=
  {p | ∃ σ : Equiv.Perm (Fin 2015), ∀ i, p i = σ i + 1}

/-- The initial tuple (1, 2, ..., 2015) -/
def initial : Fin 2015 → ℕ := fun i => i + 1

/-- The set of all tuples reachable from the initial tuple -/
inductive reachable : (Fin 2015 → ℕ) → Prop
  | init : reachable initial
  | step {a b} (k l : Fin 2015) (h : Even (a k)) :
      reachable a → b = transform a k l h → reachable b

theorem permutation_reachable :
  ∀ p ∈ permutations, reachable p :=
sorry

end permutation_reachable_l788_78859


namespace number_of_hens_l788_78806

theorem number_of_hens (total_animals : ℕ) (total_feet : ℕ) (hens : ℕ) (cows : ℕ) : 
  total_animals = 60 →
  total_feet = 200 →
  total_animals = hens + cows →
  total_feet = 2 * hens + 4 * cows →
  hens = 20 := by
sorry

end number_of_hens_l788_78806


namespace panda_babies_l788_78812

theorem panda_babies (total_pandas : ℕ) (pregnancy_rate : ℚ) : 
  total_pandas = 16 → 
  pregnancy_rate = 1/4 → 
  (total_pandas / 2 : ℚ) * pregnancy_rate = 2 :=
sorry

end panda_babies_l788_78812


namespace square_sum_given_difference_and_product_l788_78865

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : x - y = 12) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 162 := by
sorry

end square_sum_given_difference_and_product_l788_78865


namespace xyz_bound_l788_78862

theorem xyz_bound (x y z : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (hz_pos : z > 0)
  (hx_bound : x ≤ 2) (hy_bound : y ≤ 3) (hsum : x + y + z = 11) :
  x * y * z ≤ 36 := by
  sorry

end xyz_bound_l788_78862


namespace complex_product_sum_l788_78801

theorem complex_product_sum (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) 
  (h2 : (1 + i) * (2 + i) = a + b * i) : a + b = 4 := by
  sorry

end complex_product_sum_l788_78801


namespace total_assembly_time_l788_78880

def chairs : ℕ := 7
def tables : ℕ := 3
def bookshelves : ℕ := 2
def lamps : ℕ := 4

def chair_time : ℕ := 4
def table_time : ℕ := 8
def bookshelf_time : ℕ := 12
def lamp_time : ℕ := 2

theorem total_assembly_time :
  chairs * chair_time + tables * table_time + bookshelves * bookshelf_time + lamps * lamp_time = 84 := by
  sorry

end total_assembly_time_l788_78880


namespace pages_copied_for_thirty_dollars_l788_78822

/-- Given the rate of copying 5 pages for 8 cents, prove that $30 (3000 cents) will allow copying 1875 pages. -/
theorem pages_copied_for_thirty_dollars :
  let rate : ℚ := 5 / 8 -- pages per cent
  let total_cents : ℕ := 3000 -- $30 in cents
  (rate * total_cents : ℚ) = 1875 := by sorry

end pages_copied_for_thirty_dollars_l788_78822


namespace op_35_77_l788_78808

-- Define the operation @
def op (a b : ℕ+) : ℚ := (a * b) / (a + b)

-- Theorem statement
theorem op_35_77 : op 35 77 = 2695 / 112 := by
  sorry

end op_35_77_l788_78808


namespace group_selection_count_l788_78867

theorem group_selection_count : 
  let total_students : ℕ := 7
  let male_students : ℕ := 4
  let female_students : ℕ := 3
  let group_size : ℕ := 3
  (Nat.choose total_students group_size) - 
  (Nat.choose male_students group_size) - 
  (Nat.choose female_students group_size) = 30 := by
sorry

end group_selection_count_l788_78867


namespace parabola_point_distance_l788_78842

theorem parabola_point_distance (a : ℝ) : 
  (∀ x y : ℝ, y^2 = 4*x → (x - a)^2 + y^2 ≥ a^2) → a ≤ 2 := by
  sorry

end parabola_point_distance_l788_78842


namespace remainder_problem_l788_78856

theorem remainder_problem (N : ℤ) : 
  N % 19 = 7 → N % 20 = 6 := by
sorry

end remainder_problem_l788_78856


namespace ellipse_m_range_l788_78804

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the equation of the curve -/
def curve_equation (m : ℝ) (p : Point) : Prop :=
  m * (p.x^2 + p.y^2 + 2*p.y + 1) = (p.x - 2*p.y + 3)^2

/-- Defines what it means for the curve to be an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (p : Point), curve_equation m p ↔ 
    (p.x - h)^2 / a^2 + (p.y - k)^2 / b^2 = 1
  where
    h := 0  -- center x-coordinate
    k := -1 -- center y-coordinate

/-- The main theorem stating the range of m for which the curve is an ellipse -/
theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse m ↔ m > 5 :=
sorry

end ellipse_m_range_l788_78804


namespace subset_union_subset_l788_78884

theorem subset_union_subset (U M N : Set α) : M ⊆ U → N ⊆ U → (M ∪ N) ⊆ U := by sorry

end subset_union_subset_l788_78884


namespace smallest_n_for_perfect_cube_sum_l788_78834

/-- Checks if a number is a perfect cube -/
def isPerfectCube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m ^ 3

/-- Checks if three natural numbers are distinct and nonzero -/
def areDistinctNonzero (a b c : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a

/-- The main theorem statement -/
theorem smallest_n_for_perfect_cube_sum : 
  (∃ a b c : ℕ, areDistinctNonzero a b c ∧ 
    10 = a + b + c ∧ 
    isPerfectCube ((a + b) * (b + c) * (c + a))) ∧ 
  (∀ n : ℕ, n < 10 → 
    ¬(∃ a b c : ℕ, areDistinctNonzero a b c ∧ 
      n = a + b + c ∧ 
      isPerfectCube ((a + b) * (b + c) * (c + a)))) :=
by sorry

end smallest_n_for_perfect_cube_sum_l788_78834


namespace sheilas_monthly_savings_l788_78802

/-- Calculates the monthly savings amount given the initial savings, family contribution, 
    savings period in years, and final amount in the piggy bank. -/
def monthlySavings (initialSavings familyContribution : ℕ) (savingsPeriodYears : ℕ) 
    (finalAmount : ℕ) : ℚ :=
  let totalInitialAmount := initialSavings + familyContribution
  let amountToSave := finalAmount - totalInitialAmount
  let monthsInPeriod := savingsPeriodYears * 12
  (amountToSave : ℚ) / (monthsInPeriod : ℚ)

/-- Theorem stating that Sheila's monthly savings is $276 -/
theorem sheilas_monthly_savings : 
  monthlySavings 3000 7000 4 23248 = 276 := by
  sorry

end sheilas_monthly_savings_l788_78802


namespace tangent_line_at_pi_third_l788_78809

noncomputable def f (x : ℝ) : ℝ := (1/2) * x + Real.sin x

def tangent_line_equation (x y : ℝ) : Prop :=
  6 * x - 6 * y + 3 * Real.sqrt 3 - Real.pi = 0

theorem tangent_line_at_pi_third :
  tangent_line_equation (π/3) (f (π/3)) :=
sorry

end tangent_line_at_pi_third_l788_78809


namespace animal_shelter_cats_l788_78838

theorem animal_shelter_cats (total : ℕ) (cats dogs : ℕ) : 
  total = 60 →
  cats = dogs + 20 →
  cats + dogs = total →
  cats = 40 := by
sorry

end animal_shelter_cats_l788_78838


namespace grid_difference_theorem_l788_78848

def Grid := Fin 8 → Fin 8 → Fin 64

def adjacent (x₁ y₁ x₂ y₂ : Fin 8) : Prop :=
  (x₁ = x₂ ∧ (y₁.val + 1 = y₂.val ∨ y₂.val + 1 = y₁.val)) ∨
  (y₁ = y₂ ∧ (x₁.val + 1 = x₂.val ∨ x₂.val + 1 = x₁.val))

theorem grid_difference_theorem (g : Grid) (h : Function.Injective g) :
    ∃ (x₁ y₁ x₂ y₂ : Fin 8), adjacent x₁ y₁ x₂ y₂ ∧ 
    (g x₁ y₁).val.succ.succ.succ.succ ≤ (g x₂ y₂).val ∨ 
    (g x₂ y₂).val.succ.succ.succ.succ ≤ (g x₁ y₁).val := by
  sorry

end grid_difference_theorem_l788_78848


namespace arithmetic_sequence_iff_t_eq_zero_l788_78876

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) (t : ℝ) : ℝ := n^2 + 5*n + t

/-- The nth term of the sequence -/
def a (n : ℕ) (t : ℝ) : ℝ :=
  if n = 1 then S 1 t
  else S n t - S (n-1) t

/-- A sequence is arithmetic if the difference between consecutive terms is constant -/
def is_arithmetic_sequence (f : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, f (n+1) - f n = d

theorem arithmetic_sequence_iff_t_eq_zero (t : ℝ) :
  is_arithmetic_sequence (λ n => a n t) ↔ t = 0 := by
  sorry

end arithmetic_sequence_iff_t_eq_zero_l788_78876


namespace integral_sqrt_4_minus_x_squared_plus_2x_l788_78839

open MeasureTheory Interval Real

theorem integral_sqrt_4_minus_x_squared_plus_2x : 
  ∫ x in (-2)..2, (Real.sqrt (4 - x^2) + 2*x) = 2*π := by sorry

end integral_sqrt_4_minus_x_squared_plus_2x_l788_78839


namespace inequality_solution_set_l788_78837

theorem inequality_solution_set (x : ℝ) :
  (3 * x - 7 ≤ 2) ↔ (x ≤ 3) := by sorry

end inequality_solution_set_l788_78837


namespace cube_equation_solution_l788_78890

theorem cube_equation_solution : ∃ (N : ℕ), N > 0 ∧ 26^3 * 65^3 = 10^3 * N^3 ∧ N = 169 := by
  sorry

end cube_equation_solution_l788_78890


namespace angle_alpha_properties_l788_78887

def angle_alpha (α : Real) : Prop :=
  ∃ (x y : Real), x = 1 ∧ y = Real.sqrt 3 ∧ x = Real.cos α ∧ y = Real.sin α

theorem angle_alpha_properties (α : Real) (h : angle_alpha α) :
  (Real.sin (π - α) - Real.sin (π / 2 + α) = (Real.sqrt 3 - 1) / 2) ∧
  (∃ k : ℤ, α = 2 * π * (k : Real) + π / 3) :=
sorry

end angle_alpha_properties_l788_78887


namespace volume_ADBFE_l788_78841

-- Define the pyramid ABCD
def Pyramid (A B C D : Point) : Set Point := sorry

-- Define the volume of a set of points
def volume : Set Point → ℝ := sorry

-- Define a median of a triangle
def isMedian (E : Point) (triangle : Set Point) : Prop := sorry

-- Define a midpoint of a line segment
def isMidpoint (F : Point) (segment : Set Point) : Prop := sorry

theorem volume_ADBFE (A B C D E F : Point) 
  (hpyramid : Pyramid A B C D)
  (hmedian : isMedian E {A, B, C})
  (hmidpoint : isMidpoint F {D, C})
  (hvolume : volume (Pyramid A B C D) = 40) :
  volume {A, D, B, F, E} = (3/4) * volume (Pyramid A B C D) := by
  sorry

end volume_ADBFE_l788_78841


namespace square_sum_of_two_integers_l788_78852

theorem square_sum_of_two_integers (x y : ℕ+) 
  (h1 : x * y + x + y = 17)
  (h2 : x^2 * y + x * y^2 = 72) : 
  x^2 + y^2 = 65 := by
  sorry

end square_sum_of_two_integers_l788_78852


namespace last_two_digits_product_l788_78875

theorem last_two_digits_product (n : ℕ) : 
  (n % 100 ≥ 10) →  -- Ensure n has at least two digits
  (n % 6 = 0) →     -- n is divisible by 6
  ((n % 100) / 10 + n % 10 = 15) →  -- Sum of last two digits is 15
  (((n % 100) / 10) * (n % 10) = 56 ∨ ((n % 100) / 10) * (n % 10) = 54) :=
by sorry

end last_two_digits_product_l788_78875


namespace negation_of_existence_quadratic_always_positive_l788_78886

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem quadratic_always_positive : 
  (¬ ∃ x : ℝ, x^2 + x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + x + 2 > 0) := by sorry

end negation_of_existence_quadratic_always_positive_l788_78886


namespace function_min_value_l788_78829

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

-- State the theorem
theorem function_min_value 
  (m : ℝ) 
  (h_max : ∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≤ 3) :
  ∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -37 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≥ -37 :=
sorry

end function_min_value_l788_78829


namespace optimal_numbering_scheme_l788_78832

/-- Represents a numbering scheme for a population --/
structure NumberingScheme where
  start : Nat
  digits : Nat

/-- Checks if a numbering scheme is valid for a given population size --/
def isValidScheme (populationSize : Nat) (scheme : NumberingScheme) : Prop :=
  scheme.start = 0 ∧
  scheme.digits = 3 ∧
  10 ^ scheme.digits > populationSize

/-- Theorem stating the optimal numbering scheme for the given conditions --/
theorem optimal_numbering_scheme
  (populationSize : Nat)
  (sampleSize : Nat)
  (h1 : populationSize = 106)
  (h2 : sampleSize = 10)
  (h3 : sampleSize < populationSize) :
  ∃ (scheme : NumberingScheme),
    isValidScheme populationSize scheme ∧
    scheme.start = 0 ∧
    scheme.digits = 3 :=
  sorry

end optimal_numbering_scheme_l788_78832


namespace netGainDifference_l788_78816

/-- Represents an applicant for a job position -/
structure Applicant where
  salary : ℕ
  revenue : ℕ
  trainingMonths : ℕ
  trainingCostPerMonth : ℕ
  hiringBonusPercent : ℕ

/-- Calculates the net gain for the company from an applicant -/
def netGain (a : Applicant) : ℕ :=
  a.revenue - a.salary - (a.trainingMonths * a.trainingCostPerMonth) - (a.salary * a.hiringBonusPercent / 100)

/-- The first applicant's details -/
def applicant1 : Applicant :=
  { salary := 42000
    revenue := 93000
    trainingMonths := 3
    trainingCostPerMonth := 1200
    hiringBonusPercent := 0 }

/-- The second applicant's details -/
def applicant2 : Applicant :=
  { salary := 45000
    revenue := 92000
    trainingMonths := 0
    trainingCostPerMonth := 0
    hiringBonusPercent := 1 }

/-- Theorem stating the difference in net gain between the two applicants -/
theorem netGainDifference : netGain applicant1 - netGain applicant2 = 850 := by
  sorry

end netGainDifference_l788_78816


namespace solve_for_s_l788_78892

theorem solve_for_s (t : ℚ) (h1 : 7 * ((t / 2) + 3) + 6 * t = 156) : (t / 2) + 3 = 192 / 19 := by
  sorry

end solve_for_s_l788_78892


namespace original_number_is_seven_l788_78872

theorem original_number_is_seven : ∃ x : ℝ, 3 * x - 5 = 16 ∧ x = 7 := by
  sorry

end original_number_is_seven_l788_78872


namespace equation_equality_l788_78824

theorem equation_equality : 3 * 6524 = 8254 * 3 := by
  sorry

end equation_equality_l788_78824


namespace chef_used_41_apples_l788_78897

/-- The number of apples the chef used to make pies -/
def apples_used (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Theorem: Given the initial number of apples and the remaining number of apples,
    prove that the number of apples used is 41. -/
theorem chef_used_41_apples (initial : ℕ) (remaining : ℕ)
  (h1 : initial = 43)
  (h2 : remaining = 2) :
  apples_used initial remaining = 41 := by
  sorry

end chef_used_41_apples_l788_78897


namespace line_BM_equation_angle_ABM_equals_ABN_l788_78889

-- Define the parabola C
def C (x y : ℝ) : Prop := y^2 = 2*x

-- Define points A and B
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (-2, 0)

-- Define a line l passing through A
def l (t : ℝ) (x y : ℝ) : Prop := x = t*y + 2

-- Define points M and N as intersections of l and C
def M (t : ℝ) : ℝ × ℝ := sorry
def N (t : ℝ) : ℝ × ℝ := sorry

-- Theorem 1: When l is perpendicular to x-axis, equation of BM
theorem line_BM_equation (t : ℝ) : 
  t = 0 → (
    let (x₁, y₁) := M t
    (x₁ - 2*y₁ + 2 = 0) ∨ (x₁ + 2*y₁ + 2 = 0)
  ) := by sorry

-- Theorem 2: ∠ABM = ∠ABN for any line l
theorem angle_ABM_equals_ABN (t : ℝ) :
  let (x₁, y₁) := M t
  let (x₂, y₂) := N t
  (y₁ / (x₁ + 2)) + (y₂ / (x₂ + 2)) = 0 := by sorry

end line_BM_equation_angle_ABM_equals_ABN_l788_78889


namespace f_lower_bound_and_g_inequality_l788_78813

noncomputable section

def f (x : ℝ) := x - Real.log x

def g (x : ℝ) := x^3 + x^2 * (f x) - 16*x

theorem f_lower_bound_and_g_inequality {x : ℝ} (hx : x > 0) :
  f x ≥ 1 ∧ g x > -20 := by sorry

end f_lower_bound_and_g_inequality_l788_78813


namespace tennis_tournament_matches_l788_78850

theorem tennis_tournament_matches (total_players : Nat) (seeded_players : Nat) : 
  total_players = 128 → seeded_players = 32 → total_players - 1 = 127 :=
by
  sorry

#check tennis_tournament_matches

end tennis_tournament_matches_l788_78850


namespace exponential_base_theorem_l788_78891

theorem exponential_base_theorem (a : ℝ) (h : a > 0) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, a^x ≤ max a a⁻¹) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, min a a⁻¹ ≤ a^x) ∧
  (max a a⁻¹ - min a a⁻¹ = 1) →
  a = (Real.sqrt 5 + 1) / 2 ∨ a = (Real.sqrt 5 - 1) / 2 :=
by sorry

end exponential_base_theorem_l788_78891


namespace master_wang_parts_per_day_l788_78855

/-- The number of parts Master Wang processed per day -/
def parts_per_day (a : ℕ) : ℚ :=
  (a + 3 : ℚ) / 8

/-- Theorem stating that the number of parts processed per day is (a + 3) / 8 -/
theorem master_wang_parts_per_day (a : ℕ) :
  parts_per_day a = (a + 3 : ℚ) / 8 := by
  sorry

#check master_wang_parts_per_day

end master_wang_parts_per_day_l788_78855


namespace keith_pears_l788_78845

theorem keith_pears (jason_pears mike_ate remaining_pears : ℕ) 
  (h1 : jason_pears = 46)
  (h2 : mike_ate = 12)
  (h3 : remaining_pears = 81) :
  ∃ keith_pears : ℕ, jason_pears + keith_pears - mike_ate = remaining_pears ∧ keith_pears = 47 :=
by sorry

end keith_pears_l788_78845


namespace emilys_average_speed_l788_78881

-- Define the parameters of Emily's trip
def distance1 : ℝ := 450  -- miles
def time1 : ℝ := 7.5      -- hours (7 hours 30 minutes)
def break_time : ℝ := 1   -- hour
def distance2 : ℝ := 540  -- miles
def time2 : ℝ := 8        -- hours

-- Define the total distance and time
def total_distance : ℝ := distance1 + distance2
def total_time : ℝ := time1 + break_time + time2

-- Theorem to prove
theorem emilys_average_speed :
  total_distance / total_time = 60 := by sorry

end emilys_average_speed_l788_78881


namespace corn_planting_ratio_l788_78840

/-- Represents the problem of calculating the ratio of dinner cost to total earnings for corn planting kids. -/
theorem corn_planting_ratio :
  -- Define constants based on the problem conditions
  let ears_per_row : ℕ := 70
  let seeds_per_bag : ℕ := 48
  let seeds_per_ear : ℕ := 2
  let pay_per_row : ℚ := 3/2  -- $1.5 expressed as a rational number
  let dinner_cost : ℚ := 36
  let bags_used : ℕ := 140

  -- Calculate total ears planted
  let total_ears : ℕ := (bags_used * seeds_per_bag) / seeds_per_ear

  -- Calculate rows planted
  let rows_planted : ℕ := total_ears / ears_per_row

  -- Calculate total earnings
  let total_earned : ℚ := pay_per_row * rows_planted

  -- The ratio of dinner cost to total earnings is 1/2
  dinner_cost / total_earned = 1/2 :=
by
  sorry  -- Proof is omitted as per instructions

end corn_planting_ratio_l788_78840


namespace geometric_sequence_product_l788_78827

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_prod : a 5 * a 14 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 := by
sorry

end geometric_sequence_product_l788_78827


namespace triangle_inequality_l788_78846

theorem triangle_inequality (a b c p : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_half_perimeter : p = (a + b + c) / 2) :
  a^2 * (p - a) * (p - b) + b^2 * (p - b) * (p - c) + c^2 * (p - c) * (p - a) ≤ (4 / 27) * p^4 := by
sorry

end triangle_inequality_l788_78846


namespace sqrt_simplification_l788_78805

theorem sqrt_simplification : 
  Real.sqrt 45 - 2 * Real.sqrt 5 + Real.sqrt 360 / Real.sqrt 2 = Real.sqrt 245 := by
  sorry

end sqrt_simplification_l788_78805


namespace fraction_simplification_l788_78895

theorem fraction_simplification :
  (5 : ℚ) / ((8 : ℚ) / 13 + (1 : ℚ) / 13) = 65 / 9 := by
  sorry

end fraction_simplification_l788_78895


namespace triangle_angle_c_l788_78858

theorem triangle_angle_c (A B C : Real) (a b c : Real) :
  -- Conditions
  a + b + c = Real.sqrt 2 + 1 →  -- Perimeter condition
  (1/2) * a * b * Real.sin C = (1/6) * Real.sin C →  -- Area condition
  Real.sin A + Real.sin B = Real.sqrt 2 * Real.sin C →  -- Sine relation
  -- Conclusion
  C = π / 3 := by
  sorry

end triangle_angle_c_l788_78858


namespace cartesian_product_A_B_l788_78817

def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}

theorem cartesian_product_A_B :
  A ×ˢ B = {(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)} := by
  sorry

end cartesian_product_A_B_l788_78817


namespace uninterrupted_viewing_time_movie_problem_solution_l788_78885

/-- Calculates the uninterrupted viewing time at the end of a movie given the total viewing time,
    initial viewing periods, and rewind times. -/
theorem uninterrupted_viewing_time 
  (total_time : ℕ) 
  (first_viewing : ℕ) 
  (first_rewind : ℕ) 
  (second_viewing : ℕ) 
  (second_rewind : ℕ) : 
  total_time - (first_viewing + second_viewing + first_rewind + second_rewind) = 
  total_time - ((first_viewing + second_viewing) + (first_rewind + second_rewind)) :=
by sorry

/-- Proves that the uninterrupted viewing time at the end of the movie is 20 minutes
    given the specific conditions from the problem. -/
theorem movie_problem_solution 
  (total_time : ℕ) 
  (first_viewing : ℕ) 
  (first_rewind : ℕ) 
  (second_viewing : ℕ) 
  (second_rewind : ℕ) 
  (h1 : total_time = 120) 
  (h2 : first_viewing = 35) 
  (h3 : first_rewind = 5) 
  (h4 : second_viewing = 45) 
  (h5 : second_rewind = 15) : 
  total_time - (first_viewing + second_viewing + first_rewind + second_rewind) = 20 :=
by sorry

end uninterrupted_viewing_time_movie_problem_solution_l788_78885


namespace height_of_pillar_D_l788_78888

/-- Regular hexagon with pillars -/
structure HexagonWithPillars where
  -- Side length of the hexagon
  side_length : ℝ
  -- Heights of pillars at A, B, C
  height_A : ℝ
  height_B : ℝ
  height_C : ℝ

/-- Theorem: Height of pillar at D in a regular hexagon with given pillar heights -/
theorem height_of_pillar_D (h : HexagonWithPillars) 
  (h_side : h.side_length = 10)
  (h_A : h.height_A = 8)
  (h_B : h.height_B = 11)
  (h_C : h.height_C = 12) : 
  ∃ (z : ℝ), z = 5 ∧ 
  ((-15 * Real.sqrt 3) * (-10) + 20 * 0 + (50 * Real.sqrt 3) * z = 400 * Real.sqrt 3) := by
  sorry

#check height_of_pillar_D

end height_of_pillar_D_l788_78888


namespace pencil_profit_problem_l788_78814

theorem pencil_profit_problem (total_pencils : ℕ) (buy_price sell_price : ℚ) (desired_profit : ℚ) :
  total_pencils = 1500 →
  buy_price = 1/10 →
  sell_price = 1/4 →
  desired_profit = 100 →
  ∃ (pencils_sold : ℕ), 
    pencils_sold ≤ total_pencils ∧
    sell_price * pencils_sold - buy_price * total_pencils = desired_profit ∧
    pencils_sold = 1000 :=
by sorry

end pencil_profit_problem_l788_78814


namespace spatial_relationships_l788_78882

/-- Two lines are non-coincident -/
def non_coincident_lines (l m : Line) : Prop := l ≠ m

/-- Two planes are non-coincident -/
def non_coincident_planes (α β : Plane) : Prop := α ≠ β

/-- A line is perpendicular to a plane -/
def line_perp_plane (l : Line) (α : Plane) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line) (α : Plane) : Prop := sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Line) (α : Plane) : Prop := sorry

/-- Two planes are perpendicular -/
def planes_perp (α β : Plane) : Prop := sorry

/-- Two lines are perpendicular -/
def lines_perp (l m : Line) : Prop := sorry

theorem spatial_relationships (l m : Line) (α β : Plane) 
  (h1 : non_coincident_lines l m) (h2 : non_coincident_planes α β) :
  (lines_perp l m ∧ line_perp_plane l α ∧ line_perp_plane m β → planes_perp α β) ∧
  (line_perp_plane l β ∧ planes_perp α β → line_parallel_plane l α ∨ line_in_plane l α) :=
sorry

end spatial_relationships_l788_78882


namespace quadratic_real_roots_condition_l788_78811

theorem quadratic_real_roots_condition (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 4 * x + 2 = 0) ↔ (k ≤ 2 ∧ k ≠ 0) :=
by sorry

end quadratic_real_roots_condition_l788_78811


namespace shadow_boundary_is_constant_l788_78868

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- The equation of the shadow boundary for a sphere -/
def shadowBoundary (s : Sphere) (lightSource : Point3D) : ℝ → ℝ := fun x => -2

/-- Theorem stating that the shadow boundary is y = -2 for the given sphere and light source -/
theorem shadow_boundary_is_constant (s : Sphere) (lightSource : Point3D) :
  s.center = Point3D.mk 0 0 2 →
  s.radius = 2 →
  lightSource = Point3D.mk 0 1 3 →
  ∀ x, shadowBoundary s lightSource x = -2 := by
  sorry

#check shadow_boundary_is_constant

end shadow_boundary_is_constant_l788_78868


namespace two_numbers_difference_l788_78864

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 23) : |x - y| = 22 := by
  sorry

end two_numbers_difference_l788_78864


namespace system_solution_l788_78896

theorem system_solution : 
  ∀ x : ℝ, (3 * x^2 = Real.sqrt (36 * x^2) ∧ 3 * x^2 + 21 = 24 * x) ↔ (x = 7 ∨ x = 1) :=
by sorry

end system_solution_l788_78896


namespace sum_f_1_to_10_l788_78853

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic_3 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 3) = f x

theorem sum_f_1_to_10 (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_periodic : is_periodic_3 f) 
  (h_f_neg_1 : f (-1) = 1) : 
  f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
sorry

end sum_f_1_to_10_l788_78853


namespace cube_sphere_volume_l788_78877

theorem cube_sphere_volume (cube_volume : Real) (h : cube_volume = 8) :
  ∃ (sphere_volume : Real), sphere_volume = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end cube_sphere_volume_l788_78877


namespace min_value_problem_l788_78815

theorem min_value_problem (a b c d e f g h : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (pos_e : 0 < e) (pos_f : 0 < f) (pos_g : 0 < g) (pos_h : 0 < h)
  (h1 : a * b * c * d = 8)
  (h2 : e * f * g * h = 16)
  (h3 : a + b + c + d = e * f * g) :
  64 ≤ (a*e)^2 + (b*f)^2 + (c*g)^2 + (d*h)^2 ∧ 
  ∃ (a' b' c' d' e' f' g' h' : ℝ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 
    0 < e' ∧ 0 < f' ∧ 0 < g' ∧ 0 < h' ∧
    a' * b' * c' * d' = 8 ∧
    e' * f' * g' * h' = 16 ∧
    a' + b' + c' + d' = e' * f' * g' ∧
    (a'*e')^2 + (b'*f')^2 + (c'*g')^2 + (d'*h')^2 = 64 :=
by sorry

end min_value_problem_l788_78815


namespace range_of_a_l788_78844

-- Define the propositions p and q
def p (a x : ℝ) : Prop := a - 4 < x ∧ x < a + 4
def q (x : ℝ) : Prop := (x - 2) * (x - 3) > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, p a x → q x) →
  (a ≤ -2 ∨ a ≥ 7) :=
sorry

end range_of_a_l788_78844


namespace polynomial_division_remainder_l788_78878

theorem polynomial_division_remainder (x : ℝ) :
  ∃ q : ℝ → ℝ, x^5 + x^2 + 3 = (x - 3)^2 * q x + 219 := by
  sorry

end polynomial_division_remainder_l788_78878


namespace electronic_shop_purchase_cost_l788_78831

/-- The price of a smartphone in dollars -/
def smartphone_price : ℕ := 300

/-- The price difference between a personal computer and a smartphone in dollars -/
def pc_price_difference : ℕ := 500

/-- The price of a personal computer in dollars -/
def pc_price : ℕ := smartphone_price + pc_price_difference

/-- The price of an advanced tablet in dollars -/
def tablet_price : ℕ := smartphone_price + pc_price

/-- The total cost of buying one of each product in dollars -/
def total_cost : ℕ := smartphone_price + pc_price + tablet_price

theorem electronic_shop_purchase_cost : total_cost = 2200 := by
  sorry

end electronic_shop_purchase_cost_l788_78831


namespace rectangle_area_l788_78823

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 186) : L * B = 2030 := by
  sorry

end rectangle_area_l788_78823


namespace third_angle_measure_l788_78871

-- Define a triangle type
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.angle1 > 0 ∧ t.angle2 > 0 ∧ t.angle3 > 0 ∧
  t.angle1 + t.angle2 + t.angle3 = 180

-- Theorem statement
theorem third_angle_measure (t : Triangle) :
  is_valid_triangle t → t.angle1 = 25 → t.angle2 = 70 → t.angle3 = 85 := by
  sorry

end third_angle_measure_l788_78871


namespace intersection_M_N_l788_78800

-- Define the sets M and N
def M : Set ℝ := {x | x - 2 > 0}
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 + 1)}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | x > 2} := by sorry

end intersection_M_N_l788_78800


namespace no_guaranteed_primes_l788_78828

theorem no_guaranteed_primes (n : ℕ) (h : n > 1) :
  ∀ p : ℕ, Prime p → (p ∉ Set.Ioo (n.factorial) (n.factorial + 2*n)) :=
sorry

end no_guaranteed_primes_l788_78828


namespace cosine_sum_upper_bound_l788_78860

theorem cosine_sum_upper_bound (α β γ : Real) 
  (h : Real.sin α + Real.sin β + Real.sin γ ≥ 2) : 
  Real.cos α + Real.cos β + Real.cos γ ≤ Real.sqrt 5 := by
  sorry

end cosine_sum_upper_bound_l788_78860


namespace expression_simplification_l788_78870

theorem expression_simplification (b y : ℝ) (hb : b > 0) (hy : y > 0) :
  (Real.sqrt (b^2 + y^2) - (y^2 - b^2) / Real.sqrt (b^2 + y^2)) / (b + y^2) = 2 * b^2 / (b + y^2) := by
sorry

end expression_simplification_l788_78870


namespace complete_square_quadratic_l788_78849

theorem complete_square_quadratic (x : ℝ) : 
  ∃ (a b : ℝ), x^2 + 6*x - 3 = 0 ↔ (x + a)^2 = b ∧ b = 12 := by
  sorry

end complete_square_quadratic_l788_78849


namespace typing_time_proof_l788_78833

def original_speed : ℕ := 212
def speed_reduction : ℕ := 40
def document_length : ℕ := 3440

theorem typing_time_proof :
  (document_length : ℚ) / (original_speed - speed_reduction) = 20 := by
  sorry

end typing_time_proof_l788_78833


namespace quadratic_at_most_one_solution_l788_78854

theorem quadratic_at_most_one_solution (a : ℝ) :
  (∃! x : ℝ, a * x^2 + 3 * x + 1 = 0) ∨ (∀ x y : ℝ, a * x^2 + 3 * x + 1 = 0 → a * y^2 + 3 * y + 1 = 0 → x = y) ↔
  a = 0 ∨ a ≥ 9/4 := by
  sorry

end quadratic_at_most_one_solution_l788_78854


namespace gcd_of_polynomial_and_multiple_l788_78866

theorem gcd_of_polynomial_and_multiple : ∀ y : ℤ, 
  9240 ∣ y → 
  Int.gcd ((5*y+3)*(11*y+2)*(17*y+8)*(4*y+7)) y = 168 := by
  sorry

end gcd_of_polynomial_and_multiple_l788_78866


namespace physics_value_l788_78879

def letterValue (n : Nat) : Int :=
  match n % 9 with
  | 0 => 0
  | 1 => 2
  | 2 => 3
  | 3 => 2
  | 4 => 0
  | 5 => -1
  | 6 => -2
  | 7 => -3
  | 8 => -2
  | _ => -1

def wordValue (word : List Nat) : Int :=
  List.sum (List.map letterValue word)

theorem physics_value :
  wordValue [16, 8, 25, 19, 9, 3, 19] = 1 := by
  sorry

end physics_value_l788_78879


namespace pyramid_sphere_radius_l788_78869

-- Define the pyramid
structure RegularQuadrilateralPyramid where
  base_side : ℝ
  lateral_edge : ℝ

-- Define the spheres
structure Sphere where
  radius : ℝ

-- Define the problem
def pyramid_problem (p : RegularQuadrilateralPyramid) (q1 q2 : Sphere) : Prop :=
  p.base_side = 12 ∧
  p.lateral_edge = 10 ∧
  -- Q1 is inscribed in the pyramid (this is implied, not explicitly stated in Lean)
  -- Q2 touches Q1 and all lateral faces (this is implied, not explicitly stated in Lean)
  q2.radius = 6 * Real.sqrt 7 / 49

-- Theorem statement
theorem pyramid_sphere_radius 
  (p : RegularQuadrilateralPyramid) 
  (q1 q2 : Sphere) 
  (h : pyramid_problem p q1 q2) : 
  q2.radius = 6 * Real.sqrt 7 / 49 := by
  sorry


end pyramid_sphere_radius_l788_78869


namespace sqrt_2a_plus_b_equals_3_l788_78863

theorem sqrt_2a_plus_b_equals_3 (a b : ℝ) 
  (h1 : (2*a - 1) = 9)
  (h2 : a - 2*b + 1 = 8) :
  Real.sqrt (2*a + b) = 3 := by
sorry

end sqrt_2a_plus_b_equals_3_l788_78863


namespace arithmetic_sequence_third_term_l788_78810

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_third_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_first : a 1 = 2)
  (h_fifth : a 5 = a 4 + 2) :
  a 3 = 6 := by
sorry

end arithmetic_sequence_third_term_l788_78810


namespace four_digit_number_problem_l788_78873

theorem four_digit_number_problem (a b c d : Nat) : 
  (a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9) →  -- Ensuring each digit is between 0 and 9
  (a ≠ 0) →  -- Ensuring 'a' is not 0 (as it's a four-digit number)
  (1000 * a + 100 * b + 10 * c + d) - (100 * a + 10 * b + c) - (10 * a + b) - a = 1787 →
  (1000 * a + 100 * b + 10 * c + d = 2009 ∨ 1000 * a + 100 * b + 10 * c + d = 2010) :=
by sorry


end four_digit_number_problem_l788_78873


namespace population_increase_rate_is_10_percent_l788_78818

/-- The population increase rate given initial and final populations -/
def population_increase_rate (initial : ℕ) (final : ℕ) : ℚ :=
  (final - initial : ℚ) / initial * 100

/-- Theorem: The population increase rate is 10% given the conditions -/
theorem population_increase_rate_is_10_percent :
  let initial_population := 260
  let final_population := 286
  population_increase_rate initial_population final_population = 10 := by
  sorry

end population_increase_rate_is_10_percent_l788_78818


namespace statements_equivalence_l788_78899

variable (α : Type)
variable (A B : α → Prop)

theorem statements_equivalence :
  (∀ x, A x → B x) ↔
  (∀ x, A x → B x) ∧
  (∀ x, A x → B x) ∧
  (∀ x, A x → B x) ∧
  (∀ x, ¬ B x → ¬ A x) :=
by sorry

end statements_equivalence_l788_78899


namespace geometric_sequence_problem_l788_78874

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

-- Theorem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 = 1 →
  a 8 = a 6 + 2 * a 4 →
  a 6 = 4 :=
by
  sorry

end geometric_sequence_problem_l788_78874


namespace rope_length_difference_l788_78803

/-- Given three ropes with lengths in ratio 4 : 5 : 6, where the shortest is 80 meters,
    prove that the sum of the longest and shortest is 100 meters more than the middle. -/
theorem rope_length_difference (shortest middle longest : ℝ) : 
  shortest = 80 ∧ 
  5 * shortest = 4 * middle ∧ 
  6 * shortest = 4 * longest →
  longest + shortest = middle + 100 := by
  sorry

end rope_length_difference_l788_78803


namespace no_fixed_points_iff_a_in_range_l788_78893

/-- A function f: ℝ → ℝ has no fixed points if for all x: ℝ, f x ≠ x -/
def has_no_fixed_points (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x ≠ x

/-- The quadratic function f(x) = x^2 + 2ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  x^2 + 2*a*x + 1

/-- Theorem stating that f(x) = x^2 + 2ax + 1 has no fixed points iff a ∈ (-1/2, 3/2) -/
theorem no_fixed_points_iff_a_in_range (a : ℝ) :
  has_no_fixed_points (f a) ↔ -1/2 < a ∧ a < 3/2 :=
sorry

end no_fixed_points_iff_a_in_range_l788_78893


namespace multiply_powers_of_x_l788_78847

theorem multiply_powers_of_x (x : ℝ) : 2 * (x^3) * (x^3) = 2 * (x^6) := by
  sorry

end multiply_powers_of_x_l788_78847


namespace larger_number_problem_l788_78820

theorem larger_number_problem (x y : ℝ) : 
  x - y = 7 → x + y = 45 → x = 26 ∧ x > y := by
  sorry

end larger_number_problem_l788_78820


namespace racks_fit_on_shelf_l788_78825

/-- Represents the number of CDs a single rack can hold -/
def cds_per_rack : ℕ := 8

/-- Represents the total number of CDs the shelf can hold -/
def total_cds : ℕ := 32

/-- Calculates the number of racks that can fit on the shelf -/
def racks_on_shelf : ℕ := total_cds / cds_per_rack

/-- Proves that the number of racks that can fit on the shelf is 4 -/
theorem racks_fit_on_shelf : racks_on_shelf = 4 := by
  sorry

end racks_fit_on_shelf_l788_78825


namespace tv_production_reduction_l788_78861

/-- Given a factory that produces televisions, calculate the percentage reduction in production from the first year to the second year. -/
theorem tv_production_reduction (daily_rate : ℕ) (second_year_total : ℕ) : 
  daily_rate = 10 →
  second_year_total = 3285 →
  (1 - (second_year_total : ℝ) / (daily_rate * 365 : ℝ)) * 100 = 10 := by
  sorry

#check tv_production_reduction

end tv_production_reduction_l788_78861
