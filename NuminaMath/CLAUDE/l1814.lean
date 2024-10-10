import Mathlib

namespace percentage_of_360_l1814_181411

theorem percentage_of_360 : (33 + 1 / 3 : ℚ) / 100 * 360 = 120 := by sorry

end percentage_of_360_l1814_181411


namespace paper_thickness_after_two_folds_l1814_181425

/-- The thickness of a paper after folding it in half a given number of times. -/
def thickness (initial : ℝ) (folds : ℕ) : ℝ :=
  initial * (2 ^ folds)

/-- Theorem: The thickness of a paper with initial thickness 0.1 mm after 2 folds is 0.4 mm. -/
theorem paper_thickness_after_two_folds :
  thickness 0.1 2 = 0.4 := by
  sorry

end paper_thickness_after_two_folds_l1814_181425


namespace third_person_profit_is_800_l1814_181449

/-- Calculates the third person's share of the profit in a joint business investment. -/
def third_person_profit (total_investment : ℕ) (investment_difference : ℕ) (total_profit : ℕ) : ℕ :=
  let first_investment := (total_investment - 3 * investment_difference) / 3
  let second_investment := first_investment + investment_difference
  let third_investment := second_investment + investment_difference
  (third_investment * total_profit) / total_investment

/-- Theorem stating that under the given conditions, the third person's profit share is 800. -/
theorem third_person_profit_is_800 :
  third_person_profit 9000 1000 1800 = 800 := by
  sorry

end third_person_profit_is_800_l1814_181449


namespace segment_ratios_l1814_181443

/-- Given line segments AC, AB, and BC, where AB consists of 3 parts and BC consists of 4 parts,
    prove the ratios of AB:AC and BC:AC. -/
theorem segment_ratios (AC AB BC : ℝ) (h1 : AB = 3) (h2 : BC = 4) (h3 : AC = AB + BC) :
  (AB / AC = 3 / 7) ∧ (BC / AC = 4 / 7) := by
  sorry

end segment_ratios_l1814_181443


namespace vector_parallel_implies_k_equals_three_l1814_181474

/-- Two vectors are parallel if their corresponding components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ), c ≠ 0 ∧ a.1 = c * b.1 ∧ a.2 = c * b.2

/-- Given two vectors a and b, where a depends on k, prove that k = 3 when a and b are parallel -/
theorem vector_parallel_implies_k_equals_three (k : ℝ) :
  let a : ℝ × ℝ := (2 - k, 3)
  let b : ℝ × ℝ := (2, -6)
  parallel a b → k = 3 := by
  sorry


end vector_parallel_implies_k_equals_three_l1814_181474


namespace interest_rate_calculation_l1814_181482

/-- Simple interest calculation function -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_rate_calculation (principal1 principal2 time1 time2 rate1 : ℝ) 
  (h1 : principal1 = 100)
  (h2 : principal2 = 600)
  (h3 : time1 = 48)
  (h4 : time2 = 4)
  (h5 : rate1 = 0.05)
  (h6 : simple_interest principal1 rate1 time1 = simple_interest principal2 ((10 : ℝ) / 100) time2) :
  ∃ (rate2 : ℝ), rate2 = (10 : ℝ) / 100 ∧ 
    simple_interest principal1 rate1 time1 = simple_interest principal2 rate2 time2 :=
by sorry

end interest_rate_calculation_l1814_181482


namespace triangle_side_a_equals_one_l1814_181413

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)

noncomputable def n (x : ℝ) : ℝ × ℝ := (-Real.cos x, Real.sqrt 3 * Real.cos x)

noncomputable def f (x : ℝ) : ℝ :=
  let m_vec := m x
  let n_vec := n x
  (m_vec.1 * (0.5 * m_vec.1 - n_vec.1)) + (m_vec.2 * (0.5 * m_vec.2 - n_vec.2))

theorem triangle_side_a_equals_one (A B C : ℝ) (a b c : ℝ) :
  f (B / 2) = 1 → b = 1 → c = Real.sqrt 3 →
  a = 1 :=
by sorry

end triangle_side_a_equals_one_l1814_181413


namespace parabola_properties_l1814_181421

-- Define the parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem parabola_properties (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hc : c > 1) 
  (h_point : parabola a b c 2 = 0) 
  (h_symmetry : -b / (2 * a) = 1/2) :
  abc < 0 ∧ 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ parabola a b c x₁ = a ∧ parabola a b c x₂ = a) ∧
  a < -1/2 := by
sorry

end parabola_properties_l1814_181421


namespace constant_fraction_iff_proportional_coefficients_l1814_181440

/-- A fraction of quadratic polynomials is constant if and only if the coefficients are proportional -/
theorem constant_fraction_iff_proportional_coefficients 
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) (h : a₂ ≠ 0) :
  (∃ k : ℝ, ∀ x : ℝ, (a₁ * x^2 + b₁ * x + c₁) / (a₂ * x^2 + b₂ * x + c₂) = k) ↔ 
  (∃ k : ℝ, a₁ = k * a₂ ∧ b₁ = k * b₂ ∧ c₁ = k * c₂) :=
sorry

end constant_fraction_iff_proportional_coefficients_l1814_181440


namespace subtracted_value_l1814_181433

theorem subtracted_value (N : ℝ) (x : ℝ) : 
  ((N - x) / 7 = 7) ∧ ((N - 2) / 13 = 4) → x = 5 := by
  sorry

end subtracted_value_l1814_181433


namespace parabola_intersection_distance_l1814_181468

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with equation y² = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola y² = 4x -/
def focus : Point := ⟨1, 0⟩

/-- The origin point -/
def origin : Point := ⟨0, 0⟩

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Main theorem -/
theorem parabola_intersection_distance 
  (A B : Point) 
  (hA : A ∈ Parabola) 
  (hB : B ∈ Parabola) 
  (hline : ∃ (m c : ℝ), A.y = m * A.x + c ∧ B.y = m * B.x + c ∧ focus.y = m * focus.x + c) 
  (harea : triangleArea A origin focus = 3 * triangleArea B origin focus) :
  distance A B = 16/3 := 
sorry

end parabola_intersection_distance_l1814_181468


namespace existence_of_index_l1814_181493

theorem existence_of_index (a : Fin 7 → ℝ) (h1 : a 1 = 0) (h7 : a 7 = 0) :
  ∃ k : Fin 5, (a k) + (a (k + 2)) ≤ (a (k + 1)) * Real.sqrt 3 := by
  sorry

end existence_of_index_l1814_181493


namespace f_five_l1814_181417

/-- A function satisfying f(xy) = 3xf(y) for all real x and y, with f(1) = 10 -/
def f : ℝ → ℝ :=
  sorry

/-- The functional equation for f -/
axiom f_eq (x y : ℝ) : f (x * y) = 3 * x * f y

/-- The value of f at 1 -/
axiom f_one : f 1 = 10

/-- The main theorem: f(5) = 150 -/
theorem f_five : f 5 = 150 := by
  sorry

end f_five_l1814_181417


namespace chocolate_box_problem_l1814_181455

theorem chocolate_box_problem (total : ℕ) (remaining : ℕ) : 
  (remaining = 28) →
  (total / 2 * 4 / 5 + total / 2 / 2 = total - remaining) →
  total = 80 := by
  sorry

end chocolate_box_problem_l1814_181455


namespace right_triangle_hypotenuse_l1814_181467

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 → b = 20 → c^2 = a^2 + b^2 → c = 25 := by
  sorry

end right_triangle_hypotenuse_l1814_181467


namespace no_real_solution_l1814_181486

theorem no_real_solution : ¬∃ (x : ℝ), |3*x + 1| + 6 = 0 := by
  sorry

end no_real_solution_l1814_181486


namespace system_solution_l1814_181454

theorem system_solution (x y u v : ℝ) : 
  x^2 + y^2 + u^2 + v^2 = 4 →
  x * y * u + y * u * v + u * v * x + v * x * y = -2 →
  x * y * u * v = -1 →
  ((x = 1 ∧ y = 1 ∧ u = 1 ∧ v = -1) ∨
   (x = 1 ∧ y = 1 ∧ u = -1 ∧ v = 1) ∨
   (x = 1 ∧ y = -1 ∧ u = 1 ∧ v = 1) ∨
   (x = -1 ∧ y = 1 ∧ u = 1 ∧ v = 1)) :=
by sorry

end system_solution_l1814_181454


namespace freds_spending_ratio_l1814_181488

/-- The ratio of Fred's movie spending to his weekly allowance -/
def movie_allowance_ratio (weekly_allowance : ℚ) (car_wash_earnings : ℚ) (final_amount : ℚ) : ℚ × ℚ :=
  let total_before_movies := final_amount + car_wash_earnings
  let movie_spending := total_before_movies - weekly_allowance
  (movie_spending, weekly_allowance)

/-- Theorem stating the ratio of Fred's movie spending to his weekly allowance -/
theorem freds_spending_ratio :
  let weekly_allowance : ℚ := 16
  let car_wash_earnings : ℚ := 6
  let final_amount : ℚ := 14
  let (numerator, denominator) := movie_allowance_ratio weekly_allowance car_wash_earnings final_amount
  numerator / denominator = 1 / 4 := by
  sorry

end freds_spending_ratio_l1814_181488


namespace necessary_not_sufficient_l1814_181480

/-- A function is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The product of two functions -/
def FunctionProduct (f g : ℝ → ℝ) : ℝ → ℝ :=
  fun x ↦ f x * g x

theorem necessary_not_sufficient :
  (∀ f g : ℝ → ℝ, (IsEven f ∧ IsEven g ∨ IsOdd f ∧ IsOdd g) →
    IsEven (FunctionProduct f g)) ∧
  (∃ f g : ℝ → ℝ, IsEven (FunctionProduct f g) ∧
    ¬(IsEven f ∧ IsEven g ∨ IsOdd f ∧ IsOdd g)) :=
by sorry

end necessary_not_sufficient_l1814_181480


namespace juniper_whiskers_l1814_181404

/-- Represents the number of whiskers for each cat -/
structure CatWhiskers where
  puffy : ℕ
  scruffy : ℕ
  buffy : ℕ
  juniper : ℕ

/-- Defines the conditions for the cat whisker problem -/
def whisker_conditions (c : CatWhiskers) : Prop :=
  c.buffy = 40 ∧
  c.puffy = 3 * c.juniper ∧
  c.puffy = c.scruffy / 2 ∧
  c.buffy = (c.puffy + c.scruffy + c.juniper) / 3

/-- Theorem stating that under the given conditions, Juniper has 12 whiskers -/
theorem juniper_whiskers (c : CatWhiskers) : 
  whisker_conditions c → c.juniper = 12 := by
  sorry

#check juniper_whiskers

end juniper_whiskers_l1814_181404


namespace dry_cleaning_time_is_ten_l1814_181450

def total_time : ℕ := 180 -- 3 hours = 180 minutes
def commute_time : ℕ := 30
def grocery_time : ℕ := 30
def dog_groomer_time : ℕ := 20
def cooking_time : ℕ := 90

def dry_cleaning_time : ℕ := total_time - commute_time - grocery_time - dog_groomer_time - cooking_time

theorem dry_cleaning_time_is_ten : dry_cleaning_time = 10 := by
  sorry

end dry_cleaning_time_is_ten_l1814_181450


namespace subtraction_with_division_l1814_181456

theorem subtraction_with_division : 3034 - (1002 / 20.04) = 2984 := by
  sorry

end subtraction_with_division_l1814_181456


namespace common_difference_is_two_l1814_181477

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

/-- The common difference of an arithmetic sequence is 2 given the condition -/
theorem common_difference_is_two (seq : ArithmeticSequence) 
    (h : seq.S 5 / 5 - seq.S 2 / 2 = 3) : 
    seq.a 2 - seq.a 1 = 2 := by
  sorry

end common_difference_is_two_l1814_181477


namespace daragh_initial_bears_l1814_181457

/-- The number of stuffed bears Daragh initially had -/
def initial_bears : ℕ := 20

/-- The number of favorite bears Daragh took out -/
def favorite_bears : ℕ := 8

/-- The number of sisters Daragh divided the remaining bears among -/
def num_sisters : ℕ := 3

/-- The number of bears Eden had before receiving more -/
def eden_bears_before : ℕ := 10

/-- The number of bears Eden had after receiving more -/
def eden_bears_after : ℕ := 14

theorem daragh_initial_bears :
  initial_bears = favorite_bears + (eden_bears_after - eden_bears_before) * num_sisters :=
by sorry

end daragh_initial_bears_l1814_181457


namespace no_common_sale_days_l1814_181476

def bookstore_sales : Set Nat :=
  {d | d ≤ 31 ∧ ∃ k, d = 4 * k}

def shoe_store_sales : Set Nat :=
  {d | d ≤ 31 ∧ ∃ n, d = 2 + 8 * n}

theorem no_common_sale_days : bookstore_sales ∩ shoe_store_sales = ∅ := by
  sorry

end no_common_sale_days_l1814_181476


namespace hockey_skates_fraction_l1814_181485

/-- Proves that the fraction of money spent on hockey skates is 1/2 --/
theorem hockey_skates_fraction (initial_amount pad_cost remaining : ℚ)
  (h1 : initial_amount = 150)
  (h2 : pad_cost = 50)
  (h3 : remaining = 25) :
  (initial_amount - pad_cost - remaining) / initial_amount = 1/2 := by
  sorry

end hockey_skates_fraction_l1814_181485


namespace kangaroo_koala_ratio_l1814_181414

theorem kangaroo_koala_ratio :
  let total_animals : ℕ := 216
  let num_kangaroos : ℕ := 180
  let num_koalas : ℕ := total_animals - num_kangaroos
  num_kangaroos / num_koalas = 5 := by
  sorry

end kangaroo_koala_ratio_l1814_181414


namespace multiply_and_subtract_problem_solution_l1814_181437

theorem multiply_and_subtract (a b c : ℕ) : a * c - b * c = (a - b) * c := by sorry

theorem problem_solution : 65 * 1313 - 25 * 1313 = 52520 := by sorry

end multiply_and_subtract_problem_solution_l1814_181437


namespace two_ladies_walk_l1814_181487

/-- The combined distance walked by two ladies in Central Park -/
def combined_distance (lady1_distance lady2_distance : ℝ) : ℝ :=
  lady1_distance + lady2_distance

/-- Theorem: The combined distance of two ladies is 12 miles when one walks twice as far as the other, and the second lady walks 4 miles -/
theorem two_ladies_walk :
  ∀ (lady1_distance lady2_distance : ℝ),
  lady2_distance = 4 →
  lady1_distance = 2 * lady2_distance →
  combined_distance lady1_distance lady2_distance = 12 :=
by
  sorry

end two_ladies_walk_l1814_181487


namespace pascal_triangle_interior_sum_l1814_181452

/-- The sum of interior numbers in a row of Pascal's Triangle -/
def sumInteriorNumbers (n : ℕ) : ℕ := 2^(n-1) - 2

theorem pascal_triangle_interior_sum :
  sumInteriorNumbers 6 = 30 →
  sumInteriorNumbers 8 = 126 := by
  sorry

end pascal_triangle_interior_sum_l1814_181452


namespace soccer_ball_max_height_l1814_181446

/-- The height of the soccer ball as a function of time -/
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

/-- Theorem stating that the maximum height reached by the soccer ball is 40 feet -/
theorem soccer_ball_max_height :
  ∃ (max : ℝ), max = 40 ∧ ∀ (t : ℝ), h t ≤ max :=
sorry

end soccer_ball_max_height_l1814_181446


namespace smallest_dual_base_palindrome_l1814_181436

/-- Checks if a number is a palindrome in the given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_base_palindrome :
  ∀ k : ℕ,
  k > 10 →
  (isPalindrome k 2 ∧ isPalindrome k 4) →
  k ≥ 17 ∧
  isPalindrome 17 2 ∧
  isPalindrome 17 4 := by
    sorry

end smallest_dual_base_palindrome_l1814_181436


namespace intersection_of_A_and_B_l1814_181490

def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B : Set ℝ := {x | 2 < x ∧ x < 4}

theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := by sorry

end intersection_of_A_and_B_l1814_181490


namespace rhombus_diagonals_not_necessarily_equal_l1814_181427

/-- Definition of a rhombus -/
structure Rhombus :=
  (sides : Fin 4 → ℝ)
  (equal_sides : ∀ i j : Fin 4, sides i = sides j)
  (perpendicular_diagonals : True)  -- We simplify this condition for the purpose of this problem

/-- Definition of diagonals of a rhombus -/
def diagonals (r : Rhombus) : Fin 2 → ℝ := sorry

/-- Theorem stating that the diagonals of a rhombus are not necessarily equal -/
theorem rhombus_diagonals_not_necessarily_equal :
  ¬ (∀ r : Rhombus, diagonals r 0 = diagonals r 1) :=
sorry

end rhombus_diagonals_not_necessarily_equal_l1814_181427


namespace circle_equation_point_on_circle_l1814_181444

/-- The standard equation of a circle with center (2, -1) passing through (-1, 3) -/
theorem circle_equation : 
  ∃ (x y : ℝ), (x - 2)^2 + (y + 1)^2 = 25 ∧ 
  ∀ (a b : ℝ), (a - 2)^2 + (b + 1)^2 = 25 ↔ (a, b) ∈ {(x, y) | (x - 2)^2 + (y + 1)^2 = 25} :=
by
  sorry

/-- The given point (-1, 3) satisfies the circle equation -/
theorem point_on_circle : (-1 - 2)^2 + (3 + 1)^2 = 25 :=
by
  sorry

end circle_equation_point_on_circle_l1814_181444


namespace least_positive_integer_multiple_of_53_l1814_181491

theorem least_positive_integer_multiple_of_53 :
  ∃ (x : ℕ+), 
    (∀ (y : ℕ+), y < x → ¬(53 ∣ (3 * y.val)^2 + 2 * 41 * (3 * y.val) + 41^2)) ∧
    (53 ∣ (3 * x.val)^2 + 2 * 41 * (3 * x.val) + 41^2) ∧
    x.val = 4 :=
by sorry

end least_positive_integer_multiple_of_53_l1814_181491


namespace customers_per_car_l1814_181497

/-- Proves that there are 5 customers in each car given the problem conditions --/
theorem customers_per_car :
  let num_cars : ℕ := 10
  let sports_sales : ℕ := 20
  let music_sales : ℕ := 30
  let total_sales : ℕ := sports_sales + music_sales
  let total_customers : ℕ := total_sales
  let customers_per_car : ℕ := total_customers / num_cars
  customers_per_car = 5 := by
  sorry

end customers_per_car_l1814_181497


namespace triangle_area_prove_triangle_area_l1814_181429

/-- Parabola equation: x^2 = 16y -/
def parabola (x y : ℝ) : Prop := x^2 = 16 * y

/-- Hyperbola equation: x^2 - y^2 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- Directrix of the parabola -/
def directrix : ℝ := -4

/-- Asymptotes of the hyperbola -/
def asymptote₁ (x : ℝ) : ℝ := x
def asymptote₂ (x : ℝ) : ℝ := -x

/-- Points where asymptotes intersect the directrix -/
def point₁ : ℝ × ℝ := (4, -4)
def point₂ : ℝ × ℝ := (-4, -4)

/-- The area of the triangle formed by the directrix and asymptotes -/
theorem triangle_area : ℝ := 16

/-- Proof that the area of the triangle is 16 -/
theorem prove_triangle_area : triangle_area = 16 := by sorry

end triangle_area_prove_triangle_area_l1814_181429


namespace concert_drive_distance_l1814_181409

/-- Calculates the remaining distance to drive given the total distance and the distance already driven. -/
def remaining_distance (total : ℕ) (driven : ℕ) : ℕ :=
  total - driven

/-- Theorem stating that for a total distance of 78 miles and a driven distance of 32 miles, 
    the remaining distance is 46 miles. -/
theorem concert_drive_distance : remaining_distance 78 32 = 46 := by
  sorry

end concert_drive_distance_l1814_181409


namespace pascal_triangle_elements_l1814_181430

/-- The number of elements in the nth row of Pascal's Triangle -/
def elementsInRow (n : ℕ) : ℕ := n + 1

/-- The sum of elements from row 0 to row n of Pascal's Triangle -/
def sumElements (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

theorem pascal_triangle_elements : sumElements 29 = 465 := by
  sorry

end pascal_triangle_elements_l1814_181430


namespace work_equivalence_first_group_size_l1814_181445

/-- The number of days it takes the first group to complete the work -/
def days_group1 : ℕ := 40

/-- The number of men in the second group -/
def men_group2 : ℕ := 20

/-- The number of days it takes the second group to complete the work -/
def days_group2 : ℕ := 68

/-- The number of men in the first group -/
def men_group1 : ℕ := 34

theorem work_equivalence :
  men_group1 * days_group1 = men_group2 * days_group2 :=
sorry

theorem first_group_size :
  men_group1 = (men_group2 * days_group2) / days_group1 :=
sorry

end work_equivalence_first_group_size_l1814_181445


namespace xiaopang_had_32_books_l1814_181492

/-- The number of books Xiaopang originally had -/
def xiaopang_books : ℕ := 32

/-- The number of books Xiaoya originally had -/
def xiaoya_books : ℕ := 16

/-- Theorem stating that Xiaopang originally had 32 books -/
theorem xiaopang_had_32_books :
  (xiaopang_books - 8 = xiaoya_books + 8) ∧
  (xiaopang_books + 4 = 3 * (xiaoya_books - 4)) →
  xiaopang_books = 32 := by
  sorry

end xiaopang_had_32_books_l1814_181492


namespace minBrokenSticks_correct_canFormSquare_15_not_canFormSquare_12_l1814_181495

/-- Given n sticks of lengths 1, 2, ..., n, this function returns the minimum number
    of sticks that need to be broken in half to form a square. If it's possible to
    form a square without breaking any sticks, it returns 0. -/
def minBrokenSticks (n : ℕ) : ℕ :=
  if n = 12 then 2
  else if n = 15 then 0
  else sorry

theorem minBrokenSticks_correct :
  (minBrokenSticks 12 = 2) ∧ (minBrokenSticks 15 = 0) := by sorry

/-- Function to check if it's possible to form a square from n sticks of lengths 1, 2, ..., n
    without breaking any sticks -/
def canFormSquare (n : ℕ) : Prop :=
  ∃ (a b c d : List ℕ), 
    (a ++ b ++ c ++ d).sum = n * (n + 1) / 2 ∧
    (∀ x ∈ a ++ b ++ c ++ d, x ≤ n) ∧
    a.sum = b.sum ∧ b.sum = c.sum ∧ c.sum = d.sum

theorem canFormSquare_15 : canFormSquare 15 := by sorry

theorem not_canFormSquare_12 : ¬ canFormSquare 12 := by sorry

end minBrokenSticks_correct_canFormSquare_15_not_canFormSquare_12_l1814_181495


namespace inequality_implication_l1814_181420

theorem inequality_implication (a b : ℝ) (h : a > b) : -6*a < -6*b := by
  sorry

end inequality_implication_l1814_181420


namespace tank_capacity_l1814_181428

theorem tank_capacity (initial_fill : Rat) (added_amount : Rat) (final_fill : Rat) :
  initial_fill = 3 / 4 →
  added_amount = 8 →
  final_fill = 9 / 10 →
  ∃ (capacity : Rat), capacity = 160 / 3 ∧ 
    final_fill * capacity - initial_fill * capacity = added_amount :=
by
  sorry

end tank_capacity_l1814_181428


namespace dust_storm_untouched_acres_l1814_181410

/-- The number of acres left untouched by a dust storm -/
def acres_untouched (total_acres dust_covered_acres : ℕ) : ℕ :=
  total_acres - dust_covered_acres

/-- Theorem stating that given a prairie of 65,057 acres and a dust storm covering 64,535 acres, 
    the number of acres left untouched is 522 -/
theorem dust_storm_untouched_acres : 
  acres_untouched 65057 64535 = 522 := by
  sorry

end dust_storm_untouched_acres_l1814_181410


namespace barney_towel_problem_l1814_181447

/-- The number of days without clean towels for Barney -/
def days_without_clean_towels (total_towels : ℕ) (towels_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  let towels_used_in_missed_week := towels_per_day * days_in_week
  let remaining_towels := total_towels - towels_used_in_missed_week
  let days_with_clean_towels := remaining_towels / towels_per_day
  days_in_week - days_with_clean_towels

/-- Theorem stating that Barney will not have clean towels for 5 days in the following week -/
theorem barney_towel_problem :
  days_without_clean_towels 18 2 7 = 5 := by
  sorry

end barney_towel_problem_l1814_181447


namespace pigeonhole_principle_balls_l1814_181415

theorem pigeonhole_principle_balls (red yellow blue : ℕ) :
  red > 0 ∧ yellow > 0 ∧ blue > 0 →
  ∃ n : ℕ, n = 4 ∧
    ∀ k : ℕ, k < n →
      ∃ f : Fin k → Fin 3,
        ∀ i j : Fin k, i ≠ j → f i = f j →
          ∃ m : ℕ, m ≥ n ∧
            ∀ g : Fin m → Fin 3,
              ∃ i j : Fin m, i ≠ j ∧ g i = g j :=
by sorry

end pigeonhole_principle_balls_l1814_181415


namespace inequality_solution_set_l1814_181494

theorem inequality_solution_set (x : ℝ) : 
  x ≠ 1 → (1 / (x - 1) ≥ -1 ↔ x ∈ Set.Ici 1 ∪ Set.Iic 0) :=
by sorry

end inequality_solution_set_l1814_181494


namespace inscribed_sphere_volume_l1814_181478

/-- A right circular cone with a sphere inscribed inside it. -/
structure ConeWithSphere where
  /-- The diameter of the cone's base in inches. -/
  base_diameter : ℝ
  /-- The vertex angle of the cross-section triangle in degrees. -/
  vertex_angle : ℝ
  /-- The sphere is tangent to the sides of the cone and rests on the table. -/
  sphere_tangent : Bool

/-- The volume of the inscribed sphere in cubic inches. -/
def sphere_volume (cone : ConeWithSphere) : ℝ := sorry

/-- Theorem stating the volume of the inscribed sphere for specific cone dimensions. -/
theorem inscribed_sphere_volume (cone : ConeWithSphere) 
  (h1 : cone.base_diameter = 24)
  (h2 : cone.vertex_angle = 90)
  (h3 : cone.sphere_tangent = true) :
  sphere_volume cone = 2304 * Real.pi := by sorry

end inscribed_sphere_volume_l1814_181478


namespace books_per_shelf_l1814_181475

theorem books_per_shelf
  (mystery_shelves : ℕ)
  (picture_shelves : ℕ)
  (total_books : ℕ)
  (h1 : mystery_shelves = 5)
  (h2 : picture_shelves = 4)
  (h3 : total_books = 54)
  (h4 : total_books % (mystery_shelves + picture_shelves) = 0)  -- Ensures even distribution
  : total_books / (mystery_shelves + picture_shelves) = 6 :=
by
  sorry

end books_per_shelf_l1814_181475


namespace total_beads_is_40_l1814_181458

-- Define the number of blue beads
def blue_beads : ℕ := 5

-- Define the number of red beads as twice the number of blue beads
def red_beads : ℕ := 2 * blue_beads

-- Define the number of white beads as the sum of blue and red beads
def white_beads : ℕ := blue_beads + red_beads

-- Define the number of silver beads
def silver_beads : ℕ := 10

-- Theorem: The total number of beads is 40
theorem total_beads_is_40 : 
  blue_beads + red_beads + white_beads + silver_beads = 40 := by
  sorry

end total_beads_is_40_l1814_181458


namespace remaining_note_denomination_l1814_181470

theorem remaining_note_denomination 
  (total_amount : ℕ) 
  (total_notes : ℕ) 
  (fifty_notes : ℕ) 
  (fifty_denomination : ℕ) :
  total_amount = 10350 →
  total_notes = 90 →
  fifty_notes = 77 →
  fifty_denomination = 50 →
  ∃ (remaining_denomination : ℕ),
    remaining_denomination * (total_notes - fifty_notes) = 
      total_amount - (fifty_notes * fifty_denomination) ∧
    remaining_denomination = 500 := by
  sorry

end remaining_note_denomination_l1814_181470


namespace induction_even_numbers_l1814_181469

theorem induction_even_numbers (P : ℕ → Prop) (k : ℕ) (h_k_even : Even k) (h_k_ge_2 : k ≥ 2) 
  (h_base : P 2) (h_inductive : ∀ m : ℕ, m ≥ 2 → Even m → P m → P (m + 2)) :
  (P k → P (k + 2)) ∧ ¬(P k → P (k + 1)) ∧ ¬(P k → P (2*k + 2)) ∧ ¬(P k → P (2*(k + 2))) :=
sorry

end induction_even_numbers_l1814_181469


namespace percentage_subtracted_from_b_l1814_181405

theorem percentage_subtracted_from_b (a b x m : ℝ) (p : ℝ) : 
  a > 0 ∧ b > 0 ∧
  a / b = 4 / 5 ∧
  x = a + 0.75 * a ∧
  m = b - (p / 100) * b ∧
  m / x = 0.14285714285714285 →
  p = 80 := by
sorry

end percentage_subtracted_from_b_l1814_181405


namespace walking_speed_proof_l1814_181465

def jack_speed (x : ℝ) := x^2 - 11*x - 22
def jill_distance (x : ℝ) := x^2 - 4*x - 12
def jill_time (x : ℝ) := x + 6

theorem walking_speed_proof :
  ∃ (x : ℝ), 
    (jack_speed x = jill_distance x / jill_time x) ∧
    (jack_speed x = 10) :=
by sorry

end walking_speed_proof_l1814_181465


namespace tom_hockey_games_l1814_181402

/-- The number of hockey games Tom attended this year -/
def games_this_year : ℕ := 4

/-- The number of hockey games Tom attended last year -/
def games_last_year : ℕ := 9

/-- The total number of hockey games Tom attended -/
def total_games : ℕ := games_this_year + games_last_year

theorem tom_hockey_games :
  total_games = 13 := by sorry

end tom_hockey_games_l1814_181402


namespace sam_puppies_l1814_181442

theorem sam_puppies (initial : ℝ) (given_away : ℝ) (h1 : initial = 6.0) (h2 : given_away = 2.0) :
  initial - given_away = 4.0 := by sorry

end sam_puppies_l1814_181442


namespace odds_against_C_l1814_181439

-- Define the type for horses
inductive Horse : Type
  | A
  | B
  | C

-- Define the race with no ties
def Race := Horse → ℕ

-- Define the odds against winning for each horse
def oddsAgainst (h : Horse) : ℚ :=
  match h with
  | Horse.A => 5/2
  | Horse.B => 3/1
  | Horse.C => 15/13  -- This is what we want to prove

-- Define the probability of winning for a horse given its odds against
def probWinning (odds : ℚ) : ℚ := 1 / (1 + odds)

-- State the theorem
theorem odds_against_C (race : Race) :
  (oddsAgainst Horse.A = 5/2) →
  (oddsAgainst Horse.B = 3/1) →
  (probWinning (oddsAgainst Horse.A) + probWinning (oddsAgainst Horse.B) + probWinning (oddsAgainst Horse.C) = 1) →
  oddsAgainst Horse.C = 15/13 := by
  sorry

end odds_against_C_l1814_181439


namespace extended_segment_coordinates_l1814_181471

/-- Given points A and B, and a point C on the line extending AB such that BC = 1/2 * AB,
    prove that the coordinates of C are (12, 12). -/
theorem extended_segment_coordinates :
  let A : ℝ × ℝ := (3, 3)
  let B : ℝ × ℝ := (9, 9)
  let C : ℝ × ℝ := (12, 12)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
  BC.1 = (1/2) * AB.1 ∧ BC.2 = (1/2) * AB.2 :=
by sorry

end extended_segment_coordinates_l1814_181471


namespace missing_digits_sum_l1814_181401

/-- Given an addition problem 7□8 + 2182 = 863□91 where □ represents a single digit (0-9),
    the sum of the two missing digits is 7. -/
theorem missing_digits_sum (d1 d2 : Nat) : 
  d1 ≤ 9 → d2 ≤ 9 → 
  708 + d1 * 10 + 2182 = 86300 + d2 * 10 + 91 →
  d1 + d2 = 7 := by
sorry

end missing_digits_sum_l1814_181401


namespace x_value_when_y_is_two_l1814_181434

theorem x_value_when_y_is_two (x y : ℚ) : 
  y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end x_value_when_y_is_two_l1814_181434


namespace proportional_set_l1814_181463

/-- A set of four positive real numbers is proportional if and only if
    the product of the extremes equals the product of the means. -/
def IsProportional (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a * d = b * c

/-- The set of line segments (2, 3, 4, 6) is proportional. -/
theorem proportional_set : IsProportional 2 3 4 6 := by
  sorry

end proportional_set_l1814_181463


namespace multiplication_subtraction_difference_l1814_181416

theorem multiplication_subtraction_difference (x : ℝ) (h : x = 10) : 3 * x - (20 - x) = 20 := by
  sorry

end multiplication_subtraction_difference_l1814_181416


namespace geometric_sequence_common_ratio_l1814_181453

/-- A geometric sequence with first term 1 and product of first three terms -8 has common ratio -2 -/
theorem geometric_sequence_common_ratio : 
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- geometric sequence property
    a 1 = 1 →                              -- first term is 1
    a 1 * a 2 * a 3 = -8 →                 -- product of first three terms is -8
    a 2 / a 1 = -2 :=                      -- common ratio is -2
by
  sorry

end geometric_sequence_common_ratio_l1814_181453


namespace max_value_of_y_l1814_181481

def y (x : ℝ) : ℝ := |x + 1| - 2 * |x| + |x - 2|

theorem max_value_of_y :
  ∃ (α : ℝ), α = 3 ∧ ∀ x, -1 ≤ x → x ≤ 2 → y x ≤ α :=
by sorry

end max_value_of_y_l1814_181481


namespace range_of_x_l1814_181438

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of f being even
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the property of f being decreasing on [0,+∞)
def IsDecreasingOnNonnegativeReals (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

-- State the theorem
theorem range_of_x (h1 : IsEven f) (h2 : IsDecreasingOnNonnegativeReals f) 
  (h3 : ∀ x > 0, f (Real.log x / Real.log 10) > f 1) :
  ∀ x > 0, f (Real.log x / Real.log 10) > f 1 → 1/10 < x ∧ x < 10 := by
  sorry

end range_of_x_l1814_181438


namespace equation_solution_l1814_181448

theorem equation_solution : ∃ x : ℕ, (81^20 + 81^20 + 81^20 + 81^20 + 81^20 + 81^20 = 3^x) ∧ x = 81 := by
  sorry

end equation_solution_l1814_181448


namespace cattle_breeder_milk_production_l1814_181403

/-- Calculates the weekly milk production for a given number of cows and daily milk production per cow. -/
def weekly_milk_production (num_cows : ℕ) (daily_production : ℕ) : ℕ :=
  num_cows * daily_production * 7

/-- Proves that the weekly milk production of 52 cows, each producing 1000 oz of milk per day, is 364,000 oz. -/
theorem cattle_breeder_milk_production :
  weekly_milk_production 52 1000 = 364000 := by
  sorry


end cattle_breeder_milk_production_l1814_181403


namespace tens_digit_of_13_pow_2047_l1814_181499

theorem tens_digit_of_13_pow_2047 : ∃ n : ℕ, 13^2047 ≡ 10 + n [ZMOD 100] :=
sorry

end tens_digit_of_13_pow_2047_l1814_181499


namespace consecutive_integers_product_120_l1814_181441

theorem consecutive_integers_product_120 :
  ∃ (a b c d e : ℤ),
    b = a + 1 ∧
    d = c + 1 ∧
    e = c + 2 ∧
    a * b = 120 ∧
    c * d * e = 120 ∧
    a + b + c + d + e = 37 :=
by sorry

end consecutive_integers_product_120_l1814_181441


namespace locus_of_point_M_l1814_181484

/-- The locus of point M given an ellipse and conditions on point P -/
theorem locus_of_point_M (x₀ y₀ x y : ℝ) : 
  (4 * x₀^2 + y₀^2 = 4) →  -- P(x₀, y₀) is on the ellipse
  ((0, -y₀) = (2*(x - x₀), -2*y)) →  -- PD = 2MD condition
  (x^2 + y^2 = 1) -- M(x, y) is on the unit circle
  := by sorry

end locus_of_point_M_l1814_181484


namespace algebra_drafting_not_geography_algebra_drafting_not_geography_eq_25_l1814_181496

theorem algebra_drafting_not_geography (total_algebra : ℕ) (both_algebra_drafting : ℕ) 
  (drafting_only : ℕ) (total_geography : ℕ) (both_algebra_drafting_geography : ℕ) : ℕ :=
  let algebra_only := total_algebra - both_algebra_drafting
  let total_one_subject := algebra_only + drafting_only
  let result := total_one_subject - both_algebra_drafting_geography
  
  have h1 : total_algebra = 30 := by sorry
  have h2 : both_algebra_drafting = 15 := by sorry
  have h3 : drafting_only = 12 := by sorry
  have h4 : total_geography = 8 := by sorry
  have h5 : both_algebra_drafting_geography = 2 := by sorry

  result

theorem algebra_drafting_not_geography_eq_25 : 
  algebra_drafting_not_geography 30 15 12 8 2 = 25 := by sorry

end algebra_drafting_not_geography_algebra_drafting_not_geography_eq_25_l1814_181496


namespace other_diagonal_length_l1814_181451

/-- A trapezoid with diagonals intersecting at a right angle -/
structure RightAngledTrapezoid where
  midline : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ
  diagonals_perpendicular : diagonal1 * diagonal2 = midline * midline * 2

/-- Theorem: In a right-angled trapezoid with midline 6.5 and one diagonal 12, the other diagonal is 5 -/
theorem other_diagonal_length (t : RightAngledTrapezoid) 
  (h1 : t.midline = 6.5) 
  (h2 : t.diagonal1 = 12) : 
  t.diagonal2 = 5 := by
sorry

end other_diagonal_length_l1814_181451


namespace total_coins_after_addition_initial_ratio_final_ratio_l1814_181472

/-- Represents a coin collection with gold and silver coins -/
structure CoinCollection where
  gold : ℕ
  silver : ℕ

/-- The initial state of the coin collection -/
def initial_collection : CoinCollection :=
  { gold := 30, silver := 90 }

/-- The final state of the coin collection after adding 15 gold coins -/
def final_collection : CoinCollection :=
  { gold := initial_collection.gold + 15, silver := initial_collection.silver }

/-- Theorem stating the total number of coins after the addition -/
theorem total_coins_after_addition :
  final_collection.gold + final_collection.silver = 135 := by
  sorry

/-- Theorem for the initial ratio of gold to silver coins -/
theorem initial_ratio :
  initial_collection.gold * 3 = initial_collection.silver := by
  sorry

/-- Theorem for the final ratio of gold to silver coins -/
theorem final_ratio :
  final_collection.gold * 2 = final_collection.silver := by
  sorry

end total_coins_after_addition_initial_ratio_final_ratio_l1814_181472


namespace parabola_line_tangency_l1814_181435

/-- 
Given a parabola y = ax^2 + 6 and a line y = 2x + k, where k is a constant,
this theorem states the condition for tangency between the parabola and the line.
-/
theorem parabola_line_tangency (a k : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + 6 ∧ y = 2 * x + k) →
  (k ≠ 6) →
  (a = -1 / (k - 6)) :=
by sorry

end parabola_line_tangency_l1814_181435


namespace pure_imaginary_implies_a_zero_l1814_181479

-- Define a complex number z
def z (a : ℝ) : ℂ := Complex.I * (1 + a * Complex.I)

-- State the theorem
theorem pure_imaginary_implies_a_zero (a : ℝ) :
  (∃ b : ℝ, z a = Complex.I * b) → a = 0 := by
  sorry

end pure_imaginary_implies_a_zero_l1814_181479


namespace solve_for_y_l1814_181431

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 4*x = y + 5) (h2 : x = 7) : y = 16 := by
  sorry

end solve_for_y_l1814_181431


namespace sqrt_sum_division_l1814_181464

theorem sqrt_sum_division (x y z : ℝ) : (2 * Real.sqrt 24 + 3 * Real.sqrt 6) / Real.sqrt 3 = 7 * Real.sqrt 2 := by
  sorry

end sqrt_sum_division_l1814_181464


namespace quadratic_integer_solutions_count_l1814_181462

theorem quadratic_integer_solutions_count : 
  ∃! (S : Finset ℚ), 
    (∀ k ∈ S, |k| < 100 ∧ 
      ∃! (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ 
        3 * (x₁ : ℚ)^2 + k * x₁ + 8 = 0 ∧ 
        3 * (x₂ : ℚ)^2 + k * x₂ + 8 = 0) ∧
    Finset.card S = 8 :=
by sorry

end quadratic_integer_solutions_count_l1814_181462


namespace regular_polygon_30_degree_central_angle_has_12_sides_l1814_181412

/-- Theorem: A regular polygon with a central angle of 30° has 12 sides. -/
theorem regular_polygon_30_degree_central_angle_has_12_sides :
  ∀ (n : ℕ) (central_angle : ℝ),
    central_angle = 30 →
    (360 : ℝ) / central_angle = n →
    n = 12 :=
by sorry

end regular_polygon_30_degree_central_angle_has_12_sides_l1814_181412


namespace profit_calculation_l1814_181418

/-- The number of pens John buys for $8 -/
def pens_bought : ℕ := 5

/-- The price John pays for pens_bought pens -/
def buy_price : ℚ := 8

/-- The number of pens John sells for $10 -/
def pens_sold : ℕ := 4

/-- The price John receives for pens_sold pens -/
def sell_price : ℚ := 10

/-- The desired profit -/
def target_profit : ℚ := 120

/-- The minimum number of pens John needs to sell to make the target profit -/
def min_pens_to_sell : ℕ := 134

theorem profit_calculation :
  ↑min_pens_to_sell * (sell_price / pens_sold - buy_price / pens_bought) ≥ target_profit ∧
  ∀ n : ℕ, n < min_pens_to_sell → ↑n * (sell_price / pens_sold - buy_price / pens_bought) < target_profit :=
by sorry

end profit_calculation_l1814_181418


namespace stating_wholesale_cost_calculation_l1814_181483

/-- The wholesale cost of a sleeping bag -/
def wholesale_cost : ℝ := 24.56

/-- The retailer's profit percentage -/
def profit_percentage : ℝ := 0.14

/-- The selling price of a sleeping bag -/
def selling_price : ℝ := 28

/-- 
Theorem stating that the wholesale cost is correct given the profit percentage and selling price
-/
theorem wholesale_cost_calculation (ε : ℝ) (h : ε > 0) : 
  ∃ (W : ℝ), W > 0 ∧ abs (W - wholesale_cost) < ε ∧ 
  W * (1 + profit_percentage) = selling_price :=
sorry

end stating_wholesale_cost_calculation_l1814_181483


namespace easter_egg_distribution_l1814_181489

theorem easter_egg_distribution (baskets : ℕ) (eggs_per_basket : ℕ) (people : ℕ) :
  baskets = 15 →
  eggs_per_basket = 12 →
  people = 20 →
  (baskets * eggs_per_basket) / people = 9 := by
  sorry

end easter_egg_distribution_l1814_181489


namespace inequality_proof_l1814_181424

theorem inequality_proof (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c := by
  sorry

end inequality_proof_l1814_181424


namespace car_speed_equation_l1814_181406

/-- Given a car traveling at speed v km/h, prove that v satisfies the equation
    v = 3600 / 49, if it takes 4 seconds longer to travel 1 km at speed v
    than at 80 km/h. -/
theorem car_speed_equation (v : ℝ) : v > 0 →
  (3600 / v = 3600 / 80 + 4) → v = 3600 / 49 := by
  sorry

end car_speed_equation_l1814_181406


namespace shift_left_sum_l1814_181460

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a quadratic function horizontally -/
def shift_left (f : QuadraticFunction) (units : ℝ) : QuadraticFunction :=
  QuadraticFunction.mk
    f.a
    (f.b + 2 * f.a * units)
    (f.a * units^2 + f.b * units + f.c)

theorem shift_left_sum (f : QuadraticFunction) :
  let g := shift_left f 6
  g.a + g.b + g.c = 156 :=
by
  sorry

end shift_left_sum_l1814_181460


namespace modulus_of_z_l1814_181407

-- Define the complex number z
variable (z : ℂ)

-- Define the condition z(1-i) = 2i
def condition : Prop := z * (1 - Complex.I) = 2 * Complex.I

-- Theorem statement
theorem modulus_of_z (h : condition z) : Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_z_l1814_181407


namespace expansion_coefficient_sum_l1814_181498

theorem expansion_coefficient_sum (n : ℕ) : 
  (∀ a b : ℝ, (3*a + 5*b)^n = 2^15) → n = 5 := by
  sorry

end expansion_coefficient_sum_l1814_181498


namespace p_satisfies_conditions_l1814_181422

/-- A quadratic polynomial that satisfies specific conditions -/
def p (x : ℝ) : ℝ := -3 * x^2 - 9 * x + 84

/-- Theorem stating that p satisfies the required conditions -/
theorem p_satisfies_conditions :
  p (-7) = 0 ∧ p 4 = 0 ∧ p 5 = -36 := by
  sorry

end p_satisfies_conditions_l1814_181422


namespace inequality_proof_l1814_181408

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hab : a ≤ b) (hbc : b ≤ c) (hcd : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := by
  sorry

end inequality_proof_l1814_181408


namespace min_average_of_four_integers_l1814_181419

theorem min_average_of_four_integers (a b c d : ℕ+) 
  (ha : a = 3 * b)
  (hc : c = b + 2)
  (hd : d ≥ 2) :
  (a + b + c + d : ℚ) / 4 ≥ 9/4 :=
sorry

end min_average_of_four_integers_l1814_181419


namespace richmond_tigers_ticket_sales_l1814_181423

theorem richmond_tigers_ticket_sales (total_tickets second_half_tickets : ℕ) 
    (h1 : total_tickets = 9570)
    (h2 : second_half_tickets = 5703) :
  total_tickets - second_half_tickets = 3867 := by
  sorry

end richmond_tigers_ticket_sales_l1814_181423


namespace reflection_segment_length_d_to_d_prime_length_l1814_181432

/-- The length of the segment from a point to its reflection over the x-axis --/
theorem reflection_segment_length (x y : ℝ) : 
  let d : ℝ × ℝ := (x, y)
  let d_reflected : ℝ × ℝ := (x, -y)
  Real.sqrt ((d.1 - d_reflected.1)^2 + (d.2 - d_reflected.2)^2) = 2 * |y| :=
by sorry

/-- The length of the segment from D(-5, 3) to its reflection D' over the x-axis is 6 --/
theorem d_to_d_prime_length : 
  let d : ℝ × ℝ := (-5, 3)
  let d_reflected : ℝ × ℝ := (-5, -3)
  Real.sqrt ((d.1 - d_reflected.1)^2 + (d.2 - d_reflected.2)^2) = 6 :=
by sorry

end reflection_segment_length_d_to_d_prime_length_l1814_181432


namespace line_plane_relationship_l1814_181426

-- Define the types for lines and planes
variable (L P : Type*)

-- Define the perpendicular relationship between lines
variable (perp_line : L → L → Prop)

-- Define the perpendicular relationship between a line and a plane
variable (perp_line_plane : L → P → Prop)

-- Define the parallel relationship between a line and a plane
variable (parallel : L → P → Prop)

-- Define the subset relationship between a line and a plane
variable (subset : L → P → Prop)

-- State the theorem
theorem line_plane_relationship 
  (a b : L) (α : P)
  (h1 : perp_line a b)
  (h2 : perp_line_plane b α) :
  subset a α ∨ parallel a α :=
sorry

end line_plane_relationship_l1814_181426


namespace younger_major_A_probability_l1814_181400

structure GraduatingClass where
  maleProportion : Real
  majorAProb : Real
  majorBProb : Real
  majorCProb : Real
  maleOlderProb : Real
  femaleOlderProb : Real
  majorAOlderProb : Real
  majorBOlderProb : Real
  majorCOlderProb : Real

def probabilityYoungerMajorA (gc : GraduatingClass) : Real :=
  gc.majorAProb * (1 - gc.majorAOlderProb)

theorem younger_major_A_probability (gc : GraduatingClass) 
  (h1 : gc.maleProportion = 0.4)
  (h2 : gc.majorAProb = 0.5)
  (h3 : gc.majorBProb = 0.3)
  (h4 : gc.majorCProb = 0.2)
  (h5 : gc.maleOlderProb = 0.5)
  (h6 : gc.femaleOlderProb = 0.3)
  (h7 : gc.majorAOlderProb = 0.6)
  (h8 : gc.majorBOlderProb = 0.4)
  (h9 : gc.majorCOlderProb = 0.2) :
  probabilityYoungerMajorA gc = 0.2 := by
  sorry

#check younger_major_A_probability

end younger_major_A_probability_l1814_181400


namespace coin_combination_l1814_181459

/-- Represents the number of different coin values that can be obtained -/
def different_values (five_cent : ℕ) (twenty_five_cent : ℕ) : ℕ :=
  75 - 4 * five_cent

theorem coin_combination (five_cent : ℕ) (twenty_five_cent : ℕ) :
  five_cent + twenty_five_cent = 15 →
  different_values five_cent twenty_five_cent = 27 →
  twenty_five_cent = 3 := by
sorry

end coin_combination_l1814_181459


namespace quadratic_integer_expression_l1814_181473

theorem quadratic_integer_expression (A B C : ℤ) :
  ∃ (k l m : ℚ), 
    (∀ x, A * x^2 + B * x + C = k * (x * (x - 1)) / 2 + l * x + m) ∧
    ((∀ x : ℤ, ∃ y : ℤ, A * x^2 + B * x + C = y) ↔ 
      (∃ (k' l' m' : ℤ), k = k' ∧ l = l' ∧ m = m')) :=
by sorry

end quadratic_integer_expression_l1814_181473


namespace remainder_problem_l1814_181461

theorem remainder_problem (y : ℤ) : 
  ∃ (k : ℤ), y = 276 * k + 42 → ∃ (m : ℤ), y = 23 * m + 19 := by
sorry

end remainder_problem_l1814_181461


namespace intersection_possibilities_l1814_181466

-- Define the sets P and Q
variable (P Q : Set ℕ)

-- Define the function f
def f (t : ℕ) : ℕ := t^2

-- State the theorem
theorem intersection_possibilities (h1 : Q = {1, 4}) 
  (h2 : ∀ t ∈ P, f t ∈ Q) : 
  P ∩ Q = {1} ∨ P ∩ Q = ∅ := by
sorry

end intersection_possibilities_l1814_181466
