import Mathlib

namespace NUMINAMATH_CALUDE_cubic_equation_has_real_root_l1918_191895

theorem cubic_equation_has_real_root :
  ∃ (x : ℝ), x^3 + 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_has_real_root_l1918_191895


namespace NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l1918_191883

/-- A function f defined piecewise on the real numbers. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then 2^x + 1 else -x^2 + a*x + 1

/-- The theorem stating the range of 'a' for which f is increasing on ℝ. -/
theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) → 2 ≤ a ∧ a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l1918_191883


namespace NUMINAMATH_CALUDE_tims_change_l1918_191840

/-- Tim's change calculation -/
theorem tims_change (initial_amount : ℕ) (spent_amount : ℕ) (change : ℕ) : 
  initial_amount = 50 → spent_amount = 45 → change = initial_amount - spent_amount → change = 5 := by
  sorry

end NUMINAMATH_CALUDE_tims_change_l1918_191840


namespace NUMINAMATH_CALUDE_range_of_a_l1918_191864

theorem range_of_a (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 3) 
  (sum_sq : a^2 + 2*b^2 + 3*c^2 + 6*d^2 = 5) : 
  1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1918_191864


namespace NUMINAMATH_CALUDE_perimeter_of_figure_l1918_191819

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A triangle defined by three points -/
structure Triangle :=
  (A B C : Point)

/-- A square defined by four points -/
structure Square :=
  (E H I J : Point)

/-- The figure ABCDEFGHIJ -/
structure Figure :=
  (A B C D E F G H I J : Point)

/-- Definition of an equilateral triangle -/
def isEquilateral (t : Triangle) : Prop :=
  sorry

/-- Definition of a midpoint -/
def isMidpoint (M A B : Point) : Prop :=
  sorry

/-- Definition of a square -/
def isSquare (s : Square) : Prop :=
  sorry

/-- Distance between two points -/
def distance (A B : Point) : ℝ :=
  sorry

/-- Perimeter of the figure -/
def perimeter (fig : Figure) : ℝ :=
  sorry

/-- Main theorem -/
theorem perimeter_of_figure (fig : Figure) :
  isEquilateral ⟨fig.A, fig.B, fig.C⟩ →
  isEquilateral ⟨fig.A, fig.D, fig.E⟩ →
  isEquilateral ⟨fig.E, fig.F, fig.G⟩ →
  isMidpoint fig.D fig.A fig.C →
  isMidpoint fig.G fig.A fig.E →
  isSquare ⟨fig.E, fig.H, fig.I, fig.J⟩ →
  distance fig.E fig.J = distance fig.D fig.E →
  distance fig.A fig.B = 6 →
  perimeter fig = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_figure_l1918_191819


namespace NUMINAMATH_CALUDE_chord_length_sqrt3_line_l1918_191811

/-- A line in the form y = mx --/
structure Line where
  m : ℝ

/-- A circle in the form (x - h)^2 + (y - k)^2 = r^2 --/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The length of a chord formed by the intersection of a line and a circle --/
def chordLength (l : Line) (c : Circle) : ℝ :=
  sorry

theorem chord_length_sqrt3_line (c : Circle) :
  c.h = 2 ∧ c.k = 0 ∧ c.r = 2 →
  chordLength { m := Real.sqrt 3 } c = 2 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_sqrt3_line_l1918_191811


namespace NUMINAMATH_CALUDE_y_decreases_as_x_increases_l1918_191850

/-- A linear function y = -2x - 3 -/
def f (x : ℝ) : ℝ := -2 * x - 3

/-- Theorem: For any two points on the graph of f, if the x-coordinate of the first point
    is less than the x-coordinate of the second point, then the y-coordinate of the first point
    is greater than the y-coordinate of the second point. -/
theorem y_decreases_as_x_increases (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : f x₁ = y₁) 
  (h2 : f x₂ = y₂) 
  (h3 : x₁ < x₂) : 
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y_decreases_as_x_increases_l1918_191850


namespace NUMINAMATH_CALUDE_shortest_distance_proof_l1918_191870

/-- Given a body moving on a horizontal plane, prove that with displacements of 4 meters
    along the x-axis and 3 meters along the y-axis, the shortest distance between
    the initial and final points is 5 meters. -/
theorem shortest_distance_proof (x y : ℝ) (hx : x = 4) (hy : y = 3) :
  Real.sqrt (x^2 + y^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_proof_l1918_191870


namespace NUMINAMATH_CALUDE_f_2018_eq_l1918_191886

open Real

/-- Sequence of functions defined recursively --/
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => λ x => sin x - cos x
  | n + 1 => λ x => deriv (f n) x

/-- The 2018th function in the sequence equals -sin(x) + cos(x) --/
theorem f_2018_eq (x : ℝ) : f 2018 x = -sin x + cos x := by
  sorry

end NUMINAMATH_CALUDE_f_2018_eq_l1918_191886


namespace NUMINAMATH_CALUDE_division_problem_l1918_191821

theorem division_problem (x y q : ℕ) : 
  y - x = 1375 →
  y = 1632 →
  y = q * x + 15 →
  q = 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1918_191821


namespace NUMINAMATH_CALUDE_intersection_equals_N_l1918_191822

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x > 0, y = x^2}
def N : Set ℝ := {y | ∃ x > 0, y = x + 2}

-- State the theorem
theorem intersection_equals_N : M ∩ N = N := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_N_l1918_191822


namespace NUMINAMATH_CALUDE_teenage_group_size_l1918_191892

theorem teenage_group_size (total_bill : ℝ) (individual_cost : ℝ) (gratuity_rate : ℝ) :
  total_bill = 840 →
  individual_cost = 100 →
  gratuity_rate = 0.2 →
  ∃ n : ℕ, n = 7 ∧ total_bill = (individual_cost * n) * (1 + gratuity_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_teenage_group_size_l1918_191892


namespace NUMINAMATH_CALUDE_total_percentage_increase_l1918_191876

/-- Calculates the total percentage increase in a purchase of three items given their initial and final prices. -/
theorem total_percentage_increase
  (book_initial : ℝ) (book_final : ℝ)
  (album_initial : ℝ) (album_final : ℝ)
  (poster_initial : ℝ) (poster_final : ℝ)
  (h1 : book_initial = 300)
  (h2 : book_final = 480)
  (h3 : album_initial = 15)
  (h4 : album_final = 20)
  (h5 : poster_initial = 5)
  (h6 : poster_final = 10) :
  (((book_final + album_final + poster_final) - (book_initial + album_initial + poster_initial)) / (book_initial + album_initial + poster_initial)) * 100 = 59.375 := by
  sorry

end NUMINAMATH_CALUDE_total_percentage_increase_l1918_191876


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1918_191806

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying the condition,
    prove that the sum of specific terms equals 2502.5. -/
theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_condition : a 3 + a 4 + a 10 + a 11 = 2002) :
  a 1 + a 5 + a 7 + a 9 + a 13 = 2502.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1918_191806


namespace NUMINAMATH_CALUDE_equation_sum_zero_l1918_191825

theorem equation_sum_zero (a b c : ℝ) 
  (h1 : a + b / c = 1) 
  (h2 : b + c / a = 1) 
  (h3 : c + a / b = 1) : 
  a * b + b * c + c * a = 0 := by
sorry

end NUMINAMATH_CALUDE_equation_sum_zero_l1918_191825


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1918_191815

theorem sufficient_not_necessary (x : ℝ) : 
  (∀ x, x > 1 → x^2 + x - 2 > 0) ∧ 
  (∃ x, x^2 + x - 2 > 0 ∧ x ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1918_191815


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l1918_191841

theorem intersection_implies_a_value (A B : Set ℝ) (a : ℝ) : 
  A = {-1, 1, 3} →
  B = {a + 2, a^2 + 4} →
  A ∩ B = {1} →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l1918_191841


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1918_191889

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 9/16) 
  (h2 : x - y = 5/16) : 
  x^2 - y^2 = 45/256 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1918_191889


namespace NUMINAMATH_CALUDE_percentage_of_mathematicians_in_it_l1918_191861

theorem percentage_of_mathematicians_in_it (total : ℝ) (mathematicians : ℝ) 
  (h1 : mathematicians > 0) 
  (h2 : total > mathematicians) 
  (h3 : 0.7 * mathematicians = 0.07 * total) : 
  mathematicians / total = 0.1 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_mathematicians_in_it_l1918_191861


namespace NUMINAMATH_CALUDE_cos_theta_value_l1918_191868

theorem cos_theta_value (x y : ℝ) (θ : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (hθ : θ ∈ Set.Ioo (π/4) (π/2))
  (h1 : y / Real.sin θ = x / Real.cos θ)
  (h2 : 10 / (x^2 + y^2) = 3 / (x * y)) :
  Real.cos θ = Real.sqrt 10 / 10 := by
sorry

end NUMINAMATH_CALUDE_cos_theta_value_l1918_191868


namespace NUMINAMATH_CALUDE_prob_at_least_one_man_l1918_191857

/-- The probability of selecting at least one man when choosing 5 people at random from a group of 12 men and 8 women -/
theorem prob_at_least_one_man (total_people : ℕ) (men : ℕ) (women : ℕ) (selection_size : ℕ) :
  total_people = men + women →
  men = 12 →
  women = 8 →
  selection_size = 5 →
  (1 : ℚ) - (women.choose selection_size : ℚ) / (total_people.choose selection_size : ℚ) = 687 / 692 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_man_l1918_191857


namespace NUMINAMATH_CALUDE_problem_statement_l1918_191842

theorem problem_statement (n : ℕ) (x m : ℝ) :
  let p := x^2 - 2*x - 8 ≤ 0
  let q := |x - 2| ≤ m
  (∀ k : ℕ, k ≤ n → ((-1:ℝ)^k * (n.choose k) = (-1)^n * (n.choose (n-k)))) →
  (
    (m = 3 ∧ p ∧ q) → -1 ≤ x ∧ x ≤ 4
  ) ∧
  (
    (∀ y : ℝ, (y^2 - 2*y - 8 ≤ 0) → |y - 2| ≤ m) → m ≥ 4
  ) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1918_191842


namespace NUMINAMATH_CALUDE_commission_calculation_l1918_191845

/-- Calculates the commission amount given a commission rate and total sales -/
def calculate_commission (rate : ℚ) (sales : ℚ) : ℚ :=
  rate * sales

theorem commission_calculation :
  let rate : ℚ := 25 / 1000  -- 2.5% expressed as a rational number
  let sales : ℚ := 600
  calculate_commission rate sales = 15 := by
  sorry

end NUMINAMATH_CALUDE_commission_calculation_l1918_191845


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1918_191880

theorem system_solution_ratio (x y z a b : ℝ) 
  (eq1 : 4 * x - 3 * y + z = a)
  (eq2 : 6 * y - 8 * x - 2 * z = b)
  (b_nonzero : b ≠ 0)
  (has_solution : ∃ (x y z : ℝ), 4 * x - 3 * y + z = a ∧ 6 * y - 8 * x - 2 * z = b) :
  a / b = -1 / 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1918_191880


namespace NUMINAMATH_CALUDE_circles_intersect_implies_equilateral_l1918_191851

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c

/-- Predicate that checks if any two circles intersect -/
def circlesIntersect (t : Triangle) : Prop :=
  t.c/2 ≤ t.a/4 + t.b/4 ∧ t.a/2 ≤ t.b/4 + t.c/4 ∧ t.b/2 ≤ t.c/4 + t.a/4

/-- Predicate that checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- Theorem stating that if circles drawn around midpoints of a triangle's sides
    with radii 1/4 of the side lengths intersect, then the triangle is equilateral -/
theorem circles_intersect_implies_equilateral (t : Triangle) :
  circlesIntersect t → isEquilateral t :=
by
  sorry


end NUMINAMATH_CALUDE_circles_intersect_implies_equilateral_l1918_191851


namespace NUMINAMATH_CALUDE_saras_quarters_l1918_191843

theorem saras_quarters (initial_quarters final_quarters : ℕ) 
  (h1 : initial_quarters = 21)
  (h2 : final_quarters = 70) : 
  final_quarters - initial_quarters = 49 := by
  sorry

end NUMINAMATH_CALUDE_saras_quarters_l1918_191843


namespace NUMINAMATH_CALUDE_triangle_positive_number_placement_l1918_191805

theorem triangle_positive_number_placement 
  (A B C : ℝ × ℝ) -- Vertices of the triangle
  (AB BC CA : ℝ)  -- Lengths of the sides
  (h_pos_AB : AB > 0)
  (h_pos_BC : BC > 0)
  (h_pos_CA : CA > 0)
  (h_triangle : AB + BC > CA ∧ BC + CA > AB ∧ CA + AB > BC) -- Triangle inequality
  : ∃ x y z : ℝ, 
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    AB = x + y ∧
    BC = y + z ∧
    CA = z + x :=
sorry

end NUMINAMATH_CALUDE_triangle_positive_number_placement_l1918_191805


namespace NUMINAMATH_CALUDE_ab_values_l1918_191812

theorem ab_values (a b : ℝ) (h : a^2*b^2 + a^2 + b^2 + 1 = 4*a*b) :
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1) := by
  sorry

end NUMINAMATH_CALUDE_ab_values_l1918_191812


namespace NUMINAMATH_CALUDE_dennis_initial_amount_l1918_191820

theorem dennis_initial_amount (shirt_cost change_received : ℕ) : 
  shirt_cost = 27 → change_received = 23 → shirt_cost + change_received = 50 := by
  sorry

end NUMINAMATH_CALUDE_dennis_initial_amount_l1918_191820


namespace NUMINAMATH_CALUDE_solution_sum_l1918_191894

theorem solution_sum (a b x y : ℝ) : 
  x = 2 ∧ y = -1 ∧ 
  a * x - 2 * y = 4 ∧ 
  3 * x + b * y = -7 →
  a + b = 14 := by
sorry

end NUMINAMATH_CALUDE_solution_sum_l1918_191894


namespace NUMINAMATH_CALUDE_length_of_diagonal_l1918_191882

/-- Given a quadrilateral ABCD with specific side lengths, prove the length of AC -/
theorem length_of_diagonal (AB DC AD : ℝ) (h1 : AB = 15) (h2 : DC = 24) (h3 : AD = 7) :
  ∃ AC : ℝ, abs (AC - Real.sqrt (417 + 112 * Real.sqrt 6)) < 0.05 :=
sorry

end NUMINAMATH_CALUDE_length_of_diagonal_l1918_191882


namespace NUMINAMATH_CALUDE_max_fruits_is_34_l1918_191800

/-- Represents the weight of an apple in grams -/
def apple_weight : ℕ := 300

/-- Represents the weight of a pear in grams -/
def pear_weight : ℕ := 200

/-- Represents the maximum weight Ana's bag can hold in grams -/
def bag_capacity : ℕ := 7000

/-- Represents the constraint on the number of apples and pears -/
def weight_constraint (m p : ℕ) : Prop :=
  m * apple_weight + p * pear_weight ≤ bag_capacity

/-- Represents the total number of fruits -/
def total_fruits (m p : ℕ) : ℕ := m + p

/-- Theorem stating that the maximum number of fruits Ana can buy is 34 -/
theorem max_fruits_is_34 : 
  ∃ (m p : ℕ), weight_constraint m p ∧ m > 0 ∧ p > 0 ∧
  total_fruits m p = 34 ∧
  ∀ (m' p' : ℕ), weight_constraint m' p' ∧ m' > 0 ∧ p' > 0 → 
    total_fruits m' p' ≤ 34 :=
sorry

end NUMINAMATH_CALUDE_max_fruits_is_34_l1918_191800


namespace NUMINAMATH_CALUDE_divisibility_problem_solutions_l1918_191859

/-- The set of solutions for the divisibility problem -/
def SolutionSet : Set (ℕ × ℕ) := {(1, 1), (1, 5), (5, 1)}

/-- The divisibility condition -/
def DivisibilityCondition (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ (m * n) ∣ ((2^(2^n) + 1) * (2^(2^m) + 1))

/-- Theorem stating that the SolutionSet contains all and only the pairs satisfying the divisibility condition -/
theorem divisibility_problem_solutions :
  ∀ m n : ℕ, DivisibilityCondition m n ↔ (m, n) ∈ SolutionSet := by
  sorry


end NUMINAMATH_CALUDE_divisibility_problem_solutions_l1918_191859


namespace NUMINAMATH_CALUDE_scheme1_higher_sale_price_l1918_191826

def original_price : ℝ := 15000

def scheme1_price (p : ℝ) : ℝ :=
  p * (1 - 0.25) * (1 - 0.15) * (1 - 0.05) * (1 + 0.30)

def scheme2_price (p : ℝ) : ℝ :=
  p * (1 - 0.40) * (1 + 0.30)

theorem scheme1_higher_sale_price :
  scheme1_price original_price > scheme2_price original_price :=
by sorry

end NUMINAMATH_CALUDE_scheme1_higher_sale_price_l1918_191826


namespace NUMINAMATH_CALUDE_exam_marks_lost_l1918_191898

theorem exam_marks_lost (total_questions : ℕ) (marks_per_correct : ℕ) (total_marks : ℕ) (correct_answers : ℕ)
  (h1 : total_questions = 80)
  (h2 : marks_per_correct = 4)
  (h3 : total_marks = 120)
  (h4 : correct_answers = 40) :
  (marks_per_correct * correct_answers - total_marks) / (total_questions - correct_answers) = 1 := by
sorry

end NUMINAMATH_CALUDE_exam_marks_lost_l1918_191898


namespace NUMINAMATH_CALUDE_chi_square_relationship_confidence_l1918_191897

/-- The critical value for 99% confidence level in this χ² test -/
def critical_value : ℝ := 6.635

/-- The observed χ² value -/
def observed_chi_square : ℝ := 8.654

/-- The confidence level as a percentage -/
def confidence_level : ℝ := 99

theorem chi_square_relationship_confidence :
  observed_chi_square > critical_value →
  confidence_level = 99 := by
sorry

end NUMINAMATH_CALUDE_chi_square_relationship_confidence_l1918_191897


namespace NUMINAMATH_CALUDE_jason_car_count_l1918_191809

theorem jason_car_count (purple : ℕ) (red : ℕ) (green : ℕ) : 
  purple = 47 →
  red = purple + 6 →
  green = 4 * red →
  purple + red + green = 312 := by
sorry

end NUMINAMATH_CALUDE_jason_car_count_l1918_191809


namespace NUMINAMATH_CALUDE_yoongi_has_fewest_apples_l1918_191838

def jungkook_apples : ℕ := 6 * 3
def yoongi_apples : ℕ := 4
def yuna_apples : ℕ := 5

theorem yoongi_has_fewest_apples :
  yoongi_apples ≤ jungkook_apples ∧ yoongi_apples ≤ yuna_apples := by
  sorry

end NUMINAMATH_CALUDE_yoongi_has_fewest_apples_l1918_191838


namespace NUMINAMATH_CALUDE_milk_production_theorem_l1918_191855

/-- Represents the milk production scenario with varying cow efficiencies -/
structure MilkProduction where
  a : ℕ  -- number of cows in original group
  b : ℝ  -- gallons of milk produced by original group
  c : ℕ  -- number of days for original group
  d : ℕ  -- number of cows in new group
  e : ℕ  -- number of days for new group
  h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0  -- ensure positive values

/-- The theorem stating the milk production for the new group -/
theorem milk_production_theorem (mp : MilkProduction) :
  let avg_rate := mp.b / (mp.a * mp.c)
  let efficient_rate := 2 * avg_rate
  let inefficient_rate := avg_rate / 2
  let new_production := mp.d * (efficient_rate * mp.a / 2 + inefficient_rate * mp.a / 2) / mp.a * mp.e
  new_production = mp.d * mp.b * mp.e / (mp.a * mp.c) := by
  sorry

#check milk_production_theorem

end NUMINAMATH_CALUDE_milk_production_theorem_l1918_191855


namespace NUMINAMATH_CALUDE_delivery_tip_cost_is_eight_l1918_191885

/-- Calculates the delivery and tip cost given grocery order details --/
def delivery_and_tip_cost (original_order : ℝ) 
                          (tomatoes_old : ℝ) (tomatoes_new : ℝ)
                          (lettuce_old : ℝ) (lettuce_new : ℝ)
                          (celery_old : ℝ) (celery_new : ℝ)
                          (total_bill : ℝ) : ℝ :=
  let price_increase := (tomatoes_new - tomatoes_old) + 
                        (lettuce_new - lettuce_old) + 
                        (celery_new - celery_old)
  let new_grocery_cost := original_order + price_increase
  total_bill - new_grocery_cost

/-- Theorem stating that the delivery and tip cost is $8.00 --/
theorem delivery_tip_cost_is_eight :
  delivery_and_tip_cost 25 0.99 2.20 1.00 1.75 1.96 2.00 35 = 8 :=
by sorry


end NUMINAMATH_CALUDE_delivery_tip_cost_is_eight_l1918_191885


namespace NUMINAMATH_CALUDE_mikes_hourly_rate_l1918_191832

/-- Given Mike's weekly earnings information, calculate his hourly rate for the second job. -/
theorem mikes_hourly_rate (total_wage : ℚ) (first_job_wage : ℚ) (second_job_hours : ℚ) 
  (h1 : total_wage = 160)
  (h2 : first_job_wage = 52)
  (h3 : second_job_hours = 12) :
  (total_wage - first_job_wage) / second_job_hours = 9 := by
sorry

#eval (160 : ℚ) - (52 : ℚ) / (12 : ℚ)

end NUMINAMATH_CALUDE_mikes_hourly_rate_l1918_191832


namespace NUMINAMATH_CALUDE_petya_wins_l1918_191813

/-- Represents the game state -/
structure GameState where
  stones : ℕ
  playerTurn : Bool  -- true for Petya, false for Vasya

/-- Defines a valid move in the game -/
def validMove (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : ℕ) : GameState :=
  { stones := state.stones - move, playerTurn := ¬state.playerTurn }

/-- Determines if the game is over -/
def gameOver (state : GameState) : Prop := state.stones = 0

/-- Defines a winning strategy for the first player -/
def winningStrategy (strategy : GameState → ℕ) : Prop :=
  ∀ (state : GameState), 
    validMove (strategy state) ∧ 
    (gameOver (applyMove state (strategy state)) ∨ 
     ∀ (opponentMove : ℕ), validMove opponentMove → 
       ¬gameOver (applyMove (applyMove state (strategy state)) opponentMove))

/-- Theorem: The first player (Petya) has a winning strategy -/
theorem petya_wins : 
  ∃ (strategy : GameState → ℕ), winningStrategy strategy ∧ 
    (∀ (state : GameState), state.stones = 111 ∧ state.playerTurn = true → 
      gameOver (applyMove state (strategy state)) ∨ 
      ∀ (opponentMove : ℕ), validMove opponentMove → 
        ¬gameOver (applyMove (applyMove state (strategy state)) opponentMove)) :=
sorry

end NUMINAMATH_CALUDE_petya_wins_l1918_191813


namespace NUMINAMATH_CALUDE_total_amount_after_ten_years_l1918_191877

/-- Calculates the total amount after compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Theorem: The total amount after 10 years with 5% annual interest rate -/
theorem total_amount_after_ten_years :
  let initial_deposit : ℝ := 100000
  let interest_rate : ℝ := 0.05
  let years : ℕ := 10
  compound_interest initial_deposit interest_rate years = initial_deposit * (1 + interest_rate) ^ years :=
by sorry

end NUMINAMATH_CALUDE_total_amount_after_ten_years_l1918_191877


namespace NUMINAMATH_CALUDE_last_letter_of_93rd_perm_l1918_191881

def word := "BRAVE"

/-- Represents a permutation of the word "BRAVE" -/
def Permutation := Fin 5 → Char

/-- The set of all permutations of "BRAVE" -/
def all_permutations : Finset Permutation :=
  sorry

/-- Dictionary order for permutations -/
def dict_order (p q : Permutation) : Prop :=
  sorry

/-- The 93rd permutation in dictionary order -/
def perm_93 : Permutation :=
  sorry

theorem last_letter_of_93rd_perm :
  (perm_93 4) = 'R' :=
sorry

end NUMINAMATH_CALUDE_last_letter_of_93rd_perm_l1918_191881


namespace NUMINAMATH_CALUDE_square_sum_nonzero_iff_not_both_zero_l1918_191867

theorem square_sum_nonzero_iff_not_both_zero (a b : ℝ) :
  a^2 + b^2 ≠ 0 ↔ ¬(a = 0 ∧ b = 0) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_nonzero_iff_not_both_zero_l1918_191867


namespace NUMINAMATH_CALUDE_num_valid_distributions_is_180_l1918_191853

/-- Represents a club -/
inductive Club
| ChunhuiLiteratureSociety
| DancersRollerSkatingClub
| BasketballHome
| GoGarden

/-- Represents a student -/
inductive Student
| A
| B
| C
| D
| E

/-- A valid distribution of students to clubs -/
def ValidDistribution := Student → Club

/-- Checks if a distribution is valid according to the problem conditions -/
def isValidDistribution (d : ValidDistribution) : Prop :=
  (∀ c : Club, ∃ s : Student, d s = c) ∧ 
  (d Student.A ≠ Club.GoGarden)

/-- The number of valid distributions -/
def numValidDistributions : ℕ := sorry

/-- The main theorem stating that the number of valid distributions is 180 -/
theorem num_valid_distributions_is_180 : numValidDistributions = 180 := by sorry

end NUMINAMATH_CALUDE_num_valid_distributions_is_180_l1918_191853


namespace NUMINAMATH_CALUDE_circle_line_intersection_l1918_191846

/-- The x-coordinates of the intersection points between a circle and a line -/
theorem circle_line_intersection
  (x1 y1 x2 y2 : ℝ)  -- Endpoints of the circle's diameter
  (m b : ℝ)  -- Line equation coefficients (y = mx + b)
  (h_distinct : (x1, y1) ≠ (x2, y2))  -- Ensure distinct endpoints
  (h_line : m = -1/2 ∧ b = 5)  -- Specific line equation
  (h_endpoints : x1 = 2 ∧ y1 = 4 ∧ x2 = 10 ∧ y2 = 8)  -- Specific endpoint coordinates
  : ∃ (x_left x_right : ℝ),
    x_left = 4.4 - 2.088 ∧
    x_right = 4.4 + 2.088 ∧
    (∀ (x y : ℝ),
      (x - (x1 + x2)/2)^2 + (y - (y1 + y2)/2)^2 = ((x2 - x1)^2 + (y2 - y1)^2)/4 ∧
      y = m * x + b →
      x = x_left ∨ x = x_right) :=
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l1918_191846


namespace NUMINAMATH_CALUDE_parallelogram_area_is_3sqrt14_l1918_191839

-- Define the complex equations
def equation1 (z : ℂ) : Prop := z^2 = 9 + 9 * Complex.I * Real.sqrt 7
def equation2 (z : ℂ) : Prop := z^2 = 3 + 3 * Complex.I * Real.sqrt 2

-- Define the solutions
def solutions : Set ℂ := {z : ℂ | equation1 z ∨ equation2 z}

-- Define the parallelogram area function
noncomputable def parallelogramArea (vertices : Set ℂ) : ℝ :=
  sorry -- Actual implementation would go here

-- Theorem statement
theorem parallelogram_area_is_3sqrt14 :
  parallelogramArea solutions = 3 * Real.sqrt 14 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_area_is_3sqrt14_l1918_191839


namespace NUMINAMATH_CALUDE_min_value_theorem_l1918_191875

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∃ (min_val : ℝ), min_val = 6 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 1 →
    1/(x-1) + 9/(y-1) ≥ min_val := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1918_191875


namespace NUMINAMATH_CALUDE_sqrt_square_eq_abs_l1918_191874

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x ^ 2) = |x| := by sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_abs_l1918_191874


namespace NUMINAMATH_CALUDE_sam_placed_twelve_crayons_l1918_191879

/-- The number of crayons Sam placed in the drawer -/
def crayons_placed (initial_crayons final_crayons : ℕ) : ℕ :=
  final_crayons - initial_crayons

/-- Theorem: Sam placed 12 crayons in the drawer -/
theorem sam_placed_twelve_crayons :
  crayons_placed 41 53 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sam_placed_twelve_crayons_l1918_191879


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1918_191871

theorem sqrt_inequality (a b : ℝ) : Real.sqrt a < Real.sqrt b → a < b := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1918_191871


namespace NUMINAMATH_CALUDE_walking_rate_problem_l1918_191810

/-- Proves that given the conditions of the problem, the walking rate when missing the train is 4 kmph -/
theorem walking_rate_problem (distance : ℝ) (early_rate : ℝ) (early_time : ℝ) (late_time : ℝ) :
  distance = 4 →
  early_rate = 5 →
  early_time = 6 →
  late_time = 6 →
  ∃ (late_rate : ℝ),
    (distance / early_rate) * 60 + early_time = (distance / late_rate) * 60 - late_time ∧
    late_rate = 4 :=
by sorry

end NUMINAMATH_CALUDE_walking_rate_problem_l1918_191810


namespace NUMINAMATH_CALUDE_same_color_plate_probability_l1918_191824

theorem same_color_plate_probability (total : ℕ) (yellow : ℕ) (green : ℕ) 
  (h1 : total = yellow + green)
  (h2 : yellow = 7)
  (h3 : green = 5) :
  (Nat.choose yellow 2 + Nat.choose green 2) / Nat.choose total 2 = 31 / 66 := by
  sorry

end NUMINAMATH_CALUDE_same_color_plate_probability_l1918_191824


namespace NUMINAMATH_CALUDE_number_divisibility_l1918_191836

theorem number_divisibility (N : ℕ) : 
  N % 5 = 0 ∧ N % 4 = 2 → N / 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_divisibility_l1918_191836


namespace NUMINAMATH_CALUDE_unique_solution_when_p_equals_two_l1918_191852

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^(1/3) + (2 - x)^(1/3)

-- State the theorem
theorem unique_solution_when_p_equals_two :
  ∃! p : ℝ, ∃! x : ℝ, f x = p :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_solution_when_p_equals_two_l1918_191852


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1918_191899

/-- The standard equation of a hyperbola given specific conditions -/
theorem hyperbola_equation (f : ℝ × ℝ) (M N : ℝ × ℝ) :
  (∃ c : ℝ, c > 0 ∧ f = (c, 0) ∧ ∀ x y : ℝ, y^2 = 4 * Real.sqrt 7 * x → (x - c)^2 + y^2 = c^2) →  -- focus coincides with parabola focus
  (M.1 - 1 = M.2 ∧ N.1 - 1 = N.2) →  -- M and N are on the line y = x - 1
  ((M.1 + N.1) / 2 = -2/3) →  -- x-coordinate of midpoint is -2/3
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ 
    ((x - f.1)^2 / (a^2 + b^2) + (y - f.2)^2 / (a^2 + b^2) = 1 ∧
     (x + f.1)^2 / (a^2 + b^2) + (y - f.2)^2 / (a^2 + b^2) = 1)) →
  ∃ x y : ℝ, x^2/2 - y^2/5 = 1 ↔
    ((x - f.1)^2 / 7 + (y - f.2)^2 / 7 = 1 ∧
     (x + f.1)^2 / 7 + (y - f.2)^2 / 7 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1918_191899


namespace NUMINAMATH_CALUDE_divisor_cube_eq_four_n_l1918_191860

/-- The number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The set of solutions to the equation d(n)^3 = 4n -/
def solution_set : Set ℕ := {2, 128, 2000}

/-- Theorem stating that n is a solution if and only if it's in the solution set -/
theorem divisor_cube_eq_four_n (n : ℕ) : 
  (num_divisors n)^3 = 4 * n ↔ n ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_divisor_cube_eq_four_n_l1918_191860


namespace NUMINAMATH_CALUDE_unique_divisible_by_nine_l1918_191807

theorem unique_divisible_by_nine : ∃! x : ℕ, 
  x ≥ 0 ∧ x ≤ 9 ∧ (13800 + x * 10 + 6) % 9 = 0 := by sorry

end NUMINAMATH_CALUDE_unique_divisible_by_nine_l1918_191807


namespace NUMINAMATH_CALUDE_food_expense_percentage_l1918_191803

/-- Proves that the percentage of salary spent on food is 32% given the specified conditions --/
theorem food_expense_percentage
  (salary : ℝ)
  (medicine_percentage : ℝ)
  (savings_percentage : ℝ)
  (savings_amount : ℝ)
  (h1 : salary = 15000)
  (h2 : medicine_percentage = 20)
  (h3 : savings_percentage = 60)
  (h4 : savings_amount = 4320)
  (h5 : savings_amount = (salary - (medicine_percentage / 100) * salary - food_expense) * (savings_percentage / 100))
  : (food_expense / salary) * 100 = 32 := by
  sorry

end NUMINAMATH_CALUDE_food_expense_percentage_l1918_191803


namespace NUMINAMATH_CALUDE_marker_distance_l1918_191818

theorem marker_distance (k : ℝ) (h_pos : k > 0) : 
  (∀ n : ℕ, ∀ m : ℕ, m - n = 4 → 
    Real.sqrt ((m - n)^2 + (m*k - n*k)^2) = 31) →
  Real.sqrt ((19 - 7)^2 + (19*k - 7*k)^2) = 93 :=
by sorry

end NUMINAMATH_CALUDE_marker_distance_l1918_191818


namespace NUMINAMATH_CALUDE_average_of_eleven_numbers_l1918_191888

theorem average_of_eleven_numbers (first_six_avg : ℝ) (last_six_avg : ℝ) (middle : ℝ) :
  first_six_avg = 10.5 →
  last_six_avg = 11.4 →
  middle = 22.5 →
  (6 * first_six_avg + 6 * last_six_avg - middle) / 11 = 9.9 := by
sorry

end NUMINAMATH_CALUDE_average_of_eleven_numbers_l1918_191888


namespace NUMINAMATH_CALUDE_furniture_purchase_cost_l1918_191856

/-- Calculate the final cost of furniture purchase --/
theorem furniture_purchase_cost :
  let table_cost : ℚ := 140
  let chair_cost : ℚ := table_cost / 7
  let sofa_cost : ℚ := 2 * table_cost
  let num_chairs : ℕ := 4
  let table_discount_rate : ℚ := 1 / 10
  let sales_tax_rate : ℚ := 7 / 100
  let exchange_rate : ℚ := 12 / 10

  let total_chair_cost : ℚ := num_chairs * chair_cost
  let discounted_table_cost : ℚ := table_cost * (1 - table_discount_rate)
  let subtotal : ℚ := discounted_table_cost + total_chair_cost + sofa_cost
  let sales_tax : ℚ := subtotal * sales_tax_rate
  let final_cost : ℚ := subtotal + sales_tax

  final_cost = 52002 / 100 := by sorry

end NUMINAMATH_CALUDE_furniture_purchase_cost_l1918_191856


namespace NUMINAMATH_CALUDE_equation_solution_l1918_191884

theorem equation_solution :
  ∃ x : ℚ, (1 / (x + 2) + 3 * x / (x + 2) + 4 / (x + 2) = 1) ∧ (x = -3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1918_191884


namespace NUMINAMATH_CALUDE_cosine_sum_twenty_degrees_l1918_191801

theorem cosine_sum_twenty_degrees : 
  Real.cos (20 * π / 180) + Real.cos (60 * π / 180) + 
  Real.cos (100 * π / 180) + Real.cos (140 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_twenty_degrees_l1918_191801


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1918_191890

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  is_arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 10 →
  a 3 = 17 →
  a 7 = 38 →
  a 4 + a 5 + a 6 = 93 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1918_191890


namespace NUMINAMATH_CALUDE_sum_factorial_units_digit_2023_l1918_191849

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def factorialUnitsDigit (n : ℕ) : ℕ :=
  if n > 4 then 0 else unitsDigit (factorial n)

def sumFactorialUnitsDigits (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => (acc + factorialUnitsDigit (i + 1)) % 10) 0

theorem sum_factorial_units_digit_2023 :
  sumFactorialUnitsDigits 2023 = 3 := by sorry

end NUMINAMATH_CALUDE_sum_factorial_units_digit_2023_l1918_191849


namespace NUMINAMATH_CALUDE_cylinder_not_triangular_front_view_l1918_191891

/-- A solid geometry object --/
inductive Solid
  | Cylinder
  | Cone
  | Tetrahedron
  | TriangularPrism

/-- The shape of a view (projection) of a solid --/
inductive ViewShape
  | Triangle
  | Rectangle

/-- The front view of a solid --/
def frontView (s : Solid) : ViewShape :=
  match s with
  | Solid.Cylinder => ViewShape.Rectangle
  | _ => ViewShape.Triangle  -- We only care about the cylinder case for this problem

/-- Theorem: A cylinder cannot have a triangular front view --/
theorem cylinder_not_triangular_front_view :
  ∀ s : Solid, s = Solid.Cylinder → frontView s ≠ ViewShape.Triangle :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_not_triangular_front_view_l1918_191891


namespace NUMINAMATH_CALUDE_sphere_tetrahedron_intersection_length_l1918_191848

/-- The total length of the intersection between a sphere and a regular tetrahedron -/
theorem sphere_tetrahedron_intersection_length 
  (edge_length : ℝ) 
  (sphere_radius : ℝ) 
  (h_edge : edge_length = 2 * Real.sqrt 6) 
  (h_radius : sphere_radius = Real.sqrt 3) : 
  ∃ (intersection_length : ℝ), 
    intersection_length = 8 * Real.sqrt 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_tetrahedron_intersection_length_l1918_191848


namespace NUMINAMATH_CALUDE_tennis_tournament_matches_l1918_191869

theorem tennis_tournament_matches (total_players : ℕ) (advanced_players : ℕ) 
  (h1 : total_players = 128)
  (h2 : advanced_players = 20)
  (h3 : total_players > advanced_players) :
  (total_players - 1 : ℕ) = 127 := by
sorry

end NUMINAMATH_CALUDE_tennis_tournament_matches_l1918_191869


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1918_191831

-- Define the complex number z
variable (z : ℂ)

-- State the theorem
theorem complex_equation_solution :
  (3 - 4*I + z)*I = 2 + I → z = -2 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1918_191831


namespace NUMINAMATH_CALUDE_sunday_newspaper_delivery_l1918_191872

theorem sunday_newspaper_delivery (total : ℕ) (difference : ℕ) 
  (h1 : total = 110)
  (h2 : difference = 20) :
  ∃ (saturday sunday : ℕ), 
    saturday + sunday = total ∧ 
    sunday = saturday + difference ∧ 
    sunday = 65 := by
  sorry

end NUMINAMATH_CALUDE_sunday_newspaper_delivery_l1918_191872


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1918_191833

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 20 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1918_191833


namespace NUMINAMATH_CALUDE_triangle_medians_area_relationship_l1918_191827

/-- Represents a triangle with three medians -/
structure Triangle where
  median1 : ℝ
  median2 : ℝ
  median3 : ℝ
  area : ℝ

/-- The theorem stating the relationship between the medians and area of the triangle -/
theorem triangle_medians_area_relationship (t : Triangle) 
  (h1 : t.median1 = 5)
  (h2 : t.median2 = 7)
  (h3 : t.area = 10 * Real.sqrt 3) :
  t.median3 = 4 * Real.sqrt 3 := by
  sorry

#check triangle_medians_area_relationship

end NUMINAMATH_CALUDE_triangle_medians_area_relationship_l1918_191827


namespace NUMINAMATH_CALUDE_expected_value_of_three_from_seven_l1918_191887

/-- The number of marbles in the bag -/
def n : ℕ := 7

/-- The number of marbles drawn -/
def k : ℕ := 3

/-- The sum of numbers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The average value of a set of k elements from 1 to n -/
def avg_value (n k : ℕ) : ℚ := (sum_to_n n : ℚ) / n * k

/-- The expected value of the sum of k randomly chosen marbles from n marbles -/
def expected_value (n k : ℕ) : ℚ := avg_value n k

theorem expected_value_of_three_from_seven :
  expected_value n k = 12 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_three_from_seven_l1918_191887


namespace NUMINAMATH_CALUDE_probability_yellow_ball_l1918_191873

def num_red_balls : ℕ := 4
def num_yellow_balls : ℕ := 7

def total_balls : ℕ := num_red_balls + num_yellow_balls

theorem probability_yellow_ball :
  (num_yellow_balls : ℚ) / (total_balls : ℚ) = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_yellow_ball_l1918_191873


namespace NUMINAMATH_CALUDE_biker_distance_difference_l1918_191834

/-- The difference in distance traveled between two bikers with different speeds over a fixed time -/
theorem biker_distance_difference (alberto_speed bjorn_speed : ℝ) (race_duration : ℝ) 
  (h1 : alberto_speed = 18)
  (h2 : bjorn_speed = 15)
  (h3 : race_duration = 6) :
  alberto_speed * race_duration - bjorn_speed * race_duration = 18 :=
by sorry

end NUMINAMATH_CALUDE_biker_distance_difference_l1918_191834


namespace NUMINAMATH_CALUDE_unique_solution_l1918_191828

theorem unique_solution (x y z : ℝ) 
  (hx : x > 4) (hy : y > 4) (hz : z > 4)
  (h : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 42) :
  x = 11 ∧ y = 9 ∧ z = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1918_191828


namespace NUMINAMATH_CALUDE_product_first_three_is_960_l1918_191862

/-- An arithmetic sequence with seventh term 20 and common difference 2 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  8 + 2 * (n - 1)

/-- The product of the first three terms of the arithmetic sequence -/
def product_first_three : ℚ :=
  (arithmetic_sequence 1) * (arithmetic_sequence 2) * (arithmetic_sequence 3)

theorem product_first_three_is_960 :
  product_first_three = 960 :=
by sorry

end NUMINAMATH_CALUDE_product_first_three_is_960_l1918_191862


namespace NUMINAMATH_CALUDE_chess_tournament_solution_l1918_191802

/-- Chess tournament with n women and 2n men -/
structure ChessTournament (n : ℕ) where
  women : Fin n
  men : Fin (2 * n)

/-- The number of games played in the tournament -/
def total_games (n : ℕ) : ℕ :=
  n * (3 * n - 1) / 2

/-- The number of games won by women -/
def women_wins (n : ℕ) : ℚ :=
  (n * (n - 1) / 2) + (17 * n^2 - 3 * n) / 8

/-- The number of games won by men -/
def men_wins (n : ℕ) : ℚ :=
  (n * (2 * n - 1)) + (3 * n / 8)

/-- The theorem stating that n must equal 3 -/
theorem chess_tournament_solution : 
  ∃ (n : ℕ), n > 0 ∧ 
  7 * (men_wins n) = 5 * (women_wins n) ∧
  (women_wins n).isInt ∧ (men_wins n).isInt :=
by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_solution_l1918_191802


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_150_75_l1918_191817

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem largest_two_digit_prime_factor_of_binomial_150_75 :
  (∃ (p : ℕ), p.Prime ∧ p ∣ binomial 150 75 ∧ 10 ≤ p ∧ p < 100) ∧
  (∀ (q : ℕ), q.Prime → q ∣ binomial 150 75 → 10 ≤ q → q < 100 → q ≤ 73) :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_150_75_l1918_191817


namespace NUMINAMATH_CALUDE_angle_with_same_terminal_side_l1918_191863

/-- Given an angle α and another angle θ, proves that if α = 1560°, 
    θ has the same terminal side as α, and -360° < θ < 360°, 
    then θ = 120° or θ = -240°. -/
theorem angle_with_same_terminal_side 
  (α θ : ℝ) 
  (h1 : α = 1560)
  (h2 : ∃ (k : ℤ), θ = 360 * k + 120)
  (h3 : -360 < θ ∧ θ < 360) :
  θ = 120 ∨ θ = -240 :=
sorry

end NUMINAMATH_CALUDE_angle_with_same_terminal_side_l1918_191863


namespace NUMINAMATH_CALUDE_zachary_pushups_count_l1918_191847

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := sorry

/-- The number of crunches Zachary did -/
def zachary_crunches : ℕ := 58

/-- Zachary did 12 more crunches than push-ups -/
axiom crunches_pushups_difference : zachary_crunches = zachary_pushups + 12

theorem zachary_pushups_count : zachary_pushups = 46 := by sorry

end NUMINAMATH_CALUDE_zachary_pushups_count_l1918_191847


namespace NUMINAMATH_CALUDE_pet_food_discount_l1918_191866

/-- Proves that the regular discount is 30% given the conditions of the problem -/
theorem pet_food_discount (msrp : ℝ) (sale_price : ℝ) (additional_discount : ℝ) :
  msrp = 45 →
  sale_price = 25.2 →
  additional_discount = 20 →
  ∃ (regular_discount : ℝ),
    sale_price = msrp * (1 - regular_discount / 100) * (1 - additional_discount / 100) ∧
    regular_discount = 30 := by
  sorry

#check pet_food_discount

end NUMINAMATH_CALUDE_pet_food_discount_l1918_191866


namespace NUMINAMATH_CALUDE_sin_symmetry_condition_l1918_191896

def is_symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem sin_symmetry_condition (φ : ℝ) :
  (φ = π / 2 → is_symmetric_about_y_axis (fun x ↦ Real.sin (x + φ))) ∧
  ¬(is_symmetric_about_y_axis (fun x ↦ Real.sin (x + φ)) → φ = π / 2) :=
by sorry

end NUMINAMATH_CALUDE_sin_symmetry_condition_l1918_191896


namespace NUMINAMATH_CALUDE_ashok_pyarelal_capital_ratio_l1918_191808

/-- Given a business investment scenario where:
  * The total loss is 1000
  * Pyarelal's loss is 900
  * Ashok's loss is the remaining amount
  * The ratio of losses is proportional to the ratio of investments

  This theorem proves that the ratio of Ashok's capital to Pyarelal's capital is 1:9.
-/
theorem ashok_pyarelal_capital_ratio 
  (total_loss : ℕ) 
  (pyarelal_loss : ℕ) 
  (h_total_loss : total_loss = 1000)
  (h_pyarelal_loss : pyarelal_loss = 900)
  (ashok_loss : ℕ := total_loss - pyarelal_loss)
  (ashok_capital pyarelal_capital : ℚ)
  (h_loss_ratio : ashok_loss / pyarelal_loss = ashok_capital / pyarelal_capital) :
  ashok_capital / pyarelal_capital = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ashok_pyarelal_capital_ratio_l1918_191808


namespace NUMINAMATH_CALUDE_isosceles_triangle_line_equation_l1918_191865

/-- An isosceles triangle AOB with given properties -/
structure IsoscelesTriangle where
  /-- Point O is at the origin -/
  O : ℝ × ℝ := (0, 0)
  /-- Point A coordinates -/
  A : ℝ × ℝ := (1, 3)
  /-- Point B is on the positive x-axis -/
  B : ℝ × ℝ
  /-- B's y-coordinate is 0 -/
  h_B_on_x_axis : B.2 = 0
  /-- B's x-coordinate is positive -/
  h_B_positive_x : B.1 > 0
  /-- AO = AB (isosceles property) -/
  h_isosceles : (A.1 - O.1)^2 + (A.2 - O.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2

/-- The equation of line AB in an isosceles triangle AOB is y-3 = -3(x-1) -/
theorem isosceles_triangle_line_equation (t : IsoscelesTriangle) :
  ∀ x y : ℝ, (y - 3 = -3 * (x - 1)) ↔ (∃ k : ℝ, x = t.A.1 + k * (t.B.1 - t.A.1) ∧ y = t.A.2 + k * (t.B.2 - t.A.2)) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_line_equation_l1918_191865


namespace NUMINAMATH_CALUDE_double_discount_reduction_l1918_191835

theorem double_discount_reduction (original_price : ℝ) (discount : ℝ) : 
  discount = 0.4 → 
  (1 - (1 - discount) * (1 - discount)) * 100 = 64 := by
sorry

end NUMINAMATH_CALUDE_double_discount_reduction_l1918_191835


namespace NUMINAMATH_CALUDE_intersection_point_of_function_and_inverse_l1918_191814

-- Define the function f
def f (b : ℤ) : ℝ → ℝ := λ x => 4 * x + b

-- State the theorem
theorem intersection_point_of_function_and_inverse
  (b : ℤ) (a : ℤ) :
  (∃ (x : ℝ), f b x = x ∧ f b (-4) = a) →
  a = -4 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_of_function_and_inverse_l1918_191814


namespace NUMINAMATH_CALUDE_line_transformation_l1918_191858

/-- The analytical expression of a line after transformation -/
def transformed_line (a b : ℝ) (dx dy : ℝ) : ℝ → ℝ := fun x ↦ a * (x + dx) + b + dy

/-- The original line y = 2x - 1 -/
def original_line : ℝ → ℝ := fun x ↦ 2 * x - 1

theorem line_transformation :
  transformed_line 2 (-1) 1 (-2) = original_line := by sorry

end NUMINAMATH_CALUDE_line_transformation_l1918_191858


namespace NUMINAMATH_CALUDE_inequality_proof_l1918_191804

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a / (b + 2*c + 3*d)) + (b / (c + 2*d + 3*a)) + (c / (d + 2*a + 3*b)) + (d / (a + 2*b + 3*c)) ≥ 2/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1918_191804


namespace NUMINAMATH_CALUDE_root_difference_of_quadratic_l1918_191830

theorem root_difference_of_quadratic (r₁ r₂ : ℝ) : 
  r₁^2 - 9*r₁ + 14 = 0 → 
  r₂^2 - 9*r₂ + 14 = 0 → 
  r₁ + r₂ = r₁ * r₂ → 
  |r₁ - r₂| = 5 := by
sorry

end NUMINAMATH_CALUDE_root_difference_of_quadratic_l1918_191830


namespace NUMINAMATH_CALUDE_chocolate_distribution_l1918_191893

theorem chocolate_distribution (pieces_per_bar : ℕ) 
  (girls_consumption_case1 girls_consumption_case2 : ℕ) 
  (boys_consumption_case1 boys_consumption_case2 : ℕ) 
  (bars_case1 bars_case2 : ℕ) : 
  pieces_per_bar = 12 →
  girls_consumption_case1 = 7 →
  boys_consumption_case1 = 2 →
  bars_case1 = 3 →
  girls_consumption_case2 = 8 →
  boys_consumption_case2 = 4 →
  bars_case2 = 4 →
  ∃ (girls boys : ℕ),
    girls_consumption_case1 * girls + boys_consumption_case1 * boys > pieces_per_bar * bars_case1 ∧
    girls_consumption_case2 * girls + boys_consumption_case2 * boys < pieces_per_bar * bars_case2 ∧
    girls = 5 ∧
    boys = 1 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l1918_191893


namespace NUMINAMATH_CALUDE_integer_sum_problem_l1918_191837

theorem integer_sum_problem (x y : ℕ+) 
  (h1 : x.val - y.val = 15)
  (h2 : x.val * y.val = 56) :
  x.val + y.val = Real.sqrt 449 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l1918_191837


namespace NUMINAMATH_CALUDE_octagon_triangle_angle_sum_l1918_191829

theorem octagon_triangle_angle_sum :
  ∀ (ABC ABD : ℝ),
  (∃ (n : ℕ), n = 8 ∧ ABC = 180 * (n - 2) / n) →
  (∃ (m : ℕ), m = 3 ∧ ABD = 180 * (m - 2) / m) →
  ABC + ABD = 195 := by
sorry

end NUMINAMATH_CALUDE_octagon_triangle_angle_sum_l1918_191829


namespace NUMINAMATH_CALUDE_farmer_cows_problem_l1918_191878

theorem farmer_cows_problem (initial_cows : ℕ) (final_cows : ℕ) (new_cows : ℕ) : 
  initial_cows = 51 →
  final_cows = 42 →
  (3 : ℚ) / 4 * (initial_cows + new_cows) = final_cows →
  new_cows = 5 := by
sorry

end NUMINAMATH_CALUDE_farmer_cows_problem_l1918_191878


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l1918_191823

/-- The quadratic function f(x) = 3(x-1)^2 + 2 -/
def f (x : ℝ) : ℝ := 3 * (x - 1)^2 + 2

/-- The axis of symmetry for f(x) is x = 1 -/
theorem axis_of_symmetry (x : ℝ) : f (1 + x) = f (1 - x) := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l1918_191823


namespace NUMINAMATH_CALUDE_positive_integer_solutions_count_l1918_191816

theorem positive_integer_solutions_count : ∃ (n : ℕ), n = 10 ∧ 
  n = (Finset.filter (λ (x : ℕ × ℕ × ℕ) => 
    x.1 + x.2.1 + x.2.2 = 6 ∧ x.1 > 0 ∧ x.2.1 > 0 ∧ x.2.2 > 0) 
    (Finset.product (Finset.range 7) (Finset.product (Finset.range 7) (Finset.range 7)))).card :=
by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_count_l1918_191816


namespace NUMINAMATH_CALUDE_jack_waiting_time_l1918_191844

/-- The total waiting time in hours for Jack's travel to Canada -/
def total_waiting_time (customs_hours : ℕ) (quarantine_days : ℕ) : ℕ :=
  customs_hours + 24 * quarantine_days

/-- Theorem stating that Jack's total waiting time is 356 hours -/
theorem jack_waiting_time :
  total_waiting_time 20 14 = 356 := by
  sorry

end NUMINAMATH_CALUDE_jack_waiting_time_l1918_191844


namespace NUMINAMATH_CALUDE_rectangle_x_coordinate_l1918_191854

/-- A rectangle with vertices (1, 0), (x, 0), (1, 2), and (x, 2) is divided into two identical
    quadrilaterals by a line passing through the origin with slope 0.2.
    This theorem proves that the x-coordinate of the second and fourth vertices is 9. -/
theorem rectangle_x_coordinate (x : ℝ) :
  (∃ (l : Set (ℝ × ℝ)),
    -- Line l passes through the origin
    (0, 0) ∈ l ∧
    -- Line l has slope 0.2
    (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l → (x₂, y₂) ∈ l → x₁ ≠ x₂ → (y₂ - y₁) / (x₂ - x₁) = 0.2) ∧
    -- Line l divides the rectangle into two identical quadrilaterals
    (∃ (m n : ℝ × ℝ), m ∈ l ∧ n ∈ l ∧
      m.1 = (1 + x) / 2 ∧ m.2 = 1 ∧
      n.1 = (1 + x) / 2 ∧ n.2 = 1)) →
  x = 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_x_coordinate_l1918_191854
