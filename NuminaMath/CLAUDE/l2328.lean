import Mathlib

namespace quadratic_equation_positive_solutions_l2328_232822

theorem quadratic_equation_positive_solutions :
  ∃! (x : ℝ), x > 0 ∧ x^2 = -6*x + 9 := by sorry

end quadratic_equation_positive_solutions_l2328_232822


namespace apples_left_l2328_232851

def apples_bought : ℕ := 15
def apples_given : ℕ := 7

theorem apples_left : apples_bought - apples_given = 8 := by
  sorry

end apples_left_l2328_232851


namespace symmetry_of_shifted_even_function_l2328_232839

-- Define a function f
variable (f : ℝ → ℝ)

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define the axis of symmetry for a function
def axis_of_symmetry (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

-- State the theorem
theorem symmetry_of_shifted_even_function :
  is_even (fun x ↦ f (x - 2)) → axis_of_symmetry f (-2) :=
by sorry

end symmetry_of_shifted_even_function_l2328_232839


namespace pinwheel_area_is_four_l2328_232848

/-- Represents a point on a 2D grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a triangle on the grid -/
structure Triangle where
  v1 : GridPoint
  v2 : GridPoint
  v3 : GridPoint

/-- Represents the pinwheel design -/
structure Pinwheel where
  center : GridPoint
  arms : List Triangle

/-- Calculates the area of a triangle using Pick's theorem -/
def triangleArea (t : Triangle) : Int :=
  sorry

/-- Calculates the total area of the pinwheel -/
def pinwheelArea (p : Pinwheel) : Int :=
  sorry

/-- The main theorem to prove -/
theorem pinwheel_area_is_four :
  let center := GridPoint.mk 3 3
  let arm1 := Triangle.mk center (GridPoint.mk 6 3) (GridPoint.mk 3 6)
  let arm2 := Triangle.mk center (GridPoint.mk 3 6) (GridPoint.mk 0 3)
  let arm3 := Triangle.mk center (GridPoint.mk 0 3) (GridPoint.mk 3 0)
  let arm4 := Triangle.mk center (GridPoint.mk 3 0) (GridPoint.mk 6 3)
  let pinwheel := Pinwheel.mk center [arm1, arm2, arm3, arm4]
  pinwheelArea pinwheel = 4 :=
sorry

end pinwheel_area_is_four_l2328_232848


namespace probability_one_of_each_interpreter_l2328_232870

def team_size : ℕ := 5
def english_interpreters : ℕ := 3
def russian_interpreters : ℕ := 2

theorem probability_one_of_each_interpreter :
  let total_combinations := Nat.choose team_size 2
  let favorable_combinations := Nat.choose english_interpreters 1 * Nat.choose russian_interpreters 1
  (favorable_combinations : ℚ) / total_combinations = 3 / 5 := by
  sorry

end probability_one_of_each_interpreter_l2328_232870


namespace smallest_number_remainder_l2328_232894

theorem smallest_number_remainder (n : ℕ) : 
  (n = 210) → 
  (n % 13 = 3) → 
  (∀ m : ℕ, m < n → m % 13 ≠ 3 ∨ m % 17 ≠ n % 17) → 
  n % 17 = 6 := by
sorry

end smallest_number_remainder_l2328_232894


namespace gcd_nine_factorial_seven_factorial_squared_l2328_232899

theorem gcd_nine_factorial_seven_factorial_squared :
  Nat.gcd (Nat.factorial 9) ((Nat.factorial 7)^2) = 362880 := by
  sorry

end gcd_nine_factorial_seven_factorial_squared_l2328_232899


namespace initial_distance_problem_l2328_232858

theorem initial_distance_problem (speed_A speed_B : ℝ) (start_time end_time : ℝ) :
  speed_A = 5 →
  speed_B = 7 →
  start_time = 1 →
  end_time = 3 →
  let time_walked := end_time - start_time
  let distance_A := speed_A * time_walked
  let distance_B := speed_B * time_walked
  let initial_distance := distance_A + distance_B
  initial_distance = 24 := by
  sorry

end initial_distance_problem_l2328_232858


namespace x_range_l2328_232889

theorem x_range (x : ℝ) 
  (h1 : 1 / x < 3) 
  (h2 : 1 / x > -4) 
  (h3 : x^2 - 1 > 0) : 
  x > 1 ∨ x < -1 := by
sorry

end x_range_l2328_232889


namespace solve_equation_l2328_232854

theorem solve_equation (x : ℝ) : (x^3)^(1/2) = 18 * 18^(1/9) → x = 18^(20/27) := by
  sorry

end solve_equation_l2328_232854


namespace radical_simplification_l2328_232830

theorem radical_simplification (p : ℝ) :
  Real.sqrt (42 * p^2) * Real.sqrt (7 * p^2) * Real.sqrt (14 * p^2) = 14 * p^3 * Real.sqrt 21 := by
  sorry

end radical_simplification_l2328_232830


namespace line_parallel_perpendicular_l2328_232821

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

-- Define lines and planes
def Line (V : Type*) [NormedAddCommGroup V] := Set V
def Plane (V : Type*) [NormedAddCommGroup V] := Set V

-- Define parallel and perpendicular relations
def parallel (l₁ l₂ : Line V) : Prop := sorry
def perpendicular (l : Line V) (p : Plane V) : Prop := sorry

-- Theorem statement
theorem line_parallel_perpendicular 
  (a b : Line V) (α : Plane V) 
  (h₁ : a ≠ b) 
  (h₂ : parallel a b) 
  (h₃ : perpendicular a α) : 
  perpendicular b α := 
sorry

end line_parallel_perpendicular_l2328_232821


namespace students_not_enrolled_l2328_232859

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 79)
  (h2 : french = 41)
  (h3 : german = 22)
  (h4 : both = 9) :
  total - (french + german - both) = 25 := by
  sorry

end students_not_enrolled_l2328_232859


namespace ratio_satisfies_condition_l2328_232802

/-- Represents the number of people in each profession -/
structure ProfessionCount where
  doctors : ℕ
  lawyers : ℕ
  engineers : ℕ

/-- The average age of the entire group -/
def groupAverage : ℝ := 45

/-- The average age of doctors -/
def doctorAverage : ℝ := 40

/-- The average age of lawyers -/
def lawyerAverage : ℝ := 55

/-- The average age of engineers -/
def engineerAverage : ℝ := 35

/-- Checks if the given profession count satisfies the average age conditions -/
def satisfiesAverageCondition (count : ProfessionCount) : Prop :=
  let totalPeople := count.doctors + count.lawyers + count.engineers
  let totalAge := doctorAverage * count.doctors + lawyerAverage * count.lawyers + engineerAverage * count.engineers
  totalAge / totalPeople = groupAverage

/-- The theorem stating that the ratio 2:2:1 satisfies the average age conditions -/
theorem ratio_satisfies_condition :
  ∃ (k : ℕ), k > 0 ∧ satisfiesAverageCondition { doctors := 2 * k, lawyers := 2 * k, engineers := k } :=
sorry

end ratio_satisfies_condition_l2328_232802


namespace not_enough_money_l2328_232815

theorem not_enough_money (book1_price book2_price available_money : ℝ) 
  (h1 : book1_price = 21.8)
  (h2 : book2_price = 19.5)
  (h3 : available_money = 40) :
  book1_price + book2_price > available_money := by
  sorry

end not_enough_money_l2328_232815


namespace division_problem_l2328_232833

theorem division_problem (L S q : ℕ) : 
  L - S = 1365 → 
  L = 1634 → 
  L = S * q + 20 → 
  q = 6 := by
sorry

end division_problem_l2328_232833


namespace bananas_permutations_count_l2328_232814

/-- The number of unique permutations of the letters in "BANANAS" -/
def bananas_permutations : ℕ := 420

/-- The total number of letters in "BANANAS" -/
def total_letters : ℕ := 7

/-- The number of occurrences of 'A' in "BANANAS" -/
def a_count : ℕ := 3

/-- The number of occurrences of 'N' in "BANANAS" -/
def n_count : ℕ := 2

/-- Theorem stating that the number of unique permutations of the letters in "BANANAS"
    is equal to 420, given the total number of letters and the counts of repeated letters. -/
theorem bananas_permutations_count : 
  bananas_permutations = (Nat.factorial total_letters) / 
    ((Nat.factorial a_count) * (Nat.factorial n_count)) := by
  sorry

end bananas_permutations_count_l2328_232814


namespace parametric_to_circle_equation_l2328_232807

/-- Given parametric equations for a curve and a relationship between parameters,
    prove that the resulting equation is that of a circle with specific center and radius,
    excluding two points on the x-axis. -/
theorem parametric_to_circle_equation 
  (u v : ℝ) (m : ℝ) (hm : m ≠ 0)
  (hx : ∀ u v, x = (1 - u^2 - v^2) / ((1 - u)^2 + v^2))
  (hy : ∀ u v, y = 2 * v / ((1 - u)^2 + v^2))
  (hv : v = m * u) :
  x^2 + (y - 1/m)^2 = 1 + 1/m^2 ∧ 
  (x ≠ 1 ∨ y ≠ 0) ∧ (x ≠ -1 ∨ y ≠ 0) := by
  sorry


end parametric_to_circle_equation_l2328_232807


namespace cube_volume_ratio_l2328_232810

theorem cube_volume_ratio (e : ℝ) (h : e > 0) :
  let small_cube_volume := e^3
  let large_cube_volume := (3*e)^3
  large_cube_volume = 27 * small_cube_volume := by
sorry

end cube_volume_ratio_l2328_232810


namespace multiple_of_x_l2328_232881

theorem multiple_of_x (x y z k : ℕ+) : 
  (k * x = 5 * y) ∧ (5 * y = 8 * z) ∧ (x + y + z = 33) → k = 40 := by
  sorry

end multiple_of_x_l2328_232881


namespace hypotenuse_area_change_l2328_232800

theorem hypotenuse_area_change
  (a b c : ℝ)
  (h_right_triangle : a^2 + b^2 = c^2)
  (h_area_increase : (a - 5) * (b + 5) / 2 = a * b / 2 + 5)
  : c^2 - ((a - 5)^2 + (b + 5)^2) = 20 :=
by sorry

end hypotenuse_area_change_l2328_232800


namespace max_value_expression_l2328_232866

theorem max_value_expression (a b : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) :
  (|7*a + 8*b - a*b| + |2*a + 8*b - 6*a*b|) / (a * Real.sqrt (1 + b^2)) ≤ 9 * Real.sqrt 2 :=
sorry

end max_value_expression_l2328_232866


namespace equation_solution_l2328_232825

theorem equation_solution : ∃! x : ℝ, (10 : ℝ)^(2*x) * (100 : ℝ)^x = (1000 : ℝ)^4 :=
  by
    use 3
    constructor
    · -- Proof that x = 3 satisfies the equation
      sorry
    · -- Proof of uniqueness
      sorry

#check equation_solution

end equation_solution_l2328_232825


namespace divisibility_condition_l2328_232846

theorem divisibility_condition (x : ℤ) : (x - 1) ∣ (x - 3) ↔ x ∈ ({-1, 0, 2, 3} : Set ℤ) := by
  sorry

end divisibility_condition_l2328_232846


namespace y_sum_theorem_l2328_232835

theorem y_sum_theorem (y₁ y₂ y₃ y₄ y₅ : ℝ) 
  (eq1 : y₁ + 3*y₂ + 6*y₃ + 10*y₄ + 15*y₅ = 3)
  (eq2 : 3*y₁ + 6*y₂ + 10*y₃ + 15*y₄ + 21*y₅ = 20)
  (eq3 : 6*y₁ + 10*y₂ + 15*y₃ + 21*y₄ + 28*y₅ = 86)
  (eq4 : 10*y₁ + 15*y₂ + 21*y₃ + 28*y₄ + 36*y₅ = 225) :
  15*y₁ + 21*y₂ + 28*y₃ + 36*y₄ + 45*y₅ = 395 := by
  sorry

end y_sum_theorem_l2328_232835


namespace gcd_f_x_l2328_232878

def f (x : ℤ) : ℤ := (5*x+3)*(8*x+2)*(12*x+7)*(3*x+11)

theorem gcd_f_x (x : ℤ) (h : ∃ k : ℤ, x = 18720 * k) : 
  Nat.gcd (Int.natAbs (f x)) (Int.natAbs x) = 462 := by
  sorry

end gcd_f_x_l2328_232878


namespace condition_sufficient_not_necessary_l2328_232804

theorem condition_sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > 0 ∧ b > 0 → a^2 + b^2 ≥ 2*a*b) ∧
  ¬(∀ a b : ℝ, a^2 + b^2 ≥ 2*a*b → a > 0 ∧ b > 0) :=
sorry

end condition_sufficient_not_necessary_l2328_232804


namespace root_sum_squares_l2328_232850

-- Define the polynomial
def p (x : ℝ) : ℝ := x^3 - 12*x^2 + 44*x - 85

-- Define the roots of the polynomial
def roots_condition (a b c : ℝ) : Prop := p a = 0 ∧ p b = 0 ∧ p c = 0

-- Theorem statement
theorem root_sum_squares (a b c : ℝ) (h : roots_condition a b c) :
  (a + b)^2 + (b + c)^2 + (c + a)^2 = 200 := by
  sorry

end root_sum_squares_l2328_232850


namespace system_solution_l2328_232827

theorem system_solution (x y b : ℝ) : 
  (4 * x + 2 * y = b) → 
  (3 * x + 7 * y = 3 * b) → 
  (x = 3) → 
  (b = 66) := by
sorry

end system_solution_l2328_232827


namespace calculate_total_income_person_total_income_l2328_232865

/-- Calculates a person's total income based on given distributions --/
theorem calculate_total_income (children_share : Real) (wife_share : Real) 
  (orphan_donation_rate : Real) (final_amount : Real) : Real :=
  let total_distributed := children_share + wife_share
  let remaining_before_donation := 1 - total_distributed
  let orphan_donation := orphan_donation_rate * remaining_before_donation
  let final_share := remaining_before_donation - orphan_donation
  final_amount / final_share

/-- Proves that the person's total income is $150,000 --/
theorem person_total_income : 
  calculate_total_income 0.25 0.35 0.1 45000 = 150000 := by
  sorry

end calculate_total_income_person_total_income_l2328_232865


namespace max_volume_side_length_l2328_232898

def sheet_length : ℝ := 90
def sheet_width : ℝ := 48

def container_volume (x : ℝ) : ℝ :=
  (sheet_length - 2 * x) * (sheet_width - 2 * x) * x

theorem max_volume_side_length :
  ∃ (x : ℝ), x > 0 ∧ x < sheet_width / 2 ∧ x < sheet_length / 2 ∧
  ∀ (y : ℝ), y > 0 → y < sheet_width / 2 → y < sheet_length / 2 →
  container_volume y ≤ container_volume x ∧
  x = 10 :=
sorry

end max_volume_side_length_l2328_232898


namespace problem_statement_l2328_232869

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  (2/(a-1) + 1/(b-2) ≥ 2) ∧ (2*a + b ≥ 8) := by
  sorry

end problem_statement_l2328_232869


namespace rectangle_width_l2328_232828

theorem rectangle_width (length width : ℝ) : 
  width = length + 3 →
  2 * length + 2 * width = 54 →
  width = 15 := by
sorry

end rectangle_width_l2328_232828


namespace even_function_property_l2328_232805

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The main theorem -/
theorem even_function_property (f : ℝ → ℝ) 
  (h_even : IsEven f) 
  (h_neg : ∀ x < 0, f x = 3 * x - 1) : 
  ∀ x > 0, f x = -3 * x - 1 := by
sorry

end even_function_property_l2328_232805


namespace field_trip_girls_l2328_232884

theorem field_trip_girls (num_vans : ℕ) (students_per_van : ℕ) (num_boys : ℕ) : 
  num_vans = 5 → 
  students_per_van = 28 → 
  num_boys = 60 → 
  num_vans * students_per_van - num_boys = 80 :=
by
  sorry

end field_trip_girls_l2328_232884


namespace total_cost_of_seeds_bottles_not_enough_l2328_232886

-- Define the given values
def seed_price : ℝ := 9.48
def seed_amount : ℝ := 3.3
def bottle_capacity : ℝ := 0.35
def num_bottles : ℕ := 9

-- Theorem for the total cost of grass seeds
theorem total_cost_of_seeds : seed_price * seed_amount = 31.284 := by sorry

-- Theorem for the insufficiency of 9 bottles
theorem bottles_not_enough : seed_amount > (bottle_capacity * num_bottles) := by sorry

end total_cost_of_seeds_bottles_not_enough_l2328_232886


namespace correct_algebraic_simplification_l2328_232826

theorem correct_algebraic_simplification (x y : ℝ) : 3 * x^2 * y - 8 * y * x^2 = -5 * x^2 * y := by
  sorry

end correct_algebraic_simplification_l2328_232826


namespace interest_rate_calculation_interest_rate_proof_l2328_232863

theorem interest_rate_calculation (initial_investment : ℝ) 
  (first_rate : ℝ) (first_duration : ℝ) (second_duration : ℝ) 
  (final_value : ℝ) : ℝ :=
  let first_growth := initial_investment * (1 + first_rate * first_duration / 12)
  let second_rate := ((final_value / first_growth - 1) * 12 / second_duration) * 100
  second_rate

theorem interest_rate_proof (initial_investment : ℝ) 
  (first_rate : ℝ) (first_duration : ℝ) (second_duration : ℝ) 
  (final_value : ℝ) :
  initial_investment = 12000 ∧ 
  first_rate = 0.08 ∧ 
  first_duration = 3 ∧ 
  second_duration = 3 ∧ 
  final_value = 12980 →
  interest_rate_calculation initial_investment first_rate first_duration second_duration final_value = 24 := by
  sorry

end interest_rate_calculation_interest_rate_proof_l2328_232863


namespace tangent_line_cubic_curve_l2328_232855

/-- Given a cubic function f(x) = x^3 + ax + b represented by curve C,
    if the line y = kx - 2 is tangent to C at point (1, 0),
    then k = 2 and f(x) = x^3 - x -/
theorem tangent_line_cubic_curve (a b k : ℝ) :
  let f : ℝ → ℝ := fun x ↦ x^3 + a*x + b
  let tangent_line : ℝ → ℝ := fun x ↦ k*x - 2
  (f 1 = 0) →
  (tangent_line 1 = 0) →
  (∀ x, tangent_line x ≤ f x) →
  (∃ x₀, x₀ ≠ 1 ∧ tangent_line x₀ = f x₀) →
  (k = 2 ∧ ∀ x, f x = x^3 - x) := by
  sorry

end tangent_line_cubic_curve_l2328_232855


namespace pyramid_height_equals_cube_volume_l2328_232817

theorem pyramid_height_equals_cube_volume (cube_edge : ℝ) (pyramid_base : ℝ) (pyramid_height : ℝ) :
  cube_edge = 5 →
  pyramid_base = 10 →
  cube_edge ^ 3 = (1 / 3) * pyramid_base ^ 2 * pyramid_height →
  pyramid_height = 3.75 := by
sorry

end pyramid_height_equals_cube_volume_l2328_232817


namespace product_of_largest_primes_l2328_232847

/-- The largest two-digit prime number -/
def largest_two_digit_prime : ℕ := 97

/-- The largest four-digit prime number -/
def largest_four_digit_prime : ℕ := 9973

/-- Theorem stating that the product of the largest two-digit prime and the largest four-digit prime is 967781 -/
theorem product_of_largest_primes : 
  largest_two_digit_prime * largest_four_digit_prime = 967781 := by
  sorry

end product_of_largest_primes_l2328_232847


namespace min_value_of_f_l2328_232813

-- Define the function f(x) = x^2 + 6x + 13
def f (x : ℝ) : ℝ := x^2 + 6*x + 13

-- Theorem: The minimum value of f(x) is 4 for all real x
theorem min_value_of_f : ∀ x : ℝ, f x ≥ 4 := by
  sorry

end min_value_of_f_l2328_232813


namespace magnitude_of_complex_fourth_power_l2328_232890

theorem magnitude_of_complex_fourth_power : 
  Complex.abs ((4 : ℂ) + (3 * Complex.I * Real.sqrt 3))^4 = 1849 := by
  sorry

end magnitude_of_complex_fourth_power_l2328_232890


namespace triangle_angle_measure_l2328_232860

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 84 →
  E = 4 * F + 18 →
  D + E + F = 180 →
  F = 15.6 := by
sorry

end triangle_angle_measure_l2328_232860


namespace cone_height_increase_l2328_232893

theorem cone_height_increase (h r : ℝ) (h' : ℝ) : 
  h > 0 → r > 0 → 
  ((1/3) * Real.pi * r^2 * h') = 2.9 * ((1/3) * Real.pi * r^2 * h) → 
  (h' - h) / h = 1.9 := by
sorry

end cone_height_increase_l2328_232893


namespace matrix_equation_proof_l2328_232871

theorem matrix_equation_proof : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -8; 9, 3]
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![43/7, -54/7; -33/14, 24/7]
  N * A = B := by sorry

end matrix_equation_proof_l2328_232871


namespace total_coins_l2328_232836

def coin_distribution (x : ℕ) : Prop :=
  let paul_coins := x
  let pete_coins := x * (x + 1) / 2
  pete_coins = 5 * paul_coins

theorem total_coins : ∃ x : ℕ, 
  coin_distribution x ∧ 
  x + 5 * x = 54 := by
  sorry

end total_coins_l2328_232836


namespace exists_valid_layout_18_rectangles_l2328_232845

/-- Represents a rectangle with width and height --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a position on a 2D grid --/
structure Position where
  x : ℕ
  y : ℕ

/-- Represents a layout of rectangles on a grid --/
def Layout := Position → Option Rectangle

/-- Checks if two positions are adjacent --/
def adjacent (p1 p2 : Position) : Prop :=
  (p1.x = p2.x ∧ (p1.y + 1 = p2.y ∨ p2.y + 1 = p1.y)) ∨
  (p1.y = p2.y ∧ (p1.x + 1 = p2.x ∨ p2.x + 1 = p1.x))

/-- Checks if two rectangles form a larger rectangle when adjacent --/
def formsLargerRectangle (r1 r2 : Rectangle) : Prop :=
  r1.width = r2.width ∨ r1.height = r2.height

/-- Checks if a layout satisfies the non-adjacency condition --/
def validLayout (l : Layout) : Prop :=
  ∀ p1 p2, adjacent p1 p2 →
    match l p1, l p2 with
    | some r1, some r2 => ¬formsLargerRectangle r1 r2
    | _, _ => True

/-- The main theorem: there exists a valid layout with 18 rectangles --/
theorem exists_valid_layout_18_rectangles :
  ∃ (l : Layout) (r : Rectangle),
    validLayout l ∧
    (∃ (positions : Finset Position), positions.card = 18 ∧
      ∀ p, p ∈ positions ↔ ∃ (smallR : Rectangle), l p = some smallR) :=
sorry

end exists_valid_layout_18_rectangles_l2328_232845


namespace number_of_bad_oranges_l2328_232812

/-- Given a basket with good and bad oranges, where the number of good oranges
    is known and the ratio of good to bad oranges is given, this theorem proves
    the number of bad oranges. -/
theorem number_of_bad_oranges
  (good_oranges : ℕ)
  (ratio_good : ℕ)
  (ratio_bad : ℕ)
  (h1 : good_oranges = 24)
  (h2 : ratio_good = 3)
  (h3 : ratio_bad = 1)
  : ∃ bad_oranges : ℕ, bad_oranges = 8 ∧ good_oranges * ratio_bad = bad_oranges * ratio_good :=
by
  sorry


end number_of_bad_oranges_l2328_232812


namespace tennis_tournament_rounds_l2328_232882

theorem tennis_tournament_rounds :
  ∀ (rounds : ℕ)
    (games_per_round : List ℕ)
    (cans_per_game : ℕ)
    (balls_per_can : ℕ)
    (total_balls : ℕ),
  games_per_round = [8, 4, 2, 1] →
  cans_per_game = 5 →
  balls_per_can = 3 →
  total_balls = 225 →
  (List.sum games_per_round * cans_per_game * balls_per_can = total_balls) →
  rounds = 4 := by
sorry

end tennis_tournament_rounds_l2328_232882


namespace field_width_proof_l2328_232816

/-- Proves that a rectangular field with given conditions has a width of 20 feet -/
theorem field_width_proof (total_tape : ℝ) (field_length : ℝ) (leftover_tape : ℝ) 
  (h1 : total_tape = 250)
  (h2 : field_length = 60)
  (h3 : leftover_tape = 90) :
  let used_tape := total_tape - leftover_tape
  let perimeter := used_tape
  let width := (perimeter - 2 * field_length) / 2
  width = 20 := by sorry

end field_width_proof_l2328_232816


namespace binary_sum_equals_11101101_l2328_232872

/-- The sum of specific binary numbers equals 11101101₂ -/
theorem binary_sum_equals_11101101 :
  (0b10101 : Nat) + (0b11 : Nat) + (0b1010 : Nat) + (0b11100 : Nat) + (0b1101 : Nat) = (0b11101101 : Nat) := by
  sorry

end binary_sum_equals_11101101_l2328_232872


namespace hexagon_tessellation_l2328_232809

-- Define a hexagon
structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ

-- Define properties of the hexagon
def is_convex (h : Hexagon) : Prop :=
  sorry

def has_parallel_opposite_sides (h : Hexagon) : Prop :=
  sorry

def parallel_sides_length_one (h : Hexagon) : Prop :=
  sorry

-- Define tessellation
def can_tessellate_plane (h : Hexagon) : Prop :=
  sorry

-- Theorem statement
theorem hexagon_tessellation :
  ∃ (h : Hexagon), 
    is_convex h ∧ 
    has_parallel_opposite_sides h ∧ 
    parallel_sides_length_one h ∧ 
    can_tessellate_plane h :=
sorry

end hexagon_tessellation_l2328_232809


namespace sequence_has_repeating_pair_l2328_232862

def is_valid_sequence (a : Fin 99 → Fin 10) : Prop :=
  ∀ n : Fin 98, (a n = 1 → a (n + 1) ≠ 2) ∧ (a n = 3 → a (n + 1) ≠ 4)

theorem sequence_has_repeating_pair (a : Fin 99 → Fin 10) (h : is_valid_sequence a) :
  ∃ k l : Fin 98, k ≠ l ∧ a k = a l ∧ a (k + 1) = a (l + 1) := by
  sorry

end sequence_has_repeating_pair_l2328_232862


namespace green_lab_coat_pairs_l2328_232806

theorem green_lab_coat_pairs 
  (total_students : ℕ) 
  (white_coat_students : ℕ) 
  (green_coat_students : ℕ) 
  (total_pairs : ℕ) 
  (white_white_pairs : ℕ) 
  (h1 : total_students = 142)
  (h2 : white_coat_students = 68)
  (h3 : green_coat_students = 74)
  (h4 : total_pairs = 71)
  (h5 : white_white_pairs = 29)
  (h6 : total_students = white_coat_students + green_coat_students)
  (h7 : total_students = 2 * total_pairs) :
  ∃ (green_green_pairs : ℕ), green_green_pairs = 32 ∧ 
    green_green_pairs + white_white_pairs + (white_coat_students - 2 * white_white_pairs) = total_pairs :=
by
  sorry

end green_lab_coat_pairs_l2328_232806


namespace divisibility_of_factorial_products_l2328_232838

theorem divisibility_of_factorial_products (a b : ℕ) : 
  Nat.Prime (a + b + 1) → 
  (∃ k : ℤ, (k = a.factorial * b.factorial + 1 ∨ k = a.factorial * b.factorial - 1) ∧ 
   (a + b + 1 : ℤ) ∣ k) := by
sorry

end divisibility_of_factorial_products_l2328_232838


namespace junior_score_l2328_232840

theorem junior_score (n : ℝ) (h_pos : n > 0) : 
  let junior_count := 0.2 * n
  let senior_count := 0.8 * n
  let total_score := 86 * n
  let senior_score := 85 * senior_count
  junior_count * (total_score - senior_score) / junior_count = 90 :=
by sorry

end junior_score_l2328_232840


namespace complex_equation_solution_l2328_232864

theorem complex_equation_solution :
  ∃ z : ℂ, z^2 - 4*z + 21 = 0 :=
by
  -- Proof goes here
  sorry

end complex_equation_solution_l2328_232864


namespace repeating_decimal_property_l2328_232829

def is_repeating_decimal_period_2 (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ 1 / n = (10 * a + b) / 99

def is_repeating_decimal_period_3 (n : ℕ) : Prop :=
  ∃ (u v w : ℕ), u < 10 ∧ v < 10 ∧ w < 10 ∧ 1 / n = (100 * u + 10 * v + w) / 999

theorem repeating_decimal_property (n : ℕ) :
  n > 0 ∧ n < 3000 ∧
  is_repeating_decimal_period_2 n ∧
  is_repeating_decimal_period_3 (n + 8) →
  601 ≤ n ∧ n ≤ 1200 := by
  sorry

end repeating_decimal_property_l2328_232829


namespace units_digit_G_100_l2328_232897

/-- Modified Fermat number -/
def G (n : ℕ) : ℕ := 3^(2^n) + 1

/-- Units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_G_100 : units_digit (G 100) = 2 := by
  sorry

end units_digit_G_100_l2328_232897


namespace distance_from_negative_two_l2328_232837

theorem distance_from_negative_two : 
  {x : ℝ | |x - (-2)| = 1} = {-3, -1} := by sorry

end distance_from_negative_two_l2328_232837


namespace z_in_second_quadrant_l2328_232808

def z : ℂ := (2 + Complex.I) * Complex.I

theorem z_in_second_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
sorry

end z_in_second_quadrant_l2328_232808


namespace milton_zoology_books_l2328_232892

theorem milton_zoology_books : 
  ∀ (z b : ℕ), 
    z + b = 80 → 
    b = 4 * z → 
    z = 16 :=
by
  sorry

end milton_zoology_books_l2328_232892


namespace no_baby_cries_iff_even_l2328_232849

/-- Represents the direction a baby is facing -/
inductive Direction
  | Right
  | Down
  | Left
  | Up

/-- Represents a position on the grid -/
structure Position where
  x : Nat
  y : Nat

/-- Represents the state of a baby on the grid -/
structure Baby where
  pos : Position
  dir : Direction

/-- The grid of babies -/
def Grid := List Baby

/-- Function to check if a position is within the grid -/
def isWithinGrid (n m : Nat) (pos : Position) : Prop :=
  0 ≤ pos.x ∧ pos.x < n ∧ 0 ≤ pos.y ∧ pos.y < m

/-- Function to move a baby according to the rules -/
def moveBaby (n m : Nat) (baby : Baby) : Baby :=
  sorry

/-- Function to check if any baby cries after a move -/
def anyCry (n m : Nat) (grid : Grid) : Prop :=
  sorry

/-- Main theorem: No baby cries if and only if n and m are even -/
theorem no_baby_cries_iff_even (n m : Nat) :
  (∀ (grid : Grid), ¬(anyCry n m grid)) ↔ (∃ (k l : Nat), n = 2 * k ∧ m = 2 * l) :=
  sorry

end no_baby_cries_iff_even_l2328_232849


namespace exam_average_is_36_l2328_232895

/-- The overall average of marks obtained by all boys in an examination. -/
def overall_average (total_boys : ℕ) (passed_boys : ℕ) (avg_passed : ℕ) (avg_failed : ℕ) : ℚ :=
  let failed_boys := total_boys - passed_boys
  ((passed_boys * avg_passed + failed_boys * avg_failed) : ℚ) / total_boys

/-- Theorem stating that the overall average of marks is 36 given the conditions. -/
theorem exam_average_is_36 :
  overall_average 120 105 39 15 = 36 := by
  sorry

end exam_average_is_36_l2328_232895


namespace height_to_radius_ratio_l2328_232885

/-- A regular triangular prism -/
structure RegularTriangularPrism where
  /-- The cosine of the dihedral angle between a face and the base -/
  cos_dihedral_angle : ℝ
  /-- The height of the prism -/
  height : ℝ
  /-- The radius of the inscribed sphere -/
  inscribed_radius : ℝ

/-- Theorem: For a regular triangular prism where the cosine of the dihedral angle 
    between a face and the base is 1/6, the ratio of the height to the radius 
    of the inscribed sphere is 7 -/
theorem height_to_radius_ratio (prism : RegularTriangularPrism) 
    (h : prism.cos_dihedral_angle = 1/6) : 
    prism.height / prism.inscribed_radius = 7 := by
  sorry

end height_to_radius_ratio_l2328_232885


namespace percentage_relation_l2328_232853

theorem percentage_relation (x y z w : ℝ) 
  (h1 : x = 1.2 * y)
  (h2 : y = 0.4 * z)
  (h3 : z = 0.7 * w) :
  x = 0.336 * w := by
sorry

end percentage_relation_l2328_232853


namespace prime_extension_l2328_232842

theorem prime_extension (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) := by
  sorry

end prime_extension_l2328_232842


namespace sum_interior_eighth_row_l2328_232811

/-- Sum of interior numbers in a row of Pascal's Triangle -/
def sum_interior (n : ℕ) : ℕ := 2^(n-1) - 2

/-- The row number where interior numbers begin in Pascal's Triangle -/
def interior_start : ℕ := 3

theorem sum_interior_eighth_row :
  sum_interior 6 = 30 →
  sum_interior 8 = 126 :=
by sorry

end sum_interior_eighth_row_l2328_232811


namespace min_lcm_x_z_l2328_232832

def problem (x y z : ℕ) : Prop :=
  Nat.lcm x y = 20 ∧ Nat.lcm y z = 28

theorem min_lcm_x_z (x y z : ℕ) (h : problem x y z) :
  Nat.lcm x z ≥ 35 :=
sorry

end min_lcm_x_z_l2328_232832


namespace negation_of_proposition_l2328_232841

theorem negation_of_proposition (x₀ : ℝ) : 
  ¬(x₀^2 + 2*x₀ + 2 ≤ 0) ↔ (x₀^2 + 2*x₀ + 2 > 0) := by sorry

end negation_of_proposition_l2328_232841


namespace intersection_of_A_and_B_l2328_232868

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end intersection_of_A_and_B_l2328_232868


namespace cake_cutting_l2328_232852

/-- Represents a rectangular grid --/
structure RectangularGrid :=
  (rows : ℕ)
  (cols : ℕ)

/-- The maximum number of pieces created by a single straight line cut in a rectangular grid --/
def max_pieces (grid : RectangularGrid) : ℕ :=
  grid.rows * grid.cols + (grid.rows + grid.cols - 1)

/-- The minimum number of straight cuts required to intersect all cells in a rectangular grid --/
def min_cuts (grid : RectangularGrid) : ℕ :=
  min grid.rows grid.cols

theorem cake_cutting (grid : RectangularGrid) 
  (h1 : grid.rows = 3) 
  (h2 : grid.cols = 5) : 
  max_pieces grid = 22 ∧ min_cuts grid = 3 := by
  sorry

#eval max_pieces ⟨3, 5⟩
#eval min_cuts ⟨3, 5⟩

end cake_cutting_l2328_232852


namespace circle_parameter_range_l2328_232823

/-- Represents the equation of a potential circle -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 4*y + 5*a = 0

/-- Determines if the equation represents a valid circle -/
def is_valid_circle (a : ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y a ↔ (x - h)^2 + (y - k)^2 = r^2

/-- The main theorem stating the range of 'a' for which the equation represents a circle -/
theorem circle_parameter_range :
  ∀ a : ℝ, is_valid_circle a ↔ (a > 4 ∨ a < 1) :=
sorry

end circle_parameter_range_l2328_232823


namespace fred_basketball_games_l2328_232801

/-- The number of basketball games Fred went to last year -/
def last_year_games : ℕ := 36

/-- The number of games less Fred went to this year compared to last year -/
def games_difference : ℕ := 11

/-- The number of basketball games Fred went to this year -/
def this_year_games : ℕ := last_year_games - games_difference

theorem fred_basketball_games : this_year_games = 25 := by
  sorry

end fred_basketball_games_l2328_232801


namespace rays_fish_market_rays_fish_market_specific_l2328_232867

/-- The number of customers who will not receive fish in Mr. Ray's fish market scenario -/
theorem rays_fish_market (total_customers : ℕ) (num_tuna : ℕ) (tuna_weight : ℕ) (customer_request : ℕ) : ℕ :=
  let total_fish := num_tuna * tuna_weight
  let served_customers := total_fish / customer_request
  total_customers - served_customers

/-- Proof of the specific scenario in Mr. Ray's fish market -/
theorem rays_fish_market_specific : rays_fish_market 100 10 200 25 = 20 := by
  sorry

end rays_fish_market_rays_fish_market_specific_l2328_232867


namespace dogs_sold_l2328_232820

theorem dogs_sold (cats : ℕ) (dogs : ℕ) (ratio : ℚ) : 
  ratio = 2 / 1 → cats = 16 → dogs = 8 := by
  sorry

end dogs_sold_l2328_232820


namespace parabola_and_line_intersection_l2328_232843

-- Define the parabola and line
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = -2*p*y
def line1 (x y : ℝ) : Prop := y = (1/2)*x - 1
def line2 (k : ℝ) (x y : ℝ) : Prop := y = k*x - 3/2

-- Define the theorem
theorem parabola_and_line_intersection 
  (p : ℝ) 
  (x_M y_M x_N y_N : ℝ) 
  (h_p : p > 0)
  (h_intersect1 : line1 x_M y_M ∧ line1 x_N y_N)
  (h_parabola1 : parabola p x_M y_M ∧ parabola p x_N y_N)
  (h_condition : (x_M + 1) * (x_N + 1) = -8)
  (k : ℝ)
  (x_A y_A x_B y_B : ℝ)
  (h_k : k ≠ 0)
  (h_intersect2 : line2 k x_A y_A ∧ line2 k x_B y_B)
  (h_parabola2 : parabola p x_A y_A ∧ parabola p x_B y_B)
  (x_A' : ℝ)
  (h_symmetric : x_A' = -x_A) :
  (∀ x y, parabola p x y ↔ x^2 = -6*y) ∧
  (∃ t : ℝ, t = (y_B - y_A) / (x_B - x_A') ∧ 
            0 = t * 0 + y_A - t * x_A' ∧
            3/2 = t * 0 + y_A - t * x_A') :=
by sorry

end parabola_and_line_intersection_l2328_232843


namespace remainder_divisibility_l2328_232857

theorem remainder_divisibility (x : ℤ) : x % 72 = 19 → x % 8 = 3 := by
  sorry

end remainder_divisibility_l2328_232857


namespace f_inequality_solution_f_minimum_value_l2328_232818

def f (x : ℝ) := |x - 2| + |x + 1|

theorem f_inequality_solution (x : ℝ) :
  f x > 4 ↔ x < -1.5 ∨ x > 2.5 := by sorry

theorem f_minimum_value :
  ∃ (a : ℝ), (∀ x, f x ≥ a) ∧ (∀ b, (∀ x, f x ≥ b) → b ≤ a) ∧ a = 3 := by sorry

end f_inequality_solution_f_minimum_value_l2328_232818


namespace rectangle_diagonals_equal_diagonals_equal_not_always_rectangle_not_rectangle_diagonals_not_equal_not_always_diagonals_not_equal_not_rectangle_l2328_232883

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define what it means for a quadrilateral to be a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  sorry

-- Define what it means for diagonals to be equal
def diagonals_equal (q : Quadrilateral) : Prop :=
  sorry

-- Theorem statements
theorem rectangle_diagonals_equal (q : Quadrilateral) :
  is_rectangle q → diagonals_equal q :=
sorry

theorem diagonals_equal_not_always_rectangle :
  ∃ q : Quadrilateral, diagonals_equal q ∧ ¬is_rectangle q :=
sorry

theorem not_rectangle_diagonals_not_equal_not_always :
  ∃ q : Quadrilateral, ¬is_rectangle q ∧ diagonals_equal q :=
sorry

theorem diagonals_not_equal_not_rectangle (q : Quadrilateral) :
  ¬diagonals_equal q → ¬is_rectangle q :=
sorry

end rectangle_diagonals_equal_diagonals_equal_not_always_rectangle_not_rectangle_diagonals_not_equal_not_always_diagonals_not_equal_not_rectangle_l2328_232883


namespace distance_is_15_miles_l2328_232891

/-- Represents the walking scenario with distance, speed, and time. -/
structure WalkScenario where
  distance : ℝ
  speed : ℝ
  time : ℝ

/-- The original walking scenario. -/
def original : WalkScenario := sorry

/-- The scenario with increased speed. -/
def increased_speed : WalkScenario := sorry

/-- The scenario with decreased speed. -/
def decreased_speed : WalkScenario := sorry

theorem distance_is_15_miles :
  (∀ s : WalkScenario, s.distance = s.speed * s.time) →
  (increased_speed.speed = original.speed + 0.5) →
  (increased_speed.time = 4/5 * original.time) →
  (decreased_speed.speed = original.speed - 0.5) →
  (decreased_speed.time = original.time + 2.5) →
  (original.distance = increased_speed.distance) →
  (original.distance = decreased_speed.distance) →
  original.distance = 15 := by
  sorry

end distance_is_15_miles_l2328_232891


namespace polynomial_negative_values_l2328_232875

theorem polynomial_negative_values (a x : ℝ) (h : 0 < x ∧ x < a) : 
  (a - x)^6 - 3*a*(a - x)^5 + 5/2*a^2*(a - x)^4 - 1/2*a^4*(a - x)^2 < 0 := by
  sorry

end polynomial_negative_values_l2328_232875


namespace complex_number_properties_l2328_232896

theorem complex_number_properties : ∃ (z : ℂ), 
  z = 2 / (Complex.I - 1) ∧ 
  z^2 = 2 * Complex.I ∧ 
  z.im = -1 := by
  sorry

end complex_number_properties_l2328_232896


namespace bird_families_count_l2328_232877

/-- The number of bird families that flew away for winter -/
def flew_away : ℕ := 32

/-- The number of bird families that stayed near the mountain -/
def stayed : ℕ := 35

/-- The initial number of bird families living near the mountain -/
def initial_families : ℕ := flew_away + stayed

theorem bird_families_count : initial_families = 67 := by
  sorry

end bird_families_count_l2328_232877


namespace division_equality_l2328_232819

theorem division_equality : (180 : ℚ) / (12 + 13 * 2) = 90 / 19 := by sorry

end division_equality_l2328_232819


namespace fraction_sum_zero_l2328_232876

theorem fraction_sum_zero (a b c : ℝ) 
  (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 := by
  sorry

end fraction_sum_zero_l2328_232876


namespace work_completion_time_l2328_232856

/-- Given two workers X and Y, where X can finish a job in 21 days and Y in 15 days,
    if Y works for 5 days and then leaves, prove that X needs 14 days to finish the remaining work. -/
theorem work_completion_time (x_rate y_rate : ℚ) (y_days : ℕ) :
  x_rate = 1 / 21 →
  y_rate = 1 / 15 →
  y_days = 5 →
  (1 - y_rate * y_days) / x_rate = 14 := by
  sorry

end work_completion_time_l2328_232856


namespace sqrt_5_is_simplest_l2328_232861

/-- A quadratic radical is considered simplest if it cannot be simplified further
    and does not have denominators under the square root. -/
def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, x = Real.sqrt y → (∀ z : ℝ, y ≠ z^2) ∧ (∀ n : ℕ, n > 1 → y ≠ Real.sqrt n)

/-- The theorem states that √5 is the simplest quadratic radical among the given options. -/
theorem sqrt_5_is_simplest :
  is_simplest_quadratic_radical (Real.sqrt 5) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 8) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (a^2)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (0.2 * b)) :=
sorry

end sqrt_5_is_simplest_l2328_232861


namespace max_value_fraction_l2328_232831

theorem max_value_fraction (x : ℝ) :
  x^4 / (x^8 + 2*x^6 - 4*x^4 + 8*x^2 + 16) ≤ 1/12 ∧
  ∃ y : ℝ, y^4 / (y^8 + 2*y^6 - 4*y^4 + 8*y^2 + 16) = 1/12 :=
by sorry

end max_value_fraction_l2328_232831


namespace williams_points_l2328_232834

/-- The number of classes in the contest -/
def num_classes : ℕ := 4

/-- Points scored by Mr. Adams' class -/
def adams_points : ℕ := 57

/-- Points scored by Mrs. Brown's class -/
def brown_points : ℕ := 49

/-- Points scored by Mrs. Daniel's class -/
def daniel_points : ℕ := 57

/-- The mean of the number of points scored -/
def mean_points : ℚ := 53.3

/-- Theorem stating that Mrs. William's class scored 50 points -/
theorem williams_points : ℕ := by
  sorry

end williams_points_l2328_232834


namespace greatest_x_value_l2328_232803

def is_prime_power (n : ℕ) : Prop :=
  ∃ p k, p.Prime ∧ k > 0 ∧ n = p ^ k

theorem greatest_x_value (x : ℕ) 
  (h1 : Nat.lcm x (Nat.lcm 15 21) = 105)
  (h2 : is_prime_power x) :
  x ≤ 7 ∧ (∀ y, y > x → is_prime_power y → Nat.lcm y (Nat.lcm 15 21) ≠ 105) :=
sorry

end greatest_x_value_l2328_232803


namespace range_of_a_l2328_232874

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x) 
  (h2 : ∃ x : ℝ, x^2 + 4*x + a = 0) : 
  Real.exp 1 ≤ a ∧ a ≤ 4 := by
  sorry

end range_of_a_l2328_232874


namespace credit_card_balance_proof_l2328_232844

/-- Calculates the new credit card balance after purchases and returns -/
def new_credit_card_balance (initial_balance groceries_cost towels_return : ℚ) : ℚ :=
  initial_balance + groceries_cost + (groceries_cost / 2) - towels_return

/-- Proves that the new credit card balance is correct given the initial conditions -/
theorem credit_card_balance_proof :
  new_credit_card_balance 126 60 45 = 171 := by
  sorry

#eval new_credit_card_balance 126 60 45

end credit_card_balance_proof_l2328_232844


namespace midpoint_distance_theorem_l2328_232873

theorem midpoint_distance_theorem (t : ℝ) : 
  let A : ℝ × ℝ := (2*t - 3, t)
  let B : ℝ × ℝ := (t - 1, 2*t + 4)
  let midpoint : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ((midpoint.1 - A.1)^2 + (midpoint.2 - A.2)^2) = (t^2 + t) / 2 →
  t = -10 := by
sorry

end midpoint_distance_theorem_l2328_232873


namespace abs_neg_two_l2328_232880

theorem abs_neg_two : |(-2 : ℤ)| = 2 := by sorry

end abs_neg_two_l2328_232880


namespace christophers_age_l2328_232879

theorem christophers_age (christopher george ford : ℕ) : 
  george = christopher + 8 →
  ford = christopher - 2 →
  christopher + george + ford = 60 →
  christopher = 18 := by
sorry

end christophers_age_l2328_232879


namespace chase_travel_time_l2328_232887

/-- Represents the travel time between Granville and Salisbury -/
def travel_time (speed : ℝ) : ℝ := sorry

theorem chase_travel_time :
  let chase_speed : ℝ := 1
  let cameron_speed : ℝ := 2 * chase_speed
  let danielle_speed : ℝ := 3 * cameron_speed
  let danielle_time : ℝ := 30

  travel_time chase_speed = 180 := by sorry

end chase_travel_time_l2328_232887


namespace common_ratio_of_geometric_sequence_l2328_232824

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem common_ratio_of_geometric_sequence
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geometric : geometric_sequence a q)
  (h_odd_product : a 1 * a 3 * a 5 * a 7 * a 9 = 2)
  (h_even_product : a 2 * a 4 * a 6 * a 8 * a 10 = 64) :
  q = 2 :=
sorry

end common_ratio_of_geometric_sequence_l2328_232824


namespace hyperbola_line_intersection_l2328_232888

/-- The hyperbola equation -/
def hyperbola (x y b : ℝ) : Prop := x^2 / 4 - y^2 / b^2 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := x + 3*y - 1 = 0

/-- The condition for intersection based on slope comparison -/
def intersection_condition (b : ℝ) : Prop := b > 2/3

/-- The theorem stating that b > 1 is sufficient but not necessary for intersection -/
theorem hyperbola_line_intersection (b : ℝ) (h : b > 0) :
  (∀ x y, hyperbola x y b → line x y → intersection_condition b) ∧
  ¬(∀ b, intersection_condition b → b > 1) :=
sorry

end hyperbola_line_intersection_l2328_232888
