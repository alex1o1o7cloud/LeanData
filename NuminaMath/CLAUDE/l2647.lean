import Mathlib

namespace collinear_vectors_k_value_l2647_264779

/-- Given two non-collinear vectors in a real vector space, 
    if certain conditions are met, then k = -8. -/
theorem collinear_vectors_k_value 
  (V : Type*) [AddCommGroup V] [Module ℝ V] 
  (e₁ e₂ : V) (k : ℝ) 
  (h_non_collinear : ¬ ∃ (r : ℝ), e₁ = r • e₂) 
  (AB CB CD : V) 
  (h_AB : AB = 2 • e₁ + k • e₂) 
  (h_CB : CB = e₁ + 3 • e₂) 
  (h_CD : CD = 2 • e₁ - e₂) 
  (h_collinear : ∃ (t : ℝ), AB = t • (CD - CB)) : 
  k = -8 := by
sorry

end collinear_vectors_k_value_l2647_264779


namespace bell_ringing_fraction_l2647_264711

theorem bell_ringing_fraction :
  let big_bell_rings : ℕ := 36
  let total_rings : ℕ := 52
  let small_bell_rings (f : ℚ) : ℚ := f * big_bell_rings + 4

  ∃ f : ℚ, f = 1/3 ∧ (↑big_bell_rings : ℚ) + small_bell_rings f = total_rings := by
  sorry

end bell_ringing_fraction_l2647_264711


namespace binomial_variance_determines_n_l2647_264759

/-- A random variable following a binomial distribution -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_variance_determines_n (ξ : BinomialDistribution) 
  (h_p : ξ.p = 0.3) 
  (h_var : variance ξ = 2.1) : 
  ξ.n = 10 := by
sorry

end binomial_variance_determines_n_l2647_264759


namespace water_displacement_squared_l2647_264753

def cube_side_length : ℝ := 12
def tank_radius : ℝ := 6
def tank_height : ℝ := 15

theorem water_displacement_squared :
  let cube_volume := cube_side_length ^ 3
  let cube_diagonal := cube_side_length * Real.sqrt 3
  cube_diagonal ≤ tank_height →
  (cube_volume ^ 2 : ℝ) = 2985984 := by sorry

end water_displacement_squared_l2647_264753


namespace exists_A_square_diff_two_l2647_264761

/-- The ceiling function -/
noncomputable def ceil (x : ℝ) : ℤ :=
  Int.floor x + 1

/-- Main theorem -/
theorem exists_A_square_diff_two :
  ∃ A : ℝ, ∀ n : ℕ, ∃ m : ℤ, |A^n - m^2| = 2 :=
sorry

end exists_A_square_diff_two_l2647_264761


namespace polynomial_degree_is_8_l2647_264793

def polynomial_degree (x : ℝ) : ℕ :=
  let expr1 := x^7
  let expr2 := x + 1/x
  let expr3 := 1 + 3/x + 5/(x^2)
  let result := expr1 * expr2 * expr3
  8  -- The degree of the resulting polynomial

theorem polynomial_degree_is_8 : 
  ∀ x : ℝ, x ≠ 0 → polynomial_degree x = 8 := by
  sorry

end polynomial_degree_is_8_l2647_264793


namespace even_function_m_value_l2647_264708

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = x^4 + (m-1)x + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  x^4 + (m-1)*x + 1

theorem even_function_m_value :
  ∀ m : ℝ, IsEven (f m) → m = 1 := by
  sorry

end even_function_m_value_l2647_264708


namespace sales_tax_percentage_l2647_264742

-- Define the prices and quantities
def tshirt_price : ℚ := 8
def sweater_price : ℚ := 18
def jacket_price : ℚ := 80
def jacket_discount : ℚ := 0.1
def tshirt_quantity : ℕ := 6
def sweater_quantity : ℕ := 4
def jacket_quantity : ℕ := 5

-- Define the total cost before tax
def total_cost_before_tax : ℚ :=
  tshirt_price * tshirt_quantity +
  sweater_price * sweater_quantity +
  jacket_price * jacket_quantity * (1 - jacket_discount)

-- Define the total cost including tax
def total_cost_with_tax : ℚ := 504

-- Theorem: The sales tax percentage is 5%
theorem sales_tax_percentage :
  ∃ (tax_rate : ℚ), 
    tax_rate = 0.05 ∧
    total_cost_with_tax = total_cost_before_tax * (1 + tax_rate) :=
sorry

end sales_tax_percentage_l2647_264742


namespace plant_equation_correct_l2647_264767

/-- Represents the structure of a plant with branches and small branches. -/
structure Plant where
  branches : ℕ
  smallBranches : ℕ

/-- The total number of parts in a plant, including the main stem. -/
def totalParts (p : Plant) : ℕ := 1 + p.branches + p.smallBranches

/-- Constructs a plant where each branch grows a specific number of small branches. -/
def makePlant (x : ℕ) : Plant :=
  { branches := x, smallBranches := x * x }

/-- Theorem stating that for some natural number x, the plant structure
    results in a total of 91 parts. -/
theorem plant_equation_correct :
  ∃ x : ℕ, totalParts (makePlant x) = 91 := by
  sorry

end plant_equation_correct_l2647_264767


namespace complement_of_union_equals_set_l2647_264746

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 3}
def N : Set Nat := {1, 2}

theorem complement_of_union_equals_set (U M N : Set Nat) 
  (hU : U = {1, 2, 3, 4, 5}) 
  (hM : M = {1, 3}) 
  (hN : N = {1, 2}) : 
  (M ∪ N)ᶜ = {4, 5} := by
  sorry

end complement_of_union_equals_set_l2647_264746


namespace flight_portion_cost_l2647_264788

theorem flight_portion_cost (total_cost ground_cost flight_additional_cost : ℕ) :
  total_cost = 1275 →
  flight_additional_cost = 625 →
  ground_cost = 325 →
  ground_cost + flight_additional_cost = 950 := by
  sorry

end flight_portion_cost_l2647_264788


namespace watermelon_seeds_l2647_264773

theorem watermelon_seeds (total_slices : ℕ) (black_seeds_per_slice : ℕ) (total_seeds : ℕ) :
  total_slices = 40 →
  black_seeds_per_slice = 20 →
  total_seeds = 1600 →
  (total_seeds - total_slices * black_seeds_per_slice) / total_slices = 20 :=
by sorry

end watermelon_seeds_l2647_264773


namespace smallest_value_of_expression_l2647_264739

-- Define the complex cube root of unity
noncomputable def ω : ℂ := sorry

-- Define the theorem
theorem smallest_value_of_expression (a b c : ℤ) :
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) →  -- non-zero integers
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) →  -- distinct integers
  (Even a ∧ Even b ∧ Even c) →  -- even integers
  (ω^3 = 1 ∧ ω ≠ 1) →  -- properties of ω
  ∃ (min : ℝ), 
    min = Real.sqrt 12 ∧
    ∀ (x y z : ℤ), 
      (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) →
      (x ≠ y ∧ y ≠ z ∧ x ≠ z) →
      (Even x ∧ Even y ∧ Even z) →
      Complex.abs (x + y • ω + z • ω^2) ≥ min :=
sorry

end smallest_value_of_expression_l2647_264739


namespace shirt_count_proof_l2647_264732

/-- The number of different colored neckties -/
def num_neckties : ℕ := 6

/-- The probability that all boxes contain matching necktie-shirt pairs -/
def matching_probability : ℝ := 0.041666666666666664

/-- The number of different colored shirts -/
def num_shirts : ℕ := 2

theorem shirt_count_proof :
  (1 / num_shirts : ℝ) ^ num_neckties = matching_probability ∧
  num_shirts = ⌈(1 / matching_probability) ^ (1 / num_neckties : ℝ)⌉ := by
  sorry

#check shirt_count_proof

end shirt_count_proof_l2647_264732


namespace parabola_shift_l2647_264771

/-- Given a parabola y = x^2 + bx + c that is shifted 4 units to the right
    and 3 units down to become y = x^2 - 4x + 3, prove that b = 4 and c = 6. -/
theorem parabola_shift (b c : ℝ) : 
  (∀ x, (x - 4)^2 + b*(x - 4) + c - 3 = x^2 - 4*x + 3) → 
  b = 4 ∧ c = 6 := by
sorry

end parabola_shift_l2647_264771


namespace sign_up_ways_for_six_students_three_projects_l2647_264796

/-- The number of ways students can sign up for projects -/
def signUpWays (numStudents : ℕ) (numProjects : ℕ) : ℕ :=
  numProjects ^ numStudents

/-- Theorem: For 6 students and 3 projects, the number of ways to sign up is 3^6 -/
theorem sign_up_ways_for_six_students_three_projects :
  signUpWays 6 3 = 3^6 := by
  sorry

end sign_up_ways_for_six_students_three_projects_l2647_264796


namespace impossible_arrangement_l2647_264794

def numbers : List ℕ := [1, 4, 9, 16, 25, 36, 49, 64, 81]

def radial_lines : ℕ := 6

def appears_twice (n : ℕ) : Prop := ∃ (l₁ l₂ : List ℕ), l₁ ≠ l₂ ∧ n ∈ l₁ ∧ n ∈ l₂

theorem impossible_arrangement : 
  ¬∃ (arrangement : List (List ℕ)), 
    (∀ n ∈ numbers, appears_twice n) ∧ 
    (arrangement.length = radial_lines) ∧
    (∃ (s : ℕ), ∀ l ∈ arrangement, l.sum = s) :=
sorry

end impossible_arrangement_l2647_264794


namespace evaluate_expression_l2647_264747

theorem evaluate_expression (x y z : ℚ) : 
  x = 1/4 → y = 3/4 → z = 3 → x^2 * y^3 * z = 81/1024 := by
  sorry

end evaluate_expression_l2647_264747


namespace paths_H_to_J_through_I_l2647_264790

/-- The number of paths from H to I -/
def paths_H_to_I : ℕ := Nat.choose 6 1

/-- The number of paths from I to J -/
def paths_I_to_J : ℕ := Nat.choose 5 2

/-- The total number of steps from H to J -/
def total_steps : ℕ := 11

/-- Theorem stating the number of paths from H to J passing through I -/
theorem paths_H_to_J_through_I : paths_H_to_I * paths_I_to_J = 60 :=
sorry

end paths_H_to_J_through_I_l2647_264790


namespace sneezing_fit_proof_l2647_264752

/-- Calculates the number of sneezes given the duration of a sneezing fit in minutes
    and the interval between sneezes in seconds. -/
def number_of_sneezes (duration_minutes : ℕ) (interval_seconds : ℕ) : ℕ :=
  (duration_minutes * 60) / interval_seconds

/-- Proves that a 2-minute sneezing fit with sneezes every 3 seconds results in 40 sneezes. -/
theorem sneezing_fit_proof :
  number_of_sneezes 2 3 = 40 := by
  sorry

end sneezing_fit_proof_l2647_264752


namespace prime_saturated_bound_l2647_264736

def isPrimeSaturated (n : ℕ) (bound : ℕ) : Prop :=
  (Finset.prod (Nat.factors n).toFinset id) < bound

def isGreatestTwoDigitPrimeSaturated (n : ℕ) : Prop :=
  n ≤ 99 ∧ isPrimeSaturated n 96 ∧ ∀ m, m ≤ 99 → isPrimeSaturated m 96 → m ≤ n

theorem prime_saturated_bound (n : ℕ) :
  isGreatestTwoDigitPrimeSaturated 96 →
  isPrimeSaturated n (Finset.prod (Nat.factors n).toFinset id + 1) →
  Finset.prod (Nat.factors n).toFinset id < 96 :=
by sorry

end prime_saturated_bound_l2647_264736


namespace gcd_of_lcm_and_ratio_l2647_264733

theorem gcd_of_lcm_and_ratio (X Y : ℕ) (h1 : Nat.lcm X Y = 180) (h2 : X * 5 = Y * 4) :
  Nat.gcd X Y = 9 := by
  sorry

end gcd_of_lcm_and_ratio_l2647_264733


namespace linear_function_not_in_fourth_quadrant_l2647_264768

/-- A linear function with slope 1 and y-intercept 1 -/
def f (x : ℝ) : ℝ := x + 1

/-- The fourth quadrant of the Cartesian plane -/
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem linear_function_not_in_fourth_quadrant :
  ∀ x : ℝ, ¬(fourth_quadrant x (f x)) :=
sorry

end linear_function_not_in_fourth_quadrant_l2647_264768


namespace rectangular_field_diagonal_ratio_l2647_264702

theorem rectangular_field_diagonal_ratio : 
  ∀ (x y : ℝ), 
    x > 0 → y > 0 →  -- x and y are positive (representing sides of a rectangle)
    x + y - Real.sqrt (x^2 + y^2) = (2/3) * y →  -- diagonal walk saves 2/3 of longer side
    x / y = 8/9 :=  -- ratio of shorter to longer side
by
  sorry

end rectangular_field_diagonal_ratio_l2647_264702


namespace local_max_derivative_condition_l2647_264797

/-- Given a function f with derivative f'(x) = a(x+1)(x-a), 
    if f attains a local maximum at x = a, then a is in the open interval (-1, 0) -/
theorem local_max_derivative_condition (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, deriv f x = a * (x + 1) * (x - a))
  (h2 : IsLocalMax f a) :
  a ∈ Set.Ioo (-1 : ℝ) 0 := by
  sorry

end local_max_derivative_condition_l2647_264797


namespace equal_time_travel_ratio_l2647_264721

/-- The ratio of distances when travel times are equal --/
theorem equal_time_travel_ratio (x y : ℝ) (h1 : x > 0) (h2 : y > 0) :
  y / 1 = x / 1 + (x + y) / 10 → x / y = 9 / 11 := by
  sorry

#check equal_time_travel_ratio

end equal_time_travel_ratio_l2647_264721


namespace trip_time_calculation_l2647_264738

theorem trip_time_calculation (distance : ℝ) (speed1 speed2 time1 : ℝ) 
  (h1 : speed1 = 100)
  (h2 : speed2 = 50)
  (h3 : time1 = 5)
  (h4 : distance = speed1 * time1) :
  distance / speed2 = 10 := by
  sorry

end trip_time_calculation_l2647_264738


namespace prime_power_expression_l2647_264715

theorem prime_power_expression (a b : ℕ) :
  (∃ (p : ℕ) (k : ℕ), Prime p ∧ 
   (a^6 + 21*a^4*b^2 + 35*a^2*b^4 + 7*b^6) * (b^6 + 21*b^4*a^2 + 35*b^2*a^4 + 7*a^6) = p^k) ↔ 
  (∃ (i : ℕ), a = 2^i ∧ b = 2^i) :=
by sorry

end prime_power_expression_l2647_264715


namespace f_is_quadratic_l2647_264701

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² - 4 = 4 -/
def f (x : ℝ) : ℝ := x^2 - 8

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end f_is_quadratic_l2647_264701


namespace percentage_calculation_l2647_264770

theorem percentage_calculation (P : ℝ) : 
  (3/5 : ℝ) * 120 * (P/100) = 36 → P = 50 := by
  sorry

end percentage_calculation_l2647_264770


namespace survey_result_l2647_264731

theorem survey_result (total : ℕ) (tv_dislike_percent : ℚ) (both_dislike_percent : ℚ)
  (h_total : total = 1500)
  (h_tv_dislike : tv_dislike_percent = 25 / 100)
  (h_both_dislike : both_dislike_percent = 15 / 100) :
  ⌊(total : ℚ) * tv_dislike_percent * both_dislike_percent⌋ = 56 := by
  sorry

end survey_result_l2647_264731


namespace binomial_coefficient_ratio_l2647_264765

theorem binomial_coefficient_ratio (n : ℕ) : 
  (Nat.choose n 3 = 7 * Nat.choose n 1) ↔ n = 43 := by
sorry

end binomial_coefficient_ratio_l2647_264765


namespace eighth_term_is_21_l2647_264741

/-- A Fibonacci-like sequence where each number after the second is the sum of the two preceding numbers -/
def fibonacci_like_sequence (a₁ a₂ : ℕ) : ℕ → ℕ
| 0 => a₁
| 1 => a₂
| (n + 2) => fibonacci_like_sequence a₁ a₂ n + fibonacci_like_sequence a₁ a₂ (n + 1)

/-- The theorem stating that the 8th term of the specific Fibonacci-like sequence is 21 -/
theorem eighth_term_is_21 :
  ∃ (seq : ℕ → ℕ), 
    seq = fibonacci_like_sequence 1 1 ∧
    seq 7 = 21 ∧
    seq 8 = 34 ∧
    seq 9 = 55 :=
by
  sorry

end eighth_term_is_21_l2647_264741


namespace milk_dilution_l2647_264717

theorem milk_dilution (initial_volume : ℝ) (pure_milk_added : ℝ) (initial_water_percentage : ℝ) :
  initial_volume = 10 →
  pure_milk_added = 15 →
  initial_water_percentage = 5 →
  let initial_water := initial_volume * (initial_water_percentage / 100)
  let final_volume := initial_volume + pure_milk_added
  let final_water_percentage := (initial_water / final_volume) * 100
  final_water_percentage = 2 := by
  sorry

end milk_dilution_l2647_264717


namespace distance_between_points_l2647_264729

/-- The distance between points (2, 2) and (-1, -1) is 3√2 -/
theorem distance_between_points : Real.sqrt ((2 - (-1))^2 + (2 - (-1))^2) = 3 * Real.sqrt 2 := by
  sorry

end distance_between_points_l2647_264729


namespace dice_surface_sum_in_possible_sums_l2647_264775

/-- The number of dice in the arrangement -/
def num_dice : ℕ := 2012

/-- The sum of points on all faces of a standard six-sided die -/
def die_sum : ℕ := 21

/-- The sum of points on opposite faces of a die -/
def opposite_faces_sum : ℕ := 7

/-- The set of possible sums of points on the surface -/
def possible_sums : Set ℕ := {28177, 28179, 28181, 28183, 28185, 28187}

/-- Theorem: The sum of points on the surface of the arranged dice is in the set of possible sums -/
theorem dice_surface_sum_in_possible_sums :
  ∃ (x : ℕ), x ∈ possible_sums ∧
  ∃ (end_face_sum : ℕ), end_face_sum ≥ 1 ∧ end_face_sum ≤ 6 ∧
  x = num_dice * die_sum - (num_dice - 1) * opposite_faces_sum + 2 * end_face_sum :=
by
  sorry

end dice_surface_sum_in_possible_sums_l2647_264775


namespace equation_one_solution_equation_two_solution_l2647_264781

-- Equation 1
theorem equation_one_solution (x : ℝ) : 
  9 * x^2 - 25 = 0 ↔ x = 5/3 ∨ x = -5/3 := by sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) : 
  (x + 1)^3 - 27 = 0 ↔ x = 2 := by sorry

end equation_one_solution_equation_two_solution_l2647_264781


namespace hexagonal_prism_vertices_l2647_264709

/-- A prism with hexagonal bases -/
structure HexagonalPrism where
  -- The number of sides in each base
  base_sides : ℕ
  -- The number of rectangular sides
  rect_sides : ℕ
  -- The total number of vertices
  vertices : ℕ

/-- Theorem: A hexagonal prism has 12 vertices -/
theorem hexagonal_prism_vertices (p : HexagonalPrism) 
  (h1 : p.base_sides = 6)
  (h2 : p.rect_sides = 6) : 
  p.vertices = 12 := by
  sorry

end hexagonal_prism_vertices_l2647_264709


namespace gcd_of_f_is_2730_l2647_264716

-- Define the function f(n) = n^13 - n
def f (n : ℤ) : ℤ := n^13 - n

-- State the theorem
theorem gcd_of_f_is_2730 : 
  ∃ (d : ℕ), d = 2730 ∧ ∀ (n : ℤ), (f n).natAbs ∣ d ∧ 
  (∀ (m : ℕ), (∀ (k : ℤ), (f k).natAbs ∣ m) → d ∣ m) :=
sorry

end gcd_of_f_is_2730_l2647_264716


namespace back_section_total_revenue_l2647_264772

/-- Calculates the total revenue from the back section of a concert arena --/
def back_section_revenue (capacity : ℕ) (regular_price : ℚ) (half_price : ℚ) : ℚ :=
  let regular_revenue := regular_price * capacity
  let half_price_tickets := capacity / 6
  let half_price_revenue := half_price * half_price_tickets
  regular_revenue + half_price_revenue

/-- Theorem stating the total revenue from the back section --/
theorem back_section_total_revenue :
  back_section_revenue 25000 55 27.5 = 1489565 := by
  sorry

#eval back_section_revenue 25000 55 27.5

end back_section_total_revenue_l2647_264772


namespace expression_evaluation_l2647_264724

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 6 - 1
  let y : ℝ := Real.sqrt 6 + 1
  (2*x + y)^2 + (x - y)*(x + y) - 5*x*(x - y) = 45 := by
  sorry

end expression_evaluation_l2647_264724


namespace ellipse_hyperbola_shared_foci_l2647_264760

-- Define the ellipse equation
def ellipse (x y a : ℝ) : Prop := x^2 / 6 + y^2 / a^2 = 1

-- Define the hyperbola equation
def hyperbola (x y a : ℝ) : Prop := x^2 / a - y^2 / 4 = 1

-- Define the property of shared foci
def shared_foci (a : ℝ) : Prop :=
  ∀ x y : ℝ, ellipse x y a ∧ hyperbola x y a → 
    (6 - a^2).sqrt = (a + 4).sqrt

-- Theorem statement
theorem ellipse_hyperbola_shared_foci :
  ∃ a : ℝ, a > 0 ∧ shared_foci a ∧ a = 1 :=
sorry

end ellipse_hyperbola_shared_foci_l2647_264760


namespace dishes_for_equal_time_l2647_264734

/-- Represents the time taken for different chores -/
structure ChoreTime where
  sweep : ℕ  -- minutes per room
  wash : ℕ   -- minutes per dish
  laundry : ℕ -- minutes per load
  dust : ℕ   -- minutes per surface

/-- Represents the chores assigned to Anna -/
structure AnnaChores where
  rooms : ℕ
  surfaces : ℕ

/-- Represents the chores assigned to Billy -/
structure BillyChores where
  loads : ℕ
  surfaces : ℕ

/-- Calculates the total time Anna spends on chores -/
def annaTime (ct : ChoreTime) (ac : AnnaChores) : ℕ :=
  ct.sweep * ac.rooms + ct.dust * ac.surfaces

/-- Calculates the total time Billy spends on chores, excluding dishes -/
def billyTimeBeforeDishes (ct : ChoreTime) (bc : BillyChores) : ℕ :=
  ct.laundry * bc.loads + ct.dust * bc.surfaces

/-- The main theorem to prove -/
theorem dishes_for_equal_time (ct : ChoreTime) (ac : AnnaChores) (bc : BillyChores) :
  ct.sweep = 3 →
  ct.wash = 2 →
  ct.laundry = 9 →
  ct.dust = 1 →
  ac.rooms = 10 →
  ac.surfaces = 14 →
  bc.loads = 2 →
  bc.surfaces = 6 →
  ∃ (dishes : ℕ), dishes = 10 ∧
    annaTime ct ac = billyTimeBeforeDishes ct bc + ct.wash * dishes :=
by
  sorry


end dishes_for_equal_time_l2647_264734


namespace largest_certain_divisor_of_visible_product_l2647_264783

def die_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

theorem largest_certain_divisor_of_visible_product :
  ∀ (visible : Finset ℕ), visible ⊆ die_numbers → visible.card = 7 →
  ∃ (k : ℕ), (visible.prod id = 192 * k) := by
  sorry

end largest_certain_divisor_of_visible_product_l2647_264783


namespace quadratic_solution_l2647_264706

theorem quadratic_solution (h : 108 * (3/4)^2 - 35 * (3/4) - 77 = 0) :
  108 * (-23/54)^2 - 35 * (-23/54) - 77 = 0 := by
sorry

end quadratic_solution_l2647_264706


namespace total_berries_picked_l2647_264745

theorem total_berries_picked (total : ℕ) : 
  (total / 2 : ℚ) + (total / 3 : ℚ) + 7 = total → total = 42 :=
by
  sorry

end total_berries_picked_l2647_264745


namespace min_box_value_l2647_264749

/-- Given the equation (cx+d)(dx+c) = 15x^2 + ◻x + 15, where c, d, and ◻ are distinct integers,
    the minimum possible value of ◻ is 34. -/
theorem min_box_value (c d box : ℤ) : 
  (c * d + c = 15) →
  (c + d = box) →
  (c ≠ d) ∧ (c ≠ box) ∧ (d ≠ box) →
  (∀ (c' d' box' : ℤ), (c' * d' + c' = 15) → (c' + d' = box') → 
    (c' ≠ d') ∧ (c' ≠ box') ∧ (d' ≠ box') → box ≤ box') →
  box = 34 :=
by sorry

end min_box_value_l2647_264749


namespace characterize_function_l2647_264763

theorem characterize_function (n : ℕ) (hn : n ≥ 1) (hodd : Odd n) :
  ∃ (ε : Int) (d : ℕ) (c : Int),
    ε = 1 ∨ ε = -1 ∧
    d > 0 ∧
    d ∣ n ∧
    ∀ (f : ℤ → ℤ),
      (∀ (x y : ℤ), (f x - f y) ∣ (x^n - y^n)) →
      ∃ (ε' : Int) (d' : ℕ) (c' : Int),
        (ε' = 1 ∨ ε' = -1) ∧
        d' > 0 ∧
        d' ∣ n ∧
        ∀ (x : ℤ), f x = ε' * x^d' + c' :=
by sorry

end characterize_function_l2647_264763


namespace function_property_l2647_264769

def positive_integer (n : ℕ) := n > 0

theorem function_property 
  (f : ℕ → ℕ) 
  (h : ∀ n, positive_integer n → f (f n) + f (n + 1) = n + 2) : 
  ∀ n, positive_integer n → f (f n + n) = n + 1 := by
  sorry

end function_property_l2647_264769


namespace problem_statement_l2647_264791

theorem problem_statement (a b : ℝ) :
  (∀ x y : ℝ, (2 * x^2 + a * x - y + 6) - (b * x^2 - 3 * x + 5 * y - 1) = -6 * y + 7) →
  a^2 + b^2 = 13 := by
  sorry

end problem_statement_l2647_264791


namespace quadratic_function_c_bounds_l2647_264735

/-- Given a quadratic function f(x) = x² + bx + c, where b and c are real numbers,
    if 0 ≤ f(1) = f(2) ≤ 10, then 2 ≤ c ≤ 12 -/
theorem quadratic_function_c_bounds (b c : ℝ) :
  let f := fun x => x^2 + b*x + c
  (0 ≤ f 1) ∧ (f 1 = f 2) ∧ (f 2 ≤ 10) → 2 ≤ c ∧ c ≤ 12 := by
  sorry

end quadratic_function_c_bounds_l2647_264735


namespace math_books_count_l2647_264757

theorem math_books_count (total_books : ℕ) (math_book_price history_book_price : ℕ) (total_price : ℕ) :
  total_books = 250 →
  math_book_price = 7 →
  history_book_price = 9 →
  total_price = 1860 →
  ∃ (math_books history_books : ℕ),
    math_books + history_books = total_books ∧
    math_book_price * math_books + history_book_price * history_books = total_price ∧
    math_books = 195 :=
by sorry

end math_books_count_l2647_264757


namespace range_of_2a_plus_3b_l2647_264756

theorem range_of_2a_plus_3b (a b : ℝ) 
  (h1 : -1 < a + b ∧ a + b < 3) 
  (h2 : 2 < a - b ∧ a - b < 4) : 
  -9/2 < 2*a + 3*b ∧ 2*a + 3*b < 13/2 := by
sorry

end range_of_2a_plus_3b_l2647_264756


namespace collinear_vectors_cos_2theta_l2647_264774

/-- 
Given vectors AB and BC in 2D space and the condition that points A, B, and C are collinear,
prove that cos(2θ) = 7/9.
-/
theorem collinear_vectors_cos_2theta (θ : ℝ) :
  let AB : Fin 2 → ℝ := ![- 1, - 3]
  let BC : Fin 2 → ℝ := ![2 * Real.sin θ, 2]
  (∃ (k : ℝ), AB = k • BC) →
  Real.cos (2 * θ) = 7 / 9 := by
sorry

end collinear_vectors_cos_2theta_l2647_264774


namespace min_value_theorem_l2647_264722

theorem min_value_theorem (a b x : ℝ) (ha : a > 1) (hb : b > 2) (hx : x + b = 5) :
  (∀ y : ℝ, y > 1 ∧ y + b = 5 → (1 / (a - 1) + 9 / (b - 2) ≤ 1 / (y - 1) + 9 / (b - 2))) ∧
  (1 / (a - 1) + 9 / (b - 2) = 8) :=
sorry

end min_value_theorem_l2647_264722


namespace fraction_equality_implies_equality_l2647_264720

theorem fraction_equality_implies_equality (x y m : ℝ) (h : m ≠ 0) :
  x / m = y / m → x = y := by
  sorry

end fraction_equality_implies_equality_l2647_264720


namespace compare_A_B_l2647_264777

theorem compare_A_B (A B : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : (3/4) * A = (4/3) * B) : A > B := by
  sorry

end compare_A_B_l2647_264777


namespace win_trip_l2647_264725

/-- The number of chocolate bars Tom needs to sell to win the trip -/
def total_bars : ℕ := 3465

/-- The number of chocolate bars in each box -/
def bars_per_box : ℕ := 7

/-- The number of boxes Tom needs to sell to win the trip -/
def boxes_needed : ℕ := total_bars / bars_per_box

theorem win_trip : boxes_needed = 495 := by
  sorry

end win_trip_l2647_264725


namespace monotonic_increasing_interval_of_f_l2647_264782

def f (x : ℝ) := 3*x - x^3

theorem monotonic_increasing_interval_of_f :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 1, StrictMonoOn f (Set.Ioo (-1 : ℝ) 1) :=
by sorry

end monotonic_increasing_interval_of_f_l2647_264782


namespace square_difference_of_sum_and_product_l2647_264744

theorem square_difference_of_sum_and_product (x y : ℕ+) 
  (sum_eq : x + y = 22)
  (product_eq : x * y = 120) :
  x^2 - y^2 = 44 := by
  sorry

end square_difference_of_sum_and_product_l2647_264744


namespace inequality_solution_set_l2647_264751

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x : ℝ, ax - b > 0 ↔ x < 1/3) → 
  (∀ x : ℝ, (a - b) * x - (a + b) > 0 ↔ x < 2) :=
by sorry

end inequality_solution_set_l2647_264751


namespace fathers_digging_time_l2647_264713

/-- The father's digging rate in feet per hour -/
def fathersRate : ℝ := 4

/-- The depth difference between Michael's hole and twice his father's hole depth in feet -/
def depthDifference : ℝ := 400

/-- Michael's digging time in hours -/
def michaelsTime : ℝ := 700

/-- Father's digging time in hours -/
def fathersTime : ℝ := 400

theorem fathers_digging_time :
  ∀ (fathersDepth michaelsDepth : ℝ),
  michaelsDepth = 2 * fathersDepth - depthDifference →
  michaelsDepth = fathersRate * michaelsTime →
  fathersDepth = fathersRate * fathersTime :=
by sorry

end fathers_digging_time_l2647_264713


namespace avg_people_moving_rounded_l2647_264754

/-- The number of people moving to California -/
def people_moving : ℕ := 4500

/-- The time period in days -/
def days : ℕ := 5

/-- The additional hours beyond full days -/
def extra_hours : ℕ := 12

/-- Function to calculate the average people per minute -/
def avg_people_per_minute (people : ℕ) (days : ℕ) (hours : ℕ) : ℚ :=
  people / (((days * 24 + hours) * 60) : ℚ)

/-- Function to round a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

/-- Theorem stating that the average number of people moving per minute, 
    when rounded to the nearest whole number, is 1 -/
theorem avg_people_moving_rounded : 
  round_to_nearest (avg_people_per_minute people_moving days extra_hours) = 1 := by
  sorry


end avg_people_moving_rounded_l2647_264754


namespace record_cost_thomas_record_cost_l2647_264798

theorem record_cost (num_books : ℕ) (book_price : ℚ) (num_records : ℕ) (leftover : ℚ) : ℚ :=
  let total_sale := num_books * book_price
  let spent_on_records := total_sale - leftover
  spent_on_records / num_records

theorem thomas_record_cost :
  record_cost 200 1.5 75 75 = 3 := by
  sorry

end record_cost_thomas_record_cost_l2647_264798


namespace least_days_to_double_l2647_264764

/-- The least number of days for a loan to double with daily compound interest -/
theorem least_days_to_double (principal : ℝ) (rate : ℝ) (n : ℕ) : 
  principal > 0 → 
  rate > 0 → 
  principal * (1 + rate) ^ n ≥ 2 * principal → 
  principal * (1 + rate) ^ (n - 1) < 2 * principal → 
  principal = 20 → 
  rate = 0.1 → 
  n = 8 := by
  sorry

end least_days_to_double_l2647_264764


namespace walter_exceptional_days_l2647_264799

/-- Represents Walter's chore earnings over a period of days -/
structure ChoreEarnings where
  regularPay : ℕ
  exceptionalPay : ℕ
  bonusThreshold : ℕ
  bonusAmount : ℕ
  totalDays : ℕ
  totalEarnings : ℕ

/-- Calculates the number of exceptional days given ChoreEarnings -/
def exceptionalDays (ce : ChoreEarnings) : ℕ :=
  sorry

/-- Theorem stating that Walter did chores exceptionally well for 5 days -/
theorem walter_exceptional_days (ce : ChoreEarnings) 
  (h1 : ce.regularPay = 4)
  (h2 : ce.exceptionalPay = 6)
  (h3 : ce.bonusThreshold = 5)
  (h4 : ce.bonusAmount = 10)
  (h5 : ce.totalDays = 12)
  (h6 : ce.totalEarnings = 58) :
  exceptionalDays ce = 5 :=
sorry

end walter_exceptional_days_l2647_264799


namespace negation_of_universal_exponential_exponential_negation_l2647_264740

theorem negation_of_universal_exponential (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬(P x) :=
by sorry

theorem exponential_negation :
  (¬ ∀ x : ℝ, Real.exp x > 0) ↔ (∃ x : ℝ, Real.exp x ≤ 0) :=
by sorry

end negation_of_universal_exponential_exponential_negation_l2647_264740


namespace carpet_area_is_27072_l2647_264718

/-- Calculates the area of carpet required for a room with a column -/
def carpet_area (room_length room_width column_side : ℕ) : ℕ :=
  let inches_per_foot := 12
  let room_length_inches := room_length * inches_per_foot
  let room_width_inches := room_width * inches_per_foot
  let column_side_inches := column_side * inches_per_foot
  let total_area := room_length_inches * room_width_inches
  let column_area := column_side_inches * column_side_inches
  total_area - column_area

/-- Theorem: The carpet area for the given room is 27,072 square inches -/
theorem carpet_area_is_27072 :
  carpet_area 16 12 2 = 27072 := by
  sorry

end carpet_area_is_27072_l2647_264718


namespace stating_num_small_triangles_formula_l2647_264710

/-- Represents a triangle with n points inside it -/
structure TriangleWithPoints where
  n : ℕ  -- number of points inside the triangle
  no_collinear : Bool  -- property that no three points are collinear

/-- 
  Calculates the number of small triangles formed in a triangle with n internal points,
  where no three points (including the triangle's vertices) are collinear.
-/
def numSmallTriangles (t : TriangleWithPoints) : ℕ :=
  2 * t.n + 1

/-- 
  Theorem stating that for a triangle with n points inside,
  where no three points are collinear (including the triangle's vertices),
  the number of small triangles formed is 2n + 1.
-/
theorem num_small_triangles_formula (t : TriangleWithPoints) 
  (h : t.no_collinear = true) : 
  numSmallTriangles t = 2 * t.n + 1 := by
  sorry

#eval numSmallTriangles { n := 100, no_collinear := true }

end stating_num_small_triangles_formula_l2647_264710


namespace triangle_inradius_l2647_264705

/-- Given a triangle with perimeter 40 and area 50, prove that its inradius is 2.5 -/
theorem triangle_inradius (P : ℝ) (A : ℝ) (r : ℝ) 
  (h1 : P = 40) 
  (h2 : A = 50) 
  (h3 : A = r * (P / 2)) : r = 2.5 := by
  sorry

end triangle_inradius_l2647_264705


namespace nancy_shelving_problem_l2647_264707

/-- The number of romance books shelved by Nancy the librarian --/
def romance_books : ℕ := 8

/-- The total number of books on the cart --/
def total_books : ℕ := 46

/-- The number of history books shelved --/
def history_books : ℕ := 12

/-- The number of poetry books shelved --/
def poetry_books : ℕ := 4

/-- The number of Western novels shelved --/
def western_books : ℕ := 5

/-- The number of biographies shelved --/
def biography_books : ℕ := 6

theorem nancy_shelving_problem :
  romance_books = 8 ∧
  total_books = 46 ∧
  history_books = 12 ∧
  poetry_books = 4 ∧
  western_books = 5 ∧
  biography_books = 6 ∧
  (total_books - (history_books + romance_books + poetry_books)) % 2 = 0 ∧
  (total_books - (history_books + romance_books + poetry_books)) / 2 = western_books + biography_books :=
by sorry

end nancy_shelving_problem_l2647_264707


namespace lines_parallel_if_perpendicular_to_parallel_planes_l2647_264727

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism and perpendicularity relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (lineParallel : Line → Line → Prop)

-- State the theorem
theorem lines_parallel_if_perpendicular_to_parallel_planes 
  (α β : Plane) (a b : Line)
  (h_distinct_planes : α ≠ β)
  (h_distinct_lines : a ≠ b)
  (h_a_perp_α : perpendicular a α)
  (h_b_perp_β : perpendicular b β)
  (h_α_parallel_β : parallel α β) :
  lineParallel a b :=
sorry

end lines_parallel_if_perpendicular_to_parallel_planes_l2647_264727


namespace negation_of_proposition_l2647_264787

theorem negation_of_proposition (p : Prop) : 
  (¬ (∀ x : ℝ, x ≥ 0 → x - 2 > 0)) ↔ (∃ x : ℝ, x ≥ 0 ∧ x - 2 ≤ 0) :=
by sorry

end negation_of_proposition_l2647_264787


namespace alpha_value_l2647_264762

theorem alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 Real.pi) (h2 : Real.cos α = -1/2) : 
  α = 2 * Real.pi / 3 := by
  sorry

end alpha_value_l2647_264762


namespace dodecagon_hexagon_area_ratio_l2647_264714

/-- Given a regular dodecagon with area n and a hexagon ACEGIK formed by
    connecting every second vertex with area m, prove that m/n = √3 - 3/2 -/
theorem dodecagon_hexagon_area_ratio (n m : ℝ) : 
  n > 0 → -- Assuming positive area for the dodecagon
  (∃ (a : ℝ), a > 0 ∧ n = 3 * a^2 * (2 + Real.sqrt 3)) → -- Area formula for dodecagon
  (∃ (s : ℝ), s > 0 ∧ m = (3 * Real.sqrt 3 / 2) * s^2) → -- Area formula for hexagon
  m / n = Real.sqrt 3 - 3 / 2 := by
  sorry

end dodecagon_hexagon_area_ratio_l2647_264714


namespace negative_two_thousand_ten_plus_two_l2647_264792

theorem negative_two_thousand_ten_plus_two :
  (-2010 : ℤ) + 2 = -2008 := by sorry

end negative_two_thousand_ten_plus_two_l2647_264792


namespace initial_number_of_girls_l2647_264778

theorem initial_number_of_girls :
  ∀ (n : ℕ) (A : ℝ),
  n > 0 →
  (n * (A + 3) - n * A = 94 - 70) →
  n = 8 :=
by
  sorry

end initial_number_of_girls_l2647_264778


namespace otimes_twelve_nine_l2647_264723

-- Define the custom operation
def otimes (a b : ℚ) : ℚ := a + (4 * a) / (3 * b)

-- Theorem statement
theorem otimes_twelve_nine : otimes 12 9 = 13 + 7/9 := by
  sorry

end otimes_twelve_nine_l2647_264723


namespace max_value_cube_root_sum_max_value_achievable_l2647_264755

theorem max_value_cube_root_sum (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) :
  (a * b * c) ^ (1/3 : ℝ) + ((2 - a) * (2 - b) * (2 - c)) ^ (1/3 : ℝ) ≤ 2 :=
sorry

theorem max_value_achievable :
  ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧
  (a * b * c) ^ (1/3 : ℝ) + ((2 - a) * (2 - b) * (2 - c)) ^ (1/3 : ℝ) = 2 :=
sorry

end max_value_cube_root_sum_max_value_achievable_l2647_264755


namespace rectangle_length_equals_square_side_l2647_264789

/-- The length of a rectangle with width 3 cm and area equal to a square with side length 3 cm is 3 cm. -/
theorem rectangle_length_equals_square_side : 
  ∀ (length : ℝ),
  length > 0 →
  3 * length = 3 * 3 →
  length = 3 := by
sorry

end rectangle_length_equals_square_side_l2647_264789


namespace seokjin_floors_to_bookstore_l2647_264785

/-- The floor number of the bookstore -/
def bookstore_floor : ℕ := 4

/-- Seokjin's current floor number -/
def current_floor : ℕ := 1

/-- The number of floors Seokjin must go up -/
def floors_to_go_up : ℕ := bookstore_floor - current_floor

/-- Theorem stating that Seokjin must go up 3 floors to reach the bookstore -/
theorem seokjin_floors_to_bookstore : floors_to_go_up = 3 := by
  sorry

end seokjin_floors_to_bookstore_l2647_264785


namespace triangle_inequality_altitudes_l2647_264737

/-- Triangle inequality for side lengths and altitudes -/
theorem triangle_inequality_altitudes (a b c h_a h_b h_c Δ : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_ha : 0 < h_a) (h_pos_hb : 0 < h_b) (h_pos_hc : 0 < h_c)
  (h_area : 0 < Δ)
  (h_area_a : Δ = (a * h_a) / 2)
  (h_area_b : Δ = (b * h_b) / 2)
  (h_area_c : Δ = (c * h_c) / 2) :
  a * h_b + b * h_c + c * h_a ≥ 6 * Δ := by
  sorry

end triangle_inequality_altitudes_l2647_264737


namespace profit_percentage_l2647_264712

theorem profit_percentage (C S : ℝ) (h : 55 * C = 50 * S) : 
  (S - C) / C * 100 = 10 := by
sorry

end profit_percentage_l2647_264712


namespace twenty_pancakes_in_24_minutes_l2647_264728

/-- Represents the pancake production and consumption rates of a family -/
structure PancakeFamily where
  dad_rate : ℚ  -- Dad's pancake production rate per hour
  mom_rate : ℚ  -- Mom's pancake production rate per hour
  petya_rate : ℚ  -- Petya's pancake consumption rate per 15 minutes
  vasya_multiplier : ℚ  -- Vasya's consumption rate multiplier relative to Petya

/-- Calculates the minimum time (in minutes) required for at least 20 pancakes to remain uneaten -/
def min_time_for_20_pancakes (family : PancakeFamily) : ℚ :=
  sorry

/-- The main theorem stating that 24 minutes is the minimum time for 20 pancakes to remain uneaten -/
theorem twenty_pancakes_in_24_minutes (family : PancakeFamily) 
  (h1 : family.dad_rate = 70)
  (h2 : family.mom_rate = 100)
  (h3 : family.petya_rate = 10)
  (h4 : family.vasya_multiplier = 2) :
  min_time_for_20_pancakes family = 24 := by
  sorry

end twenty_pancakes_in_24_minutes_l2647_264728


namespace square_area_perimeter_ratio_l2647_264795

theorem square_area_perimeter_ratio : 
  ∀ (s₁ s₂ : ℝ), s₁ > 0 → s₂ > 0 → 
  (s₁^2 / s₂^2 = 16 / 49) → 
  ((4 * s₁) / (4 * s₂) = 4 / 7) := by
sorry

end square_area_perimeter_ratio_l2647_264795


namespace sqrt_neg_three_squared_l2647_264766

theorem sqrt_neg_three_squared : Real.sqrt ((-3)^2) = 3 := by sorry

end sqrt_neg_three_squared_l2647_264766


namespace counterexample_non_coprime_l2647_264704

theorem counterexample_non_coprime :
  ∃ (a n : ℕ+), (Nat.gcd a.val n.val ≠ 1) ∧ (a.val ^ n.val % n.val ≠ a.val % n.val) := by
  sorry

end counterexample_non_coprime_l2647_264704


namespace cost_split_theorem_l2647_264726

/-- Calculates the amount each person should pay when a group buys items and splits the cost equally -/
def calculate_cost_per_person (num_people : ℕ) (item1_count : ℕ) (item1_price : ℕ) (item2_count : ℕ) (item2_price : ℕ) : ℕ :=
  ((item1_count * item1_price + item2_count * item2_price) / num_people)

/-- Proves that when 4 friends buy 5 items at 200 won each and 7 items at 800 won each, 
    and divide the total cost equally, each person should pay 1650 won -/
theorem cost_split_theorem : 
  calculate_cost_per_person 4 5 200 7 800 = 1650 := by
  sorry

end cost_split_theorem_l2647_264726


namespace max_product_of_two_different_numbers_exists_max_product_l2647_264784

def S : Set Int := {-9, -5, -3, 0, 4, 5, 8}

theorem max_product_of_two_different_numbers (a b : Int) :
  a ∈ S → b ∈ S → a ≠ b → a * b ≤ 45 := by
  sorry

theorem exists_max_product :
  ∃ a b : Int, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a * b = 45 := by
  sorry

end max_product_of_two_different_numbers_exists_max_product_l2647_264784


namespace expression_simplification_and_evaluation_l2647_264750

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ -1 →
  (2 * x / (x + 1) - (2 * x + 6) / (x^2 - 1) / ((x + 3) / (x^2 - 2 * x + 1))) = 2 / (x + 1) ∧
  (2 / (0 + 1) = 2) := by
  sorry

end expression_simplification_and_evaluation_l2647_264750


namespace question_mark_value_l2647_264719

theorem question_mark_value : ∃ x : ℤ, 27474 + 3699 + x - 2047 = 31111 ∧ x = 1985 := by
  sorry

end question_mark_value_l2647_264719


namespace multiples_of_hundred_sequence_l2647_264703

theorem multiples_of_hundred_sequence (start : ℕ) :
  (∃ seq : Finset ℕ,
    seq.card = 10 ∧
    (∀ n ∈ seq, n % 100 = 0) ∧
    (∀ n ∈ seq, start ≤ n ∧ n ≤ 1000) ∧
    1000 ∈ seq) →
  start = 100 :=
by sorry

end multiples_of_hundred_sequence_l2647_264703


namespace inverse_f_at_3_l2647_264748

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem inverse_f_at_3 :
  ∃ (y : ℝ), y < 0 ∧ f y = 3 ∧ ∀ (z : ℝ), z < 0 ∧ f z = 3 → z = y :=
by sorry

end inverse_f_at_3_l2647_264748


namespace royal_family_children_count_l2647_264780

/-- Represents the number of years that have passed -/
def n : ℕ := sorry

/-- Represents the number of daughters -/
def d : ℕ := sorry

/-- The total number of children -/
def total_children : ℕ := d + 3

/-- The initial age of the king and queen -/
def initial_royal_age : ℕ := 35

/-- The initial total age of the children -/
def initial_children_age : ℕ := 35

/-- The combined age of the king and queen after n years -/
def royal_age_after_n_years : ℕ := 2 * initial_royal_age + 2 * n

/-- The total age of the children after n years -/
def children_age_after_n_years : ℕ := initial_children_age + total_children * n

theorem royal_family_children_count :
  (royal_age_after_n_years = children_age_after_n_years) ∧
  (total_children ≤ 20) →
  (total_children = 7 ∨ total_children = 9) :=
by sorry

end royal_family_children_count_l2647_264780


namespace total_loaves_served_l2647_264758

/-- Given that a restaurant served 0.5 loaf of wheat bread and 0.4 loaf of white bread,
    prove that the total number of loaves served is 0.9. -/
theorem total_loaves_served (wheat_bread : ℝ) (white_bread : ℝ)
    (h1 : wheat_bread = 0.5)
    (h2 : white_bread = 0.4) :
    wheat_bread + white_bread = 0.9 := by
  sorry

end total_loaves_served_l2647_264758


namespace min_value_of_expression_l2647_264730

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 * b + b^2 * c + c^2 * a = 3) :
  a^7 * b + b^7 * c + c^7 * a + a * b^3 + b * c^3 + c * a^3 ≥ 6 := by
  sorry

end min_value_of_expression_l2647_264730


namespace sum_u_v_equals_negative_42_over_77_l2647_264743

theorem sum_u_v_equals_negative_42_over_77 
  (u v : ℚ) 
  (eq1 : 3 * u - 7 * v = 17) 
  (eq2 : 5 * u + 3 * v = 1) : 
  u + v = -42 / 77 := by
sorry

end sum_u_v_equals_negative_42_over_77_l2647_264743


namespace trapezoid_side_length_l2647_264786

/-- Given a trapezoid PQRS with the following properties:
  - Area is 200 cm²
  - Altitude is 10 cm
  - PQ is 15 cm
  - RS is 20 cm
  Prove that the length of QR is 20 - 2.5√5 - 5√3 cm -/
theorem trapezoid_side_length (area : ℝ) (altitude : ℝ) (pq : ℝ) (rs : ℝ) (qr : ℝ) :
  area = 200 →
  altitude = 10 →
  pq = 15 →
  rs = 20 →
  qr = 20 - 2.5 * Real.sqrt 5 - 5 * Real.sqrt 3 :=
by sorry

end trapezoid_side_length_l2647_264786


namespace smaller_screen_diagonal_l2647_264700

theorem smaller_screen_diagonal : 
  ∃ (x : ℝ), x > 0 ∧ x^2 + 34 = 18^2 ∧ x = Real.sqrt 290 := by sorry

end smaller_screen_diagonal_l2647_264700


namespace farrah_order_proof_l2647_264776

/-- The number of matchboxes in each box -/
def matchboxes_per_box : ℕ := 20

/-- The number of match sticks in each matchbox -/
def sticks_per_matchbox : ℕ := 300

/-- The total number of match sticks ordered -/
def total_sticks : ℕ := 24000

/-- The number of boxes Farrah ordered -/
def boxes_ordered : ℕ := total_sticks / (matchboxes_per_box * sticks_per_matchbox)

theorem farrah_order_proof : boxes_ordered = 4 := by
  sorry

end farrah_order_proof_l2647_264776
