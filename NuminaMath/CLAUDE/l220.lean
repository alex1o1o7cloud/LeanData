import Mathlib

namespace NUMINAMATH_CALUDE_divisibility_of_947B_l220_22036

-- Define a function to check if a number is divisible by 3
def divisible_by_three (n : ℕ) : Prop := n % 3 = 0

-- Define a function to sum the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- Theorem statement
theorem divisibility_of_947B (B : ℕ) : 
  B < 10 →  -- B is a single digit
  (∀ (n : ℕ), divisible_by_three n ↔ divisible_by_three (sum_of_digits n)) →  -- Divisibility rule
  (divisible_by_three (9000 + 400 + 70 + B) ↔ (B = 1 ∨ B = 4 ∨ B = 7)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_947B_l220_22036


namespace NUMINAMATH_CALUDE_subtracted_value_l220_22073

theorem subtracted_value (chosen_number : ℕ) (final_result : ℕ) : 
  chosen_number = 63 → final_result = 110 → 
  (chosen_number * 4 - final_result) = 142 := by
sorry

end NUMINAMATH_CALUDE_subtracted_value_l220_22073


namespace NUMINAMATH_CALUDE_cookies_milk_ratio_l220_22040

-- Define the constants from the problem
def cookies_for_recipe : ℕ := 18
def quarts_for_recipe : ℕ := 3
def pints_per_quart : ℕ := 2
def cookies_to_bake : ℕ := 9

-- Define the function to calculate pints needed
def pints_needed (cookies : ℕ) : ℚ :=
  (cookies : ℚ) * (quarts_for_recipe * pints_per_quart : ℚ) / (cookies_for_recipe : ℚ)

-- Theorem statement
theorem cookies_milk_ratio :
  pints_needed cookies_to_bake = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookies_milk_ratio_l220_22040


namespace NUMINAMATH_CALUDE_trishas_walking_distance_l220_22053

/-- Trisha's walking distances in New York City -/
theorem trishas_walking_distance
  (total_distance : ℝ)
  (hotel_to_tshirt : ℝ)
  (h1 : total_distance = 0.8888888888888888)
  (h2 : hotel_to_tshirt = 0.6666666666666666)
  (h3 : ∃ x : ℝ, total_distance = x + x + hotel_to_tshirt) :
  ∃ x : ℝ, x = 0.1111111111111111 ∧ total_distance = x + x + hotel_to_tshirt :=
by sorry

end NUMINAMATH_CALUDE_trishas_walking_distance_l220_22053


namespace NUMINAMATH_CALUDE_problem_solution_l220_22069

theorem problem_solution (a b : ℝ) (h : |a - 1| + (2 + b)^2 = 0) : 
  (a + b)^2009 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l220_22069


namespace NUMINAMATH_CALUDE_even_function_k_value_l220_22007

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The function f(x) = kx^2 + (k-1)x + 2 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

theorem even_function_k_value :
  ∀ k : ℝ, IsEven (f k) → k = 1 := by sorry

end NUMINAMATH_CALUDE_even_function_k_value_l220_22007


namespace NUMINAMATH_CALUDE_greatest_valid_integer_l220_22066

def is_valid (n : ℕ) : Prop :=
  n < 150 ∧ Nat.gcd n 24 = 4

theorem greatest_valid_integer : 
  (∀ m, is_valid m → m ≤ 140) ∧ is_valid 140 :=
sorry

end NUMINAMATH_CALUDE_greatest_valid_integer_l220_22066


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l220_22048

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 3| = |x + 2| ∧ |x + 2| = |x - 5| ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l220_22048


namespace NUMINAMATH_CALUDE_bricklayer_wage_is_44_l220_22023

/-- Represents the hourly wage of a worker -/
structure HourlyWage where
  amount : ℝ
  nonneg : amount ≥ 0

/-- Represents the total hours worked by both workers -/
def total_hours : ℝ := 90

/-- Represents the hourly wage of the electrician -/
def electrician_wage : HourlyWage := ⟨16, by norm_num⟩

/-- Represents the total payment for both workers -/
def total_payment : ℝ := 1350

/-- Represents the hours worked by each worker -/
def individual_hours : ℝ := 22.5

/-- Theorem stating that the bricklayer's hourly wage is $44 -/
theorem bricklayer_wage_is_44 :
  ∃ (bricklayer_wage : HourlyWage),
    bricklayer_wage.amount = 44 ∧
    individual_hours * (bricklayer_wage.amount + electrician_wage.amount) = total_payment ∧
    2 * individual_hours = total_hours :=
by sorry


end NUMINAMATH_CALUDE_bricklayer_wage_is_44_l220_22023


namespace NUMINAMATH_CALUDE_olympic_volunteer_allocation_l220_22034

theorem olympic_volunteer_allocation :
  let n : ℕ := 5  -- number of volunteers
  let k : ℕ := 4  -- number of projects
  let allocations : ℕ := (n.choose 2) * (k.factorial)
  allocations = 240 :=
by sorry

end NUMINAMATH_CALUDE_olympic_volunteer_allocation_l220_22034


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_target_l220_22044

theorem sum_of_fractions_equals_target : 
  (1/3 : ℚ) + (1/2 : ℚ) + (-5/6 : ℚ) + (1/5 : ℚ) + (1/4 : ℚ) + (-9/20 : ℚ) + (2/15 : ℚ) = 
  (13333333333333333 : ℚ) / (100000000000000000 : ℚ) := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_target_l220_22044


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l220_22051

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- The theorem statement -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (-1, x)
  let b : ℝ × ℝ := (x, -4)
  parallel a b → x = 2 ∨ x = -2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l220_22051


namespace NUMINAMATH_CALUDE_range_of_a_l220_22038

-- Define a decreasing function on (-1, 1)
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y

-- Define the theorem
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_decreasing : DecreasingFunction f) 
  (h_inequality : f (1 - a) < f (2 * a - 1)) :
  0 < a ∧ a < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l220_22038


namespace NUMINAMATH_CALUDE_exists_valid_matrix_l220_22014

def is_valid_matrix (M : Matrix (Fin 4) (Fin 4) ℤ) : Prop :=
  (∀ i j, M i j ≠ 0) ∧
  (∀ i j, i + 1 < 4 → j + 1 < 4 →
    M i j + M (i + 1) j + M i (j + 1) + M (i + 1) (j + 1) = 0) ∧
  (∀ i j, i + 2 < 4 → j + 2 < 4 →
    M i j + M (i + 2) j + M i (j + 2) + M (i + 2) (j + 2) = 0) ∧
  (M 0 0 + M 0 3 + M 3 0 + M 3 3 = 0)

theorem exists_valid_matrix : ∃ M : Matrix (Fin 4) (Fin 4) ℤ, is_valid_matrix M := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_matrix_l220_22014


namespace NUMINAMATH_CALUDE_tom_ran_median_distance_l220_22055

def runners : Finset String := {"Phil", "Tom", "Pete", "Amal", "Sanjay"}

def distance : String → ℝ
| "Phil" => 4
| "Tom" => 6
| "Pete" => 2
| "Amal" => 8
| "Sanjay" => 7
| _ => 0

def isMedian (x : ℝ) (s : Finset ℝ) : Prop :=
  2 * (s.filter (· ≤ x)).card ≥ s.card ∧
  2 * (s.filter (· ≥ x)).card ≥ s.card

theorem tom_ran_median_distance :
  isMedian (distance "Tom") (runners.image distance) :=
sorry

end NUMINAMATH_CALUDE_tom_ran_median_distance_l220_22055


namespace NUMINAMATH_CALUDE_pie_division_l220_22011

theorem pie_division (total_pie : ℚ) (num_people : ℕ) : 
  total_pie = 5/6 ∧ num_people = 4 → (total_pie / num_people : ℚ) = 5/24 := by
  sorry

end NUMINAMATH_CALUDE_pie_division_l220_22011


namespace NUMINAMATH_CALUDE_x_varies_with_z_l220_22028

theorem x_varies_with_z (x y z : ℝ) (k j : ℝ) (h1 : x = k * y^2) (h2 : y = j * z^(1/3)) :
  ∃ m : ℝ, x = m * z^(2/3) :=
sorry

end NUMINAMATH_CALUDE_x_varies_with_z_l220_22028


namespace NUMINAMATH_CALUDE_solve_for_y_l220_22009

theorem solve_for_y (x y : ℝ) (h : 3 * x + 2 * y = 1) : y = (1 - 3 * x) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l220_22009


namespace NUMINAMATH_CALUDE_sum_product_inequality_l220_22002

theorem sum_product_inequality (a b c : ℝ) (h : a + b + c = 0) : a * b + b * c + c * a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l220_22002


namespace NUMINAMATH_CALUDE_complex_multiplication_l220_22059

theorem complex_multiplication (i : ℂ) : i * i = -1 → (1 - i) * (1 + 2*i) = 3 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l220_22059


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l220_22064

structure University :=
  (total_students : ℕ)
  (freshmen : ℕ)
  (sophomores : ℕ)
  (juniors : ℕ)
  (seniors : ℕ)
  (sample_size : ℕ)

def stratified_sample (u : University) : Vector ℕ 4 :=
  let sampling_ratio := u.sample_size / u.total_students
  ⟨[u.freshmen * sampling_ratio,
    u.sophomores * sampling_ratio,
    u.juniors * sampling_ratio,
    u.seniors * sampling_ratio],
   by simp⟩

theorem correct_stratified_sample (u : University) 
  (h1 : u.total_students = 8000)
  (h2 : u.freshmen = 1600)
  (h3 : u.sophomores = 3200)
  (h4 : u.juniors = 2000)
  (h5 : u.seniors = 1200)
  (h6 : u.sample_size = 400)
  (h7 : u.total_students = u.freshmen + u.sophomores + u.juniors + u.seniors) :
  stratified_sample u = ⟨[80, 160, 100, 60], by simp⟩ := by
  sorry

#check correct_stratified_sample

end NUMINAMATH_CALUDE_correct_stratified_sample_l220_22064


namespace NUMINAMATH_CALUDE_surface_area_of_slice_theorem_l220_22094

/-- Represents a right prism with isosceles triangular bases -/
structure IsoscelesPrism where
  height : ℝ
  base_length : ℝ
  side_length : ℝ

/-- Calculates the surface area of the sliced off portion of the prism -/
def surface_area_of_slice (prism : IsoscelesPrism) : ℝ :=
  sorry

/-- Theorem stating the surface area of the sliced portion -/
theorem surface_area_of_slice_theorem (prism : IsoscelesPrism) 
  (h1 : prism.height = 10)
  (h2 : prism.base_length = 10)
  (h3 : prism.side_length = 12) :
  surface_area_of_slice prism = 52.25 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_slice_theorem_l220_22094


namespace NUMINAMATH_CALUDE_ticket_sales_total_l220_22042

/-- Calculates the total amount of money collected from ticket sales -/
def totalAmountCollected (adultPrice studentPrice : ℚ) (totalTickets studentTickets : ℕ) : ℚ :=
  let adultTickets := totalTickets - studentTickets
  adultPrice * adultTickets + studentPrice * studentTickets

/-- Theorem stating that the total amount collected is $222.50 given the problem conditions -/
theorem ticket_sales_total : 
  totalAmountCollected 4 (5/2) 59 9 = 445/2 := by
  sorry

#eval totalAmountCollected 4 (5/2) 59 9

end NUMINAMATH_CALUDE_ticket_sales_total_l220_22042


namespace NUMINAMATH_CALUDE_stating_max_areas_formula_l220_22092

/-- Represents a circular disk divided by radii and secant lines -/
structure DividedDisk where
  n : ℕ
  radii_count : ℕ := 3 * n
  secant_lines : ℕ := 2
  h_positive : n > 0

/-- 
Calculates the maximum number of non-overlapping areas in a divided disk 
-/
def max_areas (disk : DividedDisk) : ℕ := 4 * disk.n + 1

/-- 
Theorem stating that the maximum number of non-overlapping areas 
in a divided disk is 4n + 1 
-/
theorem max_areas_formula (disk : DividedDisk) : 
  max_areas disk = 4 * disk.n + 1 := by sorry

end NUMINAMATH_CALUDE_stating_max_areas_formula_l220_22092


namespace NUMINAMATH_CALUDE_bakers_pastry_problem_l220_22083

/-- Baker's pastry problem -/
theorem bakers_pastry_problem (cakes_sold : ℕ) (difference : ℕ) (pastries_sold : ℕ) :
  cakes_sold = 97 →
  cakes_sold = pastries_sold + difference →
  difference = 89 →
  pastries_sold = 8 := by
  sorry

end NUMINAMATH_CALUDE_bakers_pastry_problem_l220_22083


namespace NUMINAMATH_CALUDE_show_end_time_l220_22089

/-- Represents time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  h_valid : hour < 24
  m_valid : minute < 60

/-- Represents a TV show -/
structure TVShow where
  start_time : Time
  end_time : Time
  weekday_only : Bool

def total_watch_time (s : TVShow) (days_watched : Nat) : Nat :=
  days_watched * (s.end_time.hour * 60 + s.end_time.minute - s.start_time.hour * 60 - s.start_time.minute)

theorem show_end_time (s : TVShow) 
  (h1 : s.start_time = ⟨14, 0, by norm_num, by norm_num⟩)
  (h2 : s.weekday_only = true)
  (h3 : total_watch_time s 4 = 120) :
  s.end_time = ⟨14, 30, by norm_num, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_show_end_time_l220_22089


namespace NUMINAMATH_CALUDE_x_varies_as_square_of_sin_z_l220_22021

/-- Given that x is directly proportional to the square of y, and y is directly proportional to sin(z),
    prove that x varies as the 2nd power of sin(z). -/
theorem x_varies_as_square_of_sin_z
  (x y z : ℝ)
  (hxy : ∃ k : ℝ, x = k * y^2)
  (hyz : ∃ j : ℝ, y = j * Real.sin z) :
  ∃ m : ℝ, x = m * (Real.sin z)^2 :=
sorry

end NUMINAMATH_CALUDE_x_varies_as_square_of_sin_z_l220_22021


namespace NUMINAMATH_CALUDE_inequality_solution_l220_22046

theorem inequality_solution (x : ℝ) :
  x ≠ 5 →
  (x * (x + 2) / (x - 5)^2 ≥ 15 ↔ 
    x ≤ 3.71 ∨ (x ≥ 7.14 ∧ x < 5) ∨ x > 5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l220_22046


namespace NUMINAMATH_CALUDE_plane_points_distance_l220_22061

theorem plane_points_distance (n : ℕ) (P : Fin n → ℝ × ℝ) (Q : ℝ × ℝ) 
  (h_n : n ≥ 12)
  (h_distinct : ∀ i j, i ≠ j → P i ≠ P j ∧ P i ≠ Q) :
  ∃ i : Fin n, ∃ S : Finset (Fin n), 
    S.card ≥ (n / 6 : ℕ) - 1 ∧ 
    (∀ j ∈ S, j ≠ i → dist (P j) (P i) < dist (P i) Q) :=
by sorry

end NUMINAMATH_CALUDE_plane_points_distance_l220_22061


namespace NUMINAMATH_CALUDE_system_solution_l220_22077

theorem system_solution (a b c : ℝ) : 
  (∀ x y, a * x + b * y = 2 ∧ c * x - 7 * y = 8) →
  (a * 3 + b * (-2) = 2 ∧ c * 3 - 7 * (-2) = 8) →
  (a * (-2) + b * 2 = 2) →
  (a = 4 ∧ b = 5 ∧ c = -2) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l220_22077


namespace NUMINAMATH_CALUDE_exists_same_color_distance_one_l220_22012

/-- A coloring of the plane using three colors -/
def Coloring := ℝ × ℝ → Fin 3

/-- Two points in the plane -/
def TwoPoints := (ℝ × ℝ) × (ℝ × ℝ)

/-- The distance between two points is 1 -/
def DistanceOne (p : TwoPoints) : Prop :=
  let (p1, p2) := p
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 = 1

/-- Two points have the same color -/
def SameColor (c : Coloring) (p : TwoPoints) : Prop :=
  let (p1, p2) := p
  c p1 = c p2

/-- Main theorem: In any three-coloring of the plane, there exist two points
    of the same color with distance 1 between them -/
theorem exists_same_color_distance_one :
  ∀ c : Coloring, ∃ p : TwoPoints, DistanceOne p ∧ SameColor c p := by
  sorry

end NUMINAMATH_CALUDE_exists_same_color_distance_one_l220_22012


namespace NUMINAMATH_CALUDE_plane_contains_points_and_satisfies_constraints_l220_22019

def point1 : ℝ × ℝ × ℝ := (2, -1, 3)
def point2 : ℝ × ℝ × ℝ := (0, -1, 5)
def point3 : ℝ × ℝ × ℝ := (-2, -3, 4)

def plane_equation (x y z : ℝ) : Prop := 2*x + 5*y - 2*z + 7 = 0

theorem plane_contains_points_and_satisfies_constraints :
  (plane_equation point1.1 point1.2.1 point1.2.2) ∧
  (plane_equation point2.1 point2.2.1 point2.2.2) ∧
  (plane_equation point3.1 point3.2.1 point3.2.2) ∧
  (2 > 0) ∧
  (Nat.gcd (Nat.gcd (Nat.gcd 2 5) 2) 7 = 1) :=
sorry

end NUMINAMATH_CALUDE_plane_contains_points_and_satisfies_constraints_l220_22019


namespace NUMINAMATH_CALUDE_vector_simplification_l220_22056

variable {V : Type*} [AddCommGroup V]
variable (A B C D F : V)

theorem vector_simplification :
  (C - D) + (B - C) + (A - B) = A - D ∧
  (A - B) + (D - F) + (C - D) + (B - C) + (F - A) = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_simplification_l220_22056


namespace NUMINAMATH_CALUDE_gcd_840_1764_l220_22079

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l220_22079


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l220_22085

theorem smallest_sum_of_squares (x y : ℝ) :
  (∀ a b : ℝ, x^2 + y^2 ≤ a^2 + b^2) → x^2 + y^2 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l220_22085


namespace NUMINAMATH_CALUDE_vertical_asymptote_at_four_sevenths_l220_22035

/-- The function f(x) = (2x+3)/(7x-4) has a vertical asymptote at x = 4/7 -/
theorem vertical_asymptote_at_four_sevenths :
  ∀ f : ℝ → ℝ, (∀ x : ℝ, f x = (2*x + 3) / (7*x - 4)) →
  ∃! a : ℝ, a = 4/7 ∧ ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - a| ∧ |x - a| < δ → |f x| > 1/ε := by
  sorry

end NUMINAMATH_CALUDE_vertical_asymptote_at_four_sevenths_l220_22035


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l220_22043

/-- The function f(x) = x^3 - 1/x is monotonically increasing for x > 0 -/
theorem f_monotone_increasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → x₁^3 - 1/x₁ < x₂^3 - 1/x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l220_22043


namespace NUMINAMATH_CALUDE_min_natural_numbers_for_prime_products_l220_22057

theorem min_natural_numbers_for_prime_products (p : Fin 100 → ℕ) (a : ℕ → ℕ) :
  (∀ i j, i ≠ j → p i ≠ p j) →  -- p₁, ..., p₁₀₀ are distinct
  (∀ i, Prime (p i)) →  -- p₁, ..., p₁₀₀ are prime
  (∀ i, a i > 1) →  -- Each a_i is greater than 1
  (∀ i : Fin 100, ∃ j k, a j * a k = p i * p ((i + 1) % 100)^3) →  -- Each p_i * p_{i+1}³ is a product of two a_i's
  (∃ k, ∀ i, a i ≠ 0 → i < k) →  -- There are finitely many non-zero a_i's
  (∃ k, k ≥ 150 ∧ ∀ i, a i ≠ 0 → i < k) :=  -- There are at least 150 non-zero a_i's
by sorry

end NUMINAMATH_CALUDE_min_natural_numbers_for_prime_products_l220_22057


namespace NUMINAMATH_CALUDE_cow_calf_cost_problem_l220_22017

theorem cow_calf_cost_problem (total_cost calf_cost cow_cost : ℕ) : 
  total_cost = 990 →
  cow_cost = 8 * calf_cost →
  total_cost = cow_cost + calf_cost →
  cow_cost = 880 := by
sorry

end NUMINAMATH_CALUDE_cow_calf_cost_problem_l220_22017


namespace NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l220_22025

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (x+1)(x+a) -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ (x + 1) * (x + a)

/-- If f(x) = (x+1)(x+a) is an even function, then a = -1 -/
theorem even_function_implies_a_eq_neg_one :
  ∃ a : ℝ, IsEven (f a) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l220_22025


namespace NUMINAMATH_CALUDE_initial_number_proof_l220_22010

theorem initial_number_proof (x : ℤ) : (x + 2)^2 = x^2 - 2016 → x = -505 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l220_22010


namespace NUMINAMATH_CALUDE_valid_a_values_l220_22072

def A (a : ℝ) : Set ℝ := {1, 3, a}
def B (a : ℝ) : Set ℝ := {1, a^2 - a + 1}

theorem valid_a_values (a : ℝ) : 
  (A a ⊇ B a) ↔ (a = -1 ∨ a = 2) :=
sorry

end NUMINAMATH_CALUDE_valid_a_values_l220_22072


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l220_22081

theorem quadratic_equation_solution (x : ℝ) 
  (eq : 2 * x^2 = 9 * x - 4) 
  (neq : x ≠ 4) : 
  2 * x = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l220_22081


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l220_22091

theorem sum_of_fourth_powers (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 1) : x^4 + y^4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l220_22091


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l220_22003

theorem simplify_and_evaluate (x : ℝ) (h : x ≠ 3) :
  (x^2 - x - 6) / (x - 3) = x + 2 ∧ 
  (4^2 - 4 - 6) / (4 - 3) = 6 :=
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l220_22003


namespace NUMINAMATH_CALUDE_calculator_decimal_correction_l220_22054

theorem calculator_decimal_correction (x y : ℚ) (z : ℕ) :
  x = 0.065 →
  y = 3.25 →
  z = 21125 →
  (x * y : ℚ) = 0.21125 :=
by
  sorry

end NUMINAMATH_CALUDE_calculator_decimal_correction_l220_22054


namespace NUMINAMATH_CALUDE_sum_of_three_equal_numbers_l220_22016

theorem sum_of_three_equal_numbers (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 20 → 
  a = 12 → 
  b = 24 → 
  c = d → 
  d = e → 
  c + d + e = 64 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_equal_numbers_l220_22016


namespace NUMINAMATH_CALUDE_five_digit_sum_l220_22015

theorem five_digit_sum (x : ℕ) : 
  (1 + 3 + 4 + 6 + x) * (5 * 4 * 3 * 2 * 1) = 2640 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_sum_l220_22015


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l220_22065

theorem binomial_coefficient_equality (k : ℕ) : 
  (Nat.choose 18 k = Nat.choose 18 (2 * k - 3)) ↔ (k = 3 ∨ k = 7) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l220_22065


namespace NUMINAMATH_CALUDE_unique_prime_solution_l220_22005

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem unique_prime_solution :
  ∀ p q r : ℕ,
    is_prime p ∧ is_prime q ∧ is_prime r ∧
    p * (q - r) = q + r →
    p = 5 ∧ q = 3 ∧ r = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l220_22005


namespace NUMINAMATH_CALUDE_cards_13_and_38_lowest_probability_l220_22013

/-- Represents the probability that a card is red side up after flips -/
def probability_red_up (k : ℕ) : ℚ :=
  if k ≤ 25 then
    (676 - 52 * k + 2 * k^2) / 676
  else
    (676 - 52 * (51 - k) + 2 * (51 - k)^2) / 676

/-- The total number of cards -/
def total_cards : ℕ := 50

/-- Theorem stating that cards 13 and 38 have the lowest probability of being red side up -/
theorem cards_13_and_38_lowest_probability :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ total_cards →
    probability_red_up 13 ≤ probability_red_up k ∧
    probability_red_up 38 ≤ probability_red_up k :=
sorry

end NUMINAMATH_CALUDE_cards_13_and_38_lowest_probability_l220_22013


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l220_22078

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 243 ways to distribute 5 distinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l220_22078


namespace NUMINAMATH_CALUDE_angle_sum_at_point_l220_22080

theorem angle_sum_at_point (x y : ℝ) : 
  3 * x + 6 * x + (x + y) + 4 * y = 360 → x = 0 ∧ y = 72 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_at_point_l220_22080


namespace NUMINAMATH_CALUDE_det_B_squared_minus_3B_l220_22076

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : Matrix.det ((B ^ 2) - 3 • B) = 88 := by sorry

end NUMINAMATH_CALUDE_det_B_squared_minus_3B_l220_22076


namespace NUMINAMATH_CALUDE_no_real_solutions_l220_22098

theorem no_real_solutions :
  (¬ ∃ x : ℝ, Real.sqrt (x + 1) - Real.sqrt (x - 1) = 0) ∧
  (¬ ∃ x : ℝ, Real.sqrt x - Real.sqrt (x - Real.sqrt (1 - x)) = 1) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l220_22098


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l220_22088

/-- A quadratic polynomial that satisfies specific conditions -/
def q (x : ℚ) : ℚ := (6/5) * x^2 - (4/5) * x + 8/5

/-- Theorem stating that q satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-2) = 8 ∧ q 1 = 2 ∧ q 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l220_22088


namespace NUMINAMATH_CALUDE_inequality_proof_l220_22020

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  ((a^2 / b) + (b^2 / c) + (c^2 / a) ≥ 1) ∧ (a * b + b * c + a * c ≤ 1/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l220_22020


namespace NUMINAMATH_CALUDE_restoration_time_is_minimum_l220_22068

/-- Represents the time required for a process on a handicraft -/
structure ProcessTime :=
  (shaping : ℕ)
  (painting : ℕ)

/-- The set of handicrafts -/
inductive Handicraft
  | A
  | B
  | C

/-- The time required for each handicraft -/
def handicraftTime : Handicraft → ProcessTime
  | Handicraft.A => ⟨9, 15⟩
  | Handicraft.B => ⟨16, 8⟩
  | Handicraft.C => ⟨10, 14⟩

/-- The minimum time required to complete the restoration work -/
def minRestorationTime : ℕ := 46

theorem restoration_time_is_minimum :
  minRestorationTime = 46 ∧
  ∀ (order : List Handicraft), order.length = 3 →
    (order.foldl (λ acc h => acc + (handicraftTime h).shaping) 0) +
    (List.maximum (order.map (λ h => (handicraftTime h).painting)) ).getD 0 ≥ minRestorationTime :=
  sorry

#check restoration_time_is_minimum

end NUMINAMATH_CALUDE_restoration_time_is_minimum_l220_22068


namespace NUMINAMATH_CALUDE_adi_change_l220_22027

/-- The change Adi receives when buying a pencil -/
theorem adi_change (pencil_cost : ℕ) (payment : ℕ) (change : ℕ) : 
  pencil_cost = 35 →
  payment = 100 →
  change = payment - pencil_cost →
  change = 65 :=
by
  sorry

end NUMINAMATH_CALUDE_adi_change_l220_22027


namespace NUMINAMATH_CALUDE_series_sum_l220_22024

theorem series_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let series_term (n : ℕ) := 1 / (((n - 1) * a - (n - 2) * b) * (n * a - (n - 1) * b))
  let series_sum := ∑' n, series_term n
  series_sum = 1 / ((a - b) * b) := by sorry

end NUMINAMATH_CALUDE_series_sum_l220_22024


namespace NUMINAMATH_CALUDE_intersection_implies_m_value_l220_22026

def A (m : ℝ) : Set ℝ := {1, 2, m}
def B : Set ℝ := {3, 4}

theorem intersection_implies_m_value (m : ℝ) :
  A m ∩ B = {3} → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_value_l220_22026


namespace NUMINAMATH_CALUDE_parallelogram_area_26_14_l220_22037

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 26 cm and height 14 cm is 364 square centimeters -/
theorem parallelogram_area_26_14 : parallelogram_area 26 14 = 364 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_26_14_l220_22037


namespace NUMINAMATH_CALUDE_city_population_theorem_l220_22008

/-- Given three cities with populations H, L, and C, where H > L and C = H - 5000,
    prove that if L + C = H + C - 5000, then L = H - 5000. -/
theorem city_population_theorem (H L C : ℕ) 
    (h1 : H > L) 
    (h2 : C = H - 5000) 
    (h3 : L + C = H + C - 5000) : 
  L = H - 5000 := by
  sorry

end NUMINAMATH_CALUDE_city_population_theorem_l220_22008


namespace NUMINAMATH_CALUDE_sum_remainder_mod_nine_l220_22004

theorem sum_remainder_mod_nine : (8243 + 8244 + 8245 + 8246) % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_nine_l220_22004


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_M_l220_22067

open Set Real

def M : Set ℝ := {x | ∃ y, y = log (x - 1)}
def N : Set ℝ := {y | ∃ x, y = x^2 + 1}

theorem M_intersect_N_eq_M : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_M_l220_22067


namespace NUMINAMATH_CALUDE_monotonic_subsequence_exists_l220_22086

theorem monotonic_subsequence_exists (a : Fin 10 → ℝ) (h : Function.Injective a) :
  ∃ (i j k l : Fin 10), i < j ∧ j < k ∧ k < l ∧
    ((a i ≤ a j ∧ a j ≤ a k ∧ a k ≤ a l) ∨
     (a i ≥ a j ∧ a j ≥ a k ∧ a k ≥ a l)) := by
  sorry

end NUMINAMATH_CALUDE_monotonic_subsequence_exists_l220_22086


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l220_22063

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_condition : a 2^2 + 2*a 2*a 8 + a 6*a 10 = 16) : 
  a 4 * a 6 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l220_22063


namespace NUMINAMATH_CALUDE_only_two_solutions_l220_22096

/-- Represents a solution of steers and cows --/
structure Solution :=
  (s : ℕ+)
  (c : ℕ+)

/-- Checks if a solution is valid given the budget constraint --/
def is_valid_solution (sol : Solution) : Prop :=
  30 * sol.s.val + 35 * sol.c.val = 1500

/-- The set of all valid solutions --/
def valid_solutions : Set Solution :=
  {sol : Solution | is_valid_solution sol}

/-- The theorem stating that there are only two valid solutions --/
theorem only_two_solutions :
  valid_solutions = {⟨1, 42⟩, ⟨36, 12⟩} :=
sorry

end NUMINAMATH_CALUDE_only_two_solutions_l220_22096


namespace NUMINAMATH_CALUDE_x_equals_one_l220_22050

theorem x_equals_one (y : ℝ) (a : ℝ) (x : ℝ) 
  (h1 : x + a * y = 10) 
  (h2 : y = 3) : 
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_x_equals_one_l220_22050


namespace NUMINAMATH_CALUDE_line_intersects_circle_l220_22087

/-- Given a point outside a circle, prove that a specific line intersects the circle -/
theorem line_intersects_circle (x₀ y₀ a : ℝ) (h₁ : a > 0) (h₂ : x₀^2 + y₀^2 > a^2) :
  ∃ (x y : ℝ), x^2 + y^2 = a^2 ∧ x₀*x + y₀*y = a^2 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l220_22087


namespace NUMINAMATH_CALUDE_pencil_eraser_cost_l220_22095

theorem pencil_eraser_cost :
  ∀ (p e : ℕ),
  15 * p + 5 * e = 125 →
  p > e →
  p > 0 →
  e > 0 →
  p + e = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_eraser_cost_l220_22095


namespace NUMINAMATH_CALUDE_largest_value_u3_plus_v3_l220_22082

theorem largest_value_u3_plus_v3 (u v : ℂ) 
  (h1 : Complex.abs (u + v) = 3)
  (h2 : Complex.abs (u^2 + v^2) = 10) :
  Complex.abs (u^3 + v^3) = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_largest_value_u3_plus_v3_l220_22082


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l220_22071

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- Define the theorem
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_prod : a 5 * a 7 = 2)
  (h_sum : a 2 + a 10 = 3) :
  a 12 / a 4 = 2 ∨ a 12 / a 4 = 1/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l220_22071


namespace NUMINAMATH_CALUDE_even_perfect_square_factors_count_l220_22029

/-- The number of even perfect square factors of 2^6 * 5^4 * 7^3 -/
def num_even_perfect_square_factors : ℕ :=
  sorry

/-- The given number -/
def given_number : ℕ :=
  2^6 * 5^4 * 7^3

theorem even_perfect_square_factors_count :
  num_even_perfect_square_factors = 18 :=
by sorry

end NUMINAMATH_CALUDE_even_perfect_square_factors_count_l220_22029


namespace NUMINAMATH_CALUDE_solve_equation_l220_22058

theorem solve_equation (x : ℝ) : 
  (x^4)^(1/3) = 32 * 32^(1/12) → x = 16 * 2^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l220_22058


namespace NUMINAMATH_CALUDE_vector_sign_sum_l220_22031

/-- Given a 3-dimensional vector with nonzero components, the sum of the signs of its components
    plus the sign of their product can only be 4, 0, or -2. -/
theorem vector_sign_sum (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x / |x| + y / |y| + z / |z| + (x * y * z) / |x * y * z|) ∈ ({4, 0, -2} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vector_sign_sum_l220_22031


namespace NUMINAMATH_CALUDE_arccos_sin_eight_l220_22000

-- Define the problem statement
theorem arccos_sin_eight : 
  Real.arccos (Real.sin 8) = Real.pi / 2 - 1.72 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sin_eight_l220_22000


namespace NUMINAMATH_CALUDE_prime_divisibility_l220_22052

theorem prime_divisibility (p q : Nat) : 
  Nat.Prime p → Nat.Prime q → q ∣ (3^p - 2^p) → p ∣ (q - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_l220_22052


namespace NUMINAMATH_CALUDE_two_numbers_sum_and_difference_l220_22047

theorem two_numbers_sum_and_difference (x y : ℝ) : 
  x + y = 18 ∧ x - y = 24 → x = 21 ∧ y = -3 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_and_difference_l220_22047


namespace NUMINAMATH_CALUDE_exactly_five_numbers_l220_22032

/-- A function that returns the number of ways a positive integer can be written as the sum of consecutive positive odd integers -/
def numConsecutiveOddSums (n : ℕ) : ℕ := sorry

/-- A function that checks if a positive integer is less than 100 and can be written as the sum of consecutive positive odd integers in exactly 3 different ways -/
def isValidNumber (n : ℕ) : Prop :=
  n < 100 ∧ numConsecutiveOddSums n = 3

/-- The main theorem stating that there are exactly 5 numbers satisfying the conditions -/
theorem exactly_five_numbers :
  ∃ (S : Finset ℕ), S.card = 5 ∧ ∀ n, n ∈ S ↔ isValidNumber n :=
sorry

end NUMINAMATH_CALUDE_exactly_five_numbers_l220_22032


namespace NUMINAMATH_CALUDE_combined_sticker_count_l220_22022

theorem combined_sticker_count 
  (june_initial : ℕ) 
  (bonnie_initial : ℕ) 
  (birthday_gift : ℕ) : 
  june_initial + bonnie_initial + 2 * birthday_gift = 
    (june_initial + birthday_gift) + (bonnie_initial + birthday_gift) := by
  sorry

end NUMINAMATH_CALUDE_combined_sticker_count_l220_22022


namespace NUMINAMATH_CALUDE_min_value_of_f_l220_22060

def f (x : ℝ) := x^2 - 4*x + 7

theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l220_22060


namespace NUMINAMATH_CALUDE_probability_of_drawing_two_l220_22097

def card_set : Finset ℕ := {1, 2, 2, 3, 5}

theorem probability_of_drawing_two (s : Finset ℕ := card_set) :
  (s.filter (· = 2)).card / s.card = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_of_drawing_two_l220_22097


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l220_22006

def U : Set Nat := {0, 1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {3, 4, 5, 6}

theorem intersection_complement_theorem :
  A ∩ (U \ B) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l220_22006


namespace NUMINAMATH_CALUDE_system_solution_l220_22090

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(1, 5, 2, 3), (1, 5, 3, 2), (5, 1, 2, 3), (5, 1, 3, 2), (2, 2, 2, 2),
   (2, 3, 1, 5), (2, 3, 5, 1), (3, 2, 1, 5), (3, 2, 5, 1)}

theorem system_solution (a b c d : ℕ) :
  (a * b = c + d ∧ c * d = a + b) ↔ (a, b, c, d) ∈ solution_set := by
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l220_22090


namespace NUMINAMATH_CALUDE_parallel_segment_length_l220_22062

/-- Given a triangle ABC with side AC = 8 cm, if two segments parallel to AC divide the triangle
    into three equal areas, then the length of the parallel segment closest to AC is 8√3/3. -/
theorem parallel_segment_length (A B C : ℝ × ℝ) (a b : ℝ) :
  let triangle_area := (4 : ℝ) * b
  let segment_de_length := (8 : ℝ) * Real.sqrt 6 / 3
  let segment_fg_length := (8 : ℝ) * Real.sqrt 3 / 3
  A = (0, 0) →
  B = (a, b) →
  C = (8, 0) →
  triangle_area / 3 = b * (a * Real.sqrt (8 / 3))^2 / (2 * a) →
  segment_de_length > segment_fg_length →
  segment_fg_length = 8 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_segment_length_l220_22062


namespace NUMINAMATH_CALUDE_wood_burning_problem_l220_22070

/-- Wood burning problem -/
theorem wood_burning_problem (initial_bundles morning_burned end_bundles : ℕ) 
  (h1 : initial_bundles = 10)
  (h2 : morning_burned = 4)
  (h3 : end_bundles = 3) :
  initial_bundles - morning_burned - end_bundles = 3 :=
by sorry

end NUMINAMATH_CALUDE_wood_burning_problem_l220_22070


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l220_22033

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x/y + y/z + z/x + y/x + z/y + x/z = 10) :
  (x/y + y/z + z/x) * (y/x + z/y + x/z) + (x/y * y/z * z/x) ≥ 44 := by
  sorry

-- Optionally, we can add a statement to show that the lower bound is tight
theorem min_value_achievable : 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    x/y + y/z + z/x + y/x + z/y + x/z = 10 ∧
    (x/y + y/z + z/x) * (y/x + z/y + x/z) + (x/y * y/z * z/x) = 44 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l220_22033


namespace NUMINAMATH_CALUDE_jack_additional_sweets_l220_22030

theorem jack_additional_sweets (initial_sweets : ℕ) (remaining_sweets : ℕ) : 
  initial_sweets = 22 →
  remaining_sweets = 7 →
  (initial_sweets / 2 + (initial_sweets - remaining_sweets - initial_sweets / 2) = initial_sweets - remaining_sweets) →
  initial_sweets - remaining_sweets - initial_sweets / 2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_jack_additional_sweets_l220_22030


namespace NUMINAMATH_CALUDE_hagrid_divisible_by_three_l220_22093

def HAGRID (H A G R I D : ℕ) : ℕ := 100000*H + 10000*A + 1000*G + 100*R + 10*I + D

theorem hagrid_divisible_by_three 
  (H A G R I D : ℕ) 
  (h_distinct : H ≠ A ∧ H ≠ G ∧ H ≠ R ∧ H ≠ I ∧ H ≠ D ∧ 
                A ≠ G ∧ A ≠ R ∧ A ≠ I ∧ A ≠ D ∧ 
                G ≠ R ∧ G ≠ I ∧ G ≠ D ∧ 
                R ≠ I ∧ R ≠ D ∧ 
                I ≠ D)
  (h_range : H < 10 ∧ A < 10 ∧ G < 10 ∧ R < 10 ∧ I < 10 ∧ D < 10) : 
  3 ∣ (HAGRID H A G R I D * H * A * G * R * I * D) :=
sorry

end NUMINAMATH_CALUDE_hagrid_divisible_by_three_l220_22093


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l220_22001

def repeating_decimal (n : ℕ) : ℚ := n / 9

theorem sum_of_repeating_decimals :
  repeating_decimal 7 + repeating_decimal 5 - repeating_decimal 6 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l220_22001


namespace NUMINAMATH_CALUDE_min_value_of_f_l220_22084

theorem min_value_of_f (x : ℝ) (hx : x > 0) : x + 1/x - 2 ≥ 0 ∧ (x + 1/x - 2 = 0 ↔ x = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l220_22084


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l220_22075

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Carton dimensions -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 42, height := 60 }

/-- Soap box dimensions -/
def soapBoxDimensions : BoxDimensions :=
  { length := 7, width := 6, height := 6 }

/-- Theorem: The maximum number of soap boxes that can be placed in the carton is 250 -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions) = 250 := by
  sorry

end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l220_22075


namespace NUMINAMATH_CALUDE_concyclic_projections_l220_22039

/-- Four points are concyclic if they lie on the same circle. -/
def Concyclic (A B C D : Point) : Prop := sorry

/-- The orthogonal projection of a point onto a line. -/
def OrthogonalProjection (P Q R : Point) : Point := sorry

/-- The main theorem: if A, B, C, D are concyclic, and A', C' are orthogonal projections of A, C 
    onto BD, and B', D' are orthogonal projections of B, D onto AC, then A', B', C', D' are concyclic. -/
theorem concyclic_projections 
  (A B C D : Point) 
  (h_concyclic : Concyclic A B C D) 
  (A' : Point) (h_A' : A' = OrthogonalProjection A B D)
  (C' : Point) (h_C' : C' = OrthogonalProjection C B D)
  (B' : Point) (h_B' : B' = OrthogonalProjection B A C)
  (D' : Point) (h_D' : D' = OrthogonalProjection D A C) :
  Concyclic A' B' C' D' :=
sorry

end NUMINAMATH_CALUDE_concyclic_projections_l220_22039


namespace NUMINAMATH_CALUDE_starters_combination_l220_22041

-- Define the total number of players
def total_players : ℕ := 18

-- Define the number of quadruplets
def num_quadruplets : ℕ := 4

-- Define the number of starters to choose
def num_starters : ℕ := 7

-- Define the number of quadruplets that must be in the starting lineup
def required_quadruplets : ℕ := 3

-- Define the function to calculate the number of ways to choose the starters
def choose_starters (total : ℕ) (quad : ℕ) (starters : ℕ) (req_quad : ℕ) : ℕ :=
  (Nat.choose quad req_quad) * (Nat.choose (total - quad) (starters - req_quad))

-- Theorem statement
theorem starters_combination : 
  choose_starters total_players num_quadruplets num_starters required_quadruplets = 4004 := by
  sorry

end NUMINAMATH_CALUDE_starters_combination_l220_22041


namespace NUMINAMATH_CALUDE_spinster_count_l220_22049

theorem spinster_count : 
  ∀ (spinsters cats : ℕ), 
    (spinsters : ℚ) / (cats : ℚ) = 2 / 7 →
    cats = spinsters + 35 →
    spinsters = 14 := by
  sorry

end NUMINAMATH_CALUDE_spinster_count_l220_22049


namespace NUMINAMATH_CALUDE_f_four_equals_thirtysix_l220_22045

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + 2 * f b

theorem f_four_equals_thirtysix
  (f : ℝ → ℝ)
  (h_not_zero : ∃ x, f x ≠ 0)
  (h_eq : FunctionalEquation f)
  (h_f_two : f 2 = 9) :
  f 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_f_four_equals_thirtysix_l220_22045


namespace NUMINAMATH_CALUDE_sixth_power_sum_l220_22099

theorem sixth_power_sum (x : ℝ) (hx : x ≠ 0) : x + 1/x = 1 → x^6 + 1/x^6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_sum_l220_22099


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l220_22074

theorem simplify_complex_fraction : 
  (1 / ((3 / (Real.sqrt 5 + 2)) - (1 / (Real.sqrt 4 + 1)))) = ((27 * Real.sqrt 5 + 57) / 40) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l220_22074


namespace NUMINAMATH_CALUDE_vector_decomposition_l220_22018

/-- Given vectors in R^3 -/
def x : Fin 3 → ℝ := ![5, 15, 0]
def p : Fin 3 → ℝ := ![1, 0, 5]
def q : Fin 3 → ℝ := ![-1, 3, 2]
def r : Fin 3 → ℝ := ![0, -1, 1]

/-- The decomposition of x in terms of p, q, and r -/
theorem vector_decomposition :
  x = (4 : ℝ) • p - (1 : ℝ) • q - (18 : ℝ) • r := by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l220_22018
