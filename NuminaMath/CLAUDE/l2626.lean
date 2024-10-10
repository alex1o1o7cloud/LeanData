import Mathlib

namespace min_sum_squares_l2626_262645

theorem min_sum_squares (x y z : ℝ) (h : x + 2*y + 2*z = 6) :
  ∃ (m : ℝ), (∀ a b c : ℝ, a + 2*b + 2*c = 6 → a^2 + b^2 + c^2 ≥ m) ∧
             (x^2 + y^2 + z^2 = m) ∧
             m = 4 := by
  sorry

end min_sum_squares_l2626_262645


namespace initial_crayons_l2626_262631

theorem initial_crayons (initial final added : ℕ) : 
  final = initial + added → 
  added = 6 → 
  final = 13 → 
  initial = 7 := by 
sorry

end initial_crayons_l2626_262631


namespace boat_license_count_l2626_262699

def boat_license_options : ℕ :=
  let letter_options := 3  -- A, M, or B
  let digit_options := 10  -- 0 to 9
  let digit_positions := 5
  letter_options * digit_options ^ digit_positions

theorem boat_license_count : boat_license_options = 300000 := by
  sorry

end boat_license_count_l2626_262699


namespace inverse_f_69_l2626_262615

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 + 6

-- State the theorem
theorem inverse_f_69 : f⁻¹ 69 = (21 : ℝ)^(1/3) := by sorry

end inverse_f_69_l2626_262615


namespace factorial_prime_factorization_l2626_262661

theorem factorial_prime_factorization :
  let x : ℕ := Finset.prod (Finset.range 15) (fun i => i + 1)
  ∀ (i k m p q r : ℕ),
    (i > 0 ∧ k > 0 ∧ m > 0 ∧ p > 0 ∧ q > 0 ∧ r > 0) →
    x = 2^i * 3^k * 5^m * 7^p * 11^q * 13^r →
    i + k + m + p + q + r = 29 := by
  sorry

end factorial_prime_factorization_l2626_262661


namespace unscreened_percentage_l2626_262638

/-- Calculates the percentage of unscreened part of a TV -/
theorem unscreened_percentage (tv_length tv_width screen_length screen_width : ℕ) 
  (h1 : tv_length = 6) (h2 : tv_width = 5) (h3 : screen_length = 5) (h4 : screen_width = 4) :
  (1 : ℚ) / 3 * 100 = 
    (tv_length * tv_width - screen_length * screen_width : ℚ) / (tv_length * tv_width) * 100 := by
  sorry

end unscreened_percentage_l2626_262638


namespace sqrt_sum_rational_implies_components_rational_l2626_262660

theorem sqrt_sum_rational_implies_components_rational
  (m n p : ℚ)
  (h : ∃ (q : ℚ), Real.sqrt m + Real.sqrt n + Real.sqrt p = q) :
  (∃ (r : ℚ), Real.sqrt m = r) ∧
  (∃ (s : ℚ), Real.sqrt n = s) ∧
  (∃ (t : ℚ), Real.sqrt p = t) :=
sorry

end sqrt_sum_rational_implies_components_rational_l2626_262660


namespace distinct_arrangements_l2626_262694

theorem distinct_arrangements (n : ℕ) (h : n = 8) : Nat.factorial n = 40320 := by
  sorry

end distinct_arrangements_l2626_262694


namespace area_ratio_quadrilateral_triangle_l2626_262652

-- Define the types for points and shapes
variable (Point : Type) [AddCommGroup Point] [Module ℝ Point]
variable (Quadrilateral : Type)
variable (Triangle : Type)

-- Define functions for area calculation
variable (area : Quadrilateral → ℝ)
variable (area_triangle : Triangle → ℝ)

-- Define a function to create a quadrilateral from four points
variable (make_quadrilateral : Point → Point → Point → Point → Quadrilateral)

-- Define a function to create a triangle from three points
variable (make_triangle : Point → Point → Point → Triangle)

-- Define a function to get the midpoint of two points
variable (midpoint : Point → Point → Point)

-- Define a function to extend two line segments to their intersection
variable (extend_to_intersection : Point → Point → Point → Point → Point)

-- Theorem statement
theorem area_ratio_quadrilateral_triangle 
  (A B C D : Point) 
  (ABCD : Quadrilateral) 
  (E : Point) 
  (H G : Point) 
  (EHG : Triangle) :
  ABCD = make_quadrilateral A B C D →
  E = extend_to_intersection A D B C →
  H = midpoint B D →
  G = midpoint A C →
  EHG = make_triangle E H G →
  (area_triangle EHG) / (area ABCD) = 1/4 := by sorry

end area_ratio_quadrilateral_triangle_l2626_262652


namespace betty_gave_forty_percent_l2626_262692

/-- The percentage of marbles Betty gave to Stuart -/
def percentage_given (betty_initial : ℕ) (stuart_initial : ℕ) (stuart_final : ℕ) : ℚ :=
  (stuart_final - stuart_initial : ℚ) / betty_initial * 100

/-- Theorem stating that Betty gave Stuart 40% of her marbles -/
theorem betty_gave_forty_percent :
  let betty_initial : ℕ := 60
  let stuart_initial : ℕ := 56
  let stuart_final : ℕ := 80
  percentage_given betty_initial stuart_initial stuart_final = 40 := by
sorry

end betty_gave_forty_percent_l2626_262692


namespace sequence_problem_l2626_262609

/-- Given a sequence {a_n} with sum S_n = kn^2 + n and a_10 = 39, prove a_100 = 399 -/
theorem sequence_problem (k : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = k * n^2 + n) →
  a 10 = 39 →
  a 100 = 399 := by
sorry

end sequence_problem_l2626_262609


namespace arithmetic_square_root_of_0_2_l2626_262665

theorem arithmetic_square_root_of_0_2 : ∃ x : ℝ, x^2 = 0.2 ∧ x ≠ 0.02 := by sorry

end arithmetic_square_root_of_0_2_l2626_262665


namespace fair_admission_collection_l2626_262608

theorem fair_admission_collection :
  let child_fee : ℚ := 3/2  -- $1.50 as a rational number
  let adult_fee : ℚ := 4    -- $4.00 as a rational number
  let total_people : ℕ := 2200
  let num_children : ℕ := 700
  let num_adults : ℕ := 1500
  
  (num_children : ℚ) * child_fee + (num_adults : ℚ) * adult_fee = 7050
  := by sorry

end fair_admission_collection_l2626_262608


namespace intersection_of_lines_l2626_262686

/-- The intersection point of two lines in 3D space --/
def intersection_point (A B C D : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

theorem intersection_of_lines (A B C D : ℝ × ℝ × ℝ) 
  (hA : A = (5, -3, 2)) 
  (hB : B = (15, -13, 7)) 
  (hC : C = (2, 4, -5)) 
  (hD : D = (4, -1, 15)) : 
  intersection_point A B C D = (23/3, -19/3, 7/3) := by
  sorry

end intersection_of_lines_l2626_262686


namespace cylinder_cone_dimensions_l2626_262606

theorem cylinder_cone_dimensions (r m : ℝ) : 
  r > 0 ∧ m > 0 →
  (2 * π * r * m) / (π * r * Real.sqrt (m^2 + r^2)) = 8 / 5 →
  r * m = 588 →
  m = 28 ∧ r = 21 :=
by sorry

end cylinder_cone_dimensions_l2626_262606


namespace total_results_l2626_262601

theorem total_results (average : ℝ) (first_12_avg : ℝ) (last_12_avg : ℝ) (result_13 : ℝ) :
  average = 24 →
  first_12_avg = 14 →
  last_12_avg = 17 →
  result_13 = 228 →
  ∃ (n : ℕ), n = 25 ∧ (12 * first_12_avg + result_13 + 12 * last_12_avg) / n = average :=
by
  sorry


end total_results_l2626_262601


namespace smallest_y_for_square_l2626_262695

theorem smallest_y_for_square (y : ℕ+) (M : ℤ) : 
  (∀ k : ℕ+, k < y → ¬∃ N : ℤ, (2310 : ℤ) * k = N^2) →
  (∃ N : ℤ, (2310 : ℤ) * y = N^2) →
  y = 2310 := by sorry

end smallest_y_for_square_l2626_262695


namespace vector_equality_vector_equation_solution_l2626_262684

/-- Two vectors in ℝ² are equal if their corresponding components are equal -/
theorem vector_equality (a b c d : ℝ) : (a, b) = (c, d) ↔ a = c ∧ b = d := by sorry

/-- Definition of Vector1 -/
def Vector1 (u : ℝ) : ℝ × ℝ := (3 + 5*u, -1 - 3*u)

/-- Definition of Vector2 -/
def Vector2 (v : ℝ) : ℝ × ℝ := (0 - 3*v, 2 + 4*v)

theorem vector_equation_solution :
  ∃ (u v : ℝ), Vector1 u = Vector2 v ∧ u = -3/11 ∧ v = -16/11 := by sorry

end vector_equality_vector_equation_solution_l2626_262684


namespace fg_properties_l2626_262655

open Real

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 3 * x - (a + 1) * log x

/-- The function g(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 - a * x + 4

/-- The sum of f(x) and g(x) -/
noncomputable def sum_fg (a : ℝ) (x : ℝ) : ℝ := f a x + g a x

/-- The difference of f(x) and g(x) -/
noncomputable def diff_fg (a : ℝ) (x : ℝ) : ℝ := f a x - g a x

/-- Theorem stating the conditions for monotonicity and tangency -/
theorem fg_properties :
  (∀ a : ℝ, a ≤ -1 → ∀ x > 0, Monotone (sum_fg a)) ∧
  (∃ a : ℝ, 1 < a ∧ a < 3 ∧ ∃ x > 0, diff_fg a x = 0 ∧ HasDerivAt (diff_fg a) 0 x) :=
sorry

end fg_properties_l2626_262655


namespace inequality_system_solution_l2626_262651

theorem inequality_system_solution (m : ℝ) : 
  (∃ x : ℝ, (x - m) / 2 ≥ 2 ∧ x - 4 ≤ 3 * (x - 2)) ∧ 
  (∀ x : ℤ, x < 2 → ¬((x - m) / 2 ≥ 2 ∧ x - 4 ≤ 3 * (x - 2))) →
  -3 < m ∧ m ≤ -2 := by
sorry

end inequality_system_solution_l2626_262651


namespace perpendicular_transitivity_l2626_262616

/-- In three-dimensional space -/
structure Space3D where

/-- Represent a line in 3D space -/
structure Line (S : Space3D) where

/-- Represent a plane in 3D space -/
structure Plane (S : Space3D) where

/-- Perpendicular relation between two lines -/
def Line.perp (S : Space3D) (l1 l2 : Line S) : Prop :=
  sorry

/-- Perpendicular relation between a line and a plane -/
def Line.perpToPlane (S : Space3D) (l : Line S) (p : Plane S) : Prop :=
  sorry

/-- Perpendicular relation between two planes -/
def Plane.perp (S : Space3D) (p1 p2 : Plane S) : Prop :=
  sorry

/-- The main theorem -/
theorem perpendicular_transitivity (S : Space3D) (a b : Line S) (α β : Plane S) :
  a ≠ b → α ≠ β →
  Line.perp S a b →
  Line.perpToPlane S a α →
  Line.perpToPlane S b β →
  Plane.perp S α β :=
sorry

end perpendicular_transitivity_l2626_262616


namespace park_wheels_count_l2626_262678

/-- The total number of wheels on bikes in a park -/
def total_wheels (regular_bikes children_bikes tandem_4_wheels tandem_6_wheels : ℕ) : ℕ :=
  regular_bikes * 2 + children_bikes * 4 + tandem_4_wheels * 4 + tandem_6_wheels * 6

/-- Theorem: The total number of wheels in the park is 96 -/
theorem park_wheels_count :
  total_wheels 7 11 5 3 = 96 := by
  sorry

end park_wheels_count_l2626_262678


namespace complex_equation_system_l2626_262697

theorem complex_equation_system (p q r u v w : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hu : u ≠ 0) (hv : v ≠ 0) (hw : w ≠ 0)
  (eq1 : p = (q + r) / (u - 3))
  (eq2 : q = (p + r) / (v - 3))
  (eq3 : r = (p + q) / (w - 3))
  (eq4 : u * v + u * w + v * w = 7)
  (eq5 : u + v + w = 4) :
  u * v * w = 10 := by
sorry

end complex_equation_system_l2626_262697


namespace amy_cupcakes_l2626_262622

theorem amy_cupcakes (todd_ate : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) 
  (h1 : todd_ate = 5)
  (h2 : packages = 9)
  (h3 : cupcakes_per_package = 5) :
  todd_ate + packages * cupcakes_per_package = 50 := by
  sorry

end amy_cupcakes_l2626_262622


namespace james_driving_distance_l2626_262603

/-- Calculates the total distance driven given multiple segments of a trip -/
def total_distance (speeds : List ℝ) (times : List ℝ) : ℝ :=
  List.sum (List.zipWith (· * ·) speeds times)

/-- James' driving problem -/
theorem james_driving_distance :
  let speeds : List ℝ := [30, 60, 75, 60]
  let times : List ℝ := [0.5, 0.75, 1.5, 2]
  total_distance speeds times = 292.5 := by
  sorry

end james_driving_distance_l2626_262603


namespace cake_pieces_theorem_l2626_262630

/-- The initial number of cake pieces -/
def initial_pieces : ℕ := 240

/-- The percentage of cake pieces eaten -/
def eaten_percentage : ℚ := 60 / 100

/-- The number of people who received the remaining pieces -/
def num_recipients : ℕ := 3

/-- The number of pieces each recipient received -/
def pieces_per_recipient : ℕ := 32

/-- Theorem stating that the initial number of cake pieces is correct -/
theorem cake_pieces_theorem :
  initial_pieces * (1 - eaten_percentage) = num_recipients * pieces_per_recipient := by
  sorry

end cake_pieces_theorem_l2626_262630


namespace system_solution_and_arithmetic_progression_l2626_262662

-- Define the system of equations
def system (m a b c x y z : ℝ) : Prop :=
  x + y + m*z = a ∧ x + m*y + z = b ∧ m*x + y + z = c

-- Theorem statement
theorem system_solution_and_arithmetic_progression
  (m a b c : ℝ) :
  (∃! (x y z : ℝ), system m a b c x y z) ↔ 
    (m ≠ -2 ∧ m ≠ 1) ∧
  (∀ (x y z : ℝ), system m a b c x y z → (2*y = x + z) ↔ a + c = b) :=
sorry

end system_solution_and_arithmetic_progression_l2626_262662


namespace points_to_office_theorem_l2626_262605

/-- The number of points needed to be sent to the office -/
def points_to_office : ℕ := 100

/-- Points for interrupting -/
def interrupt_points : ℕ := 5

/-- Points for insulting classmates -/
def insult_points : ℕ := 10

/-- Points for throwing things -/
def throw_points : ℕ := 25

/-- Number of times Jerry interrupted -/
def jerry_interrupts : ℕ := 2

/-- Number of times Jerry insulted classmates -/
def jerry_insults : ℕ := 4

/-- Number of times Jerry can throw things before being sent to office -/
def jerry_throws_left : ℕ := 2

/-- Theorem stating the number of points needed to be sent to the office -/
theorem points_to_office_theorem :
  points_to_office = 
    jerry_interrupts * interrupt_points +
    jerry_insults * insult_points +
    jerry_throws_left * throw_points :=
by sorry

end points_to_office_theorem_l2626_262605


namespace freds_walking_speed_l2626_262664

/-- Proves that Fred's walking speed is 4 miles per hour given the initial conditions -/
theorem freds_walking_speed 
  (initial_distance : ℝ) 
  (sams_speed : ℝ) 
  (sams_distance : ℝ) 
  (h1 : initial_distance = 40) 
  (h2 : sams_speed = 4) 
  (h3 : sams_distance = 20) : 
  (initial_distance - sams_distance) / (sams_distance / sams_speed) = 4 :=
by sorry

end freds_walking_speed_l2626_262664


namespace equation_exists_solution_l2626_262641

theorem equation_exists_solution (x : ℝ) (hx : x = 2407) :
  ∃ (y z : ℝ), x^y + y^x = z :=
sorry

end equation_exists_solution_l2626_262641


namespace work_completion_time_l2626_262677

theorem work_completion_time (a_days b_days : ℝ) (ha : a_days > 0) (hb : b_days > 0) :
  a_days = 60 → b_days = 20 → (a_days⁻¹ + b_days⁻¹)⁻¹ = 15 := by
  sorry

#check work_completion_time

end work_completion_time_l2626_262677


namespace maintenance_interval_after_additive_l2626_262667

/-- Calculates the new maintenance interval after applying an additive -/
def new_maintenance_interval (original_interval : ℕ) (increase_percentage : ℕ) : ℕ :=
  original_interval * (100 + increase_percentage) / 100

/-- Theorem: Given an original maintenance interval of 50 days and a 20% increase,
    the new maintenance interval is 60 days -/
theorem maintenance_interval_after_additive :
  new_maintenance_interval 50 20 = 60 := by
  sorry

end maintenance_interval_after_additive_l2626_262667


namespace equation_simplification_l2626_262674

theorem equation_simplification (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 1 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 1 := by
  sorry

end equation_simplification_l2626_262674


namespace loan_repayment_equality_l2626_262668

/-- Represents the loan scenario described in the problem -/
structure LoanScenario where
  M : ℝ  -- Initial loan amount in million yuan
  x : ℝ  -- Monthly repayment amount in million yuan
  r : ℝ  -- Monthly interest rate (as a decimal)
  n : ℕ  -- Number of months for repayment

/-- The theorem representing the loan repayment equality -/
theorem loan_repayment_equality (scenario : LoanScenario) 
  (h_r : scenario.r = 0.05)
  (h_n : scenario.n = 20) : 
  scenario.n * scenario.x = scenario.M * (1 + scenario.r) ^ scenario.n :=
sorry

end loan_repayment_equality_l2626_262668


namespace sandy_average_book_price_l2626_262627

/-- The average price of books bought by Sandy -/
def average_price (shop1_books shop2_books : ℕ) (shop1_cost shop2_cost : ℚ) : ℚ :=
  (shop1_cost + shop2_cost) / (shop1_books + shop2_books)

/-- Theorem: The average price Sandy paid per book is $16 -/
theorem sandy_average_book_price :
  average_price 65 55 1080 840 = 16 := by
  sorry

end sandy_average_book_price_l2626_262627


namespace five_letter_words_same_ends_l2626_262648

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The length of the words we're considering -/
def word_length : ℕ := 5

/-- The number of letters that can vary in the word -/
def variable_letters : ℕ := word_length - 2

theorem five_letter_words_same_ends : 
  alphabet_size ^ variable_letters = 456976 := by
  sorry


end five_letter_words_same_ends_l2626_262648


namespace artist_painting_difference_l2626_262647

/-- Given an artist's painting schedule over three months, prove that the difference
    between the number of pictures painted in July and June is zero. -/
theorem artist_painting_difference (june july august total : ℕ) 
    (h_june : june = 2)
    (h_august : august = 9)
    (h_total : total = 13)
    (h_sum : june + july + august = total) : july - june = 0 := by
  sorry

end artist_painting_difference_l2626_262647


namespace continued_fraction_convergents_l2626_262619

/-- Continued fraction convergents -/
theorem continued_fraction_convergents
  (P Q : ℕ → ℤ)  -- Sequences of numerators and denominators
  (a : ℕ → ℕ)    -- Sequence of continued fraction coefficients
  (h1 : ∀ k, k ≥ 2 → P k = a k * P (k-1) + P (k-2))
  (h2 : ∀ k, k ≥ 2 → Q k = a k * Q (k-1) + Q (k-2))
  (h3 : ∀ k, a k > 0) :
  (∀ k, k ≥ 2 → P k * Q (k-2) - P (k-2) * Q k = (-1)^k * a k) ∧
  (∀ k, k ≥ 1 → (P k : ℚ) / Q k - (P (k-1) : ℚ) / Q (k-1) = (-1)^(k+1) / (Q k * Q (k-1))) ∧
  (∀ n, n ≥ 1 → ∀ k, 1 ≤ k → k < n → Q k < Q (k+1)) ∧
  (∀ n, n ≥ 0 → (P 0 : ℚ) / Q 0 < (P 2 : ℚ) / Q 2 ∧ 
    (P n : ℚ) / Q n < (P (n+1) : ℚ) / Q (n+1) ∧
    (P (n+2) : ℚ) / Q (n+2) < (P (n+1) : ℚ) / Q (n+1)) ∧
  (∀ k l, k ≥ 0 → l ≥ 0 → (P (2*k) : ℚ) / Q (2*k) < (P (2*l+1) : ℚ) / Q (2*l+1)) :=
by sorry

end continued_fraction_convergents_l2626_262619


namespace student_volunteer_arrangements_l2626_262672

theorem student_volunteer_arrangements :
  let n : ℕ := 5  -- number of students
  let k : ℕ := 2  -- number of communities
  (2^n : ℕ) - k = 30 :=
by sorry

end student_volunteer_arrangements_l2626_262672


namespace zero_function_satisfies_equation_l2626_262613

theorem zero_function_satisfies_equation (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f x - f y) → (∀ x : ℝ, f x = 0) := by
  sorry

end zero_function_satisfies_equation_l2626_262613


namespace probability_at_least_one_defective_l2626_262681

theorem probability_at_least_one_defective (total : ℕ) (defective : ℕ) : 
  total = 20 → defective = 4 → 
  (1 - (total - defective) * (total - defective - 1) / (total * (total - 1))) = 7 / 19 := by
  sorry

end probability_at_least_one_defective_l2626_262681


namespace solution_set_is_open_unit_interval_l2626_262617

-- Define a decreasing function on (-2, 2)
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, -2 < x ∧ x < y ∧ y < 2 → f x > f y

-- Define the solution set
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | -2 < x ∧ x < 2 ∧ f x > f (2 - x)}

-- Theorem statement
theorem solution_set_is_open_unit_interval
  (f : ℝ → ℝ) (h : DecreasingFunction f) :
  SolutionSet f = Set.Ioo 0 1 := by
  sorry

end solution_set_is_open_unit_interval_l2626_262617


namespace root_quadratic_equation_l2626_262669

theorem root_quadratic_equation (m : ℝ) : 
  (2 * m^2 - 3 * m - 1 = 0) → (4 * m^2 - 6 * m = 2) := by
  sorry

end root_quadratic_equation_l2626_262669


namespace value_of_a_l2626_262687

-- Define the conversion rate from paise to rupees
def paise_to_rupees (paise : ℚ) : ℚ := paise / 100

-- Define the problem statement
theorem value_of_a (a : ℚ) (h : (a * 0.005) = paise_to_rupees 85) : a = 170 := by
  sorry

end value_of_a_l2626_262687


namespace polynomial_remainder_l2626_262650

theorem polynomial_remainder (x : ℝ) : (x^4 + x + 2) % (x - 3) = 86 := by
  sorry

end polynomial_remainder_l2626_262650


namespace abc_sum_and_squares_l2626_262604

theorem abc_sum_and_squares (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares_one : a^2 + b^2 + c^2 = 1) : 
  (a*b + b*c + c*a = -1/2) ∧ (a^4 + b^4 + c^4 = 1/2) := by
  sorry

end abc_sum_and_squares_l2626_262604


namespace geometric_sequence_12th_term_l2626_262611

/-- A geometric sequence is defined by its first term and common ratio. -/
structure GeometricSequence where
  a : ℝ  -- first term
  r : ℝ  -- common ratio

/-- The nth term of a geometric sequence. -/
def GeometricSequence.nthTerm (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.a * seq.r ^ (n - 1)

/-- Theorem: In a geometric sequence where the 5th term is 8 and the 9th term is 128, the 12th term is 1024. -/
theorem geometric_sequence_12th_term
  (seq : GeometricSequence)
  (h5 : seq.nthTerm 5 = 8)
  (h9 : seq.nthTerm 9 = 128) :
  seq.nthTerm 12 = 1024 := by
  sorry


end geometric_sequence_12th_term_l2626_262611


namespace parallelogram_smaller_angle_l2626_262614

theorem parallelogram_smaller_angle (smaller_angle larger_angle : ℝ) : 
  larger_angle = smaller_angle + 120 →
  smaller_angle + larger_angle + smaller_angle + larger_angle = 360 →
  smaller_angle = 30 := by
sorry

end parallelogram_smaller_angle_l2626_262614


namespace arithmetic_sequence_property_l2626_262657

theorem arithmetic_sequence_property (n : ℕ+) : 
  (∀ S : Finset ℕ, S ⊆ Finset.range 1989 → S.card = n → 
    ∃ (a d : ℕ) (H : Finset ℕ), H ⊆ S ∧ H.card = 29 ∧ 
    ∀ k, k ∈ H → ∃ i, 0 ≤ i ∧ i < 29 ∧ k = a + i * d) → 
  n > 1788 := by
sorry

end arithmetic_sequence_property_l2626_262657


namespace quadratic_inequality_always_holds_l2626_262698

theorem quadratic_inequality_always_holds (m : ℝ) :
  (∀ x : ℝ, m * x^2 - (m + 3) * x - 1 < 0) ↔ -9 < m ∧ m < -1 := by
  sorry

end quadratic_inequality_always_holds_l2626_262698


namespace sufficient_condition_range_l2626_262680

/-- p is a sufficient but not necessary condition for q -/
def is_sufficient_not_necessary (p q : Prop) : Prop :=
  (p → q) ∧ ¬(q → p)

theorem sufficient_condition_range (a : ℝ) :
  is_sufficient_not_necessary (∀ x : ℝ, 4 - x ≤ 6) (∀ x : ℝ, x > a - 1) →
  a < -1 := by
  sorry

end sufficient_condition_range_l2626_262680


namespace girls_in_class_l2626_262683

theorem girls_in_class (total : Nat) (prob : Rat) : 
  total = 25 → 
  prob = 3/25 → 
  (fun n : Nat => n * (n - 1) = prob * (total * (total - 1))) 9 → 
  total - 9 = 16 :=
by sorry

end girls_in_class_l2626_262683


namespace positive_real_equality_l2626_262610

theorem positive_real_equality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a^b = b^a) (h2 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end positive_real_equality_l2626_262610


namespace jelly_bean_probability_l2626_262629

theorem jelly_bean_probability (red orange blue : ℝ) (h1 : red = 0.25) (h2 : orange = 0.4) (h3 : blue = 0.1) :
  ∃ yellow : ℝ, yellow = 0.25 ∧ red + orange + blue + yellow = 1 := by
  sorry

end jelly_bean_probability_l2626_262629


namespace M_in_second_and_fourth_quadrants_l2626_262618

def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 * p.2 < 0}

theorem M_in_second_and_fourth_quadrants :
  ∀ p ∈ M, (p.1 > 0 ∧ p.2 < 0) ∨ (p.1 < 0 ∧ p.2 > 0) :=
by sorry

end M_in_second_and_fourth_quadrants_l2626_262618


namespace geometric_sequence_common_ratio_l2626_262696

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_eq : a 6 = a 5 + 2 * a 4) :
  ∃ q : ℝ, q = 2 ∧ ∀ n : ℕ, a (n + 1) = a n * q :=
sorry

end geometric_sequence_common_ratio_l2626_262696


namespace parallel_vectors_m_value_l2626_262643

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b (m : ℝ) : Fin 2 → ℝ := ![m, 4]

-- Define the parallelism condition
def are_parallel (u v : Fin 2 → ℝ) : Prop :=
  u 0 * v 1 = u 1 * v 0

-- Theorem statement
theorem parallel_vectors_m_value :
  ∀ m : ℝ, are_parallel a (λ i ↦ 2 * a i + b m i) → m = 2 := by
  sorry

end parallel_vectors_m_value_l2626_262643


namespace min_value_theorem_l2626_262663

theorem min_value_theorem (a b : ℝ) (h1 : a + 2*b = 2) (h2 : a > 1) (h3 : b > 0) :
  (∀ x y : ℝ, x > 1 ∧ y > 0 ∧ x + 2*y = 2 → 2/(x-1) + 1/y ≥ 2/(a-1) + 1/b) ∧
  2/(a-1) + 1/b = 8 :=
sorry

end min_value_theorem_l2626_262663


namespace complete_square_l2626_262636

theorem complete_square (b : ℝ) : ∀ x : ℝ, x^2 + b*x = (x + b/2)^2 - (b/2)^2 := by sorry

end complete_square_l2626_262636


namespace max_pages_for_25_dollars_l2626_262682

-- Define the cost per page in cents
def cost_per_page : ℕ := 3

-- Define the total amount available in cents
def total_amount : ℕ := 2500

-- Define the function to calculate the maximum number of pages
def max_pages (cost : ℕ) (total : ℕ) : ℕ := 
  (total / cost : ℕ)

-- Theorem statement
theorem max_pages_for_25_dollars : 
  max_pages cost_per_page total_amount = 833 := by
  sorry

end max_pages_for_25_dollars_l2626_262682


namespace percentage_problem_l2626_262626

theorem percentage_problem (P : ℝ) : 
  (0.5 * 640 = P / 100 * 650 + 190) → P = 20 := by
  sorry

end percentage_problem_l2626_262626


namespace power_zero_equals_one_l2626_262637

theorem power_zero_equals_one (x : ℝ) : x ^ 0 = 1 := by
  sorry

end power_zero_equals_one_l2626_262637


namespace problem_proof_l2626_262675

theorem problem_proof : Real.sqrt 8 - 4 * Real.sin (π / 4) - (1 / 3)⁻¹ = -3 := by
  sorry

end problem_proof_l2626_262675


namespace solve_colored_paper_problem_l2626_262666

def colored_paper_problem (initial : ℕ) (given_per_friend : ℕ) (num_friends : ℕ) (bought : ℕ) (current : ℕ) : Prop :=
  initial + bought - (given_per_friend * num_friends) = current

theorem solve_colored_paper_problem :
  ∃ initial : ℕ, colored_paper_problem initial 11 2 27 63 ∧ initial = 58 := by
  sorry

end solve_colored_paper_problem_l2626_262666


namespace min_distance_to_line_l2626_262658

/-- The minimum distance from integral points to the line y = (5/3)x + 4/5 -/
theorem min_distance_to_line : 
  ∃ (d : ℝ), d = Real.sqrt 34 / 85 ∧ 
  ∀ (x y : ℤ), 
    d ≤ (|(5 : ℝ) / 3 * x + 4 / 5 - y| / Real.sqrt (1 + (5 / 3)^2)) ∧
    ∃ (x₀ y₀ : ℤ), (|(5 : ℝ) / 3 * x₀ + 4 / 5 - y₀| / Real.sqrt (1 + (5 / 3)^2)) = d := by
  sorry


end min_distance_to_line_l2626_262658


namespace excess_weight_is_51_8_l2626_262688

/-- The weight of the bridge in kilograms -/
def bridge_weight : ℝ := 130

/-- Kelly's weight in kilograms -/
def kelly_weight : ℝ := 34

/-- Sam's weight in kilograms -/
def sam_weight : ℝ := 40

/-- Daisy's weight in kilograms -/
def daisy_weight : ℝ := 28

/-- Megan's weight in kilograms -/
def megan_weight : ℝ := kelly_weight * 1.1

/-- Mike's weight in kilograms -/
def mike_weight : ℝ := megan_weight + 5

/-- The total weight of all five children -/
def total_weight : ℝ := kelly_weight + sam_weight + daisy_weight + megan_weight + mike_weight

/-- Theorem stating that the excess weight is 51.8 kg -/
theorem excess_weight_is_51_8 : total_weight - bridge_weight = 51.8 := by
  sorry

end excess_weight_is_51_8_l2626_262688


namespace chairs_count_l2626_262632

/-- The number of chairs bought for the entire house -/
def total_chairs (living_room kitchen dining_room outdoor_patio : ℕ) : ℕ :=
  living_room + kitchen + dining_room + outdoor_patio

/-- Theorem stating that the total number of chairs is 29 -/
theorem chairs_count :
  total_chairs 3 6 8 12 = 29 := by
  sorry

end chairs_count_l2626_262632


namespace gcd_720_90_minus_10_l2626_262659

theorem gcd_720_90_minus_10 : Nat.gcd 720 90 - 10 = 80 := by
  sorry

end gcd_720_90_minus_10_l2626_262659


namespace inscribed_cylinder_radius_l2626_262628

/-- A right circular cylinder inscribed in a right circular cone -/
structure InscribedCylinder where
  cone_diameter : ℝ
  cone_altitude : ℝ
  cylinder_radius : ℝ
  cylinder_height : ℝ
  h_cylinder_diameter_height : cylinder_height = 2 * cylinder_radius
  h_axes_coincide : True

/-- The theorem stating the radius of the inscribed cylinder -/
theorem inscribed_cylinder_radius 
  (c : InscribedCylinder) 
  (h_cone_diameter : c.cone_diameter = 16)
  (h_cone_altitude : c.cone_altitude = 20) :
  c.cylinder_radius = 40 / 9 := by
  sorry

#check inscribed_cylinder_radius

end inscribed_cylinder_radius_l2626_262628


namespace james_glasses_cost_l2626_262646

/-- The total cost of James' new pair of glasses -/
def total_cost (frame_cost lens_cost insurance_coverage frame_discount : ℚ) : ℚ :=
  (frame_cost - frame_discount) + (lens_cost * (1 - insurance_coverage))

/-- Theorem stating the total cost for James' new pair of glasses -/
theorem james_glasses_cost :
  let frame_cost : ℚ := 200
  let lens_cost : ℚ := 500
  let insurance_coverage : ℚ := 0.8
  let frame_discount : ℚ := 50
  total_cost frame_cost lens_cost insurance_coverage frame_discount = 250 := by
sorry

end james_glasses_cost_l2626_262646


namespace unique_non_negative_twelve_quotient_l2626_262639

def pairs : List (Int × Int) := [(24, -2), (-36, 3), (144, 12), (-48, 4), (72, -6)]

theorem unique_non_negative_twelve_quotient :
  ∃! p : Int × Int, p ∈ pairs ∧ p.1 / p.2 ≠ -12 :=
by sorry

end unique_non_negative_twelve_quotient_l2626_262639


namespace solution_set_quadratic_inequality_l2626_262676

theorem solution_set_quadratic_inequality :
  {x : ℝ | 6 * x^2 + 5 * x < 4} = {x : ℝ | -4/3 < x ∧ x < 1/2} := by sorry

end solution_set_quadratic_inequality_l2626_262676


namespace independence_of_phi_l2626_262607

theorem independence_of_phi (α φ : ℝ) : 
  4 * Real.cos α * Real.cos φ * Real.cos (α - φ) + 2 * Real.sin (α - φ)^2 - Real.cos (2 * φ) = Real.cos (2 * α) + 2 := by
  sorry

end independence_of_phi_l2626_262607


namespace image_of_negative_two_three_preimage_of_two_negative_three_l2626_262621

-- Define the mapping f
def f (x y : ℝ) : ℝ × ℝ := (x + y, x * y)

-- Theorem for the image of (-2, 3)
theorem image_of_negative_two_three :
  f (-2) 3 = (1, -6) := by sorry

-- Theorem for the preimage of (2, -3)
theorem preimage_of_two_negative_three :
  {p : ℝ × ℝ | f p.1 p.2 = (2, -3)} = {(-1, 3), (3, -1)} := by sorry

end image_of_negative_two_three_preimage_of_two_negative_three_l2626_262621


namespace product_of_roots_l2626_262620

theorem product_of_roots (a b c : ℝ) (h : (5 + 3 * Real.sqrt 5) * a + (3 + Real.sqrt 5) * b + c = 0) :
  a * b = -((15 - 9 * Real.sqrt 5) / 20) := by
  sorry

end product_of_roots_l2626_262620


namespace devin_taught_calculus_four_years_l2626_262625

/-- Represents the number of years Devin taught each subject --/
structure TeachingYears where
  calculus : ℕ
  algebra : ℕ
  statistics : ℕ

/-- Defines the conditions of Devin's teaching career --/
def satisfiesConditions (y : TeachingYears) : Prop :=
  y.algebra = 2 * y.calculus ∧
  y.statistics = 5 * y.algebra ∧
  y.calculus + y.algebra + y.statistics = 52

/-- Theorem stating that given the conditions, Devin taught Calculus for 4 years --/
theorem devin_taught_calculus_four_years :
  ∃ y : TeachingYears, satisfiesConditions y ∧ y.calculus = 4 :=
by
  sorry

end devin_taught_calculus_four_years_l2626_262625


namespace vector_parallelism_l2626_262624

/-- Given two 2D vectors a and b, prove that when k*a + b is parallel to a - 3*b, k = -1/3 --/
theorem vector_parallelism (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 2))
    (h2 : b = (-3, 2))
    (h3 : ∃ (t : ℝ), t ≠ 0 ∧ k • a + b = t • (a - 3 • b)) :
  k = -1/3 := by
  sorry

end vector_parallelism_l2626_262624


namespace survey_result_l2626_262689

theorem survey_result (total : ℕ) (radio_dislike_ratio : ℚ) (music_dislike_ratio : ℚ)
  (h_total : total = 2000)
  (h_radio : radio_dislike_ratio = 1/4)
  (h_music : music_dislike_ratio = 3/20) :
  (total : ℚ) * radio_dislike_ratio * music_dislike_ratio = 75 := by
  sorry

end survey_result_l2626_262689


namespace integral_always_positive_l2626_262600

-- Define a continuous function f that is always positive
variable (f : ℝ → ℝ)
variable (hf : Continuous f)
variable (hfpos : ∀ x, f x > 0)

-- Define the integral bounds
variable (a b : ℝ)
variable (hab : a < b)

-- Theorem statement
theorem integral_always_positive :
  ∫ x in a..b, f x > 0 := by sorry

end integral_always_positive_l2626_262600


namespace parabola_sum_l2626_262633

/-- A parabola with equation y = ax^2 + bx + c, vertex at (-3, 2), and passing through (-1, -2) -/
def Parabola (a b c : ℝ) : Prop :=
  (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ (x + 3)^2 = (y - 2) / a) ∧
  a * (-1)^2 + b * (-1) + c = -2

theorem parabola_sum (a b c : ℝ) (h : Parabola a b c) : a + b + c = -14 := by
  sorry

end parabola_sum_l2626_262633


namespace all_solutions_are_powers_l2626_262671

-- Define the equation (1) as a predicate
def is_solution (p q : ℤ) : Prop := sorry

-- Define the main theorem
theorem all_solutions_are_powers (p q : ℤ) :
  p ≥ 0 ∧ q ≥ 0 ∧ is_solution p q ↔ ∃ n : ℕ, p + q * Real.sqrt 5 = (9 + 4 * Real.sqrt 5) ^ n := by
  sorry

end all_solutions_are_powers_l2626_262671


namespace correct_number_l2626_262653

theorem correct_number (x : ℤ) (h1 : x - 152 = 346) : x + 152 = 650 := by
  sorry

end correct_number_l2626_262653


namespace fraction_simplification_l2626_262642

theorem fraction_simplification (x : ℝ) (h : x ≠ 4) :
  (x^2 - 4*x) / (x^2 - 8*x + 16) = x / (x - 4) := by
  sorry

end fraction_simplification_l2626_262642


namespace total_vegetarian_consumers_is_33_l2626_262640

/-- Represents the dietary information of a family -/
structure DietaryInfo where
  only_vegetarian : ℕ
  only_non_vegetarian : ℕ
  both : ℕ
  gluten_free : ℕ
  vegan : ℕ
  non_veg_gluten_free : ℕ
  veg_gluten_free : ℕ
  both_gluten_free : ℕ
  vegan_strict_veg : ℕ
  vegan_non_veg : ℕ

/-- Calculates the total number of people consuming vegetarian dishes -/
def total_vegetarian_consumers (info : DietaryInfo) : ℕ :=
  info.only_vegetarian + info.both + info.vegan_non_veg

/-- The main theorem stating that the total number of vegetarian consumers is 33 -/
theorem total_vegetarian_consumers_is_33 (info : DietaryInfo) 
  (h1 : info.only_vegetarian = 19)
  (h2 : info.only_non_vegetarian = 9)
  (h3 : info.both = 12)
  (h4 : info.gluten_free = 6)
  (h5 : info.vegan = 5)
  (h6 : info.non_veg_gluten_free = 2)
  (h7 : info.veg_gluten_free = 3)
  (h8 : info.both_gluten_free = 1)
  (h9 : info.vegan_strict_veg = 3)
  (h10 : info.vegan_non_veg = 2) :
  total_vegetarian_consumers info = 33 := by
  sorry

end total_vegetarian_consumers_is_33_l2626_262640


namespace sum_of_roots_l2626_262670

theorem sum_of_roots (k p : ℝ) (x₁ x₂ : ℝ) :
  (4 * x₁^2 - k * x₁ - p = 0) →
  (4 * x₂^2 - k * x₂ - p = 0) →
  (x₁ ≠ x₂) →
  (x₁ + x₂ = k / 4) := by
sorry

end sum_of_roots_l2626_262670


namespace min_value_squared_sum_l2626_262644

theorem min_value_squared_sum (x y a : ℝ) : 
  x + y ≥ a →
  x - y ≤ a →
  y ≤ a →
  a > 0 →
  (∀ x' y' : ℝ, x' + y' ≥ a → x' - y' ≤ a → y' ≤ a → x'^2 + y'^2 ≥ 2) →
  (∃ x'' y'' : ℝ, x'' + y'' ≥ a ∧ x'' - y'' ≤ a ∧ y'' ≤ a ∧ x''^2 + y''^2 = 2) →
  a = 2 := by
sorry

end min_value_squared_sum_l2626_262644


namespace x_in_terms_of_abc_and_k_l2626_262690

theorem x_in_terms_of_abc_and_k 
  (k a b c x y z : ℝ) 
  (hk : k ≠ 0)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (h1 : x * y / (k * (x + y)) = a)
  (h2 : x * z / (k * (x + z)) = b)
  (h3 : y * z / (k * (y + z)) = c) :
  x = 2 * a * b * c / (k * (a * c + b * c - a * b)) :=
sorry

end x_in_terms_of_abc_and_k_l2626_262690


namespace integer_roots_of_polynomial_l2626_262679

def polynomial (x : ℤ) : ℤ := x^3 + 2*x^2 - 5*x + 30

def is_root (x : ℤ) : Prop := polynomial x = 0

def divisors_of_30 : Set ℤ := {x : ℤ | x ∣ 30 ∨ x ∣ -30}

theorem integer_roots_of_polynomial :
  {x : ℤ | is_root x} = divisors_of_30 :=
sorry

end integer_roots_of_polynomial_l2626_262679


namespace wood_bundles_problem_l2626_262654

/-- The number of wood bundles at the start of the day, given the number of bundles
    burned in the morning and afternoon, and the number left at the end of the day. -/
def initial_bundles (morning_burned : ℕ) (afternoon_burned : ℕ) (end_day_left : ℕ) : ℕ :=
  morning_burned + afternoon_burned + end_day_left

/-- Theorem stating that the initial number of wood bundles is 10, given the
    conditions from the problem. -/
theorem wood_bundles_problem :
  initial_bundles 4 3 3 = 10 := by
  sorry

end wood_bundles_problem_l2626_262654


namespace number_problem_l2626_262693

theorem number_problem : ∃ x : ℚ, (x / 6) * 12 = 8 ∧ x = 4 := by
  sorry

end number_problem_l2626_262693


namespace yellow_marbles_count_l2626_262623

/-- Represents the number of marbles of each color in Jamal's bag -/
structure MarbleCounts where
  blue : ℕ
  green : ℕ
  black : ℕ
  yellow : ℕ

/-- The probability of drawing a black marble from the bag -/
def blackMarbleProbability : ℚ := 1 / 28

/-- The total number of marbles in the bag -/
def totalMarbles (counts : MarbleCounts) : ℕ :=
  counts.blue + counts.green + counts.black + counts.yellow

/-- The theorem stating the number of yellow marbles in Jamal's bag -/
theorem yellow_marbles_count (counts : MarbleCounts) 
  (h_blue : counts.blue = 10)
  (h_green : counts.green = 5)
  (h_black : counts.black = 1)
  (h_prob : (counts.black : ℚ) / (totalMarbles counts) = blackMarbleProbability) :
  counts.yellow = 12 := by
  sorry

end yellow_marbles_count_l2626_262623


namespace bottles_per_case_is_13_l2626_262612

/-- The number of bottles of water a company produces per day -/
def daily_production : ℕ := 65000

/-- The number of cases required to hold the daily production -/
def cases_required : ℕ := 5000

/-- The number of bottles that a single case can hold -/
def bottles_per_case : ℕ := daily_production / cases_required

theorem bottles_per_case_is_13 : bottles_per_case = 13 := by
  sorry

end bottles_per_case_is_13_l2626_262612


namespace interest_rate_calculation_l2626_262656

/-- Given a principal amount and an interest rate, if the simple interest for 2 years
    is $600 and the compound interest for 2 years is $612, then the interest rate is 104%. -/
theorem interest_rate_calculation (P R : ℝ) : 
  P * R * 2 / 100 = 600 →
  P * ((1 + R / 100)^2 - 1) = 612 →
  R = 104 := by
sorry

end interest_rate_calculation_l2626_262656


namespace melissa_driving_hours_l2626_262649

/-- The number of hours Melissa spends driving in a year -/
def driving_hours_per_year (trips_per_month : ℕ) (hours_per_trip : ℕ) : ℕ :=
  trips_per_month * 12 * hours_per_trip

/-- Proof that Melissa spends 72 hours driving in a year -/
theorem melissa_driving_hours :
  driving_hours_per_year 2 3 = 72 := by
  sorry

end melissa_driving_hours_l2626_262649


namespace first_day_exceeding_500_l2626_262635

def bacteria_count (initial_count : ℕ) (growth_factor : ℕ) (days : ℕ) : ℕ :=
  initial_count * growth_factor ^ days

theorem first_day_exceeding_500 :
  let initial_count := 4
  let growth_factor := 3
  let target := 500
  (∀ d : ℕ, d < 6 → bacteria_count initial_count growth_factor d ≤ target) ∧
  (bacteria_count initial_count growth_factor 6 > target) :=
by sorry

end first_day_exceeding_500_l2626_262635


namespace product_of_four_sqrt_expressions_l2626_262691

theorem product_of_four_sqrt_expressions : 
  let a := Real.sqrt (2 - Real.sqrt 3)
  let b := Real.sqrt (2 - Real.sqrt (2 - Real.sqrt 3))
  let c := Real.sqrt (2 - Real.sqrt (2 - Real.sqrt (2 - Real.sqrt 3)))
  let d := Real.sqrt (2 + Real.sqrt (2 - Real.sqrt (2 - Real.sqrt 3)))
  a * b * c * d = 1 := by sorry

end product_of_four_sqrt_expressions_l2626_262691


namespace lucas_L10_units_digit_l2626_262602

/-- Lucas numbers sequence -/
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

/-- Units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

theorem lucas_L10_units_digit :
  unitsDigit (lucas (lucas 10)) = 4 := by
  sorry

end lucas_L10_units_digit_l2626_262602


namespace min_x_coeff_for_restricted_poly_with_specific_value_l2626_262634

/-- A polynomial with coefficients from the set {0,1,2,3,4,5} -/
def RestrictedPolynomial (P : Polynomial ℤ) : Prop :=
  ∀ i, (P.coeff i) ∈ ({0, 1, 2, 3, 4, 5} : Set ℤ)

/-- The theorem stating that if P(6) = 2013 for a restricted polynomial P,
    then the coefficient of x in P is at least 5 -/
theorem min_x_coeff_for_restricted_poly_with_specific_value
  (P : Polynomial ℤ) (h : RestrictedPolynomial P) (h2 : P.eval 6 = 2013) :
  P.coeff 1 ≥ 5 := by
  sorry

end min_x_coeff_for_restricted_poly_with_specific_value_l2626_262634


namespace bus_route_distance_bounds_l2626_262673

/-- Represents a bus route with n stops -/
structure BusRoute (n : ℕ) where
  distance_between_stops : ℝ
  (distance_positive : distance_between_stops > 0)

/-- Represents a vehicle's journey through all stops -/
def Journey (n : ℕ) := Fin n → Fin n

/-- Calculates the distance traveled in a journey -/
def distance_traveled (r : BusRoute n) (j : Journey n) : ℝ :=
  sorry

/-- Theorem stating the maximum and minimum distances for a 10-stop route -/
theorem bus_route_distance_bounds :
  ∀ (r : BusRoute 10),
    (∃ (j : Journey 10), distance_traveled r j = 50 * r.distance_between_stops) ∧
    (∃ (j : Journey 10), distance_traveled r j = 18 * r.distance_between_stops) ∧
    (∀ (j : Journey 10), 18 * r.distance_between_stops ≤ distance_traveled r j ∧ 
                         distance_traveled r j ≤ 50 * r.distance_between_stops) :=
sorry

end bus_route_distance_bounds_l2626_262673


namespace modulus_of_complex_fraction_l2626_262685

theorem modulus_of_complex_fraction (z : ℂ) : z = (1 + 2*I) / I → Complex.abs z = Real.sqrt 5 := by
  sorry

end modulus_of_complex_fraction_l2626_262685
