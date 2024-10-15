import Mathlib

namespace NUMINAMATH_CALUDE_solve_movie_problem_l2113_211366

def movie_problem (rented_movie_cost bought_movie_cost total_spent : ℚ) : Prop :=
  let num_tickets : ℕ := 2
  let other_costs : ℚ := rented_movie_cost + bought_movie_cost
  let ticket_total_cost : ℚ := total_spent - other_costs
  let ticket_cost : ℚ := ticket_total_cost / num_tickets
  ticket_cost = 10.62

theorem solve_movie_problem :
  movie_problem 1.59 13.95 36.78 := by
  sorry

end NUMINAMATH_CALUDE_solve_movie_problem_l2113_211366


namespace NUMINAMATH_CALUDE_max_consecutive_sum_less_than_1000_l2113_211304

theorem max_consecutive_sum_less_than_1000 :
  ∀ n : ℕ, n > 0 → (n * (n + 1) / 2 < 1000 ↔ n ≤ 44) :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_less_than_1000_l2113_211304


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2113_211329

def complex_number : ℂ := Complex.I * ((-2 : ℝ) + Complex.I)

theorem complex_number_in_third_quadrant :
  let z := complex_number
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2113_211329


namespace NUMINAMATH_CALUDE_assign_roles_specific_scenario_l2113_211399

/-- Represents the number of ways to assign roles in a play. -/
def assignRoles (maleRoles femaleRoles eitherRoles maleActors femaleActors : ℕ) : ℕ :=
  (maleActors.descFactorial maleRoles) *
  (femaleActors.descFactorial femaleRoles) *
  ((maleActors + femaleActors - maleRoles - femaleRoles).descFactorial eitherRoles)

/-- Theorem stating the number of ways to assign roles in the specific scenario. -/
theorem assign_roles_specific_scenario :
  assignRoles 2 2 3 4 5 = 14400 := by
  sorry

end NUMINAMATH_CALUDE_assign_roles_specific_scenario_l2113_211399


namespace NUMINAMATH_CALUDE_contribution_problem_l2113_211310

/-- The contribution problem -/
theorem contribution_problem (total_sum : ℕ) : 
  (10 : ℕ) * 300 = total_sum ∧ 
  (15 : ℕ) * (300 - 100) = total_sum := by
  sorry

#check contribution_problem

end NUMINAMATH_CALUDE_contribution_problem_l2113_211310


namespace NUMINAMATH_CALUDE_marys_final_book_count_marys_library_end_year_l2113_211370

/-- Calculates the final number of books in Mary's mystery book library after a year of changes. -/
theorem marys_final_book_count (initial : ℕ) (book_club : ℕ) (bookstore : ℕ) (yard_sales : ℕ) 
  (daughter : ℕ) (mother : ℕ) (donated : ℕ) (sold : ℕ) : ℕ :=
  initial + book_club + bookstore + yard_sales + daughter + mother - donated - sold

/-- Proves that Mary has 81 books at the end of the year given the specific conditions. -/
theorem marys_library_end_year : 
  marys_final_book_count 72 12 5 2 1 4 12 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_marys_final_book_count_marys_library_end_year_l2113_211370


namespace NUMINAMATH_CALUDE_students_in_all_three_sections_l2113_211309

/-- Represents the number of students in each section and their intersections -/
structure ClubSections where
  totalStudents : ℕ
  music : ℕ
  drama : ℕ
  dance : ℕ
  atLeastTwo : ℕ
  allThree : ℕ

/-- The theorem stating the number of students in all three sections -/
theorem students_in_all_three_sections 
  (club : ClubSections) 
  (h1 : club.totalStudents = 30)
  (h2 : club.music = 15)
  (h3 : club.drama = 18)
  (h4 : club.dance = 12)
  (h5 : club.atLeastTwo = 14)
  (h6 : ∀ s : ℕ, s ≤ club.totalStudents → s ≥ club.music ∨ s ≥ club.drama ∨ s ≥ club.dance) :
  club.allThree = 6 := by
  sorry


end NUMINAMATH_CALUDE_students_in_all_three_sections_l2113_211309


namespace NUMINAMATH_CALUDE_keith_pears_given_away_l2113_211385

/-- The number of pears Keith gave away -/
def pears_given_away (keith_initial : ℕ) (mike_initial : ℕ) (remaining : ℕ) : ℕ :=
  keith_initial + mike_initial - remaining

theorem keith_pears_given_away :
  pears_given_away 47 12 13 = 46 := by
  sorry

end NUMINAMATH_CALUDE_keith_pears_given_away_l2113_211385


namespace NUMINAMATH_CALUDE_farmer_tomatoes_l2113_211338

theorem farmer_tomatoes (initial_tomatoes : ℕ) (picked_tomatoes : ℕ) 
  (h1 : initial_tomatoes = 17)
  (h2 : initial_tomatoes - picked_tomatoes = 8) :
  picked_tomatoes = 9 := by
  sorry

end NUMINAMATH_CALUDE_farmer_tomatoes_l2113_211338


namespace NUMINAMATH_CALUDE_x_seventh_x_n_plus_one_l2113_211358

variable (x : ℝ)

-- Define the conditions
axiom x_is_root : x^2 - x - 1 = 0
axiom x_squared : x^2 = x + 1
axiom x_cubed : x^3 = 2*x + 1
axiom x_fourth : x^4 = 3*x + 2
axiom x_fifth : x^5 = 5*x + 3
axiom x_sixth : x^6 = 8*x + 5

-- Define x^n = αx + β
variable (n : ℕ) (α β : ℝ)
axiom x_nth : x^n = α*x + β

-- Theorem statements
theorem x_seventh : x^7 = 13*x + 8 := by sorry

theorem x_n_plus_one : x^(n+1) = (α + β)*x + α := by sorry

end NUMINAMATH_CALUDE_x_seventh_x_n_plus_one_l2113_211358


namespace NUMINAMATH_CALUDE_binary_product_in_base4_l2113_211301

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its base 4 representation -/
def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The first binary number: 1101₂ -/
def binary1 : List Bool := [true, true, false, true]

/-- The second binary number: 111₂ -/
def binary2 : List Bool := [true, true, true]

/-- Statement: The product of 1101₂ and 111₂ in base 4 is 311₄ -/
theorem binary_product_in_base4 :
  decimal_to_base4 (binary_to_decimal binary1 * binary_to_decimal binary2) = [3, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_binary_product_in_base4_l2113_211301


namespace NUMINAMATH_CALUDE_evaluate_expression_l2113_211352

theorem evaluate_expression (x y z w : ℚ) 
  (hx : x = 1/4) 
  (hy : y = 1/3) 
  (hz : z = 12) 
  (hw : w = -2) : 
  x^2 * y^3 * z * w = -1/18 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2113_211352


namespace NUMINAMATH_CALUDE_ellipse_equation_and_product_constant_l2113_211374

noncomputable section

-- Define the ellipse
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define eccentricity
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

-- Define the x-intercept of line QM
def x_intercept_QM (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (x₂ * y₁ - x₁ * y₂) / (y₁ + y₂)

-- Define the y-intercept of line QN
def y_intercept_QN (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (x₁ * y₂ + x₂ * y₁) / (x₁ - x₂)

-- Define the slope of line OR
def slope_OR (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (y₁ + y₂) / (x₁ + x₂)

theorem ellipse_equation_and_product_constant (a b : ℝ) 
  (h₁ : a > b) (h₂ : b > 0) (h₃ : eccentricity a b = 1/2) :
  (∃ (x y : ℝ), Ellipse 2 (Real.sqrt 3) (x, y)) ∧ 
  (∀ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ → y₁ ≠ y₂ → 
    Ellipse a b (x₁, y₁) → Ellipse a b (x₂, y₂) →
    (x_intercept_QM x₁ y₁ x₂ y₂) * (y_intercept_QN x₁ y₁ x₂ y₂) * (slope_OR x₁ y₁ x₂ y₂) = 0) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_and_product_constant_l2113_211374


namespace NUMINAMATH_CALUDE_prime_fraction_equality_l2113_211384

theorem prime_fraction_equality (A B : ℕ) : 
  Nat.Prime A → 
  Nat.Prime B → 
  A > 0 → 
  B > 0 → 
  (1 : ℚ) / A - (1 : ℚ) / B = 192 / (2005^2 - 2004^2) → 
  B = 211 := by
sorry

end NUMINAMATH_CALUDE_prime_fraction_equality_l2113_211384


namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_eq_neg_one_l2113_211315

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b = l₂.a * l₁.b

theorem parallel_lines_imply_a_eq_neg_one (a : ℝ) :
  let l₁ : Line := ⟨a, 2, 6⟩
  let l₂ : Line := ⟨1, a - 1, a^2 - 1⟩
  parallel l₁ l₂ → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_imply_a_eq_neg_one_l2113_211315


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_original_statement_l2113_211314

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x > 0, P x) ↔ (∃ x > 0, ¬ P x) :=
by sorry

theorem negation_of_original_statement :
  (¬ ∀ x > 0, x^2 - 3*x + 2 < 0) ↔ (∃ x > 0, x^2 - 3*x + 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_original_statement_l2113_211314


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2113_211323

theorem fixed_point_on_line (a : ℝ) : 
  let line := fun (x y : ℝ) => a * x + y + a + 1 = 0
  line (-1) (-1) := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2113_211323


namespace NUMINAMATH_CALUDE_expression_factorization_l2113_211307

theorem expression_factorization (a : ℝ) : 
  (8 * a^3 + 105 * a^2 + 7) - (-9 * a^3 + 16 * a^2 - 14) = a^2 * (17 * a + 89) + 21 := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2113_211307


namespace NUMINAMATH_CALUDE_mrs_hilt_apple_consumption_l2113_211341

-- Define the rate of apple consumption
def apples_per_hour : ℕ := 10

-- Define the number of hours
def total_hours : ℕ := 6

-- Theorem to prove
theorem mrs_hilt_apple_consumption :
  apples_per_hour * total_hours = 60 :=
by sorry

end NUMINAMATH_CALUDE_mrs_hilt_apple_consumption_l2113_211341


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2113_211317

/-- A geometric sequence with specified properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 1 + a 2 = 1 →
  a 2 + a 3 = 2 →
  a 6 + a 7 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2113_211317


namespace NUMINAMATH_CALUDE_books_sum_is_95_l2113_211303

/-- The total number of books Tim, Mike, Sarah, and Emily have together -/
def total_books (tim_books mike_books sarah_books emily_books : ℕ) : ℕ :=
  tim_books + mike_books + sarah_books + emily_books

/-- Theorem stating that the total number of books is 95 -/
theorem books_sum_is_95 :
  total_books 22 20 35 18 = 95 := by
  sorry

end NUMINAMATH_CALUDE_books_sum_is_95_l2113_211303


namespace NUMINAMATH_CALUDE_complex_expression_equality_l2113_211392

theorem complex_expression_equality : 
  (2 + 1/4)^(1/2) - (-9.6)^0 - (3 + 3/8)^(2/3) + (1.5)^2 + (Real.sqrt 2 * 43)^4 = 5/4 + 4 * 43^4 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l2113_211392


namespace NUMINAMATH_CALUDE_circle_path_in_right_triangle_l2113_211339

theorem circle_path_in_right_triangle (a b c : ℝ) (r : ℝ) :
  a = 5 →
  b = 12 →
  c = 13 →
  r = 2 →
  a^2 + b^2 = c^2 →
  let path_length := (a - 2*r) + (b - 2*r) + (c - 2*r)
  path_length = 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_path_in_right_triangle_l2113_211339


namespace NUMINAMATH_CALUDE_f_properties_l2113_211325

noncomputable def f (x : ℝ) : ℝ := (1 - Real.sin (2 * x)) / (Real.sin x - Real.cos x)

theorem f_properties :
  ∃ (T : ℝ) (max_value : ℝ) (max_set : Set ℝ),
    (∀ x, f (x + T) = f x) ∧  -- f has period T
    T = 2 * Real.pi ∧  -- The period is 2π
    (∀ x, f x ≤ max_value) ∧  -- max_value is an upper bound
    max_value = Real.sqrt 2 ∧  -- The maximum value is √2
    (∀ x, x ∈ max_set ↔ f x = max_value) ∧  -- max_set contains all x where f(x) is maximum
    (∀ k : ℤ, (2 * k : ℝ) * Real.pi + 3 * Real.pi / 4 ∈ max_set)  -- Characterization of max_set
    := by sorry

end NUMINAMATH_CALUDE_f_properties_l2113_211325


namespace NUMINAMATH_CALUDE_not_divides_power_minus_one_l2113_211390

theorem not_divides_power_minus_one (n : ℕ) (h : n > 1) :
  ¬(n ∣ (2^n - 1)) := by
sorry

end NUMINAMATH_CALUDE_not_divides_power_minus_one_l2113_211390


namespace NUMINAMATH_CALUDE_charity_run_donation_l2113_211322

theorem charity_run_donation (total_donation : ℕ) (race_length : ℕ) : 
  race_length = 5 ∧ 
  total_donation = 310 ∧ 
  (∃ initial_donation : ℕ, 
    total_donation = initial_donation * (2^race_length - 1)) →
  ∃ initial_donation : ℕ, initial_donation = 10 ∧
    total_donation = initial_donation * (2^race_length - 1) :=
by sorry

end NUMINAMATH_CALUDE_charity_run_donation_l2113_211322


namespace NUMINAMATH_CALUDE_sequence_property_l2113_211327

def sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (List.range n).map a |> List.sum

theorem sequence_property (a : ℕ → ℕ) :
  (∀ n : ℕ, n > 0 → sequence_sum a n = 2 * a n - 4) →
  (∀ n : ℕ, n > 0 → a n = 2^(n+1)) :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l2113_211327


namespace NUMINAMATH_CALUDE_diff_of_squares_equals_fifth_power_l2113_211393

theorem diff_of_squares_equals_fifth_power (a : ℤ) :
  ∃ x y : ℤ, x^2 - y^2 = a^5 := by
sorry

end NUMINAMATH_CALUDE_diff_of_squares_equals_fifth_power_l2113_211393


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l2113_211397

theorem smallest_number_divisibility (x : ℕ) : x = 1621432330 ↔ 
  (∀ y : ℕ, y < x → ¬(29 ∣ 5*(y+11) ∧ 53 ∣ 5*(y+11) ∧ 37 ∣ 5*(y+11) ∧ 
                     41 ∣ 5*(y+11) ∧ 47 ∣ 5*(y+11) ∧ 61 ∣ 5*(y+11))) ∧
  (29 ∣ 5*(x+11) ∧ 53 ∣ 5*(x+11) ∧ 37 ∣ 5*(x+11) ∧ 
   41 ∣ 5*(x+11) ∧ 47 ∣ 5*(x+11) ∧ 61 ∣ 5*(x+11)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l2113_211397


namespace NUMINAMATH_CALUDE_sqrt_fraction_equality_l2113_211377

theorem sqrt_fraction_equality (x : ℝ) : 
  (1 < x ∧ x ≤ 3) ↔ Real.sqrt ((3 - x) / (x - 1)) = Real.sqrt (3 - x) / Real.sqrt (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equality_l2113_211377


namespace NUMINAMATH_CALUDE_paper_width_problem_l2113_211306

theorem paper_width_problem (sheet1_length sheet1_width sheet2_length : ℝ)
  (h1 : sheet1_length = 11)
  (h2 : sheet1_width = 13)
  (h3 : sheet2_length = 11)
  (h4 : 2 * sheet1_length * sheet1_width = 2 * sheet2_length * sheet2_width + 100) :
  sheet2_width = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_paper_width_problem_l2113_211306


namespace NUMINAMATH_CALUDE_floor_tile_count_l2113_211319

/-- Represents a rectangular floor covered with square tiles. -/
structure TiledFloor where
  width : ℕ
  length : ℕ
  diagonalTiles : ℕ

/-- The conditions of our specific tiled floor. -/
def specialFloor : TiledFloor where
  width := 19
  length := 38
  diagonalTiles := 39

theorem floor_tile_count (floor : TiledFloor) 
  (h1 : floor.length = 2 * floor.width)
  (h2 : floor.diagonalTiles = 39) : 
  floor.width * floor.length = 722 := by
  sorry

#check floor_tile_count specialFloor

end NUMINAMATH_CALUDE_floor_tile_count_l2113_211319


namespace NUMINAMATH_CALUDE_base9_multiplication_addition_l2113_211328

/-- Converts a base-9 number represented as a list of digits to a natural number. -/
def base9ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 9 + d) 0

/-- Converts a natural number to its base-9 representation as a list of digits. -/
def natToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
    aux n []

/-- The main theorem to be proved -/
theorem base9_multiplication_addition :
  (base9ToNat [3, 2, 4] * base9ToNat [4, 6, 7]) + base9ToNat [1, 2, 3] =
  base9ToNat [2, 3, 4, 4, 2] := by
  sorry

#eval natToBase9 ((base9ToNat [3, 2, 4] * base9ToNat [4, 6, 7]) + base9ToNat [1, 2, 3])

end NUMINAMATH_CALUDE_base9_multiplication_addition_l2113_211328


namespace NUMINAMATH_CALUDE_tom_jogging_distance_l2113_211346

/-- The distance Tom jogs in 15 minutes given his rate -/
theorem tom_jogging_distance (rate : ℝ) (time : ℝ) (h1 : rate = 1 / 18) (h2 : time = 15) :
  rate * time = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_tom_jogging_distance_l2113_211346


namespace NUMINAMATH_CALUDE_fish_difference_l2113_211340

-- Define the sizes of the tanks
def first_tank_size : ℕ := 48
def second_tank_size : ℕ := first_tank_size / 2

-- Define the fish sizes
def first_tank_fish_size : ℕ := 3
def second_tank_fish_size : ℕ := 2

-- Calculate the number of fish in each tank
def fish_in_first_tank : ℕ := first_tank_size / first_tank_fish_size
def fish_in_second_tank : ℕ := second_tank_size / second_tank_fish_size

-- Calculate the number of fish in the first tank after one is eaten
def fish_in_first_tank_after_eating : ℕ := fish_in_first_tank - 1

-- Theorem to prove
theorem fish_difference :
  fish_in_first_tank_after_eating - fish_in_second_tank = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_fish_difference_l2113_211340


namespace NUMINAMATH_CALUDE_factor_implies_d_value_l2113_211312

-- Define the polynomial g(x)
def g (d : ℝ) (x : ℝ) : ℝ := d * x^3 + 25 * x^2 - 5 * d * x + 45

-- State the theorem
theorem factor_implies_d_value :
  ∀ d : ℝ, (∀ x : ℝ, (x + 5) ∣ g d x) → d = 6.7 :=
by sorry

end NUMINAMATH_CALUDE_factor_implies_d_value_l2113_211312


namespace NUMINAMATH_CALUDE_burj_khalifa_height_is_830_l2113_211344

/-- The height of the Eiffel Tower in meters -/
def eiffel_tower_height : ℝ := 324

/-- The difference in height between the Burj Khalifa and the Eiffel Tower in meters -/
def height_difference : ℝ := 506

/-- The height of the Burj Khalifa in meters -/
def burj_khalifa_height : ℝ := eiffel_tower_height + height_difference

theorem burj_khalifa_height_is_830 : burj_khalifa_height = 830 := by
  sorry

end NUMINAMATH_CALUDE_burj_khalifa_height_is_830_l2113_211344


namespace NUMINAMATH_CALUDE_equation_solution_l2113_211359

theorem equation_solution : 
  ∃ x : ℚ, (x^2 + 3*x + 4) / (x + 5) = x + 6 ∧ x = -13/4 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2113_211359


namespace NUMINAMATH_CALUDE_bug_path_distance_l2113_211349

theorem bug_path_distance (r : Real) (leg : Real) (h1 : r = 40) (h2 : leg = 50) :
  let diameter := 2 * r
  let other_leg := Real.sqrt (diameter ^ 2 - leg ^ 2)
  diameter + leg + other_leg = 192.45 := by
  sorry

end NUMINAMATH_CALUDE_bug_path_distance_l2113_211349


namespace NUMINAMATH_CALUDE_trapezoid_in_isosceles_triangle_l2113_211335

/-- An isosceles triangle with a trapezoid inscribed within it. -/
structure IsoscelesTriangleWithTrapezoid where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The length of the equal sides of the isosceles triangle -/
  side : ℝ
  /-- The distance from the apex to point D on side AB -/
  x : ℝ
  /-- The perimeter of the inscribed trapezoid -/
  trapezoidPerimeter : ℝ

/-- Theorem stating the condition for the inscribed trapezoid in an isosceles triangle -/
theorem trapezoid_in_isosceles_triangle 
    (t : IsoscelesTriangleWithTrapezoid) 
    (h1 : t.base = 12) 
    (h2 : t.side = 18) 
    (h3 : t.trapezoidPerimeter = 40) : 
    t.x = 6 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_in_isosceles_triangle_l2113_211335


namespace NUMINAMATH_CALUDE_order_of_expressions_l2113_211396

theorem order_of_expressions :
  let a := 2 + (1/5) * Real.log 2
  let b := 1 + 2^(1/5)
  let c := 2^(11/10)
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_order_of_expressions_l2113_211396


namespace NUMINAMATH_CALUDE_room_expansion_proof_l2113_211380

theorem room_expansion_proof (initial_length initial_width increase : ℝ)
  (h1 : initial_length = 13)
  (h2 : initial_width = 18)
  (h3 : increase = 2) :
  let new_length := initial_length + increase
  let new_width := initial_width + increase
  let single_room_area := new_length * new_width
  let total_area := 4 * single_room_area + 2 * single_room_area
  total_area = 1800 := by sorry

end NUMINAMATH_CALUDE_room_expansion_proof_l2113_211380


namespace NUMINAMATH_CALUDE_equation_solution_l2113_211394

theorem equation_solution : ∃ x : ℝ, (3 / (x^2 - 9) + x / (x - 3) = 1) ∧ (x = -4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2113_211394


namespace NUMINAMATH_CALUDE_solution_of_equation_l2113_211351

theorem solution_of_equation (x : ℝ) : x * (x - 1) = 0 ↔ x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_equation_l2113_211351


namespace NUMINAMATH_CALUDE_jeans_card_collection_l2113_211336

theorem jeans_card_collection (num_groups : ℕ) (cards_per_group : ℕ) 
  (h1 : num_groups = 9) (h2 : cards_per_group = 8) :
  num_groups * cards_per_group = 72 := by
  sorry

end NUMINAMATH_CALUDE_jeans_card_collection_l2113_211336


namespace NUMINAMATH_CALUDE_exterior_angle_theorem_l2113_211389

theorem exterior_angle_theorem (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  α + β = 150 →      -- Exterior angle is 150°
  γ = 70 →           -- One remote interior angle is 70°
  β = 80 :=          -- The other remote interior angle is 80°
by sorry

end NUMINAMATH_CALUDE_exterior_angle_theorem_l2113_211389


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l2113_211386

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, y^5 + 2*x*y = x^2 + 2*y^4 ↔ 
    (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = 0 ∧ y = 2) ∨ (x = 4 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l2113_211386


namespace NUMINAMATH_CALUDE_tomatoes_left_l2113_211382

theorem tomatoes_left (total : ℕ) (eaten_fraction : ℚ) (left : ℕ) : 
  total = 21 → 
  eaten_fraction = 1/3 →
  left = total - (total * eaten_fraction).floor →
  left = 14 := by
sorry

end NUMINAMATH_CALUDE_tomatoes_left_l2113_211382


namespace NUMINAMATH_CALUDE_ellipse_equation_l2113_211348

/-- An ellipse passing through (-√15, 5/2) with the same foci as 9x^2 + 4y^2 = 36 -/
theorem ellipse_equation (x y : ℝ) : 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 = b^2 + 5 ∧ 
   x^2 / b^2 + y^2 / a^2 = 1 ∧
   (-Real.sqrt 15)^2 / b^2 + (5/2)^2 / a^2 = 1) →
  x^2 / 20 + y^2 / 25 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2113_211348


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l2113_211334

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l2113_211334


namespace NUMINAMATH_CALUDE_expression_simplification_l2113_211324

theorem expression_simplification : 
  ((3 + 4 + 5 + 6) / 2) + ((3 * 6 + 9) / 3 + 1) = 19 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2113_211324


namespace NUMINAMATH_CALUDE_longest_segment_in_pie_sector_l2113_211391

theorem longest_segment_in_pie_sector (d : ℝ) (h : d = 12) :
  let r := d / 2
  let sector_angle := 2 * Real.pi / 3
  let chord_length := 2 * r * Real.sin (sector_angle / 2)
  chord_length ^ 2 = 108 := by sorry

end NUMINAMATH_CALUDE_longest_segment_in_pie_sector_l2113_211391


namespace NUMINAMATH_CALUDE_dagger_example_l2113_211347

-- Define the ternary operation ⋄
def dagger (a b c d e f : ℚ) : ℚ := (a * c * e) * ((d * f) / b)

-- Theorem statement
theorem dagger_example : dagger 5 9 7 2 11 5 = 3850 / 9 := by
  sorry

end NUMINAMATH_CALUDE_dagger_example_l2113_211347


namespace NUMINAMATH_CALUDE_tens_digit_of_7_pow_35_l2113_211383

-- Define a function to get the tens digit of a natural number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- State the theorem
theorem tens_digit_of_7_pow_35 : tens_digit (7^35) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_7_pow_35_l2113_211383


namespace NUMINAMATH_CALUDE_exponent_rule_multiplication_l2113_211363

theorem exponent_rule_multiplication (a : ℝ) : a^4 * a^6 = a^10 := by sorry

end NUMINAMATH_CALUDE_exponent_rule_multiplication_l2113_211363


namespace NUMINAMATH_CALUDE_sexagenary_cycle_2016_2017_l2113_211305

/-- Represents the Heavenly Stems in the Sexagenary cycle -/
inductive HeavenlyStem
| Jia | Yi | Bing | Ding | Wu | Ji | Geng | Xin | Ren | Gui

/-- Represents the Earthly Branches in the Sexagenary cycle -/
inductive EarthlyBranch
| Zi | Chou | Yin | Mao | Chen | Si | Wu | Wei | Shen | You | Xu | Hai

/-- Represents a year in the Sexagenary cycle -/
structure SexagenaryYear :=
  (stem : HeavenlyStem)
  (branch : EarthlyBranch)

/-- Returns the next Heavenly Stem in the cycle -/
def nextStem (s : HeavenlyStem) : HeavenlyStem :=
  match s with
  | HeavenlyStem.Jia => HeavenlyStem.Yi
  | HeavenlyStem.Yi => HeavenlyStem.Bing
  | HeavenlyStem.Bing => HeavenlyStem.Ding
  | HeavenlyStem.Ding => HeavenlyStem.Wu
  | HeavenlyStem.Wu => HeavenlyStem.Ji
  | HeavenlyStem.Ji => HeavenlyStem.Geng
  | HeavenlyStem.Geng => HeavenlyStem.Xin
  | HeavenlyStem.Xin => HeavenlyStem.Ren
  | HeavenlyStem.Ren => HeavenlyStem.Gui
  | HeavenlyStem.Gui => HeavenlyStem.Jia

/-- Returns the next Earthly Branch in the cycle -/
def nextBranch (b : EarthlyBranch) : EarthlyBranch :=
  match b with
  | EarthlyBranch.Zi => EarthlyBranch.Chou
  | EarthlyBranch.Chou => EarthlyBranch.Yin
  | EarthlyBranch.Yin => EarthlyBranch.Mao
  | EarthlyBranch.Mao => EarthlyBranch.Chen
  | EarthlyBranch.Chen => EarthlyBranch.Si
  | EarthlyBranch.Si => EarthlyBranch.Wu
  | EarthlyBranch.Wu => EarthlyBranch.Wei
  | EarthlyBranch.Wei => EarthlyBranch.Shen
  | EarthlyBranch.Shen => EarthlyBranch.You
  | EarthlyBranch.You => EarthlyBranch.Xu
  | EarthlyBranch.Xu => EarthlyBranch.Hai
  | EarthlyBranch.Hai => EarthlyBranch.Zi

/-- Returns the next year in the Sexagenary cycle -/
def nextYear (y : SexagenaryYear) : SexagenaryYear :=
  { stem := nextStem y.stem, branch := nextBranch y.branch }

theorem sexagenary_cycle_2016_2017 :
  ∀ (y2016 : SexagenaryYear),
    y2016.stem = HeavenlyStem.Bing ∧ y2016.branch = EarthlyBranch.Shen →
    (nextYear y2016).stem = HeavenlyStem.Ding ∧ (nextYear y2016).branch = EarthlyBranch.You :=
by sorry

end NUMINAMATH_CALUDE_sexagenary_cycle_2016_2017_l2113_211305


namespace NUMINAMATH_CALUDE_trapezoid_area_theorem_l2113_211332

/-- A trapezoid with mutually perpendicular diagonals -/
structure Trapezoid :=
  (height : ℝ)
  (diagonal : ℝ)
  (diagonals_perpendicular : Bool)

/-- The area of a trapezoid with given properties -/
def trapezoid_area (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem: The area of a trapezoid with mutually perpendicular diagonals, 
    height 4, and one diagonal of length 5 is equal to 50/3 -/
theorem trapezoid_area_theorem (t : Trapezoid) 
  (h1 : t.height = 4)
  (h2 : t.diagonal = 5)
  (h3 : t.diagonals_perpendicular = true) : 
  trapezoid_area t = 50 / 3 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_area_theorem_l2113_211332


namespace NUMINAMATH_CALUDE_min_gennadys_correct_l2113_211368

/-- Represents the number of people with a specific name at the festival -/
structure NameCount where
  alexanders : Nat
  borises : Nat
  vasilies : Nat
  gennadies : Nat

/-- Checks if the given name counts satisfy the festival conditions -/
def satisfiesConditions (counts : NameCount) : Prop :=
  counts.alexanders = 45 ∧
  counts.borises = 122 ∧
  counts.vasilies = 27 ∧
  counts.alexanders + counts.borises + counts.vasilies + counts.gennadies - 1 ≥ counts.borises

/-- The minimum number of Gennadys required for the festival -/
def minGennadys : Nat := 49

/-- Theorem stating that the minimum number of Gennadys is correct -/
theorem min_gennadys_correct :
  (∀ counts : NameCount, satisfiesConditions counts → counts.gennadies ≥ minGennadys) ∧
  (∃ counts : NameCount, satisfiesConditions counts ∧ counts.gennadies = minGennadys) := by
  sorry

#check min_gennadys_correct

end NUMINAMATH_CALUDE_min_gennadys_correct_l2113_211368


namespace NUMINAMATH_CALUDE_sixth_group_frequency_l2113_211388

/-- Given a sample of 40 data points divided into 6 groups, with the frequencies
    of the first four groups and the fifth group as specified, 
    the frequency of the sixth group is 0.2. -/
theorem sixth_group_frequency 
  (total_points : ℕ) 
  (group_count : ℕ)
  (freq_1 freq_2 freq_3 freq_4 freq_5 : ℚ) :
  total_points = 40 →
  group_count = 6 →
  freq_1 = 10 / 40 →
  freq_2 = 5 / 40 →
  freq_3 = 7 / 40 →
  freq_4 = 6 / 40 →
  freq_5 = 1 / 10 →
  ∃ freq_6 : ℚ, freq_6 = 1 - (freq_1 + freq_2 + freq_3 + freq_4 + freq_5) ∧ freq_6 = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_sixth_group_frequency_l2113_211388


namespace NUMINAMATH_CALUDE_inequality_proof_l2113_211354

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2113_211354


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2113_211369

theorem polynomial_factorization (y : ℝ) : 
  (20 * y^4 + 100 * y - 10) - (5 * y^3 - 15 * y + 10) = 5 * (4 * y^4 - y^3 + 23 * y - 4) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2113_211369


namespace NUMINAMATH_CALUDE_cistern_length_is_ten_l2113_211372

/-- Represents a cistern with given dimensions and water level --/
structure Cistern where
  length : ℝ
  width : ℝ
  waterDepth : ℝ

/-- Calculates the total wet surface area of a cistern --/
def wetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.waterDepth + 2 * c.width * c.waterDepth

/-- Theorem stating that a cistern with given dimensions has a length of 10 meters --/
theorem cistern_length_is_ten :
  ∃ (c : Cistern), c.width = 8 ∧ c.waterDepth = 1.5 ∧ wetSurfaceArea c = 134 → c.length = 10 := by
  sorry

end NUMINAMATH_CALUDE_cistern_length_is_ten_l2113_211372


namespace NUMINAMATH_CALUDE_triangle_angle_b_l2113_211345

/-- In a triangle ABC, if side a = 1, side b = √3, and angle A = 30°, then angle B = 60° -/
theorem triangle_angle_b (a b : ℝ) (A B : ℝ) : 
  a = 1 → b = Real.sqrt 3 → A = π / 6 → B = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_b_l2113_211345


namespace NUMINAMATH_CALUDE_a_always_gets_half_rule_independent_l2113_211343

/-- The game rules for counter division --/
inductive Rule
| R1  -- B takes the biggest and smallest heaps
| R2  -- B takes the two middling heaps
| R3  -- B chooses between R1 and R2

/-- The optimal number of counters A can obtain --/
def optimal_counters (N : ℕ) (r : Rule) : ℕ :=
  N / 2

/-- Theorem: A always gets ⌊N/2⌋ counters regardless of the rule --/
theorem a_always_gets_half (N : ℕ) (h : N ≥ 4) (r : Rule) :
  optimal_counters N r = N / 2 := by
  sorry

/-- Corollary: The result is independent of the chosen rule --/
theorem rule_independent (N : ℕ) (h : N ≥ 4) (r1 r2 : Rule) :
  optimal_counters N r1 = optimal_counters N r2 := by
  sorry

end NUMINAMATH_CALUDE_a_always_gets_half_rule_independent_l2113_211343


namespace NUMINAMATH_CALUDE_milk_volume_is_ten_l2113_211373

/-- The total volume of milk sold by Josephine -/
def total_milk_volume : ℝ :=
  3 * 2 + 2 * 0.75 + 5 * 0.5

/-- Theorem stating that the total volume of milk sold is 10 liters -/
theorem milk_volume_is_ten : total_milk_volume = 10 := by
  sorry

end NUMINAMATH_CALUDE_milk_volume_is_ten_l2113_211373


namespace NUMINAMATH_CALUDE_parabola_tangent_point_l2113_211379

theorem parabola_tangent_point (p q : ℤ) (h : p^2 = 4*q) :
  ∃ (a b : ℤ), (a = -p ∧ b = q) ∧ a^2 = 4*b :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_point_l2113_211379


namespace NUMINAMATH_CALUDE_power_of_i_sum_l2113_211326

theorem power_of_i_sum : ∃ (i : ℂ), i^2 = -1 ∧ i^14760 + i^14761 + i^14762 + i^14763 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_of_i_sum_l2113_211326


namespace NUMINAMATH_CALUDE_pistachios_with_opened_shells_l2113_211321

/-- Given a bag of pistachios, calculate the number of pistachios with shells and opened shells -/
theorem pistachios_with_opened_shells
  (total : ℕ)
  (shell_percent : ℚ)
  (opened_percent : ℚ)
  (h_total : total = 80)
  (h_shell : shell_percent = 95 / 100)
  (h_opened : opened_percent = 75 / 100) :
  ⌊(total : ℚ) * shell_percent * opened_percent⌋ = 57 := by
sorry

end NUMINAMATH_CALUDE_pistachios_with_opened_shells_l2113_211321


namespace NUMINAMATH_CALUDE_equation_solution_l2113_211375

theorem equation_solution :
  let f (x : ℝ) := (3 * x^2) / (x - 2) - (3 * x + 8) / 4 + (5 - 9 * x) / (x - 2) + 2
  ∀ x : ℝ, x ≠ 2 → (f x = 0 ↔ x = (5 + Real.sqrt 5) / 3 ∨ x = (5 - Real.sqrt 5) / 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2113_211375


namespace NUMINAMATH_CALUDE_power_difference_equality_l2113_211342

theorem power_difference_equality : 2^2014 - (-2)^2015 = 3 * 2^2014 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_equality_l2113_211342


namespace NUMINAMATH_CALUDE_compute_expression_l2113_211320

theorem compute_expression : 15 * (1 / 17) * 34 - (1 / 2) = 59 / 2 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2113_211320


namespace NUMINAMATH_CALUDE_simplify_expression_l2113_211313

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : x^2 + y^2 + z^2 = x*y + y*z + z*x) :
  1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2) = 3 / x^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2113_211313


namespace NUMINAMATH_CALUDE_sams_weight_l2113_211357

/-- Given the weights of Tyler, Sam, and Peter, prove Sam's weight -/
theorem sams_weight (tyler sam peter : ℕ) 
  (h1 : tyler = sam + 25)
  (h2 : peter * 2 = tyler)
  (h3 : peter = 65) : 
  sam = 105 := by sorry

end NUMINAMATH_CALUDE_sams_weight_l2113_211357


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2113_211337

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  ArithmeticSequence a → a 5 = 3 → a 6 = -2 →
  (a 3) + (a 4) + (a 5) + (a 6) + (a 7) + (a 8) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2113_211337


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2113_211355

/-- Calculates the speed of trains given their length, crossing time, and direction --/
theorem train_speed_calculation (train_length : ℝ) (crossing_time : ℝ) : 
  train_length = 120 →
  crossing_time = 12 →
  (2 * train_length) / crossing_time * 3.6 = 36 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l2113_211355


namespace NUMINAMATH_CALUDE_complement_N_intersect_M_l2113_211300

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | 0 ≤ x ∧ x < 5}

-- Define set N
def N : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem complement_N_intersect_M :
  (U \ N) ∩ M = {x : ℝ | 0 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_N_intersect_M_l2113_211300


namespace NUMINAMATH_CALUDE_total_cups_is_twenty_l2113_211381

/-- Represents the number of cups of tea drunk by each merchant -/
structure Merchants where
  sosipatra : ℕ
  olympiada : ℕ
  poliksena : ℕ

/-- Defines the conditions given in the problem -/
def satisfies_conditions (m : Merchants) : Prop :=
  m.sosipatra + m.olympiada = 11 ∧
  m.olympiada + m.poliksena = 15 ∧
  m.sosipatra + m.poliksena = 14

/-- Theorem stating that the total number of cups is 20 -/
theorem total_cups_is_twenty (m : Merchants) (h : satisfies_conditions m) :
  m.sosipatra + m.olympiada + m.poliksena = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_cups_is_twenty_l2113_211381


namespace NUMINAMATH_CALUDE_sam_tuesday_letters_l2113_211361

-- Define the number of days
def num_days : ℕ := 2

-- Define the average number of letters per day
def average_letters : ℕ := 5

-- Define the number of letters written on Wednesday
def wednesday_letters : ℕ := 3

-- Define the function to calculate the number of letters written on Tuesday
def tuesday_letters : ℕ := num_days * average_letters - wednesday_letters

-- Theorem statement
theorem sam_tuesday_letters :
  tuesday_letters = 7 := by sorry

end NUMINAMATH_CALUDE_sam_tuesday_letters_l2113_211361


namespace NUMINAMATH_CALUDE_parabola_properties_l2113_211353

-- Define the parabola function
def f (x : ℝ) : ℝ := (x - 2)^2 + 1

-- Theorem statement
theorem parabola_properties :
  -- 1. The parabola opens upwards
  (∀ x y : ℝ, f ((x + y) / 2) ≤ (f x + f y) / 2) ∧
  -- 2. The axis of symmetry is x = 2
  (∀ h : ℝ, f (2 + h) = f (2 - h)) ∧
  -- 3. The vertex is at (2, 1)
  (f 2 = 1 ∧ ∀ x : ℝ, f x ≥ 1) ∧
  -- 4. When x < 2, y decreases as x increases
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 2 → f x₁ > f x₂) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2113_211353


namespace NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l2113_211376

theorem tan_ratio_from_sin_sum_diff (p q : Real) 
  (h1 : Real.sin (p + q) = 0.6) 
  (h2 : Real.sin (p - q) = 0.3) : 
  Real.tan p / Real.tan q = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l2113_211376


namespace NUMINAMATH_CALUDE_adjacent_different_colors_l2113_211365

/-- Represents a square on the grid -/
structure Square where
  row : Fin 10
  col : Fin 10

/-- Represents the color of a piece -/
inductive Color
  | White
  | Black

/-- Represents the state of the grid at any point in the process -/
def GridState := Square → Option Color

/-- Represents a single step in the replacement process -/
structure ReplacementStep where
  removed : Square
  placed : Square

/-- The sequence of replacement steps -/
def ReplacementSequence := List ReplacementStep

/-- Two squares are adjacent if they share a common edge -/
def adjacent (s1 s2 : Square) : Prop :=
  (s1.row = s2.row ∧ (s1.col.val + 1 = s2.col.val ∨ s2.col.val + 1 = s1.col.val)) ∨
  (s1.col = s2.col ∧ (s1.row.val + 1 = s2.row.val ∨ s2.row.val + 1 = s1.row.val))

/-- The initial state of the grid with 91 white pieces -/
def initialState : GridState :=
  sorry

/-- The state of the grid after applying a sequence of replacement steps -/
def applyReplacements (initial : GridState) (steps : ReplacementSequence) : GridState :=
  sorry

/-- Theorem: There exists a point in the replacement process where two adjacent squares have different colored pieces -/
theorem adjacent_different_colors (steps : ReplacementSequence) :
  ∃ (partialSteps : ReplacementSequence) (s1 s2 : Square),
    partialSteps.length < steps.length ∧
    adjacent s1 s2 ∧
    let state := applyReplacements initialState partialSteps
    (state s1).isSome ∧ (state s2).isSome ∧ state s1 ≠ state s2 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_different_colors_l2113_211365


namespace NUMINAMATH_CALUDE_zacks_marbles_l2113_211395

theorem zacks_marbles : ∃ (M : ℕ), 
  (∃ (k : ℕ), M - 5 = 3 * k) ∧ 
  (M - (3 * 20) - 5 = 5) ∧ 
  M = 70 := by
  sorry

end NUMINAMATH_CALUDE_zacks_marbles_l2113_211395


namespace NUMINAMATH_CALUDE_sum_of_ratios_l2113_211362

def is_multiplicative (f : ℕ → ℝ) : Prop :=
  ∀ m n : ℕ, m > 0 ∧ n > 0 → f (m + n) = f m * f n

theorem sum_of_ratios (f : ℕ → ℝ) (h_mult : is_multiplicative f) (h_f1 : f 1 = 2) :
  (Finset.range 2010).sum (λ i => f (i + 2) / f (i + 1)) = 4020 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ratios_l2113_211362


namespace NUMINAMATH_CALUDE_halfway_between_fractions_l2113_211331

theorem halfway_between_fractions : (2 / 9 + 5 / 12) / 2 = 23 / 72 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_fractions_l2113_211331


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_l2113_211378

theorem quadratic_vertex_form (x : ℝ) : ∃ (a h k : ℝ), 
  3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k ∧ h = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_l2113_211378


namespace NUMINAMATH_CALUDE_min_value_expression_l2113_211356

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (min : ℝ), min = 4 * Real.sqrt 3 ∧
  ∀ (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0),
    x^2 + y^2 + 16 / x^2 + 4 * y / x ≥ min ∧
    ∃ (a₀ b₀ : ℝ) (ha₀ : a₀ ≠ 0) (hb₀ : b₀ ≠ 0),
      a₀^2 + b₀^2 + 16 / a₀^2 + 4 * b₀ / a₀ = min :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2113_211356


namespace NUMINAMATH_CALUDE_count_triples_sum_8_l2113_211364

/-- The number of ordered triples of natural numbers that sum to a given natural number. -/
def count_triples (n : ℕ) : ℕ := Nat.choose (n + 2) 2

/-- Theorem stating that the number of ordered triples (A, B, C) of natural numbers
    that satisfy A + B + C = 8 is equal to 21. -/
theorem count_triples_sum_8 : count_triples 8 = 21 := by
  sorry

end NUMINAMATH_CALUDE_count_triples_sum_8_l2113_211364


namespace NUMINAMATH_CALUDE_not_in_sequence_l2113_211308

theorem not_in_sequence : ¬∃ (n : ℕ), 24 - 2 * n = 3 := by sorry

end NUMINAMATH_CALUDE_not_in_sequence_l2113_211308


namespace NUMINAMATH_CALUDE_average_of_numbers_l2113_211371

def numbers : List ℝ := [12, 13, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_of_numbers : (numbers.sum / numbers.length : ℝ) = 114391.81818181818 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l2113_211371


namespace NUMINAMATH_CALUDE_function_composition_identity_l2113_211360

/-- Given a function f(x) = (2ax - b) / (3cx + d) where b ≠ 0, d ≠ 0, abcd ≠ 0,
    and f(f(x)) = x for all x in the domain of f, there exist real numbers b and c
    such that 3a - 2d = -4.5c - 4b -/
theorem function_composition_identity (a b c d : ℝ) : 
  b ≠ 0 → d ≠ 0 → a * b * c * d ≠ 0 → 
  (∀ x, (2 * a * ((2 * a * x - b) / (3 * c * x + d)) - b) / 
        (3 * c * ((2 * a * x - b) / (3 * c * x + d)) + d) = x) →
  ∃ (b c : ℝ), 3 * a - 2 * d = -4.5 * c - 4 * b :=
by sorry

end NUMINAMATH_CALUDE_function_composition_identity_l2113_211360


namespace NUMINAMATH_CALUDE_increasing_f_implies_t_ge_5_l2113_211333

/-- The dot product of two 2D vectors -/
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

/-- The function f(x) defined as the dot product of (x^2, x+1) and (1-x, t) -/
def f (t : ℝ) (x : ℝ) : ℝ := dot_product (x^2, x+1) (1-x, t)

/-- A function is increasing on an interval if for any two points in the interval, 
    the function value at the larger point is greater than at the smaller point -/
def is_increasing (g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → g x < g y

theorem increasing_f_implies_t_ge_5 :
  ∀ t : ℝ, is_increasing (f t) (-1) 1 → t ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_increasing_f_implies_t_ge_5_l2113_211333


namespace NUMINAMATH_CALUDE_binomial_mode_is_four_l2113_211302

/-- The number of trials in the binomial distribution -/
def n : ℕ := 20

/-- The probability of success in each trial -/
def p : ℝ := 0.2

/-- The binomial probability mass function -/
def binomialPMF (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- Theorem stating that 4 is the mode of the binomial distribution B(20, 0.2) -/
theorem binomial_mode_is_four :
  ∀ k : ℕ, k ≠ 4 → binomialPMF 4 ≥ binomialPMF k :=
sorry

end NUMINAMATH_CALUDE_binomial_mode_is_four_l2113_211302


namespace NUMINAMATH_CALUDE_significant_figures_and_precision_of_0_03020_l2113_211316

/-- Represents a decimal number with its string representation -/
structure DecimalNumber where
  representation : String
  deriving Repr

/-- Counts the number of significant figures in a decimal number -/
def countSignificantFigures (n : DecimalNumber) : Nat :=
  sorry

/-- Determines the precision of a decimal number -/
inductive Precision
  | Tenths
  | Hundredths
  | Thousandths
  | TenThousandths
  deriving Repr

def getPrecision (n : DecimalNumber) : Precision :=
  sorry

theorem significant_figures_and_precision_of_0_03020 :
  let n : DecimalNumber := { representation := "0.03020" }
  countSignificantFigures n = 4 ∧ getPrecision n = Precision.TenThousandths :=
sorry

end NUMINAMATH_CALUDE_significant_figures_and_precision_of_0_03020_l2113_211316


namespace NUMINAMATH_CALUDE_new_crew_member_weight_l2113_211311

/-- Given a crew of oarsmen, prove that replacing a crew member results in a specific weight for the new crew member. -/
theorem new_crew_member_weight
  (n : ℕ) -- Number of oarsmen
  (avg_increase : ℝ) -- Increase in average weight
  (old_weight : ℝ) -- Weight of the replaced crew member
  (h1 : n = 20) -- There are 20 oarsmen
  (h2 : avg_increase = 2) -- Average weight increases by 2 kg
  (h3 : old_weight = 40) -- The replaced crew member weighs 40 kg
  : ∃ (new_weight : ℝ), new_weight = n * avg_increase + old_weight :=
by sorry

end NUMINAMATH_CALUDE_new_crew_member_weight_l2113_211311


namespace NUMINAMATH_CALUDE_netflix_shows_l2113_211350

/-- The number of shows watched per week by Gina and her sister on Netflix. -/
def total_shows (gina_minutes : ℕ) (show_length : ℕ) (gina_ratio : ℕ) : ℕ :=
  let gina_shows := gina_minutes / show_length
  let sister_shows := gina_shows / gina_ratio
  gina_shows + sister_shows

/-- Theorem stating the total number of shows watched per week given the conditions. -/
theorem netflix_shows : total_shows 900 50 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_netflix_shows_l2113_211350


namespace NUMINAMATH_CALUDE_angle_measure_l2113_211318

/-- Given two angles AOB and BOC, proves that angle AOC is either the sum or difference of these angles. -/
theorem angle_measure (α β : ℝ) (hα : α = 30) (hβ : β = 15) :
  ∃ γ : ℝ, (γ = α + β ∨ γ = α - β) ∧ (γ = 45 ∨ γ = 15) := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l2113_211318


namespace NUMINAMATH_CALUDE_jessica_apple_pie_servings_l2113_211330

/-- Calculates the number of apples per serving in Jessica's apple pies. -/
def apples_per_serving (num_guests : ℕ) (num_pies : ℕ) (servings_per_pie : ℕ) (apples_per_guest : ℚ) : ℚ :=
  let total_apples := num_guests * apples_per_guest
  let total_servings := num_pies * servings_per_pie
  total_apples / total_servings

/-- Theorem stating that given Jessica's conditions, each serving requires 1.5 apples. -/
theorem jessica_apple_pie_servings :
  apples_per_serving 12 3 8 3 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_jessica_apple_pie_servings_l2113_211330


namespace NUMINAMATH_CALUDE_cone_height_from_cube_l2113_211387

/-- The height of a cone formed by melting a cube -/
theorem cone_height_from_cube (cube_edge : ℝ) (cone_base_area : ℝ) (cone_height : ℝ) : 
  cube_edge = 6 →
  cone_base_area = 54 →
  (cube_edge ^ 3) = (1 / 3) * cone_base_area * cone_height →
  cone_height = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_cone_height_from_cube_l2113_211387


namespace NUMINAMATH_CALUDE_fair_ride_cost_l2113_211398

theorem fair_ride_cost (total_tickets : ℕ) (spent_tickets : ℕ) (num_rides : ℕ) 
  (h1 : total_tickets = 79)
  (h2 : spent_tickets = 23)
  (h3 : num_rides = 8)
  (h4 : total_tickets ≥ spent_tickets) :
  (total_tickets - spent_tickets) / num_rides = 7 := by
sorry

end NUMINAMATH_CALUDE_fair_ride_cost_l2113_211398


namespace NUMINAMATH_CALUDE_simplify_fraction_sum_l2113_211367

theorem simplify_fraction_sum (a b c d : ℕ) : 
  a = 75 → b = 135 → 
  (∃ (k : ℕ), k * c = a ∧ k * d = b) → 
  (∀ (m : ℕ), m * c = a ∧ m * d = b → m ≤ k) →
  c + d = 14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_sum_l2113_211367
