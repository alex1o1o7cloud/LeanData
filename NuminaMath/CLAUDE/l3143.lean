import Mathlib

namespace brick_length_calculation_l3143_314372

/-- Calculates the length of a brick given wall dimensions and brick count --/
theorem brick_length_calculation (wall_length wall_height wall_thickness : ℝ)
                                 (brick_width brick_height : ℝ) (brick_count : ℕ) :
  wall_length = 750 ∧ wall_height = 600 ∧ wall_thickness = 22.5 ∧
  brick_width = 11.25 ∧ brick_height = 6 ∧ brick_count = 6000 →
  ∃ (brick_length : ℝ),
    brick_length = 25 ∧
    wall_length * wall_height * wall_thickness =
    brick_length * brick_width * brick_height * brick_count :=
by sorry

end brick_length_calculation_l3143_314372


namespace original_mean_calculation_l3143_314305

theorem original_mean_calculation (n : ℕ) (decrement : ℝ) (new_mean : ℝ) (h1 : n = 50) (h2 : decrement = 47) (h3 : new_mean = 153) :
  ∃ (original_mean : ℝ), original_mean * n = new_mean * n + decrement * n ∧ original_mean = 200 := by
  sorry

end original_mean_calculation_l3143_314305


namespace count_pairs_eq_fib_l3143_314335

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Count of pairs (α,S) with specific properties -/
def count_pairs (n : ℕ) : ℕ :=
  sorry

theorem count_pairs_eq_fib (n : ℕ) :
  count_pairs n = n! * fib (n + 1) := by
  sorry

end count_pairs_eq_fib_l3143_314335


namespace geometric_sequence_sum_l3143_314352

/-- Given a geometric sequence {a_n} where a_1 = 2 and a_1 + a_3 + a_5 = 14,
    prove that 1/a_1 + 1/a_3 + 1/a_5 = 7/8 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h1 : a 1 = 2) 
    (h2 : a 1 + a 3 + a 5 = 14) 
    (h3 : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = a n * q) :
    1 / a 1 + 1 / a 3 + 1 / a 5 = 7 / 8 := by
  sorry

end geometric_sequence_sum_l3143_314352


namespace three_two_digit_multiples_l3143_314307

theorem three_two_digit_multiples :
  (∃! (s : Finset ℕ), 
    (∀ x ∈ s, x > 0 ∧ 
      (∃! (m : Finset ℕ), 
        (∀ y ∈ m, 10 ≤ y ∧ y < 100 ∧ ∃ k, y = k * x) ∧ 
        m.card = 3)) ∧ 
    s.card = 9) := by sorry

end three_two_digit_multiples_l3143_314307


namespace solution_interval_l3143_314381

theorem solution_interval (c : ℝ) : (c / 4 ≤ 3 + c ∧ 3 + c < -3 * (1 + c)) ↔ c ∈ Set.Ici (-4) ∩ Set.Iio (-3/2) := by
  sorry

end solution_interval_l3143_314381


namespace books_bought_is_difference_melanie_books_bought_l3143_314394

/-- Represents the number of books Melanie bought at the yard sale -/
def books_bought (initial_books final_books : ℕ) : ℕ :=
  final_books - initial_books

/-- Theorem stating that the number of books bought is the difference between final and initial books -/
theorem books_bought_is_difference (initial_books final_books : ℕ) 
  (h : final_books ≥ initial_books) :
  books_bought initial_books final_books = final_books - initial_books :=
by
  sorry

/-- Melanie's initial number of books -/
def melanie_initial_books : ℕ := 41

/-- Melanie's final number of books -/
def melanie_final_books : ℕ := 87

/-- Theorem proving the number of books Melanie bought at the yard sale -/
theorem melanie_books_bought : 
  books_bought melanie_initial_books melanie_final_books = 46 :=
by
  sorry

end books_bought_is_difference_melanie_books_bought_l3143_314394


namespace tan_alpha_value_l3143_314316

theorem tan_alpha_value (α : Real) (h : Real.tan (α - π/4) = 1/6) : Real.tan α = 7/5 := by
  sorry

end tan_alpha_value_l3143_314316


namespace find_x_l3143_314336

theorem find_x : ∃ x : ℝ, (0.5 * x = 0.25 * 1500 - 30) ∧ (x = 690) := by
  sorry

end find_x_l3143_314336


namespace solution_set_l3143_314317

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : f 2 = 7)
variable (h2 : ∀ x : ℝ, deriv f x < 3)

-- Define the theorem
theorem solution_set (x : ℝ) : 
  f x < 3 * x + 1 ↔ x > 2 := by sorry

end solution_set_l3143_314317


namespace average_headcount_l3143_314309

-- Define the list of spring term headcounts
def spring_headcounts : List Nat := [11000, 10200, 10800, 11300]

-- Define the number of terms
def num_terms : Nat := 4

-- Theorem to prove the average headcount
theorem average_headcount :
  (spring_headcounts.sum / num_terms : ℚ) = 10825 := by
  sorry

end average_headcount_l3143_314309


namespace max_sin_a_value_l3143_314375

open Real

theorem max_sin_a_value (a b c : ℝ) 
  (h1 : cos a = tan b) 
  (h2 : cos b = tan c) 
  (h3 : cos c = tan a) : 
  ∃ (max_sin_a : ℝ), (∀ a' b' c' : ℝ, cos a' = tan b' → cos b' = tan c' → cos c' = tan a' → sin a' ≤ max_sin_a) ∧ max_sin_a = (sqrt 5 - 1) / 2 :=
sorry

end max_sin_a_value_l3143_314375


namespace parabola_properties_l3143_314385

/-- Properties of a parabola y = ax^2 + bx + c with a > 0, b > 0, and c < 0 -/
theorem parabola_properties (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c < 0) :
  let f := fun x => a * x^2 + b * x + c
  let vertex_x := -b / (2 * a)
  (∀ x y : ℝ, f x < f y → x < y ∨ y < x) ∧  -- Opens upwards
  vertex_x < 0 ∧                            -- Vertex in left half-plane
  f 0 < 0                                   -- Y-intercept below origin
:= by sorry

end parabola_properties_l3143_314385


namespace sum_of_roots_equation_l3143_314341

theorem sum_of_roots_equation (x : ℝ) : 
  ((-15 * x) / (x^2 - 1) = (3 * x) / (x + 1) - 9 / (x - 1)) →
  (∃ y : ℝ, (-15 * y) / (y^2 - 1) = (3 * y) / (y + 1) - 9 / (y - 1) ∧ y ≠ x) →
  x + y = -1 := by
sorry

end sum_of_roots_equation_l3143_314341


namespace second_integer_value_l3143_314378

theorem second_integer_value (n : ℝ) : 
  (n + (n + 3) = 150) → (n + 1 = 74.5) := by
  sorry

end second_integer_value_l3143_314378


namespace unique_digit_B_l3143_314359

-- Define the number as a function of B
def number (B : Nat) : Nat := 58709310 + B

-- Theorem statement
theorem unique_digit_B :
  ∀ B : Nat,
  B < 10 →
  (number B) % 2 = 0 →
  (number B) % 3 = 0 →
  (number B) % 4 = 0 →
  (number B) % 5 = 0 →
  (number B) % 6 = 0 →
  (number B) % 10 = 0 →
  B = 0 := by
  sorry

end unique_digit_B_l3143_314359


namespace salt_solution_volume_salt_solution_volume_proof_l3143_314388

/-- Given a solution with initial salt concentration of 10% that becomes 8% salt
    after adding 16 gallons of water, prove the initial volume is 64 gallons. -/
theorem salt_solution_volume : ℝ → Prop :=
  fun initial_volume =>
    let initial_salt_concentration : ℝ := 0.10
    let final_salt_concentration : ℝ := 0.08
    let added_water : ℝ := 16
    let final_volume : ℝ := initial_volume + added_water
    initial_salt_concentration * initial_volume =
      final_salt_concentration * final_volume →
    initial_volume = 64

/-- Proof of the salt_solution_volume theorem -/
theorem salt_solution_volume_proof : salt_solution_volume 64 := by
  sorry

end salt_solution_volume_salt_solution_volume_proof_l3143_314388


namespace even_function_inequality_l3143_314361

theorem even_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h_even : ∀ x, f (-x) = f x)
  (h_increasing : ∀ x y, x < y → x < 0 → f x < f y) :
  f (-2) ≥ f (a^2 - 4*a + 6) := by
  sorry

end even_function_inequality_l3143_314361


namespace point_in_fourth_quadrant_l3143_314351

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the concept of symmetry with respect to the origin
def symmetricToOrigin (p q : Point2D) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

-- Define the fourth quadrant
def inFourthQuadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

-- Theorem statement
theorem point_in_fourth_quadrant (a : ℝ) (P P_1 : Point2D) :
  a < 0 →
  P = Point2D.mk (-a^2 - 1) (-a + 3) →
  symmetricToOrigin P P_1 →
  inFourthQuadrant P_1 := by
  sorry

end point_in_fourth_quadrant_l3143_314351


namespace deepak_age_l3143_314368

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    determine Deepak's present age -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →   -- Age ratio condition
  rahul_age + 6 = 38 →                     -- Rahul's future age condition
  deepak_age = 24 := by                    -- Deepak's present age to prove
sorry

end deepak_age_l3143_314368


namespace rationality_of_x_not_necessarily_rational_l3143_314363

theorem rationality_of_x (x : ℝ) :
  (∃ (a b : ℚ), x^7 = a ∧ x^12 = b) →
  ∃ (q : ℚ), x = q :=
sorry

theorem not_necessarily_rational (x : ℝ) :
  (∃ (a b : ℚ), x^9 = a ∧ x^12 = b) →
  ¬(∀ x : ℝ, ∃ (q : ℚ), x = q) :=
sorry

end rationality_of_x_not_necessarily_rational_l3143_314363


namespace factorization_equality_l3143_314376

theorem factorization_equality (x : ℝ) :
  (x^4 - 4*x^2 + 1) * (x^4 + 3*x^2 + 1) + 10*x^4 = 
  (x + 1)^2 * (x - 1)^2 * (x^2 + x + 1) * (x^2 - x + 1) := by
  sorry

end factorization_equality_l3143_314376


namespace mixture_quantity_is_three_l3143_314310

/-- Represents the cost and quantity of a tea and coffee mixture --/
structure TeaCoffeeMixture where
  june_cost : ℝ  -- Cost per pound of both tea and coffee in June
  july_tea_cost : ℝ  -- Cost per pound of tea in July
  july_coffee_cost : ℝ  -- Cost per pound of coffee in July
  mixture_cost : ℝ  -- Total cost of the mixture in July
  mixture_quantity : ℝ  -- Quantity of the mixture in pounds

/-- Theorem stating the quantity of mixture bought given the conditions --/
theorem mixture_quantity_is_three (m : TeaCoffeeMixture) : 
  m.june_cost > 0 ∧ 
  m.july_coffee_cost = 2 * m.june_cost ∧ 
  m.july_tea_cost = 0.3 * m.june_cost ∧ 
  m.july_tea_cost = 0.3 ∧ 
  m.mixture_cost = 3.45 ∧ 
  m.mixture_quantity = m.mixture_cost / ((m.july_tea_cost + m.july_coffee_cost) / 2) →
  m.mixture_quantity = 3 := by
  sorry

#check mixture_quantity_is_three

end mixture_quantity_is_three_l3143_314310


namespace rent_increase_percentage_l3143_314308

theorem rent_increase_percentage (last_year_earnings : ℝ) : 
  let last_year_rent := 0.20 * last_year_earnings
  let this_year_earnings := 1.20 * last_year_earnings
  let this_year_rent := 0.30 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 180 := by
sorry

end rent_increase_percentage_l3143_314308


namespace units_digit_G_1000_l3143_314360

/-- The sequence G_n is defined as 3^(3^n) + 1 -/
def G (n : ℕ) : ℕ := 3^(3^n) + 1

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- Theorem: The units digit of G_1000 is 2 -/
theorem units_digit_G_1000 : units_digit (G 1000) = 2 := by sorry

end units_digit_G_1000_l3143_314360


namespace touching_x_axis_with_max_value_l3143_314301

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x

-- Define the derivative of f
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

-- Theorem statement
theorem touching_x_axis_with_max_value (a b m : ℝ) :
  m ≠ 0 →
  f a b m = 0 →
  f' a b m = 0 →
  (∀ x, f a b x ≤ 1/2) →
  (∃ x, f a b x = 1/2) →
  m = 3/2 := by
sorry

end touching_x_axis_with_max_value_l3143_314301


namespace calculation_proof_l3143_314328

theorem calculation_proof :
  (∃ x : ℝ, x ^ 2 = 2 ∧
    (Real.sqrt 6 * Real.sqrt (1/3) - Real.sqrt 16 * Real.sqrt 18 = -11 * x) ∧
    ((2 - Real.sqrt 5) * (2 + Real.sqrt 5) + (2 - Real.sqrt 2) ^ 2 = 5 - 4 * x)) := by
  sorry

end calculation_proof_l3143_314328


namespace notebook_count_l3143_314362

theorem notebook_count : ∃ (n : ℕ), n > 0 ∧ n + (n + 50) = 110 ∧ n = 30 := by sorry

end notebook_count_l3143_314362


namespace numbers_with_seven_from_1_to_800_l3143_314343

def contains_seven (n : ℕ) : Bool :=
  sorry

def count_numbers_with_seven (lower : ℕ) (upper : ℕ) : ℕ :=
  sorry

theorem numbers_with_seven_from_1_to_800 :
  count_numbers_with_seven 1 800 = 62 :=
sorry

end numbers_with_seven_from_1_to_800_l3143_314343


namespace problem_1_problem_2_l3143_314355

-- Problem 1
theorem problem_1 (x : ℝ) : 
  (4 / (x^2 - 1) - 1 = (1 - x) / (x + 1)) ↔ x = 5/2 :=
sorry

-- Problem 2
theorem problem_2 : 
  ¬∃ (x : ℝ), (2 / (x - 3) + 2 = (1 - x) / (3 - x)) :=
sorry

end problem_1_problem_2_l3143_314355


namespace sin_three_zeros_l3143_314325

/-- Given a function f(x) = sin(ωx + π/3) with ω > 0, if f has exactly 3 zeros
    in the interval [0, 2π/3], then 4 ≤ ω < 11/2 -/
theorem sin_three_zeros (ω : ℝ) (h₁ : ω > 0) :
  (∃! (zeros : Finset ℝ), zeros.card = 3 ∧
    (∀ x ∈ zeros, x ∈ Set.Icc 0 (2 * Real.pi / 3) ∧
      Real.sin (ω * x + Real.pi / 3) = 0)) →
  4 ≤ ω ∧ ω < 11 / 2 :=
by sorry

end sin_three_zeros_l3143_314325


namespace tank_capacity_l3143_314303

/-- Proves that a tank with given leak and inlet rates has a capacity of 1728 litres -/
theorem tank_capacity (leak_empty_time : ℝ) (inlet_rate : ℝ) (combined_empty_time : ℝ) 
  (h1 : leak_empty_time = 8) 
  (h2 : inlet_rate = 6) 
  (h3 : combined_empty_time = 12) : ℝ :=
by
  -- Define the capacity of the tank
  let capacity : ℝ := 1728

  -- State that the capacity is equal to 1728 litres
  have capacity_eq : capacity = 1728 := by rfl

  -- The proof would go here
  sorry


end tank_capacity_l3143_314303


namespace num_triangles_on_square_l3143_314302

/-- The number of points on each side of the square (excluding corners) -/
def points_per_side : ℕ := 7

/-- The total number of points on all sides of the square -/
def total_points : ℕ := 4 * points_per_side

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The number of different triangles formed by selecting three distinct points
    from a set of points on the sides of a square (excluding corners) -/
theorem num_triangles_on_square : 
  choose total_points 3 - 4 * (choose points_per_side 3) = 3136 := by
  sorry

end num_triangles_on_square_l3143_314302


namespace exponent_equality_l3143_314324

theorem exponent_equality (x : ℕ) : 
  2010^2011 - 2010^2009 = 2010^x * 2009 * 2011 → x = 2009 := by
  sorry

end exponent_equality_l3143_314324


namespace function_properties_l3143_314386

-- Define the function f and its derivative
def f (a b m x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + m
def f' (a b x : ℝ) : ℝ := 6 * x^2 + 2 * a * x + b

-- Main theorem
theorem function_properties (a b m : ℝ) :
  (∀ x : ℝ, f' a b x = f' a b (-1 - x)) →  -- f' is symmetric about x = -1/2
  (f' a b 1 = 0) →                         -- f'(1) = 0
  (a = 3 ∧ b = -12) ∧                      -- Part 1: values of a and b
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧     -- f has exactly three zeros
    f 3 (-12) m x₁ = 0 ∧
    f 3 (-12) m x₂ = 0 ∧
    f 3 (-12) m x₃ = 0 ∧
    ∀ x : ℝ, f 3 (-12) m x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) →
  -20 < m ∧ m < 7                          -- Part 2: range of m
  := by sorry


end function_properties_l3143_314386


namespace sum_to_zero_l3143_314397

/-- Given an initial sum of 2b - 1, where one addend is increased by 3b - 8 and another is decreased by -b - 7,
    prove that subtracting 6b - 2 from the third addend makes the total sum zero. -/
theorem sum_to_zero (b : ℝ) : 
  let initial_sum := 2*b - 1
  let increase := 3*b - 8
  let decrease := -b - 7
  let subtraction := 6*b - 2
  initial_sum + increase - decrease - subtraction = 0 := by
sorry

end sum_to_zero_l3143_314397


namespace sum_of_primes_with_square_property_l3143_314339

theorem sum_of_primes_with_square_property : ∃ (S : Finset Nat),
  (∀ p ∈ S, Nat.Prime p ∧ ∃ q, Nat.Prime q ∧ ∃ k, p^2 + p*q + q^2 = k^2) ∧
  (∀ p, Nat.Prime p → (∃ q, Nat.Prime q ∧ ∃ k, p^2 + p*q + q^2 = k^2) → p ∈ S) ∧
  S.sum id = 8 := by
  sorry

end sum_of_primes_with_square_property_l3143_314339


namespace square_of_two_minus_sqrt_three_l3143_314300

theorem square_of_two_minus_sqrt_three : (2 - Real.sqrt 3) ^ 2 = 7 - 4 * Real.sqrt 3 := by
  sorry

end square_of_two_minus_sqrt_three_l3143_314300


namespace smallest_square_with_specific_digits_l3143_314390

theorem smallest_square_with_specific_digits : 
  let n : ℕ := 666667
  ∀ m : ℕ, m < n → 
    (m ^ 2 < 444445 * 10^6) ∨ 
    (m ^ 2 ≥ 444446 * 10^6) :=
by sorry

end smallest_square_with_specific_digits_l3143_314390


namespace toothpick_grid_count_l3143_314391

/-- Calculates the number of toothpicks in a grid with missing toothpicks in regular intervals -/
def toothpick_count (length width : ℕ) (row_interval column_interval : ℕ) : ℕ :=
  let vertical_lines := length + 1
  let horizontal_lines := width + 1
  let vertical_missing := (vertical_lines / column_interval) * width
  let horizontal_missing := (horizontal_lines / row_interval) * length
  let vertical_count := vertical_lines * width - vertical_missing
  let horizontal_count := horizontal_lines * length - horizontal_missing
  vertical_count + horizontal_count

/-- The total number of toothpicks in the specified grid -/
theorem toothpick_grid_count :
  toothpick_count 45 25 5 4 = 2304 := by sorry

end toothpick_grid_count_l3143_314391


namespace planes_parallel_iff_skew_lines_parallel_l3143_314389

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (parallel : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the "contained in" relation for lines and planes
variable (contained_in : Line → Plane → Prop)

-- Define the "skew" relation for lines
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem planes_parallel_iff_skew_lines_parallel (α β : Plane) :
  parallel α β ↔
  ∃ (a b : Line),
    skew a b ∧
    contained_in a α ∧
    contained_in b β ∧
    parallel_line_plane a β ∧
    parallel_line_plane b α :=
sorry

end planes_parallel_iff_skew_lines_parallel_l3143_314389


namespace power_of_power_three_l3143_314398

theorem power_of_power_three : (3^2)^4 = 6561 := by
  sorry

end power_of_power_three_l3143_314398


namespace hotdog_eating_competition_l3143_314331

theorem hotdog_eating_competition (x y z : ℕ+) :
  y = 1 ∧
  x = z - 2 ∧
  6 * ((2*x - 3) + (3*x - y) + (4*x + z) + (x^2 - 5) + (3*y + 5*z) + (x*(y+z)) + ((x^2)+(y*z) - 2) + (x^3*y^2*z-15)) = 10000 →
  ∃ (hotdogs : ℕ), hotdogs = 6 * (x^3 * y^2 * z - 15) :=
by sorry

end hotdog_eating_competition_l3143_314331


namespace simple_interest_problem_l3143_314357

/-- Given a sum P put at simple interest rate R for 4 years, 
    if increasing the rate by 3% results in Rs. 120 more interest, 
    then P = 1000. -/
theorem simple_interest_problem (P R : ℝ) (h : P > 0) (r : R > 0) :
  (P * (R + 3) * 4) / 100 - (P * R * 4) / 100 = 120 →
  P = 1000 := by
  sorry

end simple_interest_problem_l3143_314357


namespace megan_markers_l3143_314348

theorem megan_markers (x : ℕ) : x + 109 = 326 → x = 217 := by
  sorry

end megan_markers_l3143_314348


namespace inscribed_sphere_volume_l3143_314346

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (d : ℝ) (h : d = 24) :
  let r := d * (Real.sqrt 2 - 1) / 2
  (4 / 3) * Real.pi * r^3 = (4 / 3) * Real.pi * (24 * (Real.sqrt 2 - 1))^3 := by
  sorry

end inscribed_sphere_volume_l3143_314346


namespace sqrt_fraction_simplification_l3143_314349

theorem sqrt_fraction_simplification :
  (Real.sqrt 6) / (Real.sqrt 10) = (Real.sqrt 15) / 5 := by
  sorry

end sqrt_fraction_simplification_l3143_314349


namespace notebook_cost_l3143_314382

def total_spent : ℕ := 74
def ruler_cost : ℕ := 18
def pencil_cost : ℕ := 7
def num_pencils : ℕ := 3

theorem notebook_cost :
  total_spent - (ruler_cost + num_pencils * pencil_cost) = 35 := by
  sorry

end notebook_cost_l3143_314382


namespace sequence_properties_l3143_314345

/-- A sequence where the sum of the first n terms is S_n = 2n^2 + 3n -/
def S (n : ℕ) : ℕ := 2 * n^2 + 3 * n

/-- The nth term of the sequence -/
def a (n : ℕ) : ℕ := 4 * n + 1

theorem sequence_properties :
  (∀ n : ℕ, S (n + 1) - S n = a (n + 1)) ∧
  (∀ n : ℕ, a (n + 1) - a n = 4) ∧
  (a 10 = 41) := by sorry

end sequence_properties_l3143_314345


namespace car_distance_l3143_314364

theorem car_distance (train_speed : ℝ) (car_speed_ratio : ℝ) (time_minutes : ℝ) : 
  train_speed = 120 →
  car_speed_ratio = 2/3 →
  time_minutes = 15 →
  (car_speed_ratio * train_speed) * (time_minutes / 60) = 20 := by
  sorry

end car_distance_l3143_314364


namespace integral_proof_l3143_314330

open Real

noncomputable def f (x : ℝ) : ℝ := 3*x + log (abs x) + 2*log (abs (x+1)) - log (abs (x-2))

theorem integral_proof (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -1) (h3 : x ≠ 2) : 
  deriv f x = (3*x^3 - x^2 - 12*x - 2) / (x*(x+1)*(x-2)) :=
by sorry

end integral_proof_l3143_314330


namespace equivalence_condition_l3143_314320

theorem equivalence_condition (x y : ℝ) (h : x * y ≠ 0) :
  (x + y = 0) ↔ (y / x + x / y = -2) := by
  sorry

end equivalence_condition_l3143_314320


namespace assignPositions_eq_95040_l3143_314323

/-- The number of ways to assign 5 distinct positions to 5 people chosen from a group of 12 people,
    where each person can only hold one position. -/
def assignPositions : ℕ := 12 * 11 * 10 * 9 * 8

/-- Theorem stating that the number of ways to assign the positions is 95,040. -/
theorem assignPositions_eq_95040 : assignPositions = 95040 := by
  sorry

end assignPositions_eq_95040_l3143_314323


namespace remainder_problem_l3143_314358

theorem remainder_problem (x : ℤ) (h1 : x % 62 = 7) (h2 : ∃ n : ℤ, (x + n) % 31 = 18) : 
  ∃ n : ℕ, n > 0 ∧ (x + n) % 31 = 18 ∧ ∀ m : ℕ, m > 0 ∧ (x + m) % 31 = 18 → m ≥ n :=
by
  sorry

end remainder_problem_l3143_314358


namespace coefficient_x_cubed_in_expansion_l3143_314395

theorem coefficient_x_cubed_in_expansion : 
  let expression := (fun x : ℝ => (x^2 + 1)^2 * (x - 1)^6)
  ∃ (a b c d e f g : ℝ), 
    (∀ x, expression x = a*x^9 + b*x^8 + c*x^7 + d*x^6 + e*x^5 + f*x^4 + (-32)*x^3 + g) :=
sorry

end coefficient_x_cubed_in_expansion_l3143_314395


namespace calculate_expression_l3143_314334

theorem calculate_expression : (Real.sqrt 3) ^ 0 + 2⁻¹ + Real.sqrt 2 * Real.cos (45 * π / 180) - |-(1/2)| = 2 := by
  sorry

end calculate_expression_l3143_314334


namespace container_weight_l3143_314337

/-- Given the weights of different metal bars, calculate the total weight of a container --/
theorem container_weight (copper_weight tin_weight steel_weight : ℝ) 
  (h1 : steel_weight = 2 * tin_weight)
  (h2 : steel_weight = copper_weight + 20)
  (h3 : copper_weight = 90) : 
  20 * steel_weight + 20 * copper_weight + 20 * tin_weight = 5100 := by
  sorry

#check container_weight

end container_weight_l3143_314337


namespace order_of_expressions_l3143_314373

theorem order_of_expressions (x : ℝ) :
  let a : ℝ := -x^2 - 2*x
  let b : ℝ := -2*x^2 - 2
  let c : ℝ := Real.sqrt 5 - 1
  b < a ∧ a < c := by
  sorry

end order_of_expressions_l3143_314373


namespace smallest_a_value_smallest_a_exists_l3143_314342

/-- A three-digit even number -/
def ThreeDigitEven := {n : ℕ // 100 ≤ n ∧ n ≤ 998 ∧ Even n}

/-- The sum of five three-digit even numbers -/
def SumFiveNumbers := 4306

theorem smallest_a_value (A B C D E : ThreeDigitEven) 
  (h_order : A.val < B.val ∧ B.val < C.val ∧ C.val < D.val ∧ D.val < E.val)
  (h_sum : A.val + B.val + C.val + D.val + E.val = SumFiveNumbers) :
  A.val ≥ 326 := by
  sorry

theorem smallest_a_exists :
  ∃ (A B C D E : ThreeDigitEven),
    A.val < B.val ∧ B.val < C.val ∧ C.val < D.val ∧ D.val < E.val ∧
    A.val + B.val + C.val + D.val + E.val = SumFiveNumbers ∧
    A.val = 326 := by
  sorry

end smallest_a_value_smallest_a_exists_l3143_314342


namespace shelf_rearrangement_l3143_314353

theorem shelf_rearrangement (n : ℕ) (k : ℕ) (m : ℕ) : 
  n = 8 → k = 2 → m = 4 →
  (Nat.choose n k) * ((m + 1) * m + Nat.choose (m + 1) k) = 840 := by
  sorry

end shelf_rearrangement_l3143_314353


namespace parallel_lines_theorem_l3143_314393

/-- Represents a line segment with a length -/
structure LineSegment where
  length : ℝ

/-- Represents a set of parallel lines -/
structure ParallelLines where
  ab : LineSegment
  cd : LineSegment
  ef : LineSegment
  gh : LineSegment

/-- Given conditions for the problem -/
def problem_conditions (lines : ParallelLines) : Prop :=
  lines.ab.length = 300 ∧
  lines.cd.length = 200 ∧
  lines.ef.length = (lines.ab.length + lines.cd.length) / 4 ∧
  lines.gh.length = lines.ef.length - (lines.ef.length - lines.cd.length) / 4

/-- The theorem to be proved -/
theorem parallel_lines_theorem (lines : ParallelLines) 
  (h : problem_conditions lines) : lines.gh.length = 93.75 := by
  sorry

end parallel_lines_theorem_l3143_314393


namespace power_product_equals_sum_of_exponents_l3143_314312

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^4 * a = a^5 := by
  sorry

end power_product_equals_sum_of_exponents_l3143_314312


namespace cube_volume_increase_l3143_314399

theorem cube_volume_increase (a : ℝ) (ha : a > 0) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end cube_volume_increase_l3143_314399


namespace point_in_intersection_l3143_314370

def U : Set (ℝ × ℝ) := Set.univ

def A (m : ℝ) : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 + m > 0}

def B (n : ℝ) : Set (ℝ × ℝ) := {p | p.1 + p.2 - n ≤ 0}

theorem point_in_intersection (m n : ℝ) :
  (2, 3) ∈ A m ∩ (U \ B n) ↔ m > -1 ∧ n < 5 := by
  sorry

end point_in_intersection_l3143_314370


namespace no_four_digit_numbers_with_one_eighth_property_l3143_314377

theorem no_four_digit_numbers_with_one_eighth_property : 
  ¬∃ (N : ℕ), 
    (1000 ≤ N ∧ N < 10000) ∧ 
    (∃ (a x : ℕ), 
      1 ≤ a ∧ a ≤ 9 ∧
      100 ≤ x ∧ x < 1000 ∧
      N = 1000 * a + x ∧
      x = N / 8) := by
sorry

end no_four_digit_numbers_with_one_eighth_property_l3143_314377


namespace existence_of_r_l3143_314396

/-- Two infinite sequences of rational numbers -/
def s : ℕ → ℚ := sorry
def t : ℕ → ℚ := sorry

/-- Neither sequence is constant -/
axiom not_constant_s : ∃ i j, s i ≠ s j
axiom not_constant_t : ∃ i j, t i ≠ t j

/-- For any integers i and j, (sᵢ - sⱼ)(tᵢ - tⱼ) is an integer -/
axiom product_is_integer : ∀ i j : ℕ, ∃ k : ℤ, (s i - s j) * (t i - t j) = k

/-- The main theorem to be proved -/
theorem existence_of_r : ∃ r : ℚ, 
  (∀ i j : ℕ, ∃ m : ℤ, (s i - s j) * r = m) ∧ 
  (∀ i j : ℕ, ∃ n : ℤ, (t i - t j) / r = n) :=
sorry

end existence_of_r_l3143_314396


namespace consecutive_integers_sum_1000_l3143_314392

theorem consecutive_integers_sum_1000 :
  ∃ (m k : ℕ), m > 0 ∧ k ≥ 0 ∧ (k + 1) * (2 * m + k) = 2000 := by
  sorry

end consecutive_integers_sum_1000_l3143_314392


namespace no_solution_condition_l3143_314371

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (a * x) / (x - 1) ≠ 1 / (x - 1) + 2) ↔ (a = 1 ∨ a = 2) :=
sorry

end no_solution_condition_l3143_314371


namespace max_value_inequality_max_value_achievable_l3143_314311

theorem max_value_inequality (x y : ℝ) :
  (x + 3 * y + 5) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 35 :=
by sorry

theorem max_value_achievable :
  ∃ (x y : ℝ), (x + 3 * y + 5) / Real.sqrt (x^2 + y^2 + 4) = Real.sqrt 35 :=
by sorry

end max_value_inequality_max_value_achievable_l3143_314311


namespace quadratic_inequality_range_l3143_314321

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - 3*a*x + 9 < 0) ↔ (a < -2 ∨ a > 2) := by
sorry

end quadratic_inequality_range_l3143_314321


namespace possible_values_of_5x_plus_2_l3143_314314

theorem possible_values_of_5x_plus_2 (x : ℝ) : 
  (x - 4) * (5 * x + 2) = 0 → (5 * x + 2 = 0 ∨ 5 * x + 2 = 22) :=
by sorry

end possible_values_of_5x_plus_2_l3143_314314


namespace calculate_expression_l3143_314306

theorem calculate_expression : (-Real.sqrt 6)^2 - 3 * Real.sqrt 2 * Real.sqrt 18 = -12 := by
  sorry

end calculate_expression_l3143_314306


namespace mn_inequality_characterization_l3143_314338

theorem mn_inequality_characterization :
  ∀ m n : ℕ+, 
    (1 ≤ m^n.val - n^m.val ∧ m^n.val - n^m.val ≤ m.val * n.val) ↔ 
    ((m ≥ 2 ∧ n = 1) ∨ (m = 2 ∧ n = 5) ∨ (m = 3 ∧ n = 2)) := by
  sorry

end mn_inequality_characterization_l3143_314338


namespace intersection_and_complement_when_a_is_3_range_of_a_when_M_subset_N_l3143_314354

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x - 18 ≤ 0}
def N (a : ℝ) : Set ℝ := {x | 1 - a ≤ x ∧ x ≤ 2*a + 1}

-- Theorem for part 1
theorem intersection_and_complement_when_a_is_3 :
  (M ∩ N 3 = {x | -2 ≤ x ∧ x ≤ 6}) ∧
  (Set.univ \ N 3 = {x | x < -2 ∨ x > 7}) := by
  sorry

-- Theorem for part 2
theorem range_of_a_when_M_subset_N :
  (∀ a : ℝ, M ⊆ N a) ↔ (∀ a : ℝ, a ≥ 4) := by
  sorry

end intersection_and_complement_when_a_is_3_range_of_a_when_M_subset_N_l3143_314354


namespace season_games_count_l3143_314350

/-- The number of games in a football season -/
def season_games : ℕ := 16

/-- Archie's record for touchdown passes in a season -/
def archie_record : ℕ := 89

/-- Richard's average touchdowns per game -/
def richard_avg : ℕ := 6

/-- Required average touchdowns in final two games to beat the record -/
def final_avg : ℕ := 3

/-- Number of final games -/
def final_games : ℕ := 2

theorem season_games_count :
  ∃ (x : ℕ), 
    x + final_games = season_games ∧
    richard_avg * x + final_avg * final_games > archie_record :=
by sorry

end season_games_count_l3143_314350


namespace gcf_of_36_and_54_l3143_314304

theorem gcf_of_36_and_54 : Nat.gcd 36 54 = 18 := by
  sorry

end gcf_of_36_and_54_l3143_314304


namespace complex_fraction_equality_l3143_314379

theorem complex_fraction_equality : (1 + 2*Complex.I) / (1 - Complex.I)^2 = 1 - (1/2)*Complex.I := by
  sorry

end complex_fraction_equality_l3143_314379


namespace initial_mixture_volume_l3143_314356

/-- Given a mixture of milk and water with an initial ratio of 3:2, prove that
    after adding 48 liters of water to make the new ratio 3:4, the initial
    volume of the mixture was 120 liters. -/
theorem initial_mixture_volume
  (initial_milk : ℚ) (initial_water : ℚ)
  (initial_ratio : initial_milk / initial_water = 3 / 2)
  (new_ratio : initial_milk / (initial_water + 48) = 3 / 4) :
  initial_milk + initial_water = 120 := by
sorry

end initial_mixture_volume_l3143_314356


namespace set_intersection_theorem_l3143_314333

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| > 2}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- State the theorem
theorem set_intersection_theorem :
  (Set.univ \ A) ∩ B = {x | 2 < x ∧ x ≤ 3} :=
by sorry

end set_intersection_theorem_l3143_314333


namespace complex_sum_fourth_powers_l3143_314327

theorem complex_sum_fourth_powers : 
  let z₁ : ℂ := (-1 + Complex.I * Real.sqrt 7) / 2
  let z₂ : ℂ := (-1 - Complex.I * Real.sqrt 7) / 2
  z₁^4 + z₂^4 = 1 := by sorry

end complex_sum_fourth_powers_l3143_314327


namespace pond_length_l3143_314332

/-- Given a rectangular field and a square pond, prove the length of the pond --/
theorem pond_length (field_length : ℝ) (field_width : ℝ) (pond_area : ℝ) : 
  field_length = 32 →
  field_width = field_length / 2 →
  pond_area = (field_length * field_width) / 8 →
  ∃ (pond_length : ℝ), pond_length^2 = pond_area ∧ pond_length = 8 :=
by sorry

end pond_length_l3143_314332


namespace triangle_EC_length_l3143_314369

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the points D and E
def D (t : Triangle) : ℝ × ℝ := sorry
def E (t : Triangle) : ℝ × ℝ := sorry

-- Define the length of a segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the angle between two segments
def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- Define perpendicularity
def perpendicular (p q r s : ℝ × ℝ) : Prop := sorry

theorem triangle_EC_length (t : Triangle) : 
  angle t.A t.B t.C = π/4 →          -- ∠A = 45°
  length t.B t.C = 10 →              -- BC = 10
  perpendicular (D t) t.B t.A t.C → -- BD ⊥ AC
  perpendicular (E t) t.C t.A t.B → -- CE ⊥ AB
  angle (D t) t.B t.C = 2 * angle (E t) t.C t.B → -- m∠DBC = 2m∠ECB
  length (E t) t.C = 5 * Real.sqrt 6 := by
    sorry

#check triangle_EC_length

end triangle_EC_length_l3143_314369


namespace problem_statement_l3143_314374

theorem problem_statement (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1) : 
  (ab + cd ≤ 1) ∧ (-2 ≤ a + Real.sqrt 3 * b) ∧ (a + Real.sqrt 3 * b ≤ 2) := by
  sorry

end problem_statement_l3143_314374


namespace min_distance_four_points_l3143_314313

/-- Given four points P, Q, R, and S on a line with specified distances between them,
    prove that the minimum possible distance between P and S is 0. -/
theorem min_distance_four_points (P Q R S : ℝ) 
  (h1 : |Q - P| = 12) 
  (h2 : |R - Q| = 7) 
  (h3 : |S - R| = 5) : 
  ∃ (P' Q' R' S' : ℝ), 
    |Q' - P'| = 12 ∧ 
    |R' - Q'| = 7 ∧ 
    |S' - R'| = 5 ∧ 
    |S' - P'| = 0 :=
sorry

end min_distance_four_points_l3143_314313


namespace range_of_m_l3143_314380

theorem range_of_m (x m : ℝ) : 
  (∀ x, (2 ≤ x ∧ x ≤ 3) → |x - m| < 2) →
  (∃ a b : ℝ, a < b ∧ (∀ m, a < m ∧ m < b ↔ (∀ x, (2 ≤ x ∧ x ≤ 3) → |x - m| < 2))) ∧
  (∀ a b : ℝ, (∀ m, a < m ∧ m < b ↔ (∀ x, (2 ≤ x ∧ x ≤ 3) → |x - m| < 2)) → a = 1 ∧ b = 4) :=
by sorry

end range_of_m_l3143_314380


namespace quadratic_expression_value_l3143_314344

theorem quadratic_expression_value (x : ℝ) : x = 2 → x^2 - 3*x + 2 = 0 := by
  sorry

end quadratic_expression_value_l3143_314344


namespace logarithm_product_identity_l3143_314387

theorem logarithm_product_identity (x y : ℝ) (hx : x > 0) (hy : y > 0) (hy1 : y ≠ 1) :
  Real.log x ^ 2 / Real.log (y ^ 3) *
  Real.log (y ^ 3) / Real.log (x ^ 4) *
  Real.log (x ^ 4) / Real.log (y ^ 5) *
  Real.log (y ^ 5) / Real.log (x ^ 2) =
  Real.log x / Real.log y := by
  sorry

end logarithm_product_identity_l3143_314387


namespace green_light_probability_is_five_twelfths_l3143_314384

/-- Represents the duration of each light color in seconds -/
structure TrafficLightDuration where
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- Represents the cycle time of the traffic light in seconds -/
def cycleDuration (d : TrafficLightDuration) : ℕ :=
  d.red + d.green + d.yellow

/-- The probability of seeing a green light -/
def greenLightProbability (d : TrafficLightDuration) : ℚ :=
  d.green / (cycleDuration d)

/-- Theorem stating the probability of seeing a green light
    given specific durations for each light color -/
theorem green_light_probability_is_five_twelfths :
  let d : TrafficLightDuration := { red := 30, green := 25, yellow := 5 }
  greenLightProbability d = 5 / 12 := by
  sorry

end green_light_probability_is_five_twelfths_l3143_314384


namespace two_tangents_from_origin_l3143_314315

/-- The function f(x) = -x^3 + 3x^2 + 1 -/
def f (x : ℝ) : ℝ := -x^3 + 3*x^2 + 1

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x

/-- A point (t, f(t)) on the curve y = f(x) -/
def point_on_curve (t : ℝ) : ℝ × ℝ := (t, f t)

/-- The slope of the tangent line at point (t, f(t)) -/
def tangent_slope (t : ℝ) : ℝ := f' t

/-- The equation for finding points of tangency -/
def tangency_equation (t : ℝ) : Prop := 2*t^3 - 3*t^2 + 1 = 0

theorem two_tangents_from_origin :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
    tangency_equation t₁ ∧ 
    tangency_equation t₂ ∧ 
    (∀ t, tangency_equation t → t = t₁ ∨ t = t₂) :=
sorry

end two_tangents_from_origin_l3143_314315


namespace driving_time_proof_l3143_314340

/-- Proves that given the conditions of the driving problem, the driving times for route one and route two are 2 hours and 2.5 hours respectively. -/
theorem driving_time_proof (distance_one : ℝ) (distance_two : ℝ) (time_diff : ℝ) (speed_ratio : ℝ) :
  distance_one = 180 →
  distance_two = 150 →
  time_diff = 0.5 →
  speed_ratio = 1.5 →
  ∃ (time_one time_two : ℝ),
    time_one = 2 ∧
    time_two = 2.5 ∧
    time_two = time_one + time_diff ∧
    distance_one / time_one = speed_ratio * (distance_two / time_two) :=
by sorry


end driving_time_proof_l3143_314340


namespace complex_root_property_l3143_314347

variable (a b c d e m n : ℝ)
variable (z : ℂ)

theorem complex_root_property :
  (z = m + n * Complex.I) →
  (a * z^4 + Complex.I * b * z^2 + c * z^2 + Complex.I * d * z + e = 0) →
  (a * (-m + n * Complex.I)^4 + Complex.I * b * (-m + n * Complex.I)^2 + 
   c * (-m + n * Complex.I)^2 + Complex.I * d * (-m + n * Complex.I) + e = 0) := by
  sorry

end complex_root_property_l3143_314347


namespace negation_of_universal_proposition_l3143_314319

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 > Real.log x)) ↔ (∃ x : ℝ, x^2 ≤ Real.log x) := by sorry

end negation_of_universal_proposition_l3143_314319


namespace max_chosen_squares_29x29_l3143_314322

/-- The maximum number of squares that can be chosen on an n×n chessboard 
    such that for every selected square, there exists at most one square 
    with both row and column numbers greater than or equal to the selected 
    square's row and column numbers. -/
def max_chosen_squares (n : ℕ) : ℕ :=
  if n % 2 = 0 then 3 * (n / 2) else 3 * (n / 2) + 1

/-- Theorem stating that the maximum number of chosen squares for a 29×29 chessboard is 43. -/
theorem max_chosen_squares_29x29 : max_chosen_squares 29 = 43 := by
  sorry

end max_chosen_squares_29x29_l3143_314322


namespace equation_solutions_l3143_314367

theorem equation_solutions :
  (∃ x : ℚ, 6 * x - 4 = 3 * x + 2 ∧ x = 2) ∧
  (∃ x : ℚ, x / 4 - 3 / 5 = (x + 1) / 2 ∧ x = -22 / 5) := by
  sorry

end equation_solutions_l3143_314367


namespace lucille_earnings_l3143_314318

/-- Calculates the earnings from weeding a specific area -/
def calculate_earnings (small medium large : ℕ) : ℕ :=
  4 * small + 8 * medium + 12 * large

/-- Calculates the total cost of items after discount and tax -/
def calculate_total_cost (price : ℕ) (discount_rate tax_rate : ℚ) : ℕ :=
  let discounted_price := price - (price * discount_rate).floor
  (discounted_price + (discounted_price * tax_rate).ceil).toNat

theorem lucille_earnings : 
  let flower_bed := calculate_earnings 6 3 2
  let vegetable_patch := calculate_earnings 10 2 2
  let half_grass := calculate_earnings 10 5 1
  let new_area := calculate_earnings 7 4 1
  let total_earnings := flower_bed + vegetable_patch + half_grass + new_area
  let soda_snack_cost := calculate_total_cost 149 (1/10) (12/100)
  total_earnings - soda_snack_cost = 166 := by sorry

end lucille_earnings_l3143_314318


namespace quadratic_equation_unique_root_l3143_314365

theorem quadratic_equation_unique_root (b c : ℝ) :
  (∀ x : ℝ, 3 * x^2 + b * x + c = 0 ↔ x = -4) →
  b = 24 := by
sorry

end quadratic_equation_unique_root_l3143_314365


namespace binomial_square_example_l3143_314383

theorem binomial_square_example : 34^2 + 2*(34*5) + 5^2 = 1521 := by sorry

end binomial_square_example_l3143_314383


namespace josh_marbles_l3143_314366

/-- Theorem: If Josh had 16 marbles and lost 7 marbles, he now has 9 marbles. -/
theorem josh_marbles (initial : ℕ) (lost : ℕ) (final : ℕ) 
  (h1 : initial = 16) 
  (h2 : lost = 7) 
  (h3 : final = initial - lost) : 
  final = 9 := by
  sorry

end josh_marbles_l3143_314366


namespace max_islands_is_36_l3143_314326

/-- Represents an archipelago with islands and bridges -/
structure Archipelago where
  N : Nat
  bridges : Fin N → Fin N → Bool

/-- The number of islands is at least 7 -/
def atLeastSevenIslands (a : Archipelago) : Prop :=
  a.N ≥ 7

/-- Any two islands are connected by at most one bridge -/
def atMostOneBridge (a : Archipelago) : Prop :=
  ∀ i j : Fin a.N, i ≠ j → (a.bridges i j = true → a.bridges j i = false)

/-- No more than 5 bridges lead from each island -/
def atMostFiveBridges (a : Archipelago) : Prop :=
  ∀ i : Fin a.N, (Finset.filter (fun j => a.bridges i j) (Finset.univ : Finset (Fin a.N))).card ≤ 5

/-- Among any 7 islands, there are always two that are connected by a bridge -/
def twoConnectedInSeven (a : Archipelago) : Prop :=
  ∀ s : Finset (Fin a.N), s.card = 7 →
    ∃ i j : Fin a.N, i ∈ s ∧ j ∈ s ∧ i ≠ j ∧ a.bridges i j

/-- The maximum number of islands satisfying the conditions is 36 -/
theorem max_islands_is_36 (a : Archipelago) :
    atLeastSevenIslands a →
    atMostOneBridge a →
    atMostFiveBridges a →
    twoConnectedInSeven a →
    a.N ≤ 36 := by
  sorry

end max_islands_is_36_l3143_314326


namespace sum_remainder_mod_11_l3143_314329

theorem sum_remainder_mod_11 : 
  (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 11 = 10 := by
  sorry

end sum_remainder_mod_11_l3143_314329
