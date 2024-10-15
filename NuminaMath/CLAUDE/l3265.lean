import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l3265_326587

theorem problem_solution (a b c : ℝ) (h1 : |a| = 2) (h2 : a < 1) (h3 : b * c = 1) :
  a^3 + 3 - 4*b*c = -9 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3265_326587


namespace NUMINAMATH_CALUDE_train_length_l3265_326553

/-- The length of a train given its speed, time to pass a platform, and platform length -/
theorem train_length (train_speed : ℝ) (pass_time : ℝ) (platform_length : ℝ) : 
  train_speed = 45 * (1000 / 3600) →
  pass_time = 40 →
  platform_length = 140 →
  train_speed * pass_time - platform_length = 360 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3265_326553


namespace NUMINAMATH_CALUDE_point_not_on_line_l3265_326581

theorem point_not_on_line (m b : ℝ) (h : m * b > 0) :
  ¬(0 = m * 1997 + b) :=
sorry

end NUMINAMATH_CALUDE_point_not_on_line_l3265_326581


namespace NUMINAMATH_CALUDE_largest_square_leftover_l3265_326545

def yarn_length : ℕ := 35

theorem largest_square_leftover (s : ℕ) : 
  (s * 4 ≤ yarn_length) ∧ 
  (∀ t : ℕ, t * 4 ≤ yarn_length → t ≤ s) →
  yarn_length - s * 4 = 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_square_leftover_l3265_326545


namespace NUMINAMATH_CALUDE_negation_equivalence_l3265_326519

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3265_326519


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l3265_326558

-- Define the quadratic function
def f (a b x : ℝ) := a * x^2 - (a + 1) * x + b

-- Define the solution set condition
def solution_set (a b : ℝ) :=
  ∀ x, f a b x < 0 ↔ (x < -1/2 ∨ x > 1)

-- Define the inequality for part II
def g (x m : ℝ) := x^2 + (m - 4) * x + 3 - m

-- Main theorem
theorem quadratic_inequality_problem :
  ∃ a b : ℝ,
    solution_set a b ∧
    a = -2 ∧
    b = 1 ∧
    (∀ x, (∀ m ∈ Set.Icc 0 4, g x m ≥ 0) ↔ 
      (x ≤ -1 ∨ x = 1 ∨ x ≥ 3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l3265_326558


namespace NUMINAMATH_CALUDE_first_day_of_month_l3265_326579

/-- Days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the day of the week after n days -/
def dayAfter (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => dayAfter (nextDay d) n

/-- Theorem: If the 30th day of a month is a Wednesday, then the 1st day of that month is a Tuesday -/
theorem first_day_of_month (d : DayOfWeek) : 
  dayAfter d 29 = DayOfWeek.Wednesday → d = DayOfWeek.Tuesday := by
  sorry


end NUMINAMATH_CALUDE_first_day_of_month_l3265_326579


namespace NUMINAMATH_CALUDE_borrow_three_books_l3265_326513

/-- The number of ways to borrow at least one book out of three books -/
def borrow_methods (n : ℕ) : ℕ := 2^n - 1

/-- Theorem stating that the number of ways to borrow at least one book out of three books is 7 -/
theorem borrow_three_books : borrow_methods 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_borrow_three_books_l3265_326513


namespace NUMINAMATH_CALUDE_cylinder_volume_l3265_326531

/-- The volume of a cylinder with diameter 4 cm and height 5 cm is equal to π * 20 cm³ -/
theorem cylinder_volume (π : ℝ) (h : π = Real.pi) : 
  let d : ℝ := 4 -- diameter in cm
  let h : ℝ := 5 -- height in cm
  let r : ℝ := d / 2 -- radius in cm
  let v : ℝ := π * r^2 * h -- volume formula
  v = π * 20 := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l3265_326531


namespace NUMINAMATH_CALUDE_equality_transitivity_add_polynomial_to_equation_l3265_326572

-- Statement 1: Transitivity of equality
theorem equality_transitivity (a b c : ℝ) (h1 : a = b) (h2 : b = c) : a = c := by
  sorry

-- Statement 5: Adding a polynomial to both sides of an equation
theorem add_polynomial_to_equation (f g p : ℝ → ℝ) (h : ∀ x, f x = g x) : 
  ∀ x, f x + p x = g x + p x := by
  sorry

end NUMINAMATH_CALUDE_equality_transitivity_add_polynomial_to_equation_l3265_326572


namespace NUMINAMATH_CALUDE_problem_statements_l3265_326533

theorem problem_statements :
  (∀ x : ℝ, x ≥ 0 → x + 1 + 1 / (x + 1) ≥ 2) ∧
  (∀ x : ℝ, x > 0 → (x + 1) / Real.sqrt x ≥ 2) ∧
  (∃ x : ℝ, x + 1 / x < 2) ∧
  (∀ x : ℝ, Real.sqrt (x^2 + 2) + 1 / Real.sqrt (x^2 + 2) > 2) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_statements_l3265_326533


namespace NUMINAMATH_CALUDE_birth_year_digit_sum_difference_l3265_326511

theorem birth_year_digit_sum_difference (m c d u : Nat) 
  (hm : m < 10) (hc : c < 10) (hd : d < 10) (hu : u < 10) :
  ∃ k : Int, (1000 * m + 100 * c + 10 * d + u) - (m + c + d + u) = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_birth_year_digit_sum_difference_l3265_326511


namespace NUMINAMATH_CALUDE_g_composition_equals_107_l3265_326560

/-- The function g defined as g(x) = 3x + 2 -/
def g (x : ℝ) : ℝ := 3 * x + 2

/-- Theorem stating that g(g(g(3))) = 107 -/
theorem g_composition_equals_107 : g (g (g 3)) = 107 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_equals_107_l3265_326560


namespace NUMINAMATH_CALUDE_even_function_inequality_l3265_326582

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem even_function_inequality (h1 : IsEven f) (h2 : f 2 < f 3) : f (-3) > f (-2) := by
  sorry

end NUMINAMATH_CALUDE_even_function_inequality_l3265_326582


namespace NUMINAMATH_CALUDE_tan_two_implies_sin_2theta_over_cos_squared_minus_sin_squared_l3265_326529

theorem tan_two_implies_sin_2theta_over_cos_squared_minus_sin_squared (θ : Real) 
  (h : Real.tan θ = 2) : 
  (Real.sin (2 * θ)) / (Real.cos θ ^ 2 - Real.sin θ ^ 2) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_implies_sin_2theta_over_cos_squared_minus_sin_squared_l3265_326529


namespace NUMINAMATH_CALUDE_tan_sum_special_l3265_326576

theorem tan_sum_special (θ : Real) (h : Real.tan θ = 2) : Real.tan (θ + π/4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_special_l3265_326576


namespace NUMINAMATH_CALUDE_seating_arrangement_l3265_326597

/-- Given a seating arrangement where each row seats either 6 or 9 people,
    and 57 people are to be seated with all seats occupied,
    prove that there is exactly 1 row seating 9 people. -/
theorem seating_arrangement (x y : ℕ) : 
  9 * x + 6 * y = 57 → 
  x + y > 0 →
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_seating_arrangement_l3265_326597


namespace NUMINAMATH_CALUDE_pyramid_tiers_count_l3265_326536

/-- Calculates the surface area of a pyramid with n tiers built from 1 cm³ cubes -/
def pyramidSurfaceArea (n : ℕ) : ℕ :=
  4 * n^2 + 2 * n

/-- A pyramid built from 1 cm³ cubes with a surface area of 2352 cm² has 24 tiers -/
theorem pyramid_tiers_count : ∃ n : ℕ, pyramidSurfaceArea n = 2352 ∧ n = 24 := by
  sorry

#eval pyramidSurfaceArea 24  -- Should output 2352

end NUMINAMATH_CALUDE_pyramid_tiers_count_l3265_326536


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3265_326598

theorem algebraic_expression_value (a b : ℝ) (h : 2 * a - b = 5) :
  4 * a - 2 * b + 7 = 17 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3265_326598


namespace NUMINAMATH_CALUDE_quadratic_trinomial_pairs_l3265_326507

/-- Represents a quadratic trinomial ax^2 + bx + c -/
structure QuadraticTrinomial (α : Type*) [Ring α] where
  a : α
  b : α
  c : α

/-- Checks if two numbers are roots of a quadratic trinomial -/
def areRoots {α : Type*} [Ring α] (t : QuadraticTrinomial α) (r1 r2 : α) : Prop :=
  t.a * r1 * r1 + t.b * r1 + t.c = 0 ∧ t.a * r2 * r2 + t.b * r2 + t.c = 0

theorem quadratic_trinomial_pairs 
  {α : Type*} [Field α] [CharZero α]
  (t1 t2 : QuadraticTrinomial α)
  (h1 : areRoots t2 t1.b t1.c)
  (h2 : areRoots t1 t2.b t2.c) :
  (∃ (a : α), t1 = ⟨1, a, 0⟩ ∧ t2 = ⟨1, -a, 0⟩) ∨
  (t1 = ⟨1, 1, -2⟩ ∧ t2 = ⟨1, 1, -2⟩) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_trinomial_pairs_l3265_326507


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l3265_326568

theorem root_sum_reciprocals (a b c d : ℂ) : 
  (a^4 + 8*a^3 + 9*a^2 + 5*a + 4 = 0) →
  (b^4 + 8*b^3 + 9*b^2 + 5*b + 4 = 0) →
  (c^4 + 8*c^3 + 9*c^2 + 5*c + 4 = 0) →
  (d^4 + 8*d^3 + 9*d^2 + 5*d + 4 = 0) →
  (1/(a*b) + 1/(a*c) + 1/(a*d) + 1/(b*c) + 1/(b*d) + 1/(c*d) = 9/4) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l3265_326568


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3265_326512

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), Prime p ∧ p ∣ (7^15 + 9^17) ∧ ∀ q, Prime q → q ∣ (7^15 + 9^17) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3265_326512


namespace NUMINAMATH_CALUDE_multiples_of_15_between_20_and_205_l3265_326588

theorem multiples_of_15_between_20_and_205 : 
  (Finset.filter (fun x => x % 15 = 0 ∧ x > 20 ∧ x ≤ 205) (Finset.range 206)).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_15_between_20_and_205_l3265_326588


namespace NUMINAMATH_CALUDE_lieutenant_age_l3265_326555

theorem lieutenant_age : ∃ (n : ℕ), ∃ (x : ℕ), 
  -- Initial arrangement: n rows with n+5 soldiers each
  -- New arrangement: x rows (lieutenant's age) with n+9 soldiers each
  -- Total number of soldiers remains the same
  n * (n + 5) = x * (n + 9) ∧
  -- x represents a reasonable age for a lieutenant
  x > 18 ∧ x < 65 ∧
  -- The solution
  x = 24 := by
sorry

end NUMINAMATH_CALUDE_lieutenant_age_l3265_326555


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l3265_326538

theorem absolute_value_simplification (x : ℝ) (h : x < -1) :
  |x - 2 * Real.sqrt ((x + 1)^2)| = -3 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l3265_326538


namespace NUMINAMATH_CALUDE_quebec_temperature_l3265_326502

-- Define the temperatures as integers (assuming we're working with whole numbers)
def temp_vancouver : ℤ := 22
def temp_calgary : ℤ := temp_vancouver - 19
def temp_quebec : ℤ := temp_calgary - 11

-- Theorem to prove
theorem quebec_temperature : temp_quebec = -8 := by
  sorry

end NUMINAMATH_CALUDE_quebec_temperature_l3265_326502


namespace NUMINAMATH_CALUDE_solve_for_y_l3265_326591

theorem solve_for_y (x y : ℝ) (h1 : x^2 + x + 4 = y - 4) (h2 : x = 3) : y = 20 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3265_326591


namespace NUMINAMATH_CALUDE_candy_bar_to_caramel_ratio_l3265_326532

/-- The price of caramel in dollars -/
def caramel_price : ℚ := 3

/-- The price of a candy bar as a multiple of the caramel price -/
def candy_bar_price (k : ℚ) : ℚ := k * caramel_price

/-- The price of cotton candy -/
def cotton_candy_price (k : ℚ) : ℚ := 2 * candy_bar_price k

/-- The total cost of 6 candy bars, 3 caramel, and 1 cotton candy -/
def total_cost (k : ℚ) : ℚ := 6 * candy_bar_price k + 3 * caramel_price + cotton_candy_price k

theorem candy_bar_to_caramel_ratio :
  ∃ k : ℚ, total_cost k = 57 ∧ candy_bar_price k / caramel_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_to_caramel_ratio_l3265_326532


namespace NUMINAMATH_CALUDE_angle_relation_l3265_326594

theorem angle_relation (α β : Real) : 
  π / 2 < α ∧ α < π ∧
  π / 2 < β ∧ β < π ∧
  (1 - Real.cos (2 * α)) * (1 + Real.sin β) = Real.sin (2 * α) * Real.cos β →
  2 * α + β = 5 * π / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_relation_l3265_326594


namespace NUMINAMATH_CALUDE_function_value_at_two_l3265_326574

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x ≠ 0, f x - 3 * f (1 / x) = 3^x

theorem function_value_at_two
  (f : ℝ → ℝ) (h : FunctionalEquation f) :
  f 2 = -(9 + 3 * Real.sqrt 3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l3265_326574


namespace NUMINAMATH_CALUDE_coefficient_x5_eq_11_l3265_326535

/-- The coefficient of x^5 in the expansion of (x^2 + x - 1)^5 -/
def coefficient_x5 : ℤ :=
  (Nat.choose 5 0) * (Nat.choose 5 5) -
  (Nat.choose 5 1) * (Nat.choose 4 3) +
  (Nat.choose 5 2) * (Nat.choose 3 1)

/-- Theorem stating that the coefficient of x^5 in (x^2 + x - 1)^5 is 11 -/
theorem coefficient_x5_eq_11 : coefficient_x5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x5_eq_11_l3265_326535


namespace NUMINAMATH_CALUDE_range_of_K_l3265_326557

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 2| - 1
def g (x : ℝ) : ℝ := |3 - x| + 2

-- Define the theorem
theorem range_of_K (K : ℝ) : 
  (∀ x : ℝ, f x - g x ≤ K) → K ∈ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_K_l3265_326557


namespace NUMINAMATH_CALUDE_boys_ratio_l3265_326526

theorem boys_ratio (total : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total = boys + girls) 
  (h2 : boys > 0 ∧ girls > 0) 
  (h3 : (boys : ℚ) / total = 3/5 * (girls : ℚ) / total) : 
  (boys : ℚ) / total = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_boys_ratio_l3265_326526


namespace NUMINAMATH_CALUDE_repeating_decimal_726_eq_fraction_l3265_326577

/-- The definition of a repeating decimal with period 726 -/
def repeating_decimal_726 : ℚ :=
  726 / 999

/-- Theorem stating that 0.726726726... equals 242/333 -/
theorem repeating_decimal_726_eq_fraction : repeating_decimal_726 = 242 / 333 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_726_eq_fraction_l3265_326577


namespace NUMINAMATH_CALUDE_f_neither_odd_nor_even_l3265_326544

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2

-- Define the domain of f
def domain : Set ℝ := Set.Ioc (-5) 5

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Theorem statement
theorem f_neither_odd_nor_even :
  ¬(is_odd f) ∧ ¬(is_even f) :=
sorry

end NUMINAMATH_CALUDE_f_neither_odd_nor_even_l3265_326544


namespace NUMINAMATH_CALUDE_infinite_cube_square_triples_l3265_326566

theorem infinite_cube_square_triples :
  ∃ S : Set (ℤ × ℤ × ℤ), Set.Infinite S ∧ 
  ∀ (x y z : ℤ), (x, y, z) ∈ S → x^2 + y^2 + z^2 = x^3 + y^3 + z^3 :=
by
  sorry

end NUMINAMATH_CALUDE_infinite_cube_square_triples_l3265_326566


namespace NUMINAMATH_CALUDE_angle_ADC_measure_l3265_326564

theorem angle_ADC_measure (ABC BAC BCA BAD CAD ACD BCD ADC : Real) : 
  ABC = 60 →
  BAC = BAD + CAD →
  BAD = CAD →
  BCA = ACD + BCD →
  ACD = 2 * BCD →
  BAC + ABC + BCA = 180 →
  CAD + ACD + ADC = 180 →
  ADC = 100 :=
by sorry

end NUMINAMATH_CALUDE_angle_ADC_measure_l3265_326564


namespace NUMINAMATH_CALUDE_dissected_rectangle_perimeter_l3265_326525

/-- A rectangle dissected into nine non-overlapping squares -/
structure DissectedRectangle where
  width : ℕ+
  height : ℕ+
  squares : Fin 9 → ℕ+
  sum_squares : width * height = (squares 0).val + (squares 1).val + (squares 2).val + (squares 3).val + 
                                 (squares 4).val + (squares 5).val + (squares 6).val + (squares 7).val + 
                                 (squares 8).val

/-- The perimeter of a rectangle -/
def perimeter (rect : DissectedRectangle) : ℕ :=
  2 * (rect.width + rect.height)

/-- The theorem to be proved -/
theorem dissected_rectangle_perimeter (rect : DissectedRectangle) 
  (h_coprime : Nat.Coprime rect.width rect.height) : 
  perimeter rect = 260 := by
  sorry

end NUMINAMATH_CALUDE_dissected_rectangle_perimeter_l3265_326525


namespace NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l3265_326503

theorem cos_sum_of_complex_exponentials (γ δ : ℝ) :
  Complex.exp (γ * Complex.I) = (4 / 5 : ℂ) + (3 / 5 : ℂ) * Complex.I →
  Complex.exp (δ * Complex.I) = -(5 / 13 : ℂ) + (12 / 13 : ℂ) * Complex.I →
  Real.cos (γ + δ) = -(56 / 65) := by
sorry

end NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l3265_326503


namespace NUMINAMATH_CALUDE_book_pages_l3265_326552

/-- A book with a certain number of pages -/
structure Book where
  pages : ℕ

/-- Reading progress over four days -/
structure ReadingProgress where
  day1 : Rat
  day2 : Rat
  day3 : Rat
  day4 : ℕ

/-- Theorem stating the total number of pages in the book -/
theorem book_pages (b : Book) (rp : ReadingProgress) 
  (h1 : rp.day1 = 1/2)
  (h2 : rp.day2 = 1/4)
  (h3 : rp.day3 = 1/6)
  (h4 : rp.day4 = 20)
  (h5 : rp.day1 + rp.day2 + rp.day3 + (rp.day4 : Rat) / b.pages = 1) :
  b.pages = 240 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_l3265_326552


namespace NUMINAMATH_CALUDE_sin_decreasing_omega_range_l3265_326537

theorem sin_decreasing_omega_range (ω : ℝ) (h_pos : ω > 0) :
  (∀ x ∈ Set.Icc (π / 4) (π / 2), 
    ∀ y ∈ Set.Icc (π / 4) (π / 2), 
    x < y → Real.sin (ω * x) > Real.sin (ω * y)) →
  ω ∈ Set.Icc 2 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_decreasing_omega_range_l3265_326537


namespace NUMINAMATH_CALUDE_key_chain_profit_percentage_l3265_326539

theorem key_chain_profit_percentage 
  (selling_price : ℝ)
  (old_cost new_cost : ℝ)
  (h1 : old_cost = 65)
  (h2 : new_cost = 50)
  (h3 : selling_price - new_cost = 0.5 * selling_price) :
  (selling_price - old_cost) / selling_price = 0.35 :=
by sorry

end NUMINAMATH_CALUDE_key_chain_profit_percentage_l3265_326539


namespace NUMINAMATH_CALUDE_distinct_cube_constructions_proof_l3265_326543

/-- The number of distinct ways to construct a 2 × 2 × 2 cube 
    using 6 white unit cubes and 2 black unit cubes, 
    where constructions are considered the same if one can be rotated to match the other -/
def distinct_cube_constructions : ℕ := 3

/-- The total number of unit cubes used -/
def total_cubes : ℕ := 8

/-- The number of white unit cubes -/
def white_cubes : ℕ := 6

/-- The number of black unit cubes -/
def black_cubes : ℕ := 2

/-- The dimensions of the cube -/
def cube_dimensions : Fin 3 → ℕ := λ _ => 2

/-- The order of the rotational symmetry group of a cube -/
def cube_symmetry_order : ℕ := 24

theorem distinct_cube_constructions_proof :
  distinct_cube_constructions = 3 ∧
  total_cubes = white_cubes + black_cubes ∧
  (∀ i, cube_dimensions i = 2) ∧
  cube_symmetry_order = 24 := by
  sorry

end NUMINAMATH_CALUDE_distinct_cube_constructions_proof_l3265_326543


namespace NUMINAMATH_CALUDE_simplify_expression_l3265_326599

theorem simplify_expression (x y : ℝ) : 
  (15 * x + 45 * y) + (20 * x + 58 * y) - (18 * x + 75 * y) = 17 * x + 28 * y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3265_326599


namespace NUMINAMATH_CALUDE_square_sequence_50th_term_l3265_326550

/-- Represents the number of squares in the nth figure of the sequence -/
def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- The theorem states that the 50th term of the sequence is 7651 -/
theorem square_sequence_50th_term :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 → f 50 = 7651 := by
  sorry

end NUMINAMATH_CALUDE_square_sequence_50th_term_l3265_326550


namespace NUMINAMATH_CALUDE_sets_intersection_union_l3265_326567

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2008*x - 2009 > 0}
def N (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- Define the open interval (2009, 2010]
def openInterval : Set ℝ := Set.Ioc 2009 2010

-- State the theorem
theorem sets_intersection_union (a b : ℝ) : 
  (M ∪ N a b = Set.univ) ∧ (M ∩ N a b = openInterval) → a = 2009 ∧ b = 2010 := by
  sorry

end NUMINAMATH_CALUDE_sets_intersection_union_l3265_326567


namespace NUMINAMATH_CALUDE_particle_position_at_1989_l3265_326527

/-- Represents the position of a particle on a 2D plane -/
structure Position :=
  (x : ℝ)
  (y : ℝ)

/-- Defines the movement pattern of the particle -/
def move (t : ℕ) : Position :=
  sorry

/-- The theorem to be proved -/
theorem particle_position_at_1989 :
  move 1989 = Position.mk 0 0 := by
  sorry

end NUMINAMATH_CALUDE_particle_position_at_1989_l3265_326527


namespace NUMINAMATH_CALUDE_specific_shaded_square_ratio_l3265_326523

/-- A square divided into smaller squares with a shading pattern -/
structure ShadedSquare where
  /-- The number of equal triangles in the shaded area of each quarter -/
  shaded_triangles : ℕ
  /-- The number of equal triangles in the white area of each quarter -/
  white_triangles : ℕ

/-- The ratio of shaded area to white area in a ShadedSquare -/
def shaded_to_white_ratio (s : ShadedSquare) : ℚ :=
  s.shaded_triangles / s.white_triangles

/-- Theorem stating the ratio of shaded to white area for a specific configuration -/
theorem specific_shaded_square_ratio :
  ∃ (s : ShadedSquare), s.shaded_triangles = 5 ∧ s.white_triangles = 3 ∧ 
  shaded_to_white_ratio s = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_specific_shaded_square_ratio_l3265_326523


namespace NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l3265_326518

-- Define the arithmetic sequence
def a (n : ℕ) : ℚ := 2 * n

-- Define the sequence b_n
def b (n : ℕ) : ℚ := 2^(n-1) * a n

-- Define the sum of the first n terms of b_n
def T (n : ℕ) : ℚ := (n - 1) * 2^(n + 1) + 2

-- State the theorem
theorem arithmetic_sequence_theorem (d : ℚ) (h_d : d ≠ 0) :
  (a 2 + 2 * a 4 = 20) ∧
  (∃ r : ℚ, a 3 = r * a 1 ∧ a 9 = r * a 3) →
  (∀ n : ℕ, n ≥ 1 → a n = 2 * n) ∧
  (∀ n : ℕ, T n = (n - 1) * 2^(n + 1) + 2) :=
by sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l3265_326518


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3265_326521

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_sum : ∀ n : ℕ, S n = n * (2 * a 1 + (n - 1) * d) / 2)
  (h_positive : a 1 > 0)
  (h_condition : a 9 + a 10 = a 11) :
  (d < 0) ∧
  (∀ n : ℕ, n > 14 → S n ≤ 0) ∧
  (∃ n : ℕ, n = 14 ∧ S n > 0) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3265_326521


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3265_326596

theorem complex_fraction_equality : Complex.I * 2 / (1 + Complex.I) = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3265_326596


namespace NUMINAMATH_CALUDE_students_with_B_grade_l3265_326565

def total_students : ℕ := 40

def prob_A (x : ℕ) : ℚ := (3 : ℚ) / 5 * x
def prob_B (x : ℕ) : ℚ := x
def prob_C (x : ℕ) : ℚ := (6 : ℚ) / 5 * x

theorem students_with_B_grade :
  ∃ x : ℕ, 
    x ≤ total_students ∧
    (prob_A x + prob_B x + prob_C x : ℚ) = total_students ∧
    x = 14 := by sorry

end NUMINAMATH_CALUDE_students_with_B_grade_l3265_326565


namespace NUMINAMATH_CALUDE_ruby_initial_apples_l3265_326573

/-- The number of apples Ruby has initially -/
def initial_apples : ℕ := sorry

/-- The number of apples Emily takes away -/
def apples_taken : ℕ := 55

/-- The number of apples Ruby has left -/
def apples_left : ℕ := 8

/-- Theorem stating that Ruby's initial number of apples is 63 -/
theorem ruby_initial_apples : initial_apples = 63 := by sorry

end NUMINAMATH_CALUDE_ruby_initial_apples_l3265_326573


namespace NUMINAMATH_CALUDE_computers_fixed_right_away_l3265_326505

theorem computers_fixed_right_away (total : ℕ) (unfixable_percent : ℚ) (spare_parts_percent : ℚ) :
  total = 20 →
  unfixable_percent = 20 / 100 →
  spare_parts_percent = 40 / 100 →
  (total : ℚ) * (1 - unfixable_percent - spare_parts_percent) = 8 := by
  sorry

end NUMINAMATH_CALUDE_computers_fixed_right_away_l3265_326505


namespace NUMINAMATH_CALUDE_solution_value_l3265_326586

theorem solution_value (x a : ℝ) (h : x = 5 ∧ a * x - 8 = 20 + a) : a = 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3265_326586


namespace NUMINAMATH_CALUDE_circle_k_bound_l3265_326506

/-- A circle in the Cartesian plane --/
structure Circle where
  equation : ℝ → ℝ → ℝ → Prop

/-- The equation x^2 + y^2 - 2x + y + k = 0 represents a circle --/
def isCircle (k : ℝ) : Prop :=
  ∃ (c : Circle), ∀ (x y : ℝ), c.equation x y k ↔ x^2 + y^2 - 2*x + y + k = 0

/-- If x^2 + y^2 - 2x + y + k = 0 is the equation of a circle, then k < 5/4 --/
theorem circle_k_bound (k : ℝ) : isCircle k → k < 5/4 := by
  sorry

end NUMINAMATH_CALUDE_circle_k_bound_l3265_326506


namespace NUMINAMATH_CALUDE_fencing_cost_per_meter_l3265_326580

/-- Proves that the fencing cost per meter is 60 cents for a rectangular park with given conditions -/
theorem fencing_cost_per_meter (length width : ℝ) (area perimeter total_cost : ℝ) : 
  length / width = 3 / 2 →
  area = 3750 →
  area = length * width →
  perimeter = 2 * (length + width) →
  total_cost = 150 →
  (total_cost / perimeter) * 100 = 60 :=
by sorry

end NUMINAMATH_CALUDE_fencing_cost_per_meter_l3265_326580


namespace NUMINAMATH_CALUDE_intersection_segment_equals_incircle_diameter_l3265_326570

/-- Right triangle with incircle and two circles on hypotenuse endpoints -/
structure RightTriangleWithCircles where
  -- Legs of the right triangle
  a : ℝ
  b : ℝ
  -- Hypotenuse of the right triangle
  c : ℝ
  -- Radius of the incircle
  r : ℝ
  -- The triangle is right-angled
  right_angle : a^2 + b^2 = c^2
  -- The incircle exists and touches all sides
  incircle : a + b - c = 2 * r
  -- All lengths are positive
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0
  r_pos : r > 0

/-- The length of the intersection segment equals the incircle diameter -/
theorem intersection_segment_equals_incircle_diameter 
  (t : RightTriangleWithCircles) : a + b - c = 2 * r :=
by sorry

end NUMINAMATH_CALUDE_intersection_segment_equals_incircle_diameter_l3265_326570


namespace NUMINAMATH_CALUDE_tram_speed_l3265_326551

/-- Given a tram passing an observer in 2 seconds and traversing a 96-meter tunnel in 10 seconds
    at a constant speed, the speed of the tram is 12 meters per second. -/
theorem tram_speed (passing_time : ℝ) (tunnel_length : ℝ) (tunnel_time : ℝ)
    (h1 : passing_time = 2)
    (h2 : tunnel_length = 96)
    (h3 : tunnel_time = 10) :
  ∃ (v : ℝ), v = 12 ∧ v * passing_time = v * 2 ∧ v * tunnel_time = v * 2 + tunnel_length :=
by
  sorry

#check tram_speed

end NUMINAMATH_CALUDE_tram_speed_l3265_326551


namespace NUMINAMATH_CALUDE_unique_function_property_l3265_326546

def iterateFunc (f : ℕ → ℕ) : ℕ → ℕ → ℕ
| 0, x => x
| (n + 1), x => f (iterateFunc f n x)

theorem unique_function_property (f : ℕ → ℕ) 
  (h : ∀ x y : ℕ, 0 ≤ y + f x - iterateFunc f (f y) x ∧ y + f x - iterateFunc f (f y) x ≤ 1) :
  ∀ n : ℕ, f n = n + 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_function_property_l3265_326546


namespace NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l3265_326522

theorem polynomial_sum_of_coefficients 
  (g : ℂ → ℂ) 
  (p q r s : ℝ) 
  (h1 : ∀ x, g x = x^4 + p*x^3 + q*x^2 + r*x + s)
  (h2 : g (3*I) = 0)
  (h3 : g (1 + 2*I) = 0) :
  p + q + r + s = 39 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l3265_326522


namespace NUMINAMATH_CALUDE_stations_visited_l3265_326556

theorem stations_visited (total_nails : ℕ) (nails_per_station : ℕ) (h1 : total_nails = 140) (h2 : nails_per_station = 7) :
  total_nails / nails_per_station = 20 := by
  sorry

end NUMINAMATH_CALUDE_stations_visited_l3265_326556


namespace NUMINAMATH_CALUDE_initial_angelfish_count_l3265_326583

/-- The number of fish initially in the tank -/
def initial_fish (angelfish : ℕ) : ℕ := 94 + angelfish + 89 + 58

/-- The number of fish sold -/
def sold_fish (angelfish : ℕ) : ℕ := 30 + 48 + 17 + 24

/-- The number of fish remaining after the sale -/
def remaining_fish (angelfish : ℕ) : ℕ := initial_fish angelfish - sold_fish angelfish

theorem initial_angelfish_count :
  ∃ (angelfish : ℕ), initial_fish angelfish > 0 ∧ remaining_fish angelfish = 198 ∧ angelfish = 76 := by
  sorry

end NUMINAMATH_CALUDE_initial_angelfish_count_l3265_326583


namespace NUMINAMATH_CALUDE_completing_square_result_l3265_326561

theorem completing_square_result (x : ℝ) : x^2 + 4*x + 3 = 0 ↔ (x + 2)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_result_l3265_326561


namespace NUMINAMATH_CALUDE_planted_field_fraction_l3265_326578

theorem planted_field_fraction (a b c x : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_legs : a = 6 ∧ b = 8) (h_distance : (a - 0.6*x) * (b - 0.8*x) / 2 = 3) :
  (a * b / 2 - x^2) / (a * b / 2) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_planted_field_fraction_l3265_326578


namespace NUMINAMATH_CALUDE_parabola_properties_l3265_326516

-- Define the parabola function
def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 4

-- Define the interval
def interval : Set ℝ := Set.Icc (-3) 2

-- Theorem statement
theorem parabola_properties :
  (∃ (x y : ℝ), x = -1 ∧ y = 1 ∧ ∀ (t : ℝ), f t ≥ f x) ∧ 
  (∀ (x₁ x₂ : ℝ), f x₁ = f x₂ → |x₁ + 1| = |x₂ + 1|) ∧
  (Set.Icc 1 28 = {y | ∃ x ∈ interval, f x = y}) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l3265_326516


namespace NUMINAMATH_CALUDE_average_weight_increase_l3265_326515

/-- Theorem: Increase in average weight when replacing a person in a group -/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 6 * initial_average
  let final_total := initial_total - 75 + 102
  let final_average := final_total / 6
  final_average - initial_average = 4.5 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3265_326515


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3265_326563

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x > 1}
def B : Set ℝ := {x : ℝ | x - 4 < 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x > 1} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3265_326563


namespace NUMINAMATH_CALUDE_conditions_implications_l3265_326542

-- Define the conditions
def p (a : ℝ) : Prop := ∀ x > 0, Monotone (fun x => Real.log x / Real.log (a - 1))
def q (a : ℝ) : Prop := (2 - a) / (a - 3) > 0
def r (a : ℝ) : Prop := a < 3
def s (a : ℝ) : Prop := ∀ x, ∃ y, y = Real.log (x^2 - 2*x + a) / Real.log 10

-- State the theorem
theorem conditions_implications (a : ℝ) :
  (p a → a > 2) ∧
  (q a ↔ (2 < a ∧ a < 3)) ∧
  (s a → a > 1) ∧
  ((p a → q a) ∧ ¬(q a → p a)) ∧
  ((r a → q a) ∧ ¬(q a → r a)) :=
sorry

end NUMINAMATH_CALUDE_conditions_implications_l3265_326542


namespace NUMINAMATH_CALUDE_no_integer_fourth_root_l3265_326501

theorem no_integer_fourth_root : ¬∃ (n : ℕ), n > 0 ∧ 5^4 + 12^4 + 9^4 + 8^4 = n^4 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_fourth_root_l3265_326501


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l3265_326593

/-- Given an arithmetic sequence {a_n} where a_2 + a_4 = 5, prove that a_3 = 5/2 -/
theorem arithmetic_sequence_third_term 
  (a : ℕ → ℚ) -- a is a function from natural numbers to rationals
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) -- arithmetic sequence condition
  (h_sum : a 2 + a 4 = 5) -- given condition
  : a 3 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l3265_326593


namespace NUMINAMATH_CALUDE_remainder_3456_div_97_l3265_326589

theorem remainder_3456_div_97 : 3456 % 97 = 61 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3456_div_97_l3265_326589


namespace NUMINAMATH_CALUDE_soccer_goals_mean_l3265_326517

theorem soccer_goals_mean (players_3goals : ℕ) (players_4goals : ℕ) (players_5goals : ℕ) (players_6goals : ℕ) 
  (h1 : players_3goals = 4)
  (h2 : players_4goals = 3)
  (h3 : players_5goals = 1)
  (h4 : players_6goals = 2) :
  let total_goals := 3 * players_3goals + 4 * players_4goals + 5 * players_5goals + 6 * players_6goals
  let total_players := players_3goals + players_4goals + players_5goals + players_6goals
  (total_goals : ℚ) / total_players = 4.1 := by
  sorry

end NUMINAMATH_CALUDE_soccer_goals_mean_l3265_326517


namespace NUMINAMATH_CALUDE_crackers_distribution_l3265_326547

theorem crackers_distribution (total_crackers : ℕ) (num_friends : ℕ) (crackers_per_friend : ℕ) :
  total_crackers = 81 →
  num_friends = 27 →
  total_crackers = num_friends * crackers_per_friend →
  crackers_per_friend = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_crackers_distribution_l3265_326547


namespace NUMINAMATH_CALUDE_mingming_calculation_correction_l3265_326595

theorem mingming_calculation_correction : 
  (-4 - 2/3) - (1 + 5/6) - (-18 - 1/2) + (-13 - 3/4) = -7/4 := by sorry

end NUMINAMATH_CALUDE_mingming_calculation_correction_l3265_326595


namespace NUMINAMATH_CALUDE_nickel_chocolates_l3265_326554

theorem nickel_chocolates (robert : ℕ) (difference : ℕ) (nickel : ℕ) : 
  robert = 13 → 
  robert = nickel + difference → 
  difference = 9 → 
  nickel = 4 := by sorry

end NUMINAMATH_CALUDE_nickel_chocolates_l3265_326554


namespace NUMINAMATH_CALUDE_annual_reduction_equation_l3265_326510

/-- The total cost reduction percentage over two years -/
def total_reduction : ℝ := 0.36

/-- The average annual reduction percentage -/
def x : ℝ := sorry

/-- Theorem stating the relationship between the average annual reduction and total reduction -/
theorem annual_reduction_equation : (1 - x)^2 = 1 - total_reduction := by sorry

end NUMINAMATH_CALUDE_annual_reduction_equation_l3265_326510


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l3265_326562

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 17) : 
  x^3 + y^3 = 65 := by sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l3265_326562


namespace NUMINAMATH_CALUDE_point_coordinates_l3265_326534

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of the Cartesian coordinate system -/
def fourth_quadrant (p : Point) : Prop := p.x > 0 ∧ p.y < 0

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ := |p.y|

/-- The distance from a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ := |p.x|

theorem point_coordinates :
  ∀ (A : Point),
    fourth_quadrant A →
    distance_to_x_axis A = 3 →
    distance_to_y_axis A = 6 →
    A.x = 6 ∧ A.y = -3 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l3265_326534


namespace NUMINAMATH_CALUDE_great_dane_weight_is_307_l3265_326540

/-- The weight of three dogs (chihuahua, pitbull, and great dane) -/
def total_weight : ℕ := 439

/-- The weight relationship between the pitbull and the chihuahua -/
def pitbull_weight (chihuahua_weight : ℕ) : ℕ := 3 * chihuahua_weight

/-- The weight relationship between the great dane and the pitbull -/
def great_dane_weight (pitbull_weight : ℕ) : ℕ := 3 * pitbull_weight + 10

/-- Theorem stating that the great dane weighs 307 pounds -/
theorem great_dane_weight_is_307 :
  ∃ (chihuahua_weight : ℕ),
    chihuahua_weight + pitbull_weight chihuahua_weight + great_dane_weight (pitbull_weight chihuahua_weight) = total_weight ∧
    great_dane_weight (pitbull_weight chihuahua_weight) = 307 :=
by
  sorry

end NUMINAMATH_CALUDE_great_dane_weight_is_307_l3265_326540


namespace NUMINAMATH_CALUDE_function_decomposition_even_odd_l3265_326569

theorem function_decomposition_even_odd (f : ℝ → ℝ) :
  ∃! (f₀ f₁ : ℝ → ℝ),
    (∀ x, f x = f₀ x + f₁ x) ∧
    (∀ x, f₀ (-x) = f₀ x) ∧
    (∀ x, f₁ (-x) = -f₁ x) ∧
    (∀ x, f₀ x = (1/2) * (f x + f (-x))) ∧
    (∀ x, f₁ x = (1/2) * (f x - f (-x))) := by
  sorry

end NUMINAMATH_CALUDE_function_decomposition_even_odd_l3265_326569


namespace NUMINAMATH_CALUDE_problem_one_problem_two_problem_three_problem_four_l3265_326571

-- Problem 1
theorem problem_one : (-23) - (-58) + (-17) = 18 := by sorry

-- Problem 2
theorem problem_two : (-8) / (-1 - 1/9) * 0.125 = 9/10 := by sorry

-- Problem 3
theorem problem_three : (-1/3 - 1/4 + 1/15) * (-60) = 31 := by sorry

-- Problem 4
theorem problem_four : -1^2 * |(-1/4)| + (-1/2)^3 / (-1)^2023 = -1/8 := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_problem_three_problem_four_l3265_326571


namespace NUMINAMATH_CALUDE_not_cyclically_symmetric_example_cyclically_symmetric_example_difference_of_cyclically_symmetric_triangle_angles_cyclically_symmetric_l3265_326509

-- Definition of cyclically symmetric function
def CyclicallySymmetric (f : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c, f a b c = f b c a ∧ f a b c = f c a b

-- Statement 1
theorem not_cyclically_symmetric_example :
  ¬CyclicallySymmetric (fun x y z => x^2 - y^2 + z) := by sorry

-- Statement 2
theorem cyclically_symmetric_example :
  CyclicallySymmetric (fun x y z => x^2*(y-z) + y^2*(z-x) + z^2*(x-y)) := by sorry

-- Statement 3
theorem difference_of_cyclically_symmetric (f g : ℝ → ℝ → ℝ → ℝ) :
  CyclicallySymmetric f → CyclicallySymmetric g →
  CyclicallySymmetric (fun x y z => f x y z - g x y z) := by sorry

-- Statement 4
theorem triangle_angles_cyclically_symmetric (A B C : ℝ) :
  A + B + C = π →
  CyclicallySymmetric (fun x y z => 2 + Real.cos z * Real.cos (x-y) - Real.cos z^2) := by sorry

end NUMINAMATH_CALUDE_not_cyclically_symmetric_example_cyclically_symmetric_example_difference_of_cyclically_symmetric_triangle_angles_cyclically_symmetric_l3265_326509


namespace NUMINAMATH_CALUDE_a_range_l3265_326524

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 4*x₀ + a = 0

theorem a_range (a : ℝ) (h : p a ∧ q a) : a ∈ Set.Icc (Real.exp 1) 4 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l3265_326524


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l3265_326504

theorem square_garden_perimeter (q p : ℝ) (h1 : q > 0) (h2 : p > 0) :
  q = p + 21 → p = 28 := by
  sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l3265_326504


namespace NUMINAMATH_CALUDE_total_packs_bought_l3265_326575

/-- The number of index card packs John buys for each student -/
def packs_per_student : ℕ := 2

/-- The number of classes John has -/
def num_classes : ℕ := 6

/-- The number of students in each of John's classes -/
def students_per_class : ℕ := 30

/-- Theorem: John buys 360 packs of index cards in total -/
theorem total_packs_bought : packs_per_student * num_classes * students_per_class = 360 := by
  sorry

end NUMINAMATH_CALUDE_total_packs_bought_l3265_326575


namespace NUMINAMATH_CALUDE_inequality_proof_l3265_326500

theorem inequality_proof (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_product : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3265_326500


namespace NUMINAMATH_CALUDE_ac_range_l3265_326585

-- Define the triangle
def Triangle (A B C : ℝ × ℝ) : Prop := sorry

-- Define the angle at a vertex
def Angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the length of a side
def SideLength (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ac_range (A B C : ℝ × ℝ) : 
  Triangle A B C → 
  Angle A B C < π / 2 → 
  Angle B C A < π / 2 → 
  Angle C A B < π / 2 → 
  SideLength B C = 1 → 
  Angle B A C = 2 * Angle A B C → 
  Real.sqrt 2 < SideLength A C ∧ SideLength A C < Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ac_range_l3265_326585


namespace NUMINAMATH_CALUDE_total_relaxing_is_66_l3265_326584

/-- Calculates the number of people remaining in a row after some leave --/
def remainingInRow (initial : ℕ) (leaving : ℕ) : ℕ :=
  if initial ≥ leaving then initial - leaving else 0

/-- Represents the beach scenario with 5 rows of people --/
structure BeachScenario where
  row1_initial : ℕ
  row1_leaving : ℕ
  row2_initial : ℕ
  row2_leaving : ℕ
  row3_initial : ℕ
  row3_leaving : ℕ
  row4_initial : ℕ
  row4_leaving : ℕ
  row5_initial : ℕ
  row5_leaving : ℕ

/-- Calculates the total number of people still relaxing on the beach --/
def totalRelaxing (scenario : BeachScenario) : ℕ :=
  remainingInRow scenario.row1_initial scenario.row1_leaving +
  remainingInRow scenario.row2_initial scenario.row2_leaving +
  remainingInRow scenario.row3_initial scenario.row3_leaving +
  remainingInRow scenario.row4_initial scenario.row4_leaving +
  remainingInRow scenario.row5_initial scenario.row5_leaving

/-- The given beach scenario --/
def givenScenario : BeachScenario :=
  { row1_initial := 24, row1_leaving := 7
  , row2_initial := 20, row2_leaving := 7
  , row3_initial := 18, row3_leaving := 2
  , row4_initial := 16, row4_leaving := 11
  , row5_initial := 30, row5_leaving := 15 }

/-- Theorem stating that the total number of people still relaxing is 66 --/
theorem total_relaxing_is_66 : totalRelaxing givenScenario = 66 := by
  sorry

end NUMINAMATH_CALUDE_total_relaxing_is_66_l3265_326584


namespace NUMINAMATH_CALUDE_cubic_root_complex_coefficients_l3265_326592

theorem cubic_root_complex_coefficients :
  ∀ (a b : ℝ),
  (∃ (x : ℂ), x^3 + a*x^2 + 2*x + b = 0 ∧ x = Complex.mk 2 (-3)) →
  a = -5/4 ∧ b = 143/4 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_complex_coefficients_l3265_326592


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3265_326549

theorem max_value_of_expression :
  ∃ (a b c d : ℕ),
    ({a, b, c, d} : Finset ℕ) = {1, 2, 4, 5} →
    ∀ (w x y z : ℕ),
      ({w, x, y, z} : Finset ℕ) = {1, 2, 4, 5} →
      c * a^b - d ≤ 79 ∧
      (c * a^b - d = 79 → (a = 2 ∧ b = 4 ∧ c = 5 ∧ d = 1) ∨ (a = 4 ∧ b = 2 ∧ c = 5 ∧ d = 1)) :=
by sorry

#check max_value_of_expression

end NUMINAMATH_CALUDE_max_value_of_expression_l3265_326549


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l3265_326520

theorem chess_tournament_participants (n m : ℕ) : 
  9 < n → n < 25 →  -- Total participants between 9 and 25
  (n - 2*m)^2 = n →  -- Derived equation from the condition about scoring half points against grandmasters
  (n = 16 ∧ (m = 6 ∨ m = 10)) := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l3265_326520


namespace NUMINAMATH_CALUDE_high_schooler_pairs_l3265_326559

theorem high_schooler_pairs (n : ℕ) (h : 10 ≤ n ∧ n ≤ 15) : 
  (∀ m : ℕ, 10 ≤ m ∧ m ≤ 15 → n * (n - 1) / 2 ≤ m * (m - 1) / 2) → n = 10 ∧
  (∀ m : ℕ, 10 ≤ m ∧ m ≤ 15 → m * (m - 1) / 2 ≤ n * (n - 1) / 2) → n = 15 :=
by sorry

end NUMINAMATH_CALUDE_high_schooler_pairs_l3265_326559


namespace NUMINAMATH_CALUDE_cube_difference_l3265_326541

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 53) :
  a^3 - b^3 = 385 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l3265_326541


namespace NUMINAMATH_CALUDE_basketball_team_composition_l3265_326530

-- Define the number of classes
def num_classes : ℕ := 8

-- Define the total number of players
def total_players : ℕ := 10

-- Define the function to calculate the number of composition methods
def composition_methods (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) k

-- Theorem statement
theorem basketball_team_composition :
  composition_methods (num_classes) (total_players - num_classes) = 36 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_composition_l3265_326530


namespace NUMINAMATH_CALUDE_larger_number_proof_l3265_326508

theorem larger_number_proof (L S : ℕ) (hL : L > S) :
  L - S = 1365 → L = 6 * S + 15 → L = 1635 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3265_326508


namespace NUMINAMATH_CALUDE_complex_magnitude_and_real_part_sum_of_combinations_l3265_326548

-- Problem 1
theorem complex_magnitude_and_real_part (z : ℂ) (ω : ℝ) 
  (h1 : ω = z + 1/z)
  (h2 : -1 < ω)
  (h3 : ω < 2) :
  Complex.abs z = 1 ∧ ∃ (a : ℝ), z.re = a ∧ -1/2 < a ∧ a < 1 :=
sorry

-- Problem 2
theorem sum_of_combinations : 
  (Nat.choose 5 4) + (Nat.choose 6 4) + (Nat.choose 7 4) + 
  (Nat.choose 8 4) + (Nat.choose 9 4) + (Nat.choose 10 4) = 461 :=
sorry

end NUMINAMATH_CALUDE_complex_magnitude_and_real_part_sum_of_combinations_l3265_326548


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l3265_326528

theorem binomial_expansion_properties (n : ℕ) :
  (∀ x : ℝ, x > 0 → 
    Nat.choose n 2 = Nat.choose n 6) →
  (n = 8 ∧ 
   ∀ k : ℕ, k ≤ n → (8 : ℝ) - (3 / 2 : ℝ) * k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l3265_326528


namespace NUMINAMATH_CALUDE_letter_F_transformation_l3265_326590

-- Define the position of the letter F
structure LetterFPosition where
  base : (ℝ × ℝ) -- Base endpoint
  top : (ℝ × ℝ)  -- Top endpoint

-- Define the transformations
def reflectXAxis (p : LetterFPosition) : LetterFPosition :=
  { base := (p.base.1, -p.base.2), top := (p.top.1, -p.top.2) }

def rotateCounterClockwise90 (p : LetterFPosition) : LetterFPosition :=
  { base := (-p.base.2, p.base.1), top := (-p.top.2, p.top.1) }

def rotate180 (p : LetterFPosition) : LetterFPosition :=
  { base := (-p.base.1, -p.base.2), top := (-p.top.1, -p.top.2) }

def reflectYAxis (p : LetterFPosition) : LetterFPosition :=
  { base := (-p.base.1, p.base.2), top := (-p.top.1, p.top.2) }

-- Define the initial position
def initialPosition : LetterFPosition :=
  { base := (0, -1), top := (1, 0) }

-- Define the final position
def finalPosition : LetterFPosition :=
  { base := (1, 0), top := (0, 1) }

-- Theorem statement
theorem letter_F_transformation :
  (reflectYAxis ∘ rotate180 ∘ rotateCounterClockwise90 ∘ reflectXAxis) initialPosition = finalPosition := by
  sorry

end NUMINAMATH_CALUDE_letter_F_transformation_l3265_326590


namespace NUMINAMATH_CALUDE_circle_area_through_two_points_l3265_326514

/-- The area of a circle with center P(2, -1) passing through Q(-4, 5) is 72π. -/
theorem circle_area_through_two_points :
  let P : ℝ × ℝ := (2, -1)
  let Q : ℝ × ℝ := (-4, 5)
  let r := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  π * r^2 = 72 * π := by
  sorry


end NUMINAMATH_CALUDE_circle_area_through_two_points_l3265_326514
