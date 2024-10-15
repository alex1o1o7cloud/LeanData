import Mathlib

namespace NUMINAMATH_CALUDE_f_derivative_at_negative_one_l794_79490

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

-- State the theorem
theorem f_derivative_at_negative_one (a b c : ℝ) :
  f' a b 1 = 2 → f' a b (-1) = -2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_negative_one_l794_79490


namespace NUMINAMATH_CALUDE_total_ants_employed_l794_79429

/-- The total number of ants employed for all construction tasks -/
def total_ants (carrying_red carrying_black digging_red digging_black assembling_red assembling_black : ℕ) : ℕ :=
  carrying_red + carrying_black + digging_red + digging_black + assembling_red + assembling_black

/-- Theorem stating that the total number of ants employed is 2464 -/
theorem total_ants_employed :
  total_ants 413 487 356 518 298 392 = 2464 := by
  sorry

#eval total_ants 413 487 356 518 298 392

end NUMINAMATH_CALUDE_total_ants_employed_l794_79429


namespace NUMINAMATH_CALUDE_sum_interior_angles_hexagon_l794_79471

/-- The sum of interior angles of a hexagon is 720 degrees. -/
theorem sum_interior_angles_hexagon :
  ∀ (n : ℕ) (sum_interior_angles : ℕ → ℝ),
  n = 6 →
  (∀ k : ℕ, sum_interior_angles k = (k - 2) * 180) →
  sum_interior_angles n = 720 := by
sorry

end NUMINAMATH_CALUDE_sum_interior_angles_hexagon_l794_79471


namespace NUMINAMATH_CALUDE_coefficient_of_x_power_5_l794_79447

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient function for the expansion of (x - 1/√x)^8
def coefficient (r : ℕ) : ℤ :=
  (-1)^r * (binomial 8 r)

-- Theorem statement
theorem coefficient_of_x_power_5 : coefficient 2 = 28 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_power_5_l794_79447


namespace NUMINAMATH_CALUDE_function_difference_theorem_l794_79427

theorem function_difference_theorem (f g : ℝ → ℝ) (k : ℝ) : 
  (∀ x, f x = 3 * x^2 - 1/x + 5) →
  (∀ x, g x = 2 * x^2 - k) →
  f 3 - g 3 = 6 →
  k = -23/3 := by sorry

end NUMINAMATH_CALUDE_function_difference_theorem_l794_79427


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l794_79444

theorem right_triangle_perimeter (base height : ℝ) (h_base : base = 4) (h_height : height = 3) :
  let hypotenuse := Real.sqrt (base^2 + height^2)
  base + height + hypotenuse = 12 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l794_79444


namespace NUMINAMATH_CALUDE_power_equation_solution_l794_79448

theorem power_equation_solution (a : ℝ) (k : ℝ) (h1 : a ≠ 0) : 
  (a^10 / (a^k)^4 = a^2) → k = 2 := by
sorry

end NUMINAMATH_CALUDE_power_equation_solution_l794_79448


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l794_79415

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop :=
  x^2 + 3*x*y + 2*y^2 - 14*x - 21*y + 49 = 0

-- Define the function to be maximized
def f (x y : ℝ) : ℝ := x + y

-- Theorem statement
theorem max_value_on_ellipse :
  ∃ (x₀ y₀ : ℝ), ellipse x₀ y₀ ∧
  (∀ (x y : ℝ), ellipse x y → f x y ≤ f x₀ y₀) ∧
  f x₀ y₀ = 343 / 88 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l794_79415


namespace NUMINAMATH_CALUDE_f_properties_l794_79409

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

-- Theorem statement
theorem f_properties : 
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = 4) ∧ 
  (∀ (a : ℝ), (∃ (x : ℝ), f x < a) ↔ a > 4) ∧
  (∀ (a b : ℝ), (∀ (x : ℝ), f x < a ↔ b < x ∧ x < 7/2) → a + b = 3.5) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l794_79409


namespace NUMINAMATH_CALUDE_jack_walked_4_miles_l794_79472

/-- The distance Jack walked given his walking time and rate -/
def jack_distance (time_hours : ℝ) (rate : ℝ) : ℝ :=
  time_hours * rate

theorem jack_walked_4_miles :
  let time_hours : ℝ := 1.25  -- 1 hour and 15 minutes in decimal hours
  let rate : ℝ := 3.2         -- miles per hour
  jack_distance time_hours rate = 4 := by
sorry

end NUMINAMATH_CALUDE_jack_walked_4_miles_l794_79472


namespace NUMINAMATH_CALUDE_plane_trip_distance_l794_79456

/-- Proves that if a person takes a trip a certain number of times and travels a total distance,
    then the distance of each trip is the total distance divided by the number of trips. -/
theorem plane_trip_distance (num_trips : ℝ) (total_distance : ℝ) 
    (h1 : num_trips = 32) 
    (h2 : total_distance = 8192) : 
  total_distance / num_trips = 256 := by
  sorry

#check plane_trip_distance

end NUMINAMATH_CALUDE_plane_trip_distance_l794_79456


namespace NUMINAMATH_CALUDE_smallest_three_digit_non_divisor_l794_79431

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def factorial (n : ℕ) : ℕ := Nat.factorial n

def is_divisor (a b : ℕ) : Prop := b % a = 0

theorem smallest_three_digit_non_divisor :
  ∀ k : ℕ, 100 ≤ k → k < 101 →
    is_divisor (sum_of_squares k) (factorial k) →
    ¬ is_divisor (sum_of_squares 101) (factorial 101) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_non_divisor_l794_79431


namespace NUMINAMATH_CALUDE_exactly_four_even_probability_l794_79406

def num_dice : ℕ := 8
def num_even : ℕ := 4
def prob_even : ℚ := 2/3
def prob_odd : ℚ := 1/3

theorem exactly_four_even_probability :
  (Nat.choose num_dice num_even) * (prob_even ^ num_even) * (prob_odd ^ (num_dice - num_even)) = 1120/6561 := by
  sorry

end NUMINAMATH_CALUDE_exactly_four_even_probability_l794_79406


namespace NUMINAMATH_CALUDE_perpendicular_length_is_five_l794_79470

/-- Properties of a right triangle DEF with given side lengths -/
structure RightTriangle where
  DE : ℝ
  EF : ℝ
  is_right : DE = 5 ∧ EF = 12

/-- The length of the perpendicular from the hypotenuse to the midpoint of the angle bisector -/
def perpendicular_length (t : RightTriangle) : ℝ :=
  sorry

/-- Theorem: The perpendicular length is 5 -/
theorem perpendicular_length_is_five (t : RightTriangle) :
  perpendicular_length t = 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_length_is_five_l794_79470


namespace NUMINAMATH_CALUDE_largest_number_l794_79417

theorem largest_number (a b c d e : ℚ) 
  (ha : a = 995/1000) 
  (hb : b = 9995/10000) 
  (hc : c = 99/100) 
  (hd : d = 999/1000) 
  (he : e = 9959/10000) : 
  b > a ∧ b > c ∧ b > d ∧ b > e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l794_79417


namespace NUMINAMATH_CALUDE_chris_has_12_marbles_l794_79438

-- Define the number of marbles Chris and Ryan have
def chris_marbles : ℕ := sorry
def ryan_marbles : ℕ := 28

-- Define the total number of marbles in the pile
def total_marbles : ℕ := chris_marbles + ryan_marbles

-- Define the number of marbles remaining after they take their share
def remaining_marbles : ℕ := 20

-- Theorem stating that Chris has 12 marbles
theorem chris_has_12_marbles :
  chris_marbles = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_chris_has_12_marbles_l794_79438


namespace NUMINAMATH_CALUDE_vehicle_travel_time_l794_79488

/-- 
Given two vehicles A and B traveling towards each other, prove that B's total travel time is 7.2 hours
under the following conditions:
1. They meet after 3 hours.
2. A turns back to its starting point, taking 3 hours.
3. A then turns around again and meets B after 0.5 hours.
-/
theorem vehicle_travel_time (v_A v_B : ℝ) (d : ℝ) : 
  v_A > 0 ∧ v_B > 0 ∧ d > 0 → 
  d = 3 * (v_A + v_B) →
  3 * v_A = d / 2 →
  d / 2 + 0.5 * v_A = 3.5 * v_B →
  d / v_B = 7.2 := by
sorry

end NUMINAMATH_CALUDE_vehicle_travel_time_l794_79488


namespace NUMINAMATH_CALUDE_snooker_ticket_difference_l794_79413

theorem snooker_ticket_difference :
  ∀ (vip_tickets gen_tickets : ℕ),
    vip_tickets + gen_tickets = 320 →
    40 * vip_tickets + 15 * gen_tickets = 7500 →
    gen_tickets - vip_tickets = 104 := by
  sorry

end NUMINAMATH_CALUDE_snooker_ticket_difference_l794_79413


namespace NUMINAMATH_CALUDE_late_arrivals_count_l794_79439

/-- Represents the number of people per lollipop -/
def people_per_lollipop : ℕ := 5

/-- Represents the initial number of people -/
def initial_people : ℕ := 45

/-- Represents the total number of lollipops given away -/
def total_lollipops : ℕ := 12

/-- Calculates the number of people who came in later -/
def late_arrivals : ℕ := total_lollipops * people_per_lollipop - initial_people

theorem late_arrivals_count : late_arrivals = 15 := by
  sorry

end NUMINAMATH_CALUDE_late_arrivals_count_l794_79439


namespace NUMINAMATH_CALUDE_max_pages_for_budget_l794_79462

-- Define the cost per page in cents
def cost_per_page : ℕ := 3

-- Define the budget in dollars
def budget : ℕ := 25

-- Define the function to calculate the maximum number of pages
def max_pages (cost : ℕ) (budget : ℕ) : ℕ :=
  (budget * 100) / cost

-- Theorem statement
theorem max_pages_for_budget :
  max_pages cost_per_page budget = 833 := by
  sorry

end NUMINAMATH_CALUDE_max_pages_for_budget_l794_79462


namespace NUMINAMATH_CALUDE_kelly_carrots_l794_79405

/-- The number of carrots Kelly pulled out from the first bed -/
def carrots_first_bed (total_carrots second_bed third_bed : ℕ) : ℕ :=
  total_carrots - second_bed - third_bed

/-- Theorem stating the number of carrots Kelly pulled out from the first bed -/
theorem kelly_carrots :
  carrots_first_bed (39 * 6) 101 78 = 55 := by
  sorry

end NUMINAMATH_CALUDE_kelly_carrots_l794_79405


namespace NUMINAMATH_CALUDE_exponential_inequality_l794_79493

theorem exponential_inequality (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (1 / 3 : ℝ) ^ x < (1 / 3 : ℝ) ^ y := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l794_79493


namespace NUMINAMATH_CALUDE_robin_total_pieces_l794_79414

/-- The number of gum packages Robin has -/
def gum_packages : ℕ := 21

/-- The number of candy packages Robin has -/
def candy_packages : ℕ := 45

/-- The number of mint packages Robin has -/
def mint_packages : ℕ := 30

/-- The number of gum pieces in each gum package -/
def gum_pieces_per_package : ℕ := 9

/-- The number of candy pieces in each candy package -/
def candy_pieces_per_package : ℕ := 12

/-- The number of mint pieces in each mint package -/
def mint_pieces_per_package : ℕ := 8

/-- The total number of pieces Robin has -/
def total_pieces : ℕ := gum_packages * gum_pieces_per_package + 
                        candy_packages * candy_pieces_per_package + 
                        mint_packages * mint_pieces_per_package

theorem robin_total_pieces : total_pieces = 969 := by
  sorry

end NUMINAMATH_CALUDE_robin_total_pieces_l794_79414


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l794_79441

theorem sum_of_squares_of_roots (a b c : ℝ) : 
  (3 * a^3 + 2 * a^2 - 5 * a - 15 = 0) ∧
  (3 * b^3 + 2 * b^2 - 5 * b - 15 = 0) ∧
  (3 * c^3 + 2 * c^2 - 5 * c - 15 = 0) →
  a^2 + b^2 + c^2 = -26/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l794_79441


namespace NUMINAMATH_CALUDE_consecutive_numbers_digit_sum_exists_l794_79463

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem consecutive_numbers_digit_sum_exists : ∃ (n : ℕ), 
  sumOfDigits n = 52 ∧ 
  sumOfDigits (n + 4) = 20 ∧ 
  n > 0 :=
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_digit_sum_exists_l794_79463


namespace NUMINAMATH_CALUDE_tiger_tree_trunk_length_l794_79453

/-- The length of a fallen tree trunk over which a tiger runs --/
theorem tiger_tree_trunk_length (tiger_length : ℝ) (grass_time : ℝ) (trunk_time : ℝ)
  (h_length : tiger_length = 5)
  (h_grass : grass_time = 1)
  (h_trunk : trunk_time = 5) :
  tiger_length * trunk_time = 25 := by
  sorry

end NUMINAMATH_CALUDE_tiger_tree_trunk_length_l794_79453


namespace NUMINAMATH_CALUDE_brick_length_calculation_l794_79402

theorem brick_length_calculation (courtyard_length courtyard_width : ℝ)
  (brick_width : ℝ) (total_bricks : ℕ) (h1 : courtyard_length = 18)
  (h2 : courtyard_width = 16) (h3 : brick_width = 0.1)
  (h4 : total_bricks = 14400) :
  let courtyard_area : ℝ := courtyard_length * courtyard_width * 10000
  let brick_area : ℝ := courtyard_area / total_bricks
  brick_area / brick_width = 20 := by sorry

end NUMINAMATH_CALUDE_brick_length_calculation_l794_79402


namespace NUMINAMATH_CALUDE_find_number_l794_79468

theorem find_number : ∃ x : ℝ, ((55 + x) / 7 + 40) * 5 = 555 ∧ x = 442 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l794_79468


namespace NUMINAMATH_CALUDE_parabola_directrix_tangent_circle_l794_79418

/-- Given a parabola y^2 = 2px (p > 0) with directrix x = -p/2, 
    if the directrix is tangent to the circle (x - 3)^2 + y^2 = 16, then p = 2 -/
theorem parabola_directrix_tangent_circle (p : ℝ) (h1 : p > 0) : 
  (∀ x y : ℝ, y^2 = 2*p*x → x = -p/2 → (x - 3)^2 + y^2 = 16) → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_tangent_circle_l794_79418


namespace NUMINAMATH_CALUDE_interest_difference_l794_79443

theorem interest_difference (principal rate time : ℝ) : 
  principal = 300 → 
  rate = 4 → 
  time = 8 → 
  principal - (principal * rate * time / 100) = 204 :=
by
  sorry

end NUMINAMATH_CALUDE_interest_difference_l794_79443


namespace NUMINAMATH_CALUDE_students_playing_neither_l794_79426

theorem students_playing_neither (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ)
  (h1 : total = 36)
  (h2 : football = 26)
  (h3 : tennis = 20)
  (h4 : both = 17) :
  total - (football + tennis - both) = 7 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_neither_l794_79426


namespace NUMINAMATH_CALUDE_cafeteria_pies_l794_79442

def number_of_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  ((initial_apples - handed_out) / apples_per_pie : ℕ)

theorem cafeteria_pies :
  number_of_pies 150 24 15 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l794_79442


namespace NUMINAMATH_CALUDE_intersection_of_lines_l794_79419

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (1/5, 2/5)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := y = -3 * x + 1

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := y + 1 = 7 * x

theorem intersection_of_lines :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ x' y', (line1 x' y' ∧ line2 x' y') → (x' = x ∧ y' = y) :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l794_79419


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l794_79473

theorem rectangular_box_volume 
  (face_area1 face_area2 face_area3 : ℝ) 
  (h1 : face_area1 = 18)
  (h2 : face_area2 = 50)
  (h3 : face_area3 = 45) :
  ∃ (l w h : ℝ), 
    l * w = face_area1 ∧ 
    w * h = face_area2 ∧ 
    l * h = face_area3 ∧ 
    l * w * h = 30 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l794_79473


namespace NUMINAMATH_CALUDE_absolute_value_and_exponents_l794_79412

theorem absolute_value_and_exponents :
  |(-4 : ℝ)| + (π - Real.sqrt 2)^(0 : ℝ) - (1/2 : ℝ)^(-1 : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_exponents_l794_79412


namespace NUMINAMATH_CALUDE_zero_decomposition_l794_79436

/-- Represents a base-10 arithmetic system -/
structure Base10Arithmetic where
  /-- Multiplication operation in base-10 arithmetic -/
  mul : ℤ → ℤ → ℤ
  /-- Axiom: Multiplication by zero always results in zero -/
  mul_zero : ∀ a : ℤ, mul 0 a = 0

/-- 
Theorem: In base-10 arithmetic, the only way to decompose 0 into a product 
of two integers is 0 * a = 0, where a is any integer.
-/
theorem zero_decomposition (B : Base10Arithmetic) : 
  ∀ x y : ℤ, B.mul x y = 0 → x = 0 ∨ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_decomposition_l794_79436


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l794_79479

theorem quadratic_distinct_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0) →
  c < 1/4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l794_79479


namespace NUMINAMATH_CALUDE_perfect_square_expression_l794_79487

theorem perfect_square_expression (x : ℤ) :
  ∃ d : ℤ, (4 * x + 1 - Real.sqrt (8 * x + 1 : ℝ)) / 2 = d →
  ∃ k : ℤ, d = k^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_expression_l794_79487


namespace NUMINAMATH_CALUDE_andrew_payment_l794_79449

/-- Calculate the total amount Andrew paid to the shopkeeper for grapes and mangoes. -/
theorem andrew_payment (grape_quantity : ℕ) (grape_price : ℕ) (mango_quantity : ℕ) (mango_price : ℕ) :
  grape_quantity = 11 →
  grape_price = 98 →
  mango_quantity = 7 →
  mango_price = 50 →
  grape_quantity * grape_price + mango_quantity * mango_price = 1428 :=
by
  sorry

#check andrew_payment

end NUMINAMATH_CALUDE_andrew_payment_l794_79449


namespace NUMINAMATH_CALUDE_radical_simplification_l794_79482

theorem radical_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (21 * q) = 21 * q * Real.sqrt (21 * q) :=
by sorry

end NUMINAMATH_CALUDE_radical_simplification_l794_79482


namespace NUMINAMATH_CALUDE_range_of_c_minus_b_l794_79425

/-- Represents a triangle with side lengths a, b, c and opposite angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The range of c - b in a triangle where a = 1 and C - B = π/2 -/
theorem range_of_c_minus_b (t : Triangle) 
  (h1 : t.a = 1) 
  (h2 : t.C - t.B = π/2) : 
  ∃ (l u : ℝ), l = Real.sqrt 2 / 2 ∧ u = 1 ∧ 
  ∀ x, (t.c - t.b = x) → l < x ∧ x < u :=
sorry

end NUMINAMATH_CALUDE_range_of_c_minus_b_l794_79425


namespace NUMINAMATH_CALUDE_quadratic_equation_theorem_l794_79410

/-- Quadratic equation parameters -/
structure QuadraticParams where
  m : ℝ

/-- Roots of the quadratic equation -/
structure QuadraticRoots where
  x1 : ℝ
  x2 : ℝ

/-- Theorem about the quadratic equation x^2 - (2m+3)x + m^2 + 2 = 0 -/
theorem quadratic_equation_theorem (p : QuadraticParams) (r : QuadraticRoots) : 
  /- The equation has real roots if and only if m ≥ -1/12 -/
  (∃ (x : ℝ), x^2 - (2*p.m + 3)*x + p.m^2 + 2 = 0) ↔ p.m ≥ -1/12 ∧
  
  /- If x1 and x2 are the roots of the equation and satisfy the given condition, then m = 13 -/
  (r.x1^2 - (2*p.m + 3)*r.x1 + p.m^2 + 2 = 0 ∧
   r.x2^2 - (2*p.m + 3)*r.x2 + p.m^2 + 2 = 0 ∧
   r.x1^2 + r.x2^2 = 3*r.x1*r.x2 - 14) →
  p.m = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_theorem_l794_79410


namespace NUMINAMATH_CALUDE_sum_of_squares_l794_79450

theorem sum_of_squares (a b c : ℝ) : 
  a + b + c = 21 → 
  a * b + b * c + a * c = 100 → 
  a^2 + b^2 + c^2 = 241 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l794_79450


namespace NUMINAMATH_CALUDE_modified_factor_tree_l794_79407

theorem modified_factor_tree (P X Y G Z : ℕ) : 
  P = X * Y ∧
  X = 7 * G ∧
  Y = 11 * Z ∧
  G = 7 * 4 ∧
  Z = 11 * 4 →
  P = 94864 := by
sorry

end NUMINAMATH_CALUDE_modified_factor_tree_l794_79407


namespace NUMINAMATH_CALUDE_sixth_grade_homework_forgetfulness_l794_79432

theorem sixth_grade_homework_forgetfulness
  (group_a_size : ℕ)
  (group_b_size : ℕ)
  (group_a_forget_rate : ℚ)
  (group_b_forget_rate : ℚ)
  (h1 : group_a_size = 20)
  (h2 : group_b_size = 80)
  (h3 : group_a_forget_rate = 1/5)
  (h4 : group_b_forget_rate = 3/20)
  : (((group_a_size * group_a_forget_rate + group_b_size * group_b_forget_rate) /
     (group_a_size + group_b_size)) : ℚ) = 4/25 :=
by sorry

end NUMINAMATH_CALUDE_sixth_grade_homework_forgetfulness_l794_79432


namespace NUMINAMATH_CALUDE_inequality_proof_l794_79428

theorem inequality_proof (x y z t : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : 0 < y ∧ y < 1) 
  (hz : 0 < z ∧ z < 1) 
  (ht : 0 < t ∧ t < 1) : 
  Real.sqrt (x^2 + (1-t)^2) + Real.sqrt (y^2 + (1-x)^2) + 
  Real.sqrt (z^2 + (1-y)^2) + Real.sqrt (t^2 + (1-z)^2) < 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l794_79428


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l794_79401

theorem triangle_max_perimeter (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 3 →
  (Real.sqrt 3 + a) * (Real.sin C - Real.sin A) = (a + b) * Real.sin B →
  a > 0 →
  b > 0 →
  (a + b + c : ℝ) ≤ 2 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l794_79401


namespace NUMINAMATH_CALUDE_mushroom_soup_production_l794_79420

theorem mushroom_soup_production (total_required : ℕ) (team1_production : ℕ) (team2_production : ℕ) 
  (h1 : total_required = 280)
  (h2 : team1_production = 90)
  (h3 : team2_production = 120) :
  total_required - (team1_production + team2_production) = 70 :=
by sorry

end NUMINAMATH_CALUDE_mushroom_soup_production_l794_79420


namespace NUMINAMATH_CALUDE_water_left_is_84_ounces_l794_79461

/-- Represents the water cooler problem --/
def water_cooler_problem (initial_gallons : ℕ) (ounces_per_cup : ℕ) (rows : ℕ) (chairs_per_row : ℕ) (ounces_per_gallon : ℕ) : ℕ :=
  let initial_ounces := initial_gallons * ounces_per_gallon
  let total_cups := rows * chairs_per_row
  let ounces_poured := total_cups * ounces_per_cup
  initial_ounces - ounces_poured

/-- Theorem stating that the water left in the cooler is 84 ounces --/
theorem water_left_is_84_ounces :
  water_cooler_problem 3 6 5 10 128 = 84 := by
  sorry

end NUMINAMATH_CALUDE_water_left_is_84_ounces_l794_79461


namespace NUMINAMATH_CALUDE_nail_salon_revenue_l794_79422

/-- Calculates the total money made from manicures in a nail salon --/
def total_manicure_money (manicure_cost : ℝ) (total_fingers : ℕ) (fingers_per_person : ℕ) (non_clients : ℕ) : ℝ :=
  let total_people : ℕ := total_fingers / fingers_per_person
  let clients : ℕ := total_people - non_clients
  (clients : ℝ) * manicure_cost

/-- Theorem stating the total money made from manicures in the given scenario --/
theorem nail_salon_revenue :
  total_manicure_money 20 210 10 11 = 200 := by
  sorry

end NUMINAMATH_CALUDE_nail_salon_revenue_l794_79422


namespace NUMINAMATH_CALUDE_delta_y_value_l794_79452

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 + 1

-- State the theorem
theorem delta_y_value (x Δx : ℝ) (hx : x = 1) (hΔx : Δx = 0.1) :
  f (x + Δx) - f x = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_delta_y_value_l794_79452


namespace NUMINAMATH_CALUDE_semi_circle_perimeter_specific_semi_circle_perimeter_l794_79404

/-- The perimeter of a semi-circle with radius r is equal to π * r + 2r -/
theorem semi_circle_perimeter (r : ℝ) (h : r > 0) :
  let perimeter := π * r + 2 * r
  perimeter = π * r + 2 * r :=
by sorry

/-- The perimeter of a semi-circle with radius 6.6 cm is approximately 33.93 cm -/
theorem specific_semi_circle_perimeter :
  let r : ℝ := 6.6
  let perimeter := π * r + 2 * r
  ∃ (approx : ℝ), abs (perimeter - approx) < 0.005 ∧ approx = 33.93 :=
by sorry

end NUMINAMATH_CALUDE_semi_circle_perimeter_specific_semi_circle_perimeter_l794_79404


namespace NUMINAMATH_CALUDE_mitchell_chews_145_pieces_l794_79489

/-- The number of pieces of gum Mitchell chews -/
def chewed_pieces (packets : ℕ) (pieces_per_packet : ℕ) (unchewed : ℕ) : ℕ :=
  packets * pieces_per_packet - unchewed

/-- Proof that Mitchell chews 145 pieces of gum -/
theorem mitchell_chews_145_pieces :
  chewed_pieces 15 10 5 = 145 := by
  sorry

end NUMINAMATH_CALUDE_mitchell_chews_145_pieces_l794_79489


namespace NUMINAMATH_CALUDE_number_for_B_l794_79475

/-- Given that the number for A is a, and the number for B is 1 less than twice the number for A,
    prove that the number for B can be expressed as 2a - 1. -/
theorem number_for_B (a : ℝ) : 2 * a - 1 = 2 * a - 1 := by sorry

end NUMINAMATH_CALUDE_number_for_B_l794_79475


namespace NUMINAMATH_CALUDE_car_speed_calculation_l794_79408

theorem car_speed_calculation (v : ℝ) : v > 0 → (1 / v) * 3600 = (1 / 80) * 3600 + 10 → v = 3600 / 55 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_calculation_l794_79408


namespace NUMINAMATH_CALUDE_distinguishable_cube_colorings_count_l794_79424

/-- The number of distinguishable ways to color a cube with six different colors -/
def distinguishable_cube_colorings : ℕ := 30

/-- A cube has six faces -/
def cube_faces : ℕ := 6

/-- The number of rotational symmetries of a cube -/
def cube_rotational_symmetries : ℕ := 24

/-- The total number of ways to arrange 6 colors on 6 faces -/
def total_arrangements : ℕ := 720  -- 6!

theorem distinguishable_cube_colorings_count :
  distinguishable_cube_colorings = total_arrangements / cube_rotational_symmetries :=
by sorry

end NUMINAMATH_CALUDE_distinguishable_cube_colorings_count_l794_79424


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l794_79476

/-- An isosceles triangle with side lengths 9 and 5 has a perimeter of either 19 or 23 -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive side lengths
  (a = 9 ∧ b = 5 ∧ c = 5) ∨ (a = 5 ∧ b = 9 ∧ c = 9) →  -- isosceles with sides 9 and 5
  a + b + c = 19 ∨ a + b + c = 23 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l794_79476


namespace NUMINAMATH_CALUDE_intersection_M_N_l794_79437

def M : Set ℝ := {x | 1 - 2/x < 0}
def N : Set ℝ := {x | -1 ≤ x}

theorem intersection_M_N : ∀ x : ℝ, x ∈ M ∩ N ↔ 0 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l794_79437


namespace NUMINAMATH_CALUDE_circle_area_from_sector_l794_79457

theorem circle_area_from_sector (r : ℝ) (P : ℝ) (Q : ℝ) : 
  P = 2 → -- The area of sector COD is 2
  P = (1/6) * π * r^2 → -- Area of sector COD is 1/6 of circle area
  Q = π * r^2 → -- Q is the area of the entire circle
  Q = 12 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_sector_l794_79457


namespace NUMINAMATH_CALUDE_weekend_earnings_l794_79498

def newspaper_earnings : ℕ := 16
def car_washing_earnings : ℕ := 74

theorem weekend_earnings :
  newspaper_earnings + car_washing_earnings = 90 := by sorry

end NUMINAMATH_CALUDE_weekend_earnings_l794_79498


namespace NUMINAMATH_CALUDE_positive_addition_positive_multiplication_positive_division_positive_exponentiation_positive_root_extraction_l794_79446

-- Define positive real numbers
def PositiveReal := {x : ℝ | x > 0}

-- Theorem for addition
theorem positive_addition (a b : PositiveReal) : (↑a + ↑b : ℝ) > 0 := by sorry

-- Theorem for multiplication
theorem positive_multiplication (a b : PositiveReal) : (↑a * ↑b : ℝ) > 0 := by sorry

-- Theorem for division
theorem positive_division (a b : PositiveReal) : (↑a / ↑b : ℝ) > 0 := by sorry

-- Theorem for exponentiation
theorem positive_exponentiation (a : PositiveReal) (n : ℝ) : (↑a ^ n : ℝ) > 0 := by sorry

-- Theorem for root extraction
theorem positive_root_extraction (a : PositiveReal) (n : PositiveReal) : 
  ∃ (x : ℝ), x > 0 ∧ x ^ (↑n : ℝ) = ↑a := by sorry

end NUMINAMATH_CALUDE_positive_addition_positive_multiplication_positive_division_positive_exponentiation_positive_root_extraction_l794_79446


namespace NUMINAMATH_CALUDE_modulo_equivalence_l794_79480

theorem modulo_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 54126 ≡ n [ZMOD 23] ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_modulo_equivalence_l794_79480


namespace NUMINAMATH_CALUDE_cos_240_degrees_l794_79433

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l794_79433


namespace NUMINAMATH_CALUDE_roots_of_cubic_polynomial_l794_79460

theorem roots_of_cubic_polynomial :
  let p : ℝ → ℝ := λ x => x^3 - 2*x^2 - 5*x + 6
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = -2 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_roots_of_cubic_polynomial_l794_79460


namespace NUMINAMATH_CALUDE_max_single_player_salary_is_454000_l794_79467

/-- Represents a basketball team in the semi-professional league --/
structure BasketballTeam where
  players : Nat
  minSalary : Nat
  maxTotalSalary : Nat

/-- Calculates the maximum possible salary for a single player on the team --/
def maxSinglePlayerSalary (team : BasketballTeam) : Nat :=
  team.maxTotalSalary - (team.players - 1) * team.minSalary

/-- Theorem stating the maximum possible salary for a single player --/
theorem max_single_player_salary_is_454000 :
  let team := BasketballTeam.mk 23 18000 850000
  maxSinglePlayerSalary team = 454000 := by
  sorry

#eval maxSinglePlayerSalary (BasketballTeam.mk 23 18000 850000)

end NUMINAMATH_CALUDE_max_single_player_salary_is_454000_l794_79467


namespace NUMINAMATH_CALUDE_floor_divisibility_implies_integer_l794_79403

/-- Floor function -/
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

/-- Property: For all integers m and n, if m divides n, then ⌊mr⌋ divides ⌊nr⌋ -/
def floor_divisibility_property (r : ℝ) : Prop :=
  ∀ (m n : ℤ), m ∣ n → (floor (m * r) : ℤ) ∣ (floor (n * r) : ℤ)

/-- Theorem: If r ≥ 0 satisfies the floor divisibility property, then r is an integer -/
theorem floor_divisibility_implies_integer (r : ℝ) (h1 : r ≥ 0) (h2 : floor_divisibility_property r) : ∃ (n : ℤ), r = n := by
  sorry

end NUMINAMATH_CALUDE_floor_divisibility_implies_integer_l794_79403


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l794_79499

theorem inverse_proportion_k_value (k : ℝ) (h1 : k ≠ 0) :
  (∀ x, x ≠ 0 → (k / x) = -1 ↔ x = 2) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l794_79499


namespace NUMINAMATH_CALUDE_problem_1_l794_79435

theorem problem_1 : |(-6)| - 7 + (-3) = -4 := by sorry

end NUMINAMATH_CALUDE_problem_1_l794_79435


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_cos_double_l794_79491

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 / 4 - x^2 = 1

-- Define the acute angle between asymptotes
def asymptote_angle (α : ℝ) : Prop := 
  ∃ (x y : ℝ), hyperbola x y ∧ 
  (∀ (x' y' : ℝ), hyperbola x' y' → 
    α = Real.arctan (abs (y / x)) ∧ α > 0 ∧ α < Real.pi / 2)

-- Theorem statement
theorem hyperbola_asymptote_angle_cos_double :
  ∀ α : ℝ, asymptote_angle α → Real.cos (2 * α) = -7/25 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_cos_double_l794_79491


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l794_79445

theorem gcd_lcm_sum : Nat.gcd 42 63 + Nat.lcm 48 18 = 165 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l794_79445


namespace NUMINAMATH_CALUDE_test_series_count_l794_79421

/-- The number of tests in Professor Tester's series -/
def n : ℕ := 8

/-- John's average score if he scored 97 on the last test -/
def avg_with_97 : ℚ := 90

/-- John's average score if he scored 73 on the last test -/
def avg_with_73 : ℚ := 87

/-- The score difference between the two scenarios -/
def score_diff : ℚ := 97 - 73

/-- The average difference between the two scenarios -/
def avg_diff : ℚ := avg_with_97 - avg_with_73

theorem test_series_count :
  score_diff / (n + 1 : ℚ) = avg_diff :=
sorry

end NUMINAMATH_CALUDE_test_series_count_l794_79421


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_19_l794_79454

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_19_l794_79454


namespace NUMINAMATH_CALUDE_segment_length_product_l794_79478

theorem segment_length_product (b₁ b₂ : ℝ) : 
  (((3 * b₁ - 5)^2 + (b₁ + 3)^2 = 45) ∧ 
   ((3 * b₂ - 5)^2 + (b₂ + 3)^2 = 45) ∧ 
   b₁ ≠ b₂) → 
  b₁ * b₂ = -11/10 := by
sorry

end NUMINAMATH_CALUDE_segment_length_product_l794_79478


namespace NUMINAMATH_CALUDE_expand_product_l794_79430

theorem expand_product (x : ℝ) : (3*x + 4) * (x - 2) * (x + 6) = 3*x^3 + 16*x^2 - 20*x - 48 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l794_79430


namespace NUMINAMATH_CALUDE_xyz_value_l794_79492

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) :
  x * y * z = 8 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l794_79492


namespace NUMINAMATH_CALUDE_intersection_equals_A_l794_79495

def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x ≤ 3}

theorem intersection_equals_A : A ∩ B = A := by sorry

end NUMINAMATH_CALUDE_intersection_equals_A_l794_79495


namespace NUMINAMATH_CALUDE_chef_cherries_remaining_l794_79466

theorem chef_cherries_remaining (initial_cherries used_cherries : ℕ) 
  (h1 : initial_cherries = 77)
  (h2 : used_cherries = 60) :
  initial_cherries - used_cherries = 17 := by
  sorry

end NUMINAMATH_CALUDE_chef_cherries_remaining_l794_79466


namespace NUMINAMATH_CALUDE_four_weeks_filming_time_l794_79455

/-- Calculates the total filming time in hours for a given number of weeks -/
def total_filming_time (episode_length : ℕ) (filming_factor : ℚ) (episodes_per_week : ℕ) (weeks : ℕ) : ℚ :=
  let filming_time := episode_length * (1 + filming_factor)
  let total_episodes := episodes_per_week * weeks
  (filming_time * total_episodes) / 60

theorem four_weeks_filming_time :
  total_filming_time 20 (1/2) 5 4 = 10 := by
  sorry

#eval total_filming_time 20 (1/2) 5 4

end NUMINAMATH_CALUDE_four_weeks_filming_time_l794_79455


namespace NUMINAMATH_CALUDE_ladder_wood_length_50ft_l794_79469

/-- Calculates the total length of wood needed for ladder rungs -/
def ladder_wood_length (rung_length inches_between_rungs total_height_feet : ℚ) : ℚ :=
  let inches_per_foot : ℚ := 12
  let total_height_inches : ℚ := total_height_feet * inches_per_foot
  let space_per_rung : ℚ := rung_length + inches_between_rungs
  let num_rungs : ℚ := total_height_inches / space_per_rung
  (num_rungs * rung_length) / inches_per_foot

/-- The total length of wood needed for rungs to climb 50 feet is 37.5 feet -/
theorem ladder_wood_length_50ft :
  ladder_wood_length 18 6 50 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_wood_length_50ft_l794_79469


namespace NUMINAMATH_CALUDE_negative_root_iff_negative_a_l794_79434

theorem negative_root_iff_negative_a (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 1 = 0) ↔ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_root_iff_negative_a_l794_79434


namespace NUMINAMATH_CALUDE_calculate_value_probability_l794_79458

def calculate_letters : Finset Char := {'C', 'A', 'L', 'C', 'U', 'L', 'A', 'T', 'E'}
def value_letters : Finset Char := {'V', 'A', 'L', 'U', 'E'}

theorem calculate_value_probability :
  (calculate_letters.filter (λ c => c ∈ value_letters)).card / calculate_letters.card = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_calculate_value_probability_l794_79458


namespace NUMINAMATH_CALUDE_ox_and_sheep_cost_l794_79411

theorem ox_and_sheep_cost (ox sheep : ℚ) 
  (h1 : 5 * ox + 2 * sheep = 10) 
  (h2 : 2 * ox + 8 * sheep = 8) : 
  ox = 16/9 ∧ sheep = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_ox_and_sheep_cost_l794_79411


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l794_79486

theorem binomial_expansion_theorem (n : ℕ) (a b k : ℝ) : 
  n ≥ 2 → 
  a * b ≠ 0 → 
  a = b + k → 
  k > 0 → 
  (2 : ℝ) * (n.choose 1) * (2 * b) ^ (n - 1) * k + 
  (8 : ℝ) * (n.choose 3) * (2 * b) ^ (n - 3) * k ^ 3 = 0 → 
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l794_79486


namespace NUMINAMATH_CALUDE_quadratic_equation_proof_l794_79485

theorem quadratic_equation_proof (c : ℝ) : 
  (∃ c_modified : ℝ, c = c_modified + 2 ∧ (-1)^2 + 4*(-1) + c_modified = 0) →
  c = 5 ∧ ∀ x : ℝ, x^2 + 4*x + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_proof_l794_79485


namespace NUMINAMATH_CALUDE_bombardment_death_percentage_l794_79423

/-- The percentage of people who died by bombardment in a Sri Lankan village --/
def bombardment_percentage (initial_population final_population : ℕ) (departure_rate : ℚ) : ℚ :=
  let x := (initial_population - final_population / (1 - departure_rate)) / initial_population
  x * 100

/-- Theorem stating the percentage of people who died by bombardment --/
theorem bombardment_death_percentage :
  let initial_population : ℕ := 4399
  let final_population : ℕ := 3168
  let departure_rate : ℚ := 1/5
  abs (bombardment_percentage initial_population final_population departure_rate - 9.98) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_bombardment_death_percentage_l794_79423


namespace NUMINAMATH_CALUDE_inequality_range_l794_79494

theorem inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 4 * x + a ≥ -2 * x^2 + 1) ↔ a ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l794_79494


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l794_79459

theorem polygon_sides_from_angle_sum (n : ℕ) (sum_angles : ℝ) : 
  sum_angles = 900 → (n - 2) * 180 = sum_angles → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l794_79459


namespace NUMINAMATH_CALUDE_cos_180_degrees_l794_79497

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l794_79497


namespace NUMINAMATH_CALUDE_range_of_a_l794_79483

-- Define the statements p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x - 1 < 0

def q (a : ℝ) : Prop := (3 / (a - 1)) + 1 < 0

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (¬(p a ∨ q a)) → (a ≤ -4 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l794_79483


namespace NUMINAMATH_CALUDE_addition_subtraction_reduces_system_l794_79481

/-- A method for solving systems of linear equations with two variables -/
inductive SolvingMethod
| Substitution
| AdditionSubtraction

/-- Represents a system of linear equations with two variables -/
structure LinearSystem :=
  (equations : List (LinearEquation))

/-- Represents a linear equation -/
structure LinearEquation :=
  (coefficients : List ℝ)
  (constant : ℝ)

/-- A function that determines if a method reduces a system to a single variable -/
def reduces_to_single_variable (method : SolvingMethod) (system : LinearSystem) : Prop :=
  sorry

/-- The theorem stating that the addition-subtraction method reduces a system to a single variable -/
theorem addition_subtraction_reduces_system :
  ∀ (system : LinearSystem),
    reduces_to_single_variable SolvingMethod.AdditionSubtraction system :=
  sorry

end NUMINAMATH_CALUDE_addition_subtraction_reduces_system_l794_79481


namespace NUMINAMATH_CALUDE_loops_per_day_l794_79465

def weekly_goal : ℕ := 3500
def track_length : ℕ := 50
def days_in_week : ℕ := 7

theorem loops_per_day : 
  ∀ (goal : ℕ) (track : ℕ) (days : ℕ),
  goal = weekly_goal → 
  track = track_length → 
  days = days_in_week →
  (goal / track) / days = 10 := by sorry

end NUMINAMATH_CALUDE_loops_per_day_l794_79465


namespace NUMINAMATH_CALUDE_largest_package_size_l794_79416

theorem largest_package_size (ming_pencils catherine_pencils lucas_pencils : ℕ) 
  (h_ming : ming_pencils = 48)
  (h_catherine : catherine_pencils = 36)
  (h_lucas : lucas_pencils = 60) :
  Nat.gcd ming_pencils (Nat.gcd catherine_pencils lucas_pencils) = 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l794_79416


namespace NUMINAMATH_CALUDE_impossibility_of_all_powers_of_two_l794_79477

/-- Represents a card with a natural number -/
structure Card where
  value : ℕ

/-- Represents the state of the table at any given time -/
structure TableState where
  cards : List Card
  oddCount : ℕ

/-- The procedure of creating a new card from existing cards -/
def createNewCard (state : TableState) : Card :=
  sorry

/-- The evolution of the table state over time -/
def evolveTable (initialState : TableState) : ℕ → TableState
  | 0 => initialState
  | n + 1 => let prevState := evolveTable initialState n
              let newCard := createNewCard prevState
              { cards := newCard :: prevState.cards,
                oddCount := if newCard.value % 2 = 1 then prevState.oddCount + 1 else prevState.oddCount }

/-- Checks if a number is divisible by 2^d -/
def isDivisibleByPowerOfTwo (n d : ℕ) : Bool :=
  n % (2^d) = 0

theorem impossibility_of_all_powers_of_two :
  ∀ (initialCards : List Card),
    initialCards.length = 100 →
    (initialCards.filter (λ c => c.value % 2 = 1)).length = 28 →
    ∃ (d : ℕ), ∀ (t : ℕ),
      ¬∃ (card : Card),
        card ∈ (evolveTable { cards := initialCards, oddCount := 28 } t).cards ∧
        isDivisibleByPowerOfTwo card.value d :=
  sorry

end NUMINAMATH_CALUDE_impossibility_of_all_powers_of_two_l794_79477


namespace NUMINAMATH_CALUDE_asymptote_sum_l794_79400

/-- Represents a rational function -/
structure RationalFunction where
  numerator : Polynomial ℝ
  denominator : Polynomial ℝ

/-- Counts the number of holes in the graph of a rational function -/
noncomputable def count_holes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of vertical asymptotes of a rational function -/
noncomputable def count_vertical_asymptotes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of horizontal asymptotes of a rational function -/
noncomputable def count_horizontal_asymptotes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of oblique asymptotes of a rational function -/
noncomputable def count_oblique_asymptotes (f : RationalFunction) : ℕ := sorry

/-- The main theorem -/
theorem asymptote_sum (f : RationalFunction) 
  (h : f.numerator = Polynomial.X^2 + 4*Polynomial.X + 3 ∧ 
       f.denominator = Polynomial.X^3 + 2*Polynomial.X^2 - 3*Polynomial.X) : 
  count_holes f + 2 * count_vertical_asymptotes f + 
  3 * count_horizontal_asymptotes f + 4 * count_oblique_asymptotes f = 8 := by
  sorry

end NUMINAMATH_CALUDE_asymptote_sum_l794_79400


namespace NUMINAMATH_CALUDE_f_value_at_3_l794_79464

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (x + Real.sqrt (x^2 + 1)) + a * x^7 + b * x^3 - 4

theorem f_value_at_3 (a b : ℝ) (h : f a b (-3) = 4) : f a b 3 = -12 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_3_l794_79464


namespace NUMINAMATH_CALUDE_possible_x_values_l794_79440

def M (x : ℝ) : Set ℝ := {-2, 3*x^2 + 3*x - 4, x^2 + x - 4}

theorem possible_x_values (x : ℝ) : 2 ∈ M x → x = 2 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_possible_x_values_l794_79440


namespace NUMINAMATH_CALUDE_x_over_y_equals_one_l794_79484

theorem x_over_y_equals_one (x y : ℝ) 
  (h1 : 1 < (x - y) / (x + y)) 
  (h2 : (x - y) / (x + y) < 3) 
  (h3 : ∃ n : ℤ, x / y = n) : 
  x / y = 1 := by
sorry

end NUMINAMATH_CALUDE_x_over_y_equals_one_l794_79484


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l794_79474

-- Define the repeating decimal 0.343434...
def repeating_decimal : ℚ := 34 / 99

-- Theorem statement
theorem reciprocal_of_repeating_decimal :
  (repeating_decimal⁻¹ : ℚ) = 99 / 34 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l794_79474


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l794_79451

theorem cubic_roots_sum (a b c : ℝ) : 
  (3 * a^3 - 6 * a^2 + 99 * a - 2 = 0) →
  (3 * b^3 - 6 * b^2 + 99 * b - 2 = 0) →
  (3 * c^3 - 6 * c^2 + 99 * c - 2 = 0) →
  (a + b - 2)^3 + (b + c - 2)^3 + (c + a - 2)^3 = -196 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l794_79451


namespace NUMINAMATH_CALUDE_annies_plants_leaves_l794_79496

/-- Calculates the total number of leaves for Annie's plants -/
def total_leaves (basil_pots rosemary_pots thyme_pots : ℕ) 
                 (basil_leaves rosemary_leaves thyme_leaves : ℕ) : ℕ :=
  basil_pots * basil_leaves + rosemary_pots * rosemary_leaves + thyme_pots * thyme_leaves

/-- Proves that Annie's plants have a total of 354 leaves -/
theorem annies_plants_leaves : 
  total_leaves 3 9 6 4 18 30 = 354 := by
  sorry

#eval total_leaves 3 9 6 4 18 30

end NUMINAMATH_CALUDE_annies_plants_leaves_l794_79496
