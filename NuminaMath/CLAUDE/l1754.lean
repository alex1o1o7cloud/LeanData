import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_roots_is_18_l1754_175432

def is_symmetric_about_3 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (3 + x) = f (3 - x)

def has_six_distinct_roots_in_arithmetic_sequence (f : ℝ → ℝ) : Prop :=
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℝ) (d : ℝ),
    r₁ < r₂ ∧ r₂ < r₃ ∧ r₃ < r₄ ∧ r₄ < r₅ ∧ r₅ < r₆ ∧
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0 ∧ f r₄ = 0 ∧ f r₅ = 0 ∧ f r₆ = 0 ∧
    r₂ - r₁ = d ∧ r₃ - r₂ = d ∧ r₄ - r₃ = d ∧ r₅ - r₄ = d ∧ r₆ - r₅ = d

theorem sum_of_roots_is_18 (f : ℝ → ℝ) 
    (h_sym : is_symmetric_about_3 f)
    (h_roots : has_six_distinct_roots_in_arithmetic_sequence f) :
    ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℝ),
      f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0 ∧ f r₄ = 0 ∧ f r₅ = 0 ∧ f r₆ = 0 ∧
      r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_18_l1754_175432


namespace NUMINAMATH_CALUDE_infinite_prime_divisors_l1754_175445

/-- A sequence of positive integers where no term divides another -/
def NonDivisibleSequence (a : ℕ → ℕ) : Prop :=
  ∀ i j, i ≠ j → ¬(a i ∣ a j)

/-- The set of primes dividing at least one term of the sequence -/
def PrimeDivisorsSet (a : ℕ → ℕ) : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∃ i, p ∣ a i}

theorem infinite_prime_divisors (a : ℕ → ℕ) 
    (h : NonDivisibleSequence a) : Set.Infinite (PrimeDivisorsSet a) := by
  sorry

end NUMINAMATH_CALUDE_infinite_prime_divisors_l1754_175445


namespace NUMINAMATH_CALUDE_problem_solution_l1754_175414

def f (a : ℝ) (x : ℝ) : ℝ := |x| * (x - a)

theorem problem_solution :
  (∀ x, f a x = -f a (-x)) → a = 0 ∧
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 2 → f a x ≤ f a y) → a ≤ 0 ∧
  ∃ a, a < 0 ∧ (∀ x, -1 ≤ x ∧ x ≤ 1/2 → f a x ≤ 2) ∧
     (∃ x, -1 ≤ x ∧ x ≤ 1/2 ∧ f a x = 2) ∧
     a = -3 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1754_175414


namespace NUMINAMATH_CALUDE_quadruple_solution_l1754_175433

theorem quadruple_solution (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a * b * c * d = 1)
  (h2 : a^2012 + 2012 * b = 2012 * c + d^2012)
  (h3 : 2012 * a + b^2012 = c^2012 + 2012 * d) :
  ∃ t : ℝ, t > 0 ∧ a = t ∧ b = 1/t ∧ c = 1/t ∧ d = t :=
sorry

end NUMINAMATH_CALUDE_quadruple_solution_l1754_175433


namespace NUMINAMATH_CALUDE_nonnegative_fraction_implies_nonnegative_x_l1754_175475

theorem nonnegative_fraction_implies_nonnegative_x (x : ℝ) :
  (x^2 + 2*x^4 - 3*x^5) / (x + 2*x^3 - 3*x^4) ≥ 0 → x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_nonnegative_fraction_implies_nonnegative_x_l1754_175475


namespace NUMINAMATH_CALUDE_jerrie_situp_minutes_l1754_175481

/-- The number of sit-ups Barney can do in one minute -/
def barney_situps : ℕ := 45

/-- The number of sit-ups Carrie can do in one minute -/
def carrie_situps : ℕ := 2 * barney_situps

/-- The number of sit-ups Jerrie can do in one minute -/
def jerrie_situps : ℕ := carrie_situps + 5

/-- The number of minutes Barney does sit-ups -/
def barney_minutes : ℕ := 1

/-- The number of minutes Carrie does sit-ups -/
def carrie_minutes : ℕ := 2

/-- The total number of sit-ups performed by all three people -/
def total_situps : ℕ := 510

/-- Theorem stating that Jerrie did sit-ups for 3 minutes -/
theorem jerrie_situp_minutes :
  ∃ (j : ℕ), j * jerrie_situps + barney_minutes * barney_situps + carrie_minutes * carrie_situps = total_situps ∧ j = 3 :=
by sorry

end NUMINAMATH_CALUDE_jerrie_situp_minutes_l1754_175481


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_l1754_175472

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |x - 1|

-- Part 1
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x ≥ 2} = {x : ℝ | x ≤ 2/3 ∨ x ≥ 2} := by sorry

-- Part 2
theorem range_of_a :
  (∀ x : ℝ, f a x ≥ 5 - x) → a ≥ 6 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_l1754_175472


namespace NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_solve_equation_3_solve_equation_4_l1754_175418

-- Problem 1
theorem solve_equation_1 : ∃ x : ℝ, (3 * x^2 - 32 * x - 48 = 0) ↔ (x = 12 ∨ x = -4/3) := by sorry

-- Problem 2
theorem solve_equation_2 : ∃ x : ℝ, (4 * x^2 + x - 3 = 0) ↔ (x = 3/4 ∨ x = -1) := by sorry

-- Problem 3
theorem solve_equation_3 : ∃ x : ℝ, ((3 * x + 1)^2 - 4 = 0) ↔ (x = 1/3 ∨ x = -1) := by sorry

-- Problem 4
theorem solve_equation_4 : ∃ x : ℝ, (9 * (x - 2)^2 = 4 * (x + 1)^2) ↔ (x = 8 ∨ x = 4/5) := by sorry

end NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_solve_equation_3_solve_equation_4_l1754_175418


namespace NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l1754_175488

-- Define a quadratic equation
def quadratic_equation (k : ℤ) (x : ℤ) : Prop := x^2 - 65*x + k = 0

-- Define primality
def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m > 1 → m < n → ¬(n % m = 0)

-- Theorem statement
theorem no_prime_roots_for_quadratic :
  ¬∃ k : ℤ, ∃ x y : ℤ, 
    x ≠ y ∧ 
    quadratic_equation k x ∧ 
    quadratic_equation k y ∧
    is_prime x ∧ 
    is_prime y :=
sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l1754_175488


namespace NUMINAMATH_CALUDE_product_of_cosines_l1754_175434

theorem product_of_cosines : 
  (1 + Real.cos (π / 12)) * (1 + Real.cos (5 * π / 12)) * 
  (1 + Real.cos (7 * π / 12)) * (1 + Real.cos (11 * π / 12)) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cosines_l1754_175434


namespace NUMINAMATH_CALUDE_place_mat_length_l1754_175436

/-- The radius of the round table -/
def table_radius : ℝ := 5

/-- The width of each place mat -/
def mat_width : ℝ := 1.5

/-- The number of place mats -/
def num_mats : ℕ := 4

/-- The theorem stating the length of each place mat -/
theorem place_mat_length :
  ∃ y : ℝ,
    y > 0 ∧
    y = 0.75 ∧
    (y + 2.5 * Real.sqrt 2 - mat_width / 2)^2 + (mat_width / 2)^2 = table_radius^2 ∧
    ∀ (i : Fin num_mats),
      ∃ (x y : ℝ),
        x^2 + y^2 = table_radius^2 ∧
        (x - mat_width / 2)^2 + (y - (y + 2.5 * Real.sqrt 2 - mat_width / 2))^2 = table_radius^2 :=
by
  sorry

end NUMINAMATH_CALUDE_place_mat_length_l1754_175436


namespace NUMINAMATH_CALUDE_function_value_at_two_l1754_175484

/-- Given a function f: ℝ → ℝ such that f(x) = ax^5 + bx^3 + cx + 8 for some real constants a, b, and c, 
    and f(-2) = 10, prove that f(2) = 6. -/
theorem function_value_at_two (a b c : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^5 + b * x^3 + c * x + 8)
    (h2 : f (-2) = 10) : 
  f 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l1754_175484


namespace NUMINAMATH_CALUDE_exam_scores_sum_l1754_175455

theorem exam_scores_sum (scores : List ℝ) :
  scores.length = 6 ∧
  65 ∈ scores ∧ 75 ∈ scores ∧ 85 ∈ scores ∧ 95 ∈ scores ∧
  scores.sum / scores.length = 80 →
  ∃ x y, x ∈ scores ∧ y ∈ scores ∧ x + y = 160 :=
by sorry

end NUMINAMATH_CALUDE_exam_scores_sum_l1754_175455


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1754_175405

theorem p_necessary_not_sufficient_for_q (a b : ℝ) :
  (∀ a b, a^2 + b^2 ≠ 0 → a * b = 0) ∧
  ¬(∀ a b, a * b = 0 → a^2 + b^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1754_175405


namespace NUMINAMATH_CALUDE_projectile_speed_calculation_l1754_175435

/-- 
Given two projectiles launched simultaneously 1455 km apart, with one traveling at 500 km/h,
prove that the speed of the other projectile is 470 km/h if they meet after 90 minutes.
-/
theorem projectile_speed_calculation (distance : ℝ) (time : ℝ) (speed2 : ℝ) (speed1 : ℝ) : 
  distance = 1455 → 
  time = 1.5 → 
  speed2 = 500 → 
  speed1 = 470 → 
  distance = (speed1 + speed2) * time :=
by sorry

end NUMINAMATH_CALUDE_projectile_speed_calculation_l1754_175435


namespace NUMINAMATH_CALUDE_f_10_equals_756_l1754_175483

def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 5*x + 6

theorem f_10_equals_756 : f 10 = 756 := by
  sorry

end NUMINAMATH_CALUDE_f_10_equals_756_l1754_175483


namespace NUMINAMATH_CALUDE_cone_surface_area_l1754_175406

/-- The surface area of a cone with base radius 1 and slant height 3 is 4π. -/
theorem cone_surface_area : 
  let r : ℝ := 1  -- base radius
  let l : ℝ := 3  -- slant height
  let S : ℝ := π * r * (r + l)  -- surface area formula
  S = 4 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l1754_175406


namespace NUMINAMATH_CALUDE_fibonacci_gcd_property_l1754_175495

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_gcd_property :
  Nat.gcd (fib 2017) (fib 99 * fib 101 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_gcd_property_l1754_175495


namespace NUMINAMATH_CALUDE_cos_squared_pi_sixth_plus_alpha_half_l1754_175476

theorem cos_squared_pi_sixth_plus_alpha_half (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos (π / 6 + α / 2) ^ 2 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_pi_sixth_plus_alpha_half_l1754_175476


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l1754_175463

theorem chinese_remainder_theorem_example (x : ℤ) : 
  x ≡ 2 [ZMOD 7] → x ≡ 3 [ZMOD 6] → x ≡ 9 [ZMOD 42] := by
  sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l1754_175463


namespace NUMINAMATH_CALUDE_pizza_count_l1754_175452

def num_toppings : ℕ := 8

def zero_topping_pizzas : ℕ := 1

def one_topping_pizzas (n : ℕ) : ℕ := n

def two_topping_pizzas (n : ℕ) : ℕ := n.choose 2

def total_pizzas (n : ℕ) : ℕ :=
  zero_topping_pizzas + one_topping_pizzas n + two_topping_pizzas n

theorem pizza_count : total_pizzas num_toppings = 37 := by
  sorry

end NUMINAMATH_CALUDE_pizza_count_l1754_175452


namespace NUMINAMATH_CALUDE_siblings_difference_l1754_175473

theorem siblings_difference (masud_siblings : ℕ) : 
  masud_siblings = 60 →
  let janet_siblings := 4 * masud_siblings - 60
  let carlos_siblings := (3 * masud_siblings) / 4
  janet_siblings - carlos_siblings = 45 := by
sorry

end NUMINAMATH_CALUDE_siblings_difference_l1754_175473


namespace NUMINAMATH_CALUDE_square_exterior_points_l1754_175407

/-- Given a square ABCD with side length 15 and exterior points G and H, prove that GH^2 = 1126 - 1030√2 -/
theorem square_exterior_points (A B C D G H K : ℝ × ℝ) : 
  let side_length : ℝ := 15
  let bg : ℝ := 7
  let dh : ℝ := 7
  let ag : ℝ := 13
  let ch : ℝ := 13
  let dk : ℝ := 8
  -- Square ABCD conditions
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = side_length^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = side_length^2 ∧
  (D.1 - C.1)^2 + (D.2 - C.2)^2 = side_length^2 ∧
  (A.1 - D.1)^2 + (A.2 - D.2)^2 = side_length^2 ∧
  -- Exterior points conditions
  (G.1 - B.1)^2 + (G.2 - B.2)^2 = bg^2 ∧
  (H.1 - D.1)^2 + (H.2 - D.2)^2 = dh^2 ∧
  (G.1 - A.1)^2 + (G.2 - A.2)^2 = ag^2 ∧
  (H.1 - C.1)^2 + (H.2 - C.2)^2 = ch^2 ∧
  -- K on extension of BD
  (K.1 - D.1)^2 + (K.2 - D.2)^2 = dk^2 ∧
  (K.1 - B.1) / (D.1 - B.1) = (K.2 - B.2) / (D.2 - B.2) ∧
  (K.1 - D.1) / (B.1 - D.1) > 1 →
  (G.1 - H.1)^2 + (G.2 - H.2)^2 = 1126 - 1030 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_exterior_points_l1754_175407


namespace NUMINAMATH_CALUDE_stream_speed_l1754_175448

/-- The speed of a stream given boat travel times and distances -/
theorem stream_speed (downstream_distance upstream_distance : ℝ) 
  (time : ℝ) (h1 : downstream_distance = 84) (h2 : upstream_distance = 48) 
  (h3 : time = 2) : ∃ s : ℝ, s = 9 ∧ 
  ∃ b : ℝ, downstream_distance = (b + s) * time ∧ 
           upstream_distance = (b - s) * time :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l1754_175448


namespace NUMINAMATH_CALUDE_simultaneous_congruences_l1754_175467

theorem simultaneous_congruences (x : ℤ) :
  x % 2 = 1 ∧ x % 3 = 2 ∧ x % 5 = 3 ∧ x % 7 = 4 → x % 210 = 53 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_congruences_l1754_175467


namespace NUMINAMATH_CALUDE_stream_speed_l1754_175491

/-- Given a canoe that rows upstream at 6 km/hr and downstream at 10 km/hr, 
    the speed of the stream is 2 km/hr. -/
theorem stream_speed (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed = 6)
  (h2 : downstream_speed = 10) :
  ∃ (canoe_speed stream_speed : ℝ),
    canoe_speed - stream_speed = upstream_speed ∧
    canoe_speed + stream_speed = downstream_speed ∧
    stream_speed = 2 := by
  sorry


end NUMINAMATH_CALUDE_stream_speed_l1754_175491


namespace NUMINAMATH_CALUDE_f_at_three_l1754_175424

def f (x : ℝ) : ℝ := 5 * x^3 + 3 * x^2 + 7 * x - 2

theorem f_at_three : f 3 = 181 := by
  sorry

end NUMINAMATH_CALUDE_f_at_three_l1754_175424


namespace NUMINAMATH_CALUDE_y1_less_than_y2_l1754_175494

/-- A linear function f(x) = x + 4 -/
def f (x : ℝ) : ℝ := x + 4

/-- The theorem states that for two points on the graph of f,
    if the x-coordinate of the first point is less than the x-coordinate of the second point,
    then the y-coordinate of the first point is less than the y-coordinate of the second point. -/
theorem y1_less_than_y2 (y1 y2 : ℝ) 
    (h1 : f (-1/2) = y1) 
    (h2 : f 1 = y2) : 
  y1 < y2 := by
  sorry

end NUMINAMATH_CALUDE_y1_less_than_y2_l1754_175494


namespace NUMINAMATH_CALUDE_height_comparison_l1754_175466

theorem height_comparison (a b : ℝ) (h : a = 0.8 * b) : b = 1.25 * a := by
  sorry

end NUMINAMATH_CALUDE_height_comparison_l1754_175466


namespace NUMINAMATH_CALUDE_hcf_of_48_and_64_l1754_175499

theorem hcf_of_48_and_64 : 
  let a := 48
  let b := 64
  let lcm := 192
  Nat.lcm a b = lcm → Nat.gcd a b = 16 := by
sorry

end NUMINAMATH_CALUDE_hcf_of_48_and_64_l1754_175499


namespace NUMINAMATH_CALUDE_increasing_quadratic_implies_a_ge_five_l1754_175426

/-- Given a function f(x) = -x^2 + 2(a-1)x + 2 that is increasing on the interval (-∞, 4),
    prove that a ≥ 5. -/
theorem increasing_quadratic_implies_a_ge_five (a : ℝ) :
  (∀ x < 4, Monotone (fun x => -x^2 + 2*(a-1)*x + 2)) →
  a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_increasing_quadratic_implies_a_ge_five_l1754_175426


namespace NUMINAMATH_CALUDE_two_consecutive_late_charges_l1754_175438

theorem two_consecutive_late_charges (original_bill : ℝ) (late_charge_rate : ℝ) : 
  original_bill = 500 →
  late_charge_rate = 0.02 →
  (original_bill * (1 + late_charge_rate) * (1 + late_charge_rate)) = 520.20 := by
  sorry

end NUMINAMATH_CALUDE_two_consecutive_late_charges_l1754_175438


namespace NUMINAMATH_CALUDE_two_identical_digits_in_2_pow_30_l1754_175431

theorem two_identical_digits_in_2_pow_30 :
  ∃ (d : ℕ) (i j : ℕ), i ≠ j ∧ i < 10 ∧ j < 10 ∧
  (2^30 / 10^i) % 10 = d ∧ (2^30 / 10^j) % 10 = d :=
by
  have h1 : 2^30 > 10^9 := sorry
  have h2 : 2^30 < 8 * 10^9 := sorry
  have pigeonhole : ∀ (n m : ℕ), n > m → 
    ∃ (k : ℕ), k < n ∧ (∃ (i j : ℕ), i < m ∧ j < m ∧ i ≠ j ∧
    (n / 10^i) % 10 = k ∧ (n / 10^j) % 10 = k) := sorry
  sorry


end NUMINAMATH_CALUDE_two_identical_digits_in_2_pow_30_l1754_175431


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l1754_175492

theorem area_between_concentric_circles 
  (R r : ℝ) 
  (h_positive_R : R > 0) 
  (h_positive_r : r > 0) 
  (h_R_greater_r : R > r) 
  (h_tangent : r^2 + 5^2 = R^2) : 
  π * (R^2 - r^2) = 25 * π := by
sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l1754_175492


namespace NUMINAMATH_CALUDE_bakery_boxes_l1754_175470

theorem bakery_boxes (total_muffins : ℕ) (muffins_per_box : ℕ) (available_boxes : ℕ) : 
  total_muffins = 95 → 
  muffins_per_box = 5 → 
  available_boxes = 10 → 
  (total_muffins - available_boxes * muffins_per_box + muffins_per_box - 1) / muffins_per_box = 9 := by
  sorry

end NUMINAMATH_CALUDE_bakery_boxes_l1754_175470


namespace NUMINAMATH_CALUDE_sin_eq_sin_sin_solution_count_l1754_175401

theorem sin_eq_sin_sin_solution_count :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ Real.arcsin 0.99 ∧ Real.sin x = Real.sin (Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_sin_eq_sin_sin_solution_count_l1754_175401


namespace NUMINAMATH_CALUDE_sector_properties_l1754_175468

/-- Given a sector OAB with central angle 120° and radius 6, 
    prove the length of arc AB and the area of segment AOB -/
theorem sector_properties :
  let angle : Real := 120 * π / 180
  let radius : Real := 6
  let arc_length : Real := radius * angle
  let sector_area : Real := (1 / 2) * radius * arc_length
  let triangle_area : Real := (1 / 2) * radius * radius * Real.sin angle
  let segment_area : Real := sector_area - triangle_area
  arc_length = 4 * π ∧ segment_area = 12 * π - 9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_properties_l1754_175468


namespace NUMINAMATH_CALUDE_rectangular_plot_ratio_l1754_175419

/-- Proves that for a rectangular plot with given area and breadth, the ratio of length to breadth is 3:1 -/
theorem rectangular_plot_ratio (area : ℝ) (breadth : ℝ) (length : ℝ) : 
  area = 972 →
  breadth = 18 →
  area = length * breadth →
  length / breadth = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_ratio_l1754_175419


namespace NUMINAMATH_CALUDE_united_additional_charge_value_l1754_175441

/-- Represents the additional charge per minute for United Telephone -/
def united_additional_charge : ℝ := sorry

/-- The base rate for United Telephone -/
def united_base_rate : ℝ := 11

/-- The base rate for Atlantic Call -/
def atlantic_base_rate : ℝ := 12

/-- The additional charge per minute for Atlantic Call -/
def atlantic_additional_charge : ℝ := 0.2

/-- The number of minutes for which the bills are equal -/
def equal_bill_minutes : ℝ := 20

theorem united_additional_charge_value : 
  (united_base_rate + equal_bill_minutes * united_additional_charge = 
   atlantic_base_rate + equal_bill_minutes * atlantic_additional_charge) → 
  united_additional_charge = 0.25 := by sorry

end NUMINAMATH_CALUDE_united_additional_charge_value_l1754_175441


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_on_interval_l1754_175486

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^5 - 5 * x^3

-- State the theorem
theorem f_monotone_decreasing_on_interval :
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_on_interval_l1754_175486


namespace NUMINAMATH_CALUDE_function_equality_l1754_175446

theorem function_equality (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (x + y) + y ≤ f (f (f x))) → 
  ∃ c : ℝ, ∀ x : ℝ, f x = c - x :=
sorry

end NUMINAMATH_CALUDE_function_equality_l1754_175446


namespace NUMINAMATH_CALUDE_total_ebook_readers_l1754_175427

/-- The number of eBook readers Anna bought -/
def anna_readers : ℕ := 50

/-- The number of eBook readers John bought initially -/
def john_initial_readers : ℕ := anna_readers - 15

/-- The number of eBook readers John lost -/
def john_lost_readers : ℕ := 3

/-- The number of eBook readers John has after losing some -/
def john_final_readers : ℕ := john_initial_readers - john_lost_readers

/-- The total number of eBook readers John and Anna have together -/
def total_readers : ℕ := anna_readers + john_final_readers

theorem total_ebook_readers :
  total_readers = 82 :=
by sorry

end NUMINAMATH_CALUDE_total_ebook_readers_l1754_175427


namespace NUMINAMATH_CALUDE_prank_combinations_l1754_175496

/-- The number of choices for each day of the week --/
def monday_choices : ℕ := 1
def tuesday_choices : ℕ := 4
def wednesday_choices : ℕ := 7
def thursday_choices : ℕ := 5
def friday_choices : ℕ := 1

/-- The total number of combinations --/
def total_combinations : ℕ := monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices

/-- Theorem stating that the total number of combinations is 140 --/
theorem prank_combinations : total_combinations = 140 := by
  sorry

end NUMINAMATH_CALUDE_prank_combinations_l1754_175496


namespace NUMINAMATH_CALUDE_max_fraction_value_l1754_175489

theorem max_fraction_value (A B : ℝ) (h1 : A + B = 2020) (h2 : A / B < 1 / 4) :
  A / B ≤ 403 / 1617 :=
sorry

end NUMINAMATH_CALUDE_max_fraction_value_l1754_175489


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l1754_175420

theorem quadratic_roots_expression (p q : ℝ) : 
  (3 * p ^ 2 + 9 * p - 21 = 0) →
  (3 * q ^ 2 + 9 * q - 21 = 0) →
  (3 * p - 4) * (6 * q - 8) = 14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l1754_175420


namespace NUMINAMATH_CALUDE_output_after_year_formula_l1754_175413

/-- Calculates the output after 12 months given an initial output and monthly growth rate -/
def outputAfterYear (a : ℝ) (p : ℝ) : ℝ := a * (1 + p) ^ 12

/-- Theorem stating that the output after 12 months is equal to a(1+p)^12 -/
theorem output_after_year_formula (a : ℝ) (p : ℝ) :
  outputAfterYear a p = a * (1 + p) ^ 12 := by sorry

end NUMINAMATH_CALUDE_output_after_year_formula_l1754_175413


namespace NUMINAMATH_CALUDE_cosine_product_identity_l1754_175460

theorem cosine_product_identity (n : ℕ) (hn : n = 7 ∨ n = 9) : 
  Real.cos (2 * Real.pi / n) * Real.cos (4 * Real.pi / n) * Real.cos (8 * Real.pi / n) = 
  (-1 : ℝ) ^ ((n - 1) / 2) * (1 / 8) :=
by sorry

end NUMINAMATH_CALUDE_cosine_product_identity_l1754_175460


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1754_175450

/-- Given a line l with y-intercept 1 and perpendicular to y = (1/2)x, 
    prove that the equation of l is y = -2x + 1 -/
theorem perpendicular_line_equation (l : Set (ℝ × ℝ)) 
  (y_intercept : (0, 1) ∈ l)
  (perpendicular : ∀ (x y : ℝ), (x, y) ∈ l → (y - 1) = m * x → m * (1/2) = -1) :
  ∀ (x y : ℝ), (x, y) ∈ l ↔ y = -2 * x + 1 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1754_175450


namespace NUMINAMATH_CALUDE_line_equations_l1754_175443

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

-- Define point P
def point_P : ℝ × ℝ := (2, 0)

-- Define the property of line l1
def line_l1_property (l : ℝ → ℝ → Prop) : Prop :=
  l point_P.1 point_P.2 ∧
  ∃ a b : ℝ, ∀ x y : ℝ, l x y ↔ (x = a ∨ b*x - y = 0) ∧
  ∃ x1 y1 x2 y2 : ℝ,
    circle_C x1 y1 ∧ circle_C x2 y2 ∧ l x1 y1 ∧ l x2 y2 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = 32

-- Define the property of line l2
def line_l2_property (l : ℝ → ℝ → Prop) : Prop :=
  (∀ x y z w : ℝ, l x y ∧ l z w → y - x = w - z) ∧
  ∃ x1 y1 x2 y2 : ℝ,
    circle_C x1 y1 ∧ circle_C x2 y2 ∧ l x1 y1 ∧ l x2 y2 ∧
    x1*x2 + y1*y2 = 0

-- Theorem statement
theorem line_equations :
  ∃ l1 l2 : ℝ → ℝ → Prop,
    line_l1_property l1 ∧ line_l2_property l2 ∧
    (∀ x y : ℝ, l1 x y ↔ (x = 2 ∨ 3*x - 4*y - 6 = 0)) ∧
    (∀ x y : ℝ, l2 x y ↔ (x - y - 4 = 0 ∨ x - y + 1 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_line_equations_l1754_175443


namespace NUMINAMATH_CALUDE_tan_70_plus_tan_50_minus_sqrt3_tan_70_tan_50_l1754_175417

theorem tan_70_plus_tan_50_minus_sqrt3_tan_70_tan_50 :
  Real.tan (70 * π / 180) + Real.tan (50 * π / 180) - Real.sqrt 3 * Real.tan (70 * π / 180) * Real.tan (50 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_70_plus_tan_50_minus_sqrt3_tan_70_tan_50_l1754_175417


namespace NUMINAMATH_CALUDE_sheet_width_correct_l1754_175478

/-- The width of a rectangular metallic sheet, given specific conditions -/
def sheet_width : ℝ :=
  let length : ℝ := 48
  let cut_size : ℝ := 8
  let box_volume : ℝ := 5120
  let width : ℝ := 36
  width

/-- Theorem stating that the sheet_width satisfies the given conditions -/
theorem sheet_width_correct : 
  let length : ℝ := 48
  let cut_size : ℝ := 8
  let box_volume : ℝ := 5120
  let width := sheet_width
  box_volume = (length - 2 * cut_size) * (width - 2 * cut_size) * cut_size :=
by sorry

end NUMINAMATH_CALUDE_sheet_width_correct_l1754_175478


namespace NUMINAMATH_CALUDE_fourth_largest_divisor_l1754_175485

def n : ℕ := 1234560000

-- Define a function to get the list of divisors
def divisors (m : ℕ) : List ℕ := sorry

-- Define a function to get the nth largest element from a list
def nthLargest (l : List ℕ) (k : ℕ) : ℕ := sorry

theorem fourth_largest_divisor :
  nthLargest (divisors n) 4 = 154320000 := by sorry

end NUMINAMATH_CALUDE_fourth_largest_divisor_l1754_175485


namespace NUMINAMATH_CALUDE_simplify_expression_l1754_175454

theorem simplify_expression (a b : ℝ) : 
  -2 * (a^3 - 3*b^2) + 4 * (-b^2 + a^3) = 2*a^3 + 2*b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1754_175454


namespace NUMINAMATH_CALUDE_minimum_width_for_garden_l1754_175409

-- Define the garden width as a real number
variable (w : ℝ)

-- Define the conditions of the problem
def garden_length (w : ℝ) : ℝ := w + 10
def garden_area (w : ℝ) : ℝ := w * garden_length w
def area_constraint (w : ℝ) : Prop := garden_area w ≥ 150

-- Theorem statement
theorem minimum_width_for_garden :
  (∀ x : ℝ, x > 0 → area_constraint x → x ≥ 10) ∧ area_constraint 10 :=
sorry

end NUMINAMATH_CALUDE_minimum_width_for_garden_l1754_175409


namespace NUMINAMATH_CALUDE_cost_per_song_l1754_175471

/-- Calculates the cost per song given monthly music purchase, average song length, and annual expenditure -/
theorem cost_per_song 
  (monthly_hours : ℝ) 
  (song_length_minutes : ℝ) 
  (annual_cost : ℝ) 
  (h1 : monthly_hours = 20)
  (h2 : song_length_minutes = 3)
  (h3 : annual_cost = 2400) : 
  annual_cost / (monthly_hours * 12 * 60 / song_length_minutes) = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_song_l1754_175471


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1754_175447

/-- An isosceles triangle with sides of length 8 and 3 has a perimeter of 19 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 8 ∧ b = 8 ∧ c = 3 →
  a + b > c ∧ b + c > a ∧ a + c > b →
  a + b + c = 19 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1754_175447


namespace NUMINAMATH_CALUDE_reflection_sum_l1754_175482

/-- Given a point C with coordinates (3, y) that is reflected over the line y = x to point D,
    the sum of all coordinate values of C and D is equal to 2y + 6. -/
theorem reflection_sum (y : ℝ) : 
  let C := (3, y)
  let D := (y, 3)
  (C.1 + C.2 + D.1 + D.2) = 2 * y + 6 := by
  sorry

end NUMINAMATH_CALUDE_reflection_sum_l1754_175482


namespace NUMINAMATH_CALUDE_smallest_k_for_inequality_l1754_175449

theorem smallest_k_for_inequality : 
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (a : ℝ) (n : ℕ), 0 ≤ a ∧ a ≤ 1 ∧ n > 0 → a^k * (1-a)^n < 1 / (n+1)^3) ∧
  (∀ (k' : ℕ), 0 < k' ∧ k' < k → 
    ∃ (a : ℝ) (n : ℕ), 0 ≤ a ∧ a ≤ 1 ∧ n > 0 ∧ a^k' * (1-a)^n ≥ 1 / (n+1)^3) ∧
  k = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_inequality_l1754_175449


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_l1754_175425

theorem regular_polygon_interior_angle (C : ℕ) : 
  C > 2 → (288 : ℝ) = (C - 2 : ℝ) * 180 / C → C = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_l1754_175425


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l1754_175444

theorem radical_conjugate_sum_product (a b : ℝ) 
  (h1 : (a + Real.sqrt b) + (a - Real.sqrt b) = 4)
  (h2 : (a + Real.sqrt b) * (a - Real.sqrt b) = 9) : 
  a + b = -3 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l1754_175444


namespace NUMINAMATH_CALUDE_connor_score_l1754_175410

theorem connor_score (connor amy jason : ℕ) : 
  (amy = connor + 4) →
  (jason = 2 * amy) →
  (connor + amy + jason = 20) →
  connor = 2 := by sorry

end NUMINAMATH_CALUDE_connor_score_l1754_175410


namespace NUMINAMATH_CALUDE_sum_of_ages_l1754_175498

theorem sum_of_ages (henry_age jill_age : ℕ) : 
  henry_age = 23 →
  jill_age = 17 →
  henry_age - 11 = 2 * (jill_age - 11) →
  henry_age + jill_age = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l1754_175498


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l1754_175474

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by sorry

theorem increase_80_by_150_percent :
  80 * (1 + 150 / 100) = 200 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l1754_175474


namespace NUMINAMATH_CALUDE_unique_number_l1754_175411

theorem unique_number : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  n % 13 = 0 ∧ 
  n > 26 ∧ 
  n % 7 = 0 ∧ 
  n % 10 ≠ 6 ∧ 
  n % 10 ≠ 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_l1754_175411


namespace NUMINAMATH_CALUDE_eighth_term_of_happy_sequence_l1754_175404

def happy_sequence (n : ℕ) : ℚ := (-1)^n * (n : ℚ) / 2^n

theorem eighth_term_of_happy_sequence :
  happy_sequence 8 = 1/32 := by sorry

end NUMINAMATH_CALUDE_eighth_term_of_happy_sequence_l1754_175404


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1754_175400

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {1, 2}

theorem intersection_of_M_and_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1754_175400


namespace NUMINAMATH_CALUDE_complement_of_A_union_B_l1754_175408

open Set

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 5}
def B : Set ℝ := {x : ℝ | x^2 ≥ 4}

-- State the theorem
theorem complement_of_A_union_B :
  (A ∪ B)ᶜ = Ioc (-2) (-1) := by sorry

end NUMINAMATH_CALUDE_complement_of_A_union_B_l1754_175408


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1754_175465

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 4 * x + a > 0) ↔ a > 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1754_175465


namespace NUMINAMATH_CALUDE_boat_upstream_distance_l1754_175493

/-- Proves that a boat traveling 11 km downstream in one hour with a still water speed of 8 km/h
    will travel 5 km upstream in one hour. -/
theorem boat_upstream_distance
  (boat_speed : ℝ)
  (downstream_distance : ℝ)
  (h1 : boat_speed = 8)
  (h2 : downstream_distance = 11) :
  boat_speed - (downstream_distance - boat_speed) = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_boat_upstream_distance_l1754_175493


namespace NUMINAMATH_CALUDE_abc_inequality_l1754_175421

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b * c * (a + b + c + Real.sqrt (a^2 + b^2 + c^2))) / 
  ((a^2 + b^2 + c^2) * (a * b + b * c + a * c)) ≤ (3 + Real.sqrt 3) / 9 :=
sorry

end NUMINAMATH_CALUDE_abc_inequality_l1754_175421


namespace NUMINAMATH_CALUDE_shaded_to_unshaded_ratio_l1754_175469

/-- Represents a rectangular grid with specific properties -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)
  (qr_length : ℕ)
  (st_length : ℕ)
  (rstu_height : ℕ)

/-- Calculates the area of a right triangle given its base and height -/
def triangle_area (base height : ℕ) : ℚ :=
  (base * height : ℚ) / 2

/-- Calculates the area of a rectangle given its width and height -/
def rectangle_area (width height : ℕ) : ℕ :=
  width * height

/-- Calculates the shaded area of the grid -/
def shaded_area (g : Grid) : ℚ :=
  triangle_area g.qr_length g.height +
  triangle_area g.st_length (g.height - g.rstu_height) +
  rectangle_area (g.st_length) g.rstu_height

/-- Calculates the total area of the grid -/
def total_area (g : Grid) : ℕ :=
  rectangle_area g.width g.height

/-- Calculates the unshaded area of the grid -/
def unshaded_area (g : Grid) : ℚ :=
  (total_area g : ℚ) - shaded_area g

/-- Theorem stating the ratio of shaded to unshaded area -/
theorem shaded_to_unshaded_ratio (g : Grid) (h1 : g.width = 9) (h2 : g.height = 4)
    (h3 : g.qr_length = 3) (h4 : g.st_length = 4) (h5 : g.rstu_height = 2) :
    (shaded_area g) / (unshaded_area g) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_shaded_to_unshaded_ratio_l1754_175469


namespace NUMINAMATH_CALUDE_water_difference_l1754_175437

theorem water_difference (s h : ℝ) 
  (h1 : s > h) 
  (h2 : (s - 0.43) - (h + 0.43) = 0.88) : 
  s - h = 1.74 := by
  sorry

end NUMINAMATH_CALUDE_water_difference_l1754_175437


namespace NUMINAMATH_CALUDE_equation_solution_l1754_175451

theorem equation_solution : 
  ∃! x : ℚ, (x - 15) / 3 = (3 * x + 10) / 8 :=
by
  use (-150)
  constructor
  · -- Prove that x = -150 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

end NUMINAMATH_CALUDE_equation_solution_l1754_175451


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_range_l1754_175442

/-- The range of m for which the line 2kx-y+1=0 always intersects the ellipse x²/9 + y²/m = 1 -/
theorem line_ellipse_intersection_range :
  ∀ (k : ℝ), (∀ (x y : ℝ), 2 * k * x - y + 1 = 0 → x^2 / 9 + y^2 / m = 1) →
  m ∈ Set.Icc 1 9 ∪ Set.Ioi 9 :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_range_l1754_175442


namespace NUMINAMATH_CALUDE_polynomial_uniqueness_l1754_175440

def Q (x : ℝ) (Q0 Q1 Q2 : ℝ) : ℝ := Q0 + Q1 * x + Q2 * x^2

theorem polynomial_uniqueness (Q0 Q1 Q2 : ℝ) :
  Q (-1) Q0 Q1 Q2 = -3 →
  Q 3 Q0 Q1 Q2 = 5 →
  ∀ x, Q x Q0 Q1 Q2 = 3 * x^2 + 7 * x - 5 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_uniqueness_l1754_175440


namespace NUMINAMATH_CALUDE_disjoint_sets_range_l1754_175497

def set_A : Set (ℝ × ℝ) := {p | p.2 = -|p.1| - 2}

def set_B (a : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - a)^2 + p.2^2 = a^2}

theorem disjoint_sets_range (a : ℝ) :
  set_A ∩ set_B a = ∅ ↔ -2 * Real.sqrt 2 - 2 < a ∧ a < 2 * Real.sqrt 2 + 2 :=
sorry

end NUMINAMATH_CALUDE_disjoint_sets_range_l1754_175497


namespace NUMINAMATH_CALUDE_line_divides_polygon_equally_l1754_175457

/-- Polygon type representing a closed shape with vertices --/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Line type representing a line in slope-intercept form --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Calculate the area of a polygon using the shoelace formula --/
def polygonArea (p : Polygon) : ℝ := sorry

/-- Check if a point lies on a line --/
def pointOnLine (l : Line) (p : ℝ × ℝ) : Prop := sorry

/-- Check if a line divides a polygon into two equal areas --/
def dividesEqualArea (l : Line) (p : Polygon) : Prop := sorry

/-- The main theorem --/
theorem line_divides_polygon_equally (polygon : Polygon) (line : Line) :
  polygon.vertices = [(0, 0), (0, 6), (4, 6), (4, 4), (6, 4), (6, 0)] →
  line.slope = -1/3 →
  line.intercept = 11/3 →
  pointOnLine line (2, 3) →
  dividesEqualArea line polygon := by
  sorry

end NUMINAMATH_CALUDE_line_divides_polygon_equally_l1754_175457


namespace NUMINAMATH_CALUDE_find_B_l1754_175459

theorem find_B (A C B : ℤ) (h1 : C - A = 204) (h2 : A = 520) (h3 : B + 179 = C) : B = 545 := by
  sorry

end NUMINAMATH_CALUDE_find_B_l1754_175459


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1754_175477

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point
  (l : Line)
  (p : Point)
  (hl : l.a = 3 ∧ l.b = -2 ∧ l.c = 1)
  (hp : p.x = 1 ∧ p.y = 1) :
  ∃ (l' : Line), l'.a = 3 ∧ l'.b = -2 ∧ l'.c = -1 ∧
    p.liesOn l' ∧ l'.isParallel l :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1754_175477


namespace NUMINAMATH_CALUDE_pens_distribution_l1754_175402

/-- The number of pens in each pack -/
def pens_per_pack : ℕ := 3

/-- The number of packs Kendra has -/
def kendra_packs : ℕ := 4

/-- The number of packs Tony has -/
def tony_packs : ℕ := 2

/-- The number of pens Kendra and Tony each keep for themselves -/
def pens_kept_each : ℕ := 2

/-- The total number of friends who will receive pens -/
def friends_receiving_pens : ℕ := 
  kendra_packs * pens_per_pack + tony_packs * pens_per_pack - 2 * pens_kept_each

theorem pens_distribution :
  friends_receiving_pens = 14 := by
  sorry

end NUMINAMATH_CALUDE_pens_distribution_l1754_175402


namespace NUMINAMATH_CALUDE_alaya_fruit_salads_l1754_175403

def fruit_salad_problem (alaya : ℕ) (angel : ℕ) : Prop :=
  angel = 2 * alaya ∧ alaya + angel = 600

theorem alaya_fruit_salads :
  ∃ (alaya : ℕ), fruit_salad_problem alaya (2 * alaya) ∧ alaya = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_alaya_fruit_salads_l1754_175403


namespace NUMINAMATH_CALUDE_max_value_expression_l1754_175415

theorem max_value_expression (x₁ x₂ x₃ x₄ : ℝ)
  (h₁ : 0 ≤ x₁ ∧ x₁ ≤ 1)
  (h₂ : 0 ≤ x₂ ∧ x₂ ≤ 1)
  (h₃ : 0 ≤ x₃ ∧ x₃ ≤ 1)
  (h₄ : 0 ≤ x₄ ∧ x₄ ≤ 1) :
  (∀ y₁ y₂ y₃ y₄ : ℝ,
    0 ≤ y₁ ∧ y₁ ≤ 1 →
    0 ≤ y₂ ∧ y₂ ≤ 1 →
    0 ≤ y₃ ∧ y₃ ≤ 1 →
    0 ≤ y₄ ∧ y₄ ≤ 1 →
    1 - (1 - y₁) * (1 - y₂) * (1 - y₃) * (1 - y₄) ≤ 1 - (1 - x₁) * (1 - x₂) * (1 - x₃) * (1 - x₄)) ∧
  1 - (1 - x₁) * (1 - x₂) * (1 - x₃) * (1 - x₄) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1754_175415


namespace NUMINAMATH_CALUDE_function_inequality_range_l1754_175453

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem function_inequality_range 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_mono : monotone_increasing_on f (Set.Ici 0)) :
  {x : ℝ | f (2*x - 1) < f (1/3)} = Set.Ioo (1/3) (2/3) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_range_l1754_175453


namespace NUMINAMATH_CALUDE_inequality_proof_l1754_175480

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (2 * x^2) / (y + z) + (2 * y^2) / (z + x) + (2 * z^2) / (x + y) ≥ x + y + z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1754_175480


namespace NUMINAMATH_CALUDE_system_solutions_correct_l1754_175461

theorem system_solutions_correct :
  -- Problem 1
  (∃ x y : ℚ, x + y = 4 ∧ 2 * x - y = 5) ∧
  -- Problem 2
  (∃ m n : ℚ, 3 * m - 4 * n = 4 ∧ m / 2 + n / 6 = 1) ∧
  -- Solutions
  (3 + 1 = 4 ∧ 2 * 3 - 1 = 5) ∧
  (3 * (28 / 15) - 4 * (2 / 5) = 4 ∧ (28 / 15) / 2 + (2 / 5) / 6 = 1) :=
by sorry


end NUMINAMATH_CALUDE_system_solutions_correct_l1754_175461


namespace NUMINAMATH_CALUDE_problem_solution_l1754_175429

theorem problem_solution : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * 
  (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) * Nat.factorial 7 = 
  (5^128 - 4^128) * 5040 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1754_175429


namespace NUMINAMATH_CALUDE_farey_sequence_mediant_l1754_175430

theorem farey_sequence_mediant (r s : ℕ+) : 
  (6:ℚ)/11 < r/s ∧ r/s < (5:ℚ)/9 ∧ 
  (∀ (r' s' : ℕ+), (6:ℚ)/11 < r'/s' ∧ r'/s' < (5:ℚ)/9 → s ≤ s') →
  s - r = 9 :=
by sorry

end NUMINAMATH_CALUDE_farey_sequence_mediant_l1754_175430


namespace NUMINAMATH_CALUDE_total_water_consumed_l1754_175462

-- Define the conversion rate from quarts to ounces
def quart_to_ounce : ℚ := 32

-- Define the amount of water in the bottle (in quarts)
def bottle_quarts : ℚ := 3/2

-- Define the amount of water in the can (in ounces)
def can_ounces : ℚ := 12

-- Theorem statement
theorem total_water_consumed (quart_to_ounce bottle_quarts can_ounces : ℚ) :
  quart_to_ounce * bottle_quarts + can_ounces = 60 :=
sorry

end NUMINAMATH_CALUDE_total_water_consumed_l1754_175462


namespace NUMINAMATH_CALUDE_garden_flower_distribution_l1754_175423

theorem garden_flower_distribution :
  ∀ (total_flowers white_flowers red_flowers white_roses white_tulips red_roses red_tulips : ℕ),
  total_flowers = 100 →
  white_flowers = 60 →
  red_flowers = total_flowers - white_flowers →
  white_roses = (3 * white_flowers) / 5 →
  white_tulips = white_flowers - white_roses →
  red_tulips = red_flowers / 2 →
  red_roses = red_flowers - red_tulips →
  (white_tulips + red_tulips) * 100 / total_flowers = 44 ∧
  (white_roses + red_roses) * 100 / total_flowers = 56 :=
by sorry

end NUMINAMATH_CALUDE_garden_flower_distribution_l1754_175423


namespace NUMINAMATH_CALUDE_aaron_age_proof_l1754_175422

def has_all_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → ∃ k : ℕ, n / 10^k % 10 = d

theorem aaron_age_proof :
  ∃! m : ℕ,
    1000 ≤ m^3 ∧ m^3 < 10000 ∧
    100000 ≤ m^4 ∧ m^4 < 1000000 ∧
    has_all_digits (m^3 + m^4) ∧
    m = 18 := by
  sorry

end NUMINAMATH_CALUDE_aaron_age_proof_l1754_175422


namespace NUMINAMATH_CALUDE_chocolate_division_l1754_175428

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_given : ℕ) :
  total_chocolate = 75 / 7 →
  num_piles = 5 →
  piles_given = 2 →
  (total_chocolate / num_piles) * piles_given = 30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l1754_175428


namespace NUMINAMATH_CALUDE_system_solution_l1754_175458

theorem system_solution :
  ∃ (x y z t : ℂ),
    x + y = 10 ∧
    z + t = 5 ∧
    x * y = z * t ∧
    x^3 + y^3 + z^3 + t^3 = 1080 ∧
    x = 5 + Real.sqrt 17 ∧
    y = 5 - Real.sqrt 17 ∧
    z = (5 + Complex.I * Real.sqrt 7) / 2 ∧
    t = (5 - Complex.I * Real.sqrt 7) / 2 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1754_175458


namespace NUMINAMATH_CALUDE_sport_water_amount_l1754_175456

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation ratio -/
def sport_ratio : DrinkRatio :=
  { flavoring := standard_ratio.flavoring,
    corn_syrup := standard_ratio.corn_syrup / 3,
    water := standard_ratio.water * 2 }

/-- Amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℚ := 4

/-- Theorem stating the amount of water in the sport formulation -/
theorem sport_water_amount :
  (sport_corn_syrup * sport_ratio.water) / sport_ratio.corn_syrup = 15 := by
  sorry

end NUMINAMATH_CALUDE_sport_water_amount_l1754_175456


namespace NUMINAMATH_CALUDE_max_area_triangle_l1754_175464

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the maximum area of the triangle is √2 + 1 when a*sin(C) = c*cos(A) and a = 2 -/
theorem max_area_triangle (a b c A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Sides are positive
  a * Real.sin C = c * Real.cos A ∧  -- Given condition
  a = 2 →  -- Given condition
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧  -- Area formula
              ∀ (S' : ℝ), S' = (1/2) * b * c * Real.sin A → S' ≤ S) ∧  -- S is maximum
  ((1/2) * b * c * Real.sin A ≤ Real.sqrt 2 + 1) ∧  -- Upper bound
  (∃ b' c', (1/2) * b' * c' * Real.sin A = Real.sqrt 2 + 1)  -- Maximum is achievable
  := by sorry

end NUMINAMATH_CALUDE_max_area_triangle_l1754_175464


namespace NUMINAMATH_CALUDE_find_A_l1754_175490

theorem find_A (A B : ℕ) (h1 : A + B = 1149) (h2 : A = 8 * B + 24) : A = 1024 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l1754_175490


namespace NUMINAMATH_CALUDE_club_selection_count_l1754_175412

/-- A club with members of different genders and service lengths -/
structure Club where
  total_members : Nat
  boys : Nat
  girls : Nat
  longest_serving_boys : Nat
  longest_serving_girls : Nat

/-- The conditions for selecting president and vice-president -/
def valid_selection (c : Club) : Prop :=
  c.total_members = 30 ∧
  c.boys = 15 ∧
  c.girls = 15 ∧
  c.longest_serving_boys = 6 ∧
  c.longest_serving_girls = 6

/-- The number of ways to select a president and vice-president -/
def selection_count (c : Club) : Nat :=
  c.longest_serving_boys * c.girls + c.longest_serving_girls * c.boys

/-- Theorem stating the number of ways to select president and vice-president -/
theorem club_selection_count (c : Club) :
  valid_selection c → selection_count c = 180 := by
  sorry

end NUMINAMATH_CALUDE_club_selection_count_l1754_175412


namespace NUMINAMATH_CALUDE_nine_sided_polygon_odd_spanning_diagonals_l1754_175416

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- We don't need to define the specifics of a regular polygon for this problem

/-- The number of diagonals in a regular polygon that span an odd number of vertices between their endpoints -/
def oddSpanningDiagonals (p : RegularPolygon n) : ℕ :=
  sorry  -- Definition to be implemented

/-- Theorem stating that a regular nine-sided polygon has 18 diagonals spanning an odd number of vertices -/
theorem nine_sided_polygon_odd_spanning_diagonals :
  ∀ (p : RegularPolygon 9), oddSpanningDiagonals p = 18 :=
by sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_odd_spanning_diagonals_l1754_175416


namespace NUMINAMATH_CALUDE_simplify_sqrt_fraction_simplify_sqrt_expression_l1754_175439

-- Part 1
theorem simplify_sqrt_fraction :
  (Real.sqrt 5 + 1) / (Real.sqrt 5 - 1) = (3 + Real.sqrt 5) / 2 := by sorry

-- Part 2
theorem simplify_sqrt_expression :
  Real.sqrt 12 * Real.sqrt 2 / Real.sqrt ((-3)^2) = 2 * Real.sqrt 6 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_fraction_simplify_sqrt_expression_l1754_175439


namespace NUMINAMATH_CALUDE_negation_of_forall_geq_zero_l1754_175487

theorem negation_of_forall_geq_zero (R : Type*) [OrderedRing R] :
  (¬ (∀ x : R, x^2 - 3 ≥ 0)) ↔ (∃ x₀ : R, x₀^2 - 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_geq_zero_l1754_175487


namespace NUMINAMATH_CALUDE_mitchell_gum_chewing_l1754_175479

/-- 
Given 8 packets of gum with 7 pieces each, and leaving 2 pieces unchewed,
prove that the number of pieces chewed is equal to 54.
-/
theorem mitchell_gum_chewing (packets : Nat) (pieces_per_packet : Nat) (unchewed : Nat) : 
  packets = 8 → pieces_per_packet = 7 → unchewed = 2 →
  packets * pieces_per_packet - unchewed = 54 := by
sorry

end NUMINAMATH_CALUDE_mitchell_gum_chewing_l1754_175479
