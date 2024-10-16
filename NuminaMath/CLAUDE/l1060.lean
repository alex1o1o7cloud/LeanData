import Mathlib

namespace NUMINAMATH_CALUDE_nested_radical_value_l1060_106087

/-- The value of the infinite nested radical sqrt(3 - sqrt(3 - sqrt(3 - ...))) -/
noncomputable def nestedRadical : ℝ :=
  Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt 3))))))

/-- Theorem stating that the nested radical equals (-1 + √13) / 2 -/
theorem nested_radical_value : nestedRadical = (-1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_value_l1060_106087


namespace NUMINAMATH_CALUDE_annual_output_scientific_notation_l1060_106038

/-- The annual output of the photovoltaic power station in kWh -/
def annual_output : ℝ := 448000

/-- The scientific notation representation of the annual output -/
def scientific_notation : ℝ := 4.48 * (10 ^ 5)

theorem annual_output_scientific_notation : annual_output = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_annual_output_scientific_notation_l1060_106038


namespace NUMINAMATH_CALUDE_smallest_divisible_m_l1060_106026

theorem smallest_divisible_m : ∃ (m : ℕ),
  (∀ k < m, ¬(k + 9 ∣ k^3 - 90)) ∧ (m + 9 ∣ m^3 - 90) ∧ m = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_m_l1060_106026


namespace NUMINAMATH_CALUDE_initial_dolphins_count_l1060_106066

/-- The initial number of dolphins in the ocean -/
def initial_dolphins : ℕ := 65

/-- The number of dolphins joining from the river -/
def joining_dolphins : ℕ := 3 * initial_dolphins

/-- The total number of dolphins after joining -/
def total_dolphins : ℕ := 260

theorem initial_dolphins_count : initial_dolphins = 65 :=
  by sorry

end NUMINAMATH_CALUDE_initial_dolphins_count_l1060_106066


namespace NUMINAMATH_CALUDE_polynomial_expansion_value_l1060_106094

/-- The value of a in the expansion of (x+y)^7 -/
theorem polynomial_expansion_value (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧ a + b = 1 ∧ 
  (21 * a^5 * b^2 = 35 * a^4 * b^3) →
  a = 5/8 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_value_l1060_106094


namespace NUMINAMATH_CALUDE_jeds_change_l1060_106053

/-- Given the conditions of Jed's board game purchase, prove the number of $5 bills received as change. -/
theorem jeds_change (num_games : ℕ) (game_cost : ℕ) (payment : ℕ) (change_bill : ℕ) : 
  num_games = 6 → 
  game_cost = 15 → 
  payment = 100 → 
  change_bill = 5 → 
  (payment - num_games * game_cost) / change_bill = 2 := by
sorry

end NUMINAMATH_CALUDE_jeds_change_l1060_106053


namespace NUMINAMATH_CALUDE_variation_problem_l1060_106096

/-- Given that R varies directly as S and inversely as T^2, prove that when R = 50 and T = 5, S = 5000/3 -/
theorem variation_problem (c : ℝ) (R S T : ℝ → ℝ) (t : ℝ) :
  (∀ t, R t = c * S t / (T t)^2) →  -- Relationship between R, S, and T
  R 0 = 3 →                        -- Initial condition for R
  S 0 = 16 →                       -- Initial condition for S
  T 0 = 2 →                        -- Initial condition for T
  R t = 50 →                       -- New value for R
  T t = 5 →                        -- New value for T
  S t = 5000 / 3 := by             -- Prove that S equals 5000/3
sorry


end NUMINAMATH_CALUDE_variation_problem_l1060_106096


namespace NUMINAMATH_CALUDE_shirley_sold_54_boxes_l1060_106003

/-- The number of cases Shirley needs to deliver -/
def num_cases : ℕ := 9

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 6

/-- The total number of boxes Shirley sold -/
def total_boxes : ℕ := num_cases * boxes_per_case

theorem shirley_sold_54_boxes : total_boxes = 54 := by
  sorry

end NUMINAMATH_CALUDE_shirley_sold_54_boxes_l1060_106003


namespace NUMINAMATH_CALUDE_average_weight_of_children_l1060_106062

def regression_equation (x : ℝ) : ℝ := 2 * x + 7

def children_ages : List ℝ := [2, 3, 3, 5, 2, 6, 7, 3, 4, 5]

theorem average_weight_of_children :
  let weights := children_ages.map regression_equation
  (weights.sum / weights.length) = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_of_children_l1060_106062


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l1060_106044

/-- Given two algebraic terms are like terms, prove their exponents sum to 6 -/
theorem like_terms_exponent_sum (a b : ℝ) (m n : ℕ) : 
  (∃ (k : ℝ), k * a^m * b^2 = (1/2) * a^5 * b^(n+1)) → m + n = 6 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l1060_106044


namespace NUMINAMATH_CALUDE_sum_x_y_equals_three_l1060_106057

def A (x : ℝ) : Set ℝ := {2, x}
def B (x y : ℝ) : Set ℝ := {x*y, 1}

theorem sum_x_y_equals_three (x y : ℝ) : A x = B x y → x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_three_l1060_106057


namespace NUMINAMATH_CALUDE_system_solution_l1060_106001

theorem system_solution :
  ∀ x y z : ℝ,
  (x * y = z * (x + y + z) ∧
   y * z = 4 * x * (x + y + z) ∧
   z * x = 9 * y * (x + y + z)) →
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨
   ∃ t : ℝ, t ≠ 0 ∧ x = -3 * t ∧ y = -2 * t ∧ z = 6 * t) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1060_106001


namespace NUMINAMATH_CALUDE_valid_arrangements_five_people_l1060_106056

/-- The number of people in the arrangement -/
def n : ℕ := 5

/-- The number of ways to arrange n people such that at least one of two specific people (A and B) is at one of the ends -/
def validArrangements (n : ℕ) : ℕ :=
  n.factorial - (n - 2).factorial * (n - 2).factorial

theorem valid_arrangements_five_people :
  validArrangements n = 84 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_five_people_l1060_106056


namespace NUMINAMATH_CALUDE_min_value_theorem_l1060_106092

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 2) :
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 2 → (x + y) / (x^2 * y * z) ≤ (a + b) / (a^2 * b * c)) →
  (x + y) / (x^2 * y * z) = 13.5 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1060_106092


namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_l1060_106082

theorem cubic_root_sum_squares (p q r : ℝ) (x : ℝ → ℝ) 
  (hx : ∀ t, x t = t^3 - p*t^2 + q*t - r) : 
  ∃ (r s t : ℝ), (x r = 0 ∧ x s = 0 ∧ x t = 0) ∧ 
  (r^2 + s^2 + t^2 = p^2 - 2*q) := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_l1060_106082


namespace NUMINAMATH_CALUDE_congruence_problem_l1060_106091

theorem congruence_problem : ∃ n : ℤ, 0 ≤ n ∧ n < 9 ∧ -1234 ≡ n [ZMOD 9] ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1060_106091


namespace NUMINAMATH_CALUDE_lucy_shell_count_l1060_106007

/-- Lucy's shell counting problem -/
theorem lucy_shell_count (initial_shells final_shells : ℕ) 
  (h1 : initial_shells = 68) 
  (h2 : final_shells = 89) : 
  final_shells - initial_shells = 21 := by
  sorry

end NUMINAMATH_CALUDE_lucy_shell_count_l1060_106007


namespace NUMINAMATH_CALUDE_pq_divides_3p_minus_1_q_minus_1_l1060_106097

theorem pq_divides_3p_minus_1_q_minus_1 (p q : ℕ+) :
  (p * q : ℕ) ∣ (3 * (p - 1) * (q - 1) : ℕ) ↔
  ((p = 6 ∧ q = 5) ∨ (p = 5 ∧ q = 6) ∨
   (p = 9 ∧ q = 4) ∨ (p = 4 ∧ q = 9) ∨
   (p = 3 ∧ q = 2) ∨ (p = 2 ∧ q = 3)) :=
by sorry

end NUMINAMATH_CALUDE_pq_divides_3p_minus_1_q_minus_1_l1060_106097


namespace NUMINAMATH_CALUDE_existence_of_point_S_l1060_106010

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a triangle in 3D space -/
structure Triangle where
  A : Point3D
  B : Point3D
  C : Point3D

/-- Checks if two triangles are congruent -/
def are_congruent (t1 t2 : Triangle) : Prop := sorry

/-- Checks if a triangle is parallel to a plane -/
def is_parallel_to_plane (t : Triangle) (p : Plane) : Prop := sorry

/-- Finds the intersection point of a line and a plane -/
def line_plane_intersection (p1 p2 : Point3D) (plane : Plane) : Point3D := sorry

/-- The main theorem -/
theorem existence_of_point_S (α : Plane) (ABC MNP : Triangle) 
  (h : ¬ is_parallel_to_plane ABC α) : 
  ∃ (S : Point3D), 
    let A' := line_plane_intersection S ABC.A α
    let B' := line_plane_intersection S ABC.B α
    let C' := line_plane_intersection S ABC.C α
    let A'B'C' : Triangle := ⟨A', B', C'⟩
    are_congruent A'B'C' MNP := by
  sorry

end NUMINAMATH_CALUDE_existence_of_point_S_l1060_106010


namespace NUMINAMATH_CALUDE_function_properties_l1060_106002

-- Define the function f(x)
def f (d : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + d

-- State the theorem
theorem function_properties :
  ∃ (d : ℝ), 
    (∀ x ∈ Set.Icc (-2 : ℝ) 2, f d x ≥ -4) ∧ 
    (∃ x ∈ Set.Icc (-2 : ℝ) 2, f d x = -4) →
    d = 1 ∧ 
    (∀ x ∈ Set.Icc (-2 : ℝ) 2, f d x ≤ 23) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 2, f d x = 23) :=
by
  sorry


end NUMINAMATH_CALUDE_function_properties_l1060_106002


namespace NUMINAMATH_CALUDE_problem_1_l1060_106064

theorem problem_1 (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x / y - y / x - (x^2 + y^2) / (x * y) = -2 * y / x :=
sorry

end NUMINAMATH_CALUDE_problem_1_l1060_106064


namespace NUMINAMATH_CALUDE_order_of_numbers_l1060_106035

theorem order_of_numbers : 7^(3/10) > 0.3^7 ∧ 0.3^7 > Real.log 0.3 := by
  sorry

end NUMINAMATH_CALUDE_order_of_numbers_l1060_106035


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1060_106027

theorem complex_fraction_simplification :
  let a := (5 + 4/45) - (4 + 1/6)
  let b := 5 + 8/15
  let c := (4 + 2/3) + 0.75
  let d := 3 + 9/13
  let e := 34 + 2/7
  let f := 0.3
  let g := 0.01
  let h := 70
  (a / b) / (c * d) * e + (f / g) / h + 2/7 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1060_106027


namespace NUMINAMATH_CALUDE_solution_range_l1060_106076

def A : Set ℝ := {x | (x + 1) / (x - 3) ≤ 0 ∧ x ≠ 3}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x - 3*a^2 ≤ 0}

theorem solution_range (a : ℝ) : 
  (∀ x, x ∈ B a → x ∈ A) ↔ -1/3 ≤ a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_solution_range_l1060_106076


namespace NUMINAMATH_CALUDE_eleven_to_fourth_l1060_106032

theorem eleven_to_fourth (n : ℕ) (h : n = 4) : 11^n = 14641 := by
  have h1 : 11 = 10 + 1 := by rfl
  sorry

end NUMINAMATH_CALUDE_eleven_to_fourth_l1060_106032


namespace NUMINAMATH_CALUDE_sin_squared_simplification_l1060_106088

theorem sin_squared_simplification (x y : ℝ) : 
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.sin y ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_simplification_l1060_106088


namespace NUMINAMATH_CALUDE_roses_cut_l1060_106050

theorem roses_cut (initial_roses initial_orchids final_roses final_orchids : ℕ) 
  (h1 : initial_roses = 13)
  (h2 : initial_orchids = 84)
  (h3 : final_roses = 14)
  (h4 : final_orchids = 91) :
  final_roses - initial_roses = 1 := by
  sorry

end NUMINAMATH_CALUDE_roses_cut_l1060_106050


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l1060_106067

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : 450 ∣ n^2) : 
  ∀ d : ℕ, d > 0 ∧ d ∣ n → d ≤ 30 :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l1060_106067


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l1060_106016

/-- The equation of an ellipse -/
def ellipse_equation (x y : ℝ) : Prop := x^2 / 2 + y^2 / 4 = 2

/-- The focal length of an ellipse -/
def focal_length : ℝ := 4

/-- Theorem: The focal length of the ellipse defined by x^2/2 + y^2/4 = 2 is equal to 4 -/
theorem ellipse_focal_length :
  ∀ x y : ℝ, ellipse_equation x y → focal_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l1060_106016


namespace NUMINAMATH_CALUDE_kennel_dogs_l1060_106034

/-- Given a kennel with cats and dogs, prove the number of dogs. -/
theorem kennel_dogs (cats dogs : ℕ) : 
  (cats : ℚ) / dogs = 2 / 3 →  -- ratio of cats to dogs is 2:3
  cats = dogs - 6 →            -- 6 fewer cats than dogs
  dogs = 18 := by              -- prove that there are 18 dogs
sorry

end NUMINAMATH_CALUDE_kennel_dogs_l1060_106034


namespace NUMINAMATH_CALUDE_smallest_factorial_not_divisible_by_62_l1060_106098

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_factorial_not_divisible_by_62 :
  (∀ n : ℕ, n < 31 → is_factor 62 (factorial n)) ∧
  ¬ is_factor 62 (factorial 31) ∧
  (∀ k : ℕ, k < 62 → (∃ m : ℕ, is_factor k (factorial m)) ∨ is_prime k) ∧
  ¬ is_prime 62 := by
  sorry

end NUMINAMATH_CALUDE_smallest_factorial_not_divisible_by_62_l1060_106098


namespace NUMINAMATH_CALUDE_first_day_visitors_l1060_106090

/-- Given the initial stock and restock amount, calculate the number of people who showed up on the first day -/
theorem first_day_visitors (initial_stock : ℕ) (first_restock : ℕ) (cans_per_person : ℕ) : 
  initial_stock = 2000 →
  first_restock = 1500 →
  cans_per_person = 1 →
  (initial_stock - first_restock) / cans_per_person = 500 := by
  sorry

#check first_day_visitors

end NUMINAMATH_CALUDE_first_day_visitors_l1060_106090


namespace NUMINAMATH_CALUDE_prime_power_divisibility_l1060_106083

theorem prime_power_divisibility (p n : ℕ) (hp : Prime p) (h : p ∣ n^2020) : p^2020 ∣ n^2020 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_divisibility_l1060_106083


namespace NUMINAMATH_CALUDE_coffee_percentage_contribution_l1060_106059

def pancake_price : ℚ := 4
def bacon_price : ℚ := 2
def egg_price : ℚ := 3/2
def coffee_price : ℚ := 1

def pancake_sold : ℕ := 60
def bacon_sold : ℕ := 90
def egg_sold : ℕ := 75
def coffee_sold : ℕ := 50

def total_sales : ℚ := 
  pancake_price * pancake_sold + 
  bacon_price * bacon_sold + 
  egg_price * egg_sold + 
  coffee_price * coffee_sold

def coffee_contribution : ℚ := coffee_price * coffee_sold / total_sales

theorem coffee_percentage_contribution : 
  coffee_contribution * 100 = 858/100 := by sorry

end NUMINAMATH_CALUDE_coffee_percentage_contribution_l1060_106059


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l1060_106028

/-- Given a journey with the following properties:
  * Total distance is 112 km
  * Total time is 5 hours
  * The first half is traveled at 21 km/hr
  Prove that the speed for the second half is 24 km/hr -/
theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ)
  (h1 : total_distance = 112)
  (h2 : total_time = 5)
  (h3 : first_half_speed = 21)
  : (2 * total_distance) / (2 * total_time - total_distance / first_half_speed) = 24 :=
by sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l1060_106028


namespace NUMINAMATH_CALUDE_at_least_two_solved_five_l1060_106068

/-- Represents a participant in the math competition -/
structure Participant where
  solved : Finset (Fin 6)

/-- Represents the math competition -/
structure MathCompetition where
  participants : Finset Participant
  num_problems : Nat
  num_problems_eq : num_problems = 6
  any_two_solved : ∀ i j : Fin 6, i ≠ j →
    (participants.filter (λ p => i ∈ p.solved ∧ j ∈ p.solved)).card >
    (2 / 5 : ℚ) * participants.card
  no_all_solved : ∀ p : Participant, p ∈ participants → p.solved.card < 6

theorem at_least_two_solved_five (mc : MathCompetition) :
  (mc.participants.filter (λ p => p.solved.card = 5)).card ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_solved_five_l1060_106068


namespace NUMINAMATH_CALUDE_circle_angle_distance_sum_l1060_106063

-- Define the circle and points
def Circle : Type := ℝ × ℝ → Prop
def Point : Type := ℝ × ℝ

-- Define the angle
def Angle : Type := Point → Point → Point → Prop

-- Define the distance function
def distance (p q : Point) : ℝ := sorry

-- Define the line segment
def LineSegment (p q : Point) : Point → Prop := sorry

-- State the theorem
theorem circle_angle_distance_sum
  (circle : Circle)
  (angle : Angle)
  (A B C : Point)
  (h1 : circle A ∧ circle B ∧ circle C)
  (h2 : angle A B C)
  (h3 : ∀ p, LineSegment A B p → distance C p = 8)
  (h4 : ∃ (d1 d2 : ℝ), d1 = d2 + 30 ∧
        (∀ p, angle A B p → (distance C p = d1 ∨ distance C p = d2))) :
  ∃ (d1 d2 : ℝ), d1 + d2 = 34 ∧
    (∀ p, angle A B p → (distance C p = d1 ∨ distance C p = d2)) :=
sorry

end NUMINAMATH_CALUDE_circle_angle_distance_sum_l1060_106063


namespace NUMINAMATH_CALUDE_smallest_c_for_g_range_five_l1060_106055

/-- The function g(x) defined in the problem -/
def g (c : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 4 * x + c

/-- Theorem stating that 7 is the smallest value of c such that 5 is in the range of g(x) -/
theorem smallest_c_for_g_range_five :
  ∀ c : ℝ, (∃ x : ℝ, g c x = 5) ↔ c ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_for_g_range_five_l1060_106055


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l1060_106048

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 2^x) / x^2

theorem derivative_f_at_one :
  deriv f 1 = 2 * Real.log 2 - 3 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l1060_106048


namespace NUMINAMATH_CALUDE_high_school_population_l1060_106020

theorem high_school_population (total_sample : ℕ) (first_grade_sample : ℕ) (second_grade_sample : ℕ) (third_grade_population : ℕ) : 
  total_sample = 36 → 
  first_grade_sample = 15 → 
  second_grade_sample = 12 → 
  third_grade_population = 900 → 
  (total_sample : ℚ) / (first_grade_sample + second_grade_sample + (total_sample - first_grade_sample - second_grade_sample)) = 
  (total_sample - first_grade_sample - second_grade_sample : ℚ) / third_grade_population → 
  (total_sample : ℕ) * (third_grade_population / (total_sample - first_grade_sample - second_grade_sample)) = 3600 :=
by sorry

end NUMINAMATH_CALUDE_high_school_population_l1060_106020


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1060_106012

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1060_106012


namespace NUMINAMATH_CALUDE_sally_balloons_l1060_106023

/-- The number of blue balloons each person has -/
structure Balloons where
  joan : ℕ
  sally : ℕ
  jessica : ℕ

/-- The total number of blue balloons -/
def total_balloons (b : Balloons) : ℕ := b.joan + b.sally + b.jessica

/-- Theorem stating Sally's number of balloons -/
theorem sally_balloons (b : Balloons) 
  (h1 : b.joan = 9)
  (h2 : b.jessica = 2)
  (h3 : total_balloons b = 16) :
  b.sally = 5 := by
  sorry

end NUMINAMATH_CALUDE_sally_balloons_l1060_106023


namespace NUMINAMATH_CALUDE_exactly_two_true_l1060_106042

-- Define the propositions
def proposition1 : Prop :=
  (∀ x, x^2 - 3*x + 2 = 0 → x = 2 ∨ x = 1) →
  (∀ x, x^2 - 3*x + 2 ≠ 0 → x ≠ 2 ∨ x ≠ 1)

def proposition2 : Prop :=
  (∀ x > 1, x^2 - 1 > 0) →
  (∃ x > 1, x^2 - 1 ≤ 0)

def proposition3 (p q : Prop) : Prop :=
  (¬p ∧ ¬q → ¬(p ∨ q)) ∧ ¬(¬(p ∨ q) → ¬p ∧ ¬q)

-- Theorem stating that exactly two propositions are true
theorem exactly_two_true :
  (¬proposition1 ∧ proposition2 ∧ proposition3 True False) ∨
  (¬proposition1 ∧ proposition2 ∧ proposition3 False True) ∨
  (proposition2 ∧ proposition3 True False ∧ proposition3 False True) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_true_l1060_106042


namespace NUMINAMATH_CALUDE_manufacturing_cost_calculation_l1060_106031

/-- The manufacturing cost of a shoe -/
def manufacturing_cost : ℝ := sorry

/-- The transportation cost for 100 shoes -/
def transportation_cost_100 : ℝ := 500

/-- The selling price of a shoe -/
def selling_price : ℝ := 222

/-- The gain percentage on the selling price -/
def gain_percentage : ℝ := 20

theorem manufacturing_cost_calculation : 
  manufacturing_cost = 180 := by
  sorry

end NUMINAMATH_CALUDE_manufacturing_cost_calculation_l1060_106031


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_nonnegative_reals_l1060_106018

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.sqrt x}
def N : Set ℝ := {y | ∃ x, y = x^2 - 2}

-- State the theorem
theorem M_intersect_N_equals_nonnegative_reals :
  M ∩ N = Set.Ici (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_nonnegative_reals_l1060_106018


namespace NUMINAMATH_CALUDE_sphere_box_height_l1060_106086

/-- A rectangular box with a large sphere and eight smaller spheres -/
structure SphereBox where
  h : ℝ
  large_sphere_radius : ℝ
  small_sphere_radius : ℝ
  box_width : ℝ
  box_length : ℝ
  num_small_spheres : ℕ

/-- The configuration of spheres in the box satisfies the given conditions -/
def valid_configuration (box : SphereBox) : Prop :=
  box.large_sphere_radius = 3 ∧
  box.small_sphere_radius = 1.5 ∧
  box.box_width = 6 ∧
  box.box_length = 6 ∧
  box.num_small_spheres = 8 ∧
  ∀ (small_sphere : Fin box.num_small_spheres),
    (∃ (side1 side2 side3 : ℝ), 
      side1 + side2 + side3 = box.box_width + box.box_length + box.h) ∧
    (box.large_sphere_radius + box.small_sphere_radius = 
      box.box_width / 2 - box.small_sphere_radius)

/-- The height of the box is 9 given the valid configuration -/
theorem sphere_box_height (box : SphereBox) :
  valid_configuration box → box.h = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_box_height_l1060_106086


namespace NUMINAMATH_CALUDE_right_triangle_ab_length_l1060_106040

/-- Given a right triangle ABC in the x-y plane where:
    - Angle B is 90 degrees
    - Length of AC is 25
    - Slope of line segment AC is 4/3
    Prove that the length of AB is 15 -/
theorem right_triangle_ab_length 
  (A B C : ℝ × ℝ) -- Points in the plane
  (right_angle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0) -- B is a right angle
  (ac_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 25) -- Length of AC is 25
  (ac_slope : (C.2 - A.2) / (C.1 - A.1) = 4/3) -- Slope of AC is 4/3
  : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ab_length_l1060_106040


namespace NUMINAMATH_CALUDE_problem_statement_l1060_106084

theorem problem_statement : 
  Real.sqrt 12 + |1 - Real.sqrt 3| + (π - 2023)^0 = 3 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1060_106084


namespace NUMINAMATH_CALUDE_negative_east_equals_positive_west_l1060_106074

-- Define the direction type
inductive Direction
| East
| West

-- Define a function to represent movement
def move (distance : Int) (direction : Direction) : Int :=
  match direction with
  | Direction.East => distance
  | Direction.West => -distance

-- Theorem statement
theorem negative_east_equals_positive_west :
  move (-8) Direction.East = move 8 Direction.West :=
by sorry

end NUMINAMATH_CALUDE_negative_east_equals_positive_west_l1060_106074


namespace NUMINAMATH_CALUDE_S_formula_l1060_106005

def N (n : ℕ+) : ℕ+ :=
  sorry

def S (n : ℕ) : ℕ :=
  sorry

theorem S_formula (n : ℕ) : S n = (4^n + 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_S_formula_l1060_106005


namespace NUMINAMATH_CALUDE_molar_mass_calculation_l1060_106072

/-- Given a chemical compound where 3 moles weigh 168 grams, prove that its molar mass is 56 grams per mole. -/
theorem molar_mass_calculation (mass : ℝ) (moles : ℝ) (h1 : mass = 168) (h2 : moles = 3) :
  mass / moles = 56 := by
  sorry

end NUMINAMATH_CALUDE_molar_mass_calculation_l1060_106072


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l1060_106014

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 25 cm and height 15 cm is 375 square centimeters -/
theorem parallelogram_area_example : parallelogram_area 25 15 = 375 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l1060_106014


namespace NUMINAMATH_CALUDE_bowl_weight_after_refill_l1060_106081

theorem bowl_weight_after_refill (empty_bowl_weight : ℕ) 
  (day1_food day2_food day3_food day4_food : ℕ) :
  let total_food := day1_food + day2_food + day3_food + day4_food
  empty_bowl_weight + total_food = 
    empty_bowl_weight + day1_food + day2_food + day3_food + day4_food :=
by sorry

end NUMINAMATH_CALUDE_bowl_weight_after_refill_l1060_106081


namespace NUMINAMATH_CALUDE_triangle_side_length_l1060_106045

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = 5 →
  c = 8 →
  B = Real.pi / 3 →  -- 60° in radians
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) →
  b = 7 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1060_106045


namespace NUMINAMATH_CALUDE_log_base_conversion_l1060_106017

theorem log_base_conversion (a : ℝ) (h : Real.log 16 / Real.log 14 = a) :
  Real.log 14 / Real.log 8 = 4 / (3 * a) := by
  sorry

end NUMINAMATH_CALUDE_log_base_conversion_l1060_106017


namespace NUMINAMATH_CALUDE_rectangle_diagonal_in_hexagon_l1060_106070

/-- A regular hexagon with side length 2 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- A rectangle inside the hexagon -/
structure Rectangle (h : RegularHexagon) :=
  (length : ℝ)
  (width : ℝ)
  (inside_hexagon : length + width ≤ h.side_length)

/-- Two congruent rectangles inside the hexagon -/
structure CongruentRectangles (h : RegularHexagon) :=
  (rect1 : Rectangle h)
  (rect2 : Rectangle h)
  (congruent : rect1.length = rect2.length ∧ rect1.width = rect2.width)

/-- The theorem to be proved -/
theorem rectangle_diagonal_in_hexagon 
  (h : RegularHexagon) 
  (r : CongruentRectangles h) : 
  Real.sqrt (r.rect1.length ^ 2 + r.rect1.width ^ 2) = 2 :=
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_in_hexagon_l1060_106070


namespace NUMINAMATH_CALUDE_max_surrounding_squares_l1060_106004

/-- Represents a square in 2D space -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- Predicate to check if two squares are non-overlapping -/
def non_overlapping (s1 s2 : Square) : Prop :=
  sorry

/-- Function to count the number of non-overlapping squares around a central square -/
def count_surrounding_squares (central : Square) (surrounding : List Square) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of non-overlapping squares 
    that can be placed around a given square is 8 -/
theorem max_surrounding_squares (central : Square) (surrounding : List Square) :
  count_surrounding_squares central surrounding ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_surrounding_squares_l1060_106004


namespace NUMINAMATH_CALUDE_base6_addition_subtraction_l1060_106065

/-- Converts a base 6 number to its decimal representation -/
def base6ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to its base 6 representation -/
def decimalToBase6 (n : ℕ) : ℕ := sorry

theorem base6_addition_subtraction :
  decimalToBase6 ((base6ToDecimal 35 + base6ToDecimal 14) - base6ToDecimal 20) = 33 := by sorry

end NUMINAMATH_CALUDE_base6_addition_subtraction_l1060_106065


namespace NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l1060_106085

theorem shaded_fraction_of_rectangle (length width : ℕ) (shaded_area : ℚ) :
  length = 15 →
  width = 20 →
  shaded_area = (1 / 2 : ℚ) * (1 / 4 : ℚ) * (length * width : ℚ) →
  shaded_area / (length * width : ℚ) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l1060_106085


namespace NUMINAMATH_CALUDE_semicircle_radius_l1060_106041

/-- Given a semi-circle with perimeter 180 cm, its radius is 180 / (π + 2) cm. -/
theorem semicircle_radius (P : ℝ) (h : P = 180) :
  P = π * r + 2 * r → r = 180 / (π + 2) :=
by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l1060_106041


namespace NUMINAMATH_CALUDE_smallest_positive_integer_1729m_78945n_l1060_106069

theorem smallest_positive_integer_1729m_78945n :
  ∃ (m n : ℤ), 1729 * m + 78945 * n = (1 : ℤ) ∧
  ∀ (k : ℤ), k > 0 → (∃ (x y : ℤ), 1729 * x + 78945 * y = k) → k ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_1729m_78945n_l1060_106069


namespace NUMINAMATH_CALUDE_inequality_proof_l1060_106058

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a * b + b * c + c * a = 1) :
  3 * Real.rpow (1 / (a * b * c) + 6 * (a + b + c)) (1/3) ≤ Real.rpow 3 (1/3) / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1060_106058


namespace NUMINAMATH_CALUDE_max_sum_reciprocal_ninth_l1060_106047

theorem max_sum_reciprocal_ninth (a b : ℕ+) (h : (a : ℚ)⁻¹ + (b : ℚ)⁻¹ = (9 : ℚ)⁻¹) :
  (a : ℕ) + b ≤ 100 ∧ ∃ (a' b' : ℕ+), (a' : ℚ)⁻¹ + (b' : ℚ)⁻¹ = (9 : ℚ)⁻¹ ∧ (a' : ℕ) + b' = 100 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_reciprocal_ninth_l1060_106047


namespace NUMINAMATH_CALUDE_wrong_observation_value_l1060_106019

theorem wrong_observation_value 
  (n : ℕ) 
  (initial_mean correct_value new_mean : ℝ) 
  (h1 : n = 50)
  (h2 : initial_mean = 32)
  (h3 : correct_value = 48)
  (h4 : new_mean = 32.5) :
  ∃ wrong_value : ℝ,
    (n : ℝ) * new_mean = (n : ℝ) * initial_mean - wrong_value + correct_value ∧
    wrong_value = 23 := by
sorry

end NUMINAMATH_CALUDE_wrong_observation_value_l1060_106019


namespace NUMINAMATH_CALUDE_base6_243_equals_base10_99_l1060_106037

/-- Converts a base 6 number represented as a list of digits to base 10 -/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

theorem base6_243_equals_base10_99 :
  base6ToBase10 [3, 4, 2] = 99 := by
  sorry

end NUMINAMATH_CALUDE_base6_243_equals_base10_99_l1060_106037


namespace NUMINAMATH_CALUDE_square_root_of_64_l1060_106052

theorem square_root_of_64 : ∃ x : ℝ, x^2 = 64 ↔ x = 8 ∨ x = -8 := by sorry

end NUMINAMATH_CALUDE_square_root_of_64_l1060_106052


namespace NUMINAMATH_CALUDE_range_of_x_range_of_a_l1060_106078

-- Define the conditions
def p (x : ℝ) := x^2 - x - 2 ≤ 0
def q (x : ℝ) := (x - 3) / x < 0
def r (x a : ℝ) := (x - (a + 1)) * (x + (2 * a - 1)) ≤ 0

-- Question 1
theorem range_of_x (x : ℝ) (h1 : p x) (h2 : q x) : x ∈ Set.Ioc 0 2 := by sorry

-- Question 2
theorem range_of_a (a : ℝ) 
  (h1 : ∀ x, p x → r x a) 
  (h2 : ∃ x, r x a ∧ ¬p x) 
  (h3 : a > 0) : 
  a > 1 := by sorry

end NUMINAMATH_CALUDE_range_of_x_range_of_a_l1060_106078


namespace NUMINAMATH_CALUDE_kindergarten_sample_size_l1060_106013

/-- Represents a kindergarten with students and a height measurement sample -/
structure Kindergarten where
  total_students : ℕ
  sample_size : ℕ

/-- Defines the sample size of a kindergarten height measurement -/
def sample_size (k : Kindergarten) : ℕ := k.sample_size

/-- Theorem: The sample size of the kindergarten height measurement is 31 -/
theorem kindergarten_sample_size :
  ∀ (k : Kindergarten),
  k.total_students = 310 →
  k.sample_size = 31 →
  sample_size k = 31 := by
  sorry

end NUMINAMATH_CALUDE_kindergarten_sample_size_l1060_106013


namespace NUMINAMATH_CALUDE_pizza_slices_eaten_l1060_106025

theorem pizza_slices_eaten 
  (small_pizza_slices : ℕ) 
  (large_pizza_slices : ℕ) 
  (slices_left_per_person : ℕ) 
  (num_people : ℕ) : 
  small_pizza_slices + large_pizza_slices - (slices_left_per_person * num_people) = 
  (small_pizza_slices + large_pizza_slices) - (slices_left_per_person * num_people) :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_eaten_l1060_106025


namespace NUMINAMATH_CALUDE_double_reflection_of_F_l1060_106009

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem double_reflection_of_F (F : ℝ × ℝ) (h : F = (-2, 1)) :
  (reflect_over_x_axis ∘ reflect_over_y_axis) F = (2, -1) := by
  sorry

end NUMINAMATH_CALUDE_double_reflection_of_F_l1060_106009


namespace NUMINAMATH_CALUDE_earth_circumference_scientific_notation_l1060_106033

theorem earth_circumference_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ |a| ∧ |a| < 10 ∧
    n = 6 ∧
    4010000 = a * (10 : ℝ) ^ n :=
by sorry

end NUMINAMATH_CALUDE_earth_circumference_scientific_notation_l1060_106033


namespace NUMINAMATH_CALUDE_double_price_increase_rate_l1060_106029

/-- The rate of price increase that, when applied twice, doubles the original price -/
theorem double_price_increase_rate : 
  ∃ x : ℝ, (1 + x) * (1 + x) = 2 ∧ x > 0 :=
by sorry

end NUMINAMATH_CALUDE_double_price_increase_rate_l1060_106029


namespace NUMINAMATH_CALUDE_marks_speeding_ticket_cost_l1060_106080

/-- Calculates the total amount owed for a speeding ticket -/
def speeding_ticket_cost (base_fine speed_limit actual_speed additional_penalty_per_mph : ℕ)
  (school_zone : Bool) (court_costs lawyer_fee_per_hour lawyer_hours : ℕ) : ℕ :=
  let speed_difference := actual_speed - speed_limit
  let additional_penalty := speed_difference * additional_penalty_per_mph
  let total_fine := base_fine + additional_penalty
  let doubled_fine := if school_zone then 2 * total_fine else total_fine
  let fine_with_court_costs := doubled_fine + court_costs
  let lawyer_fees := lawyer_fee_per_hour * lawyer_hours
  fine_with_court_costs + lawyer_fees

/-- Theorem: Mark's speeding ticket cost is $820 -/
theorem marks_speeding_ticket_cost :
  speeding_ticket_cost 50 30 75 2 true 300 80 3 = 820 := by
  sorry

end NUMINAMATH_CALUDE_marks_speeding_ticket_cost_l1060_106080


namespace NUMINAMATH_CALUDE_rakesh_salary_l1060_106099

/-- Rakesh's salary calculation -/
theorem rakesh_salary (salary : ℝ) : 
  (salary * (1 - 0.15) * (1 - 0.30) = 2380) → salary = 4000 := by
  sorry

end NUMINAMATH_CALUDE_rakesh_salary_l1060_106099


namespace NUMINAMATH_CALUDE_integer_list_mean_mode_relation_l1060_106054

theorem integer_list_mean_mode_relation : 
  ∀ x : ℕ, 
  x ≤ 100 → 
  x > 0 →
  let list := [20, x, x, x, x]
  let mean := (20 + 4 * x) / 5
  let mode := x
  mean = 2 * mode → 
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_integer_list_mean_mode_relation_l1060_106054


namespace NUMINAMATH_CALUDE_sin_minus_cos_value_l1060_106043

theorem sin_minus_cos_value (α : Real) 
  (h : ∃ (r : Real), r * (Real.cos (α - π/4)) = -1 ∧ r * (Real.sin (α - π/4)) = Real.sqrt 2) : 
  Real.sin α - Real.cos α = 2 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_sin_minus_cos_value_l1060_106043


namespace NUMINAMATH_CALUDE_total_desks_is_1776_total_desks_within_capacity_l1060_106089

/-- Represents the total number of classrooms in the school. -/
def total_classrooms : ℕ := 50

/-- Represents the number of desks in classrooms of type 1. -/
def desks_type1 : ℕ := 45

/-- Represents the number of desks in classrooms of type 2. -/
def desks_type2 : ℕ := 38

/-- Represents the number of desks in classrooms of type 3. -/
def desks_type3 : ℕ := 32

/-- Represents the number of desks in classrooms of type 4. -/
def desks_type4 : ℕ := 25

/-- Represents the fraction of classrooms of type 1. -/
def fraction_type1 : ℚ := 3 / 10

/-- Represents the fraction of classrooms of type 2. -/
def fraction_type2 : ℚ := 1 / 4

/-- Represents the fraction of classrooms of type 3. -/
def fraction_type3 : ℚ := 1 / 5

/-- Represents the maximum student capacity allowed by regulations. -/
def max_capacity : ℕ := 1800

/-- Theorem stating that the total number of desks in the school is 1776. -/
theorem total_desks_is_1776 : 
  (↑total_classrooms * fraction_type1).floor * desks_type1 +
  (↑total_classrooms * fraction_type2).floor * desks_type2 +
  (↑total_classrooms * fraction_type3).floor * desks_type3 +
  (total_classrooms - 
    (↑total_classrooms * fraction_type1).floor - 
    (↑total_classrooms * fraction_type2).floor - 
    (↑total_classrooms * fraction_type3).floor) * desks_type4 = 1776 :=
by sorry

/-- Theorem stating that the total number of desks does not exceed the maximum capacity. -/
theorem total_desks_within_capacity : 
  (↑total_classrooms * fraction_type1).floor * desks_type1 +
  (↑total_classrooms * fraction_type2).floor * desks_type2 +
  (↑total_classrooms * fraction_type3).floor * desks_type3 +
  (total_classrooms - 
    (↑total_classrooms * fraction_type1).floor - 
    (↑total_classrooms * fraction_type2).floor - 
    (↑total_classrooms * fraction_type3).floor) * desks_type4 ≤ max_capacity :=
by sorry

end NUMINAMATH_CALUDE_total_desks_is_1776_total_desks_within_capacity_l1060_106089


namespace NUMINAMATH_CALUDE_min_perimeter_noncongruent_isosceles_triangles_l1060_106000

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  side : ℕ  -- Length of equal sides
  base : ℕ  -- Length of the base
  is_isosceles : side > base

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.side + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * Real.sqrt ((t.side : ℝ)^2 - ((t.base : ℝ) / 2)^2) / 2

/-- Theorem: Minimum perimeter of two noncongruent integer-sided isosceles triangles -/
theorem min_perimeter_noncongruent_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    9 * t2.base = 8 * t1.base ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      9 * s2.base = 8 * s1.base →
      perimeter t1 ≤ perimeter s1 ∧
    perimeter t1 = 842 :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_noncongruent_isosceles_triangles_l1060_106000


namespace NUMINAMATH_CALUDE_manufacturing_quality_probability_l1060_106030

theorem manufacturing_quality_probability 
  (defect_rate1 : ℝ) 
  (defect_rate2 : ℝ) 
  (h1 : defect_rate1 = 0.03) 
  (h2 : defect_rate2 = 0.05) 
  (independent : True) -- Representing the independence of processes
  : (1 - defect_rate1) * (1 - defect_rate2) = 0.9215 := by
  sorry

end NUMINAMATH_CALUDE_manufacturing_quality_probability_l1060_106030


namespace NUMINAMATH_CALUDE_dagger_example_l1060_106075

def dagger (m n p q : ℚ) : ℚ := (m + n) * (p + q) * (q / n)

theorem dagger_example : dagger (5/9) (7/4) = 616/9 := by
  sorry

end NUMINAMATH_CALUDE_dagger_example_l1060_106075


namespace NUMINAMATH_CALUDE_solve_for_a_l1060_106060

theorem solve_for_a : ∃ a : ℝ, (2 * (3 - 1) - a = 0) ∧ a = 4 := by sorry

end NUMINAMATH_CALUDE_solve_for_a_l1060_106060


namespace NUMINAMATH_CALUDE_valid_numbers_l1060_106046

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  let tens := n / 10
  let ones := n % 10
  2 * (tens + ones) = tens * ones

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {63, 44, 36} := by sorry

end NUMINAMATH_CALUDE_valid_numbers_l1060_106046


namespace NUMINAMATH_CALUDE_larger_number_problem_l1060_106095

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 52) (h2 : x = 3 * y) (h3 : x > 0) (h4 : y > 0) : x = 39 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1060_106095


namespace NUMINAMATH_CALUDE_outfits_count_l1060_106022

/-- The number of different outfits that can be made with a given number of shirts, ties, and shoes. -/
def num_outfits (shirts : ℕ) (ties : ℕ) (shoes : ℕ) : ℕ := shirts * ties * shoes

/-- Theorem: Given 8 shirts, 7 ties, and 4 pairs of shoes, the total number of different possible outfits is 224. -/
theorem outfits_count : num_outfits 8 7 4 = 224 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l1060_106022


namespace NUMINAMATH_CALUDE_f_min_at_3_l1060_106039

/-- The function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- The theorem states that f(x) attains its minimum value when x = 3 -/
theorem f_min_at_3 : ∀ x : ℝ, f 3 ≤ f x := by sorry

end NUMINAMATH_CALUDE_f_min_at_3_l1060_106039


namespace NUMINAMATH_CALUDE_intersection_A_B_l1060_106093

-- Define set A
def A : Set ℝ := {x : ℝ | |x| < 2}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 5*x - 6 < 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1060_106093


namespace NUMINAMATH_CALUDE_tank_capacity_l1060_106036

theorem tank_capacity : 
  ∀ (T : ℝ), 
  (T > 0) →
  ((9/10 : ℝ) * T - (3/4 : ℝ) * T = 5) →
  T = 100/3 := by
sorry

end NUMINAMATH_CALUDE_tank_capacity_l1060_106036


namespace NUMINAMATH_CALUDE_domain_of_composed_function_l1060_106079

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-2) 2

-- State the theorem
theorem domain_of_composed_function :
  (∀ x ∈ domain_f, f x ≠ 0) →
  {x : ℝ | f (2*x + 1) ≠ 0} = Set.Icc (-3/2) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_domain_of_composed_function_l1060_106079


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l1060_106061

theorem factor_difference_of_squares (y : ℝ) : 25 - 16 * y^2 = (5 - 4*y) * (5 + 4*y) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l1060_106061


namespace NUMINAMATH_CALUDE_f_derivative_l1060_106051

noncomputable def f (x : ℝ) : ℝ := 2 + x * Real.cos x

theorem f_derivative : 
  deriv f = λ x => Real.cos x - x * Real.sin x :=
sorry

end NUMINAMATH_CALUDE_f_derivative_l1060_106051


namespace NUMINAMATH_CALUDE_relay_arrangement_count_l1060_106073

def relay_arrangements (n : ℕ) (k : ℕ) (a b : ℕ) : ℕ :=
  sorry

theorem relay_arrangement_count : relay_arrangements 6 4 1 4 = 252 := by
  sorry

end NUMINAMATH_CALUDE_relay_arrangement_count_l1060_106073


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1060_106049

theorem complex_magnitude_problem (w : ℂ) (h : w^2 = 45 - 21*I) : 
  Complex.abs w = (2466 : ℝ)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1060_106049


namespace NUMINAMATH_CALUDE_exterior_angle_of_regular_polygon_l1060_106008

theorem exterior_angle_of_regular_polygon (n : ℕ) (h : n > 2) :
  (180 * (n - 2) = 720) → (360 / n = 60) := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_of_regular_polygon_l1060_106008


namespace NUMINAMATH_CALUDE_train_speed_problem_l1060_106021

/-- Proves that the speed of the first train is 20 kmph given the problem conditions -/
theorem train_speed_problem (distance : ℝ) (speed_second : ℝ) (time_first : ℝ) (time_second : ℝ) 
  (h1 : distance = 200)
  (h2 : speed_second = 25)
  (h3 : time_first = 5)
  (h4 : time_second = 4) :
  ∃ (speed_first : ℝ), speed_first * time_first + speed_second * time_second = distance ∧ speed_first = 20 := by
  sorry

#check train_speed_problem

end NUMINAMATH_CALUDE_train_speed_problem_l1060_106021


namespace NUMINAMATH_CALUDE_all_figures_on_page_20_only_figures_in_figure5_on_page_20_l1060_106071

/-- Represents a geometric figure in the book --/
structure GeometricFigure where
  page : Nat

/-- Represents the collection of figures shown in Figure 5 --/
def Figure5 : Set GeometricFigure := sorry

/-- The property that distinguishes the figures in Figure 5 --/
def DistinguishingProperty (f : GeometricFigure) : Prop :=
  f.page = 20

/-- Theorem stating that all figures in Figure 5 have the distinguishing property --/
theorem all_figures_on_page_20 :
  ∀ f ∈ Figure5, DistinguishingProperty f :=
sorry

/-- Theorem stating that no other figures have this property --/
theorem only_figures_in_figure5_on_page_20 :
  ∀ f : GeometricFigure, DistinguishingProperty f → f ∈ Figure5 :=
sorry

end NUMINAMATH_CALUDE_all_figures_on_page_20_only_figures_in_figure5_on_page_20_l1060_106071


namespace NUMINAMATH_CALUDE_karen_tagalong_boxes_l1060_106024

/-- The number of Tagalong boxes Karen sold -/
def total_boxes (cases : ℕ) (boxes_per_case : ℕ) : ℕ :=
  cases * boxes_per_case

/-- Theorem stating that Karen sold 36 boxes of Tagalongs -/
theorem karen_tagalong_boxes : total_boxes 3 12 = 36 := by
  sorry

end NUMINAMATH_CALUDE_karen_tagalong_boxes_l1060_106024


namespace NUMINAMATH_CALUDE_congruence_solution_l1060_106077

theorem congruence_solution :
  ∃ n : ℕ, 0 ≤ n ∧ n < 53 ∧ (14 * n) % 53 = 9 % 53 ∧ n = 36 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l1060_106077


namespace NUMINAMATH_CALUDE_systematic_sample_first_product_l1060_106006

/-- Represents a systematic sample from a range of numbered products. -/
structure SystematicSample where
  total_products : ℕ
  sample_size : ℕ
  sample_interval : ℕ
  first_product : ℕ

/-- Creates a systematic sample given the total number of products and sample size. -/
def create_systematic_sample (total_products sample_size : ℕ) : SystematicSample :=
  { total_products := total_products,
    sample_size := sample_size,
    sample_interval := total_products / sample_size,
    first_product := 1 }

/-- Checks if a given product number is in the systematic sample. -/
def is_in_sample (s : SystematicSample) (product_number : ℕ) : Prop :=
  ∃ k, 0 ≤ k ∧ k < s.sample_size ∧ product_number = s.first_product + k * s.sample_interval

/-- Theorem: In a systematic sample of size 5 from 80 products, 
    if product 42 is in the sample, then the first product's number is 10. -/
theorem systematic_sample_first_product :
  let s := create_systematic_sample 80 5
  is_in_sample s 42 → s.first_product = 10 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_first_product_l1060_106006


namespace NUMINAMATH_CALUDE_min_value_a_min_value_a_achievable_l1060_106015

theorem min_value_a (a : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π / 2 → 4 + 2 * Real.sin θ * Real.cos θ - a * Real.sin θ - a * Real.cos θ ≤ 0) → 
  a ≥ 4 :=
by sorry

theorem min_value_a_achievable : 
  ∃ a : ℝ, a = 4 ∧ (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π / 2 → 4 + 2 * Real.sin θ * Real.cos θ - a * Real.sin θ - a * Real.cos θ ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_min_value_a_achievable_l1060_106015


namespace NUMINAMATH_CALUDE_circle_intersections_l1060_106011

/-- A circle C with equation x^2 + y^2 - 2x - 4y - 4 = 0 -/
def C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y - 4 = 0

/-- x₁ and x₂ are x-coordinates of intersection points with x-axis -/
def x_intersections (x₁ x₂ : ℝ) : Prop := C x₁ 0 ∧ C x₂ 0 ∧ x₁ ≠ x₂

/-- y₁ and y₂ are y-coordinates of intersection points with y-axis -/
def y_intersections (y₁ y₂ : ℝ) : Prop := C 0 y₁ ∧ C 0 y₂ ∧ y₁ ≠ y₂

theorem circle_intersections 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (hx : x_intersections x₁ x₂) 
  (hy : y_intersections y₁ y₂) : 
  abs (x₁ - x₂) = 2 * Real.sqrt 5 ∧ 
  y₁ + y₂ = 4 ∧ 
  x₁ * x₂ = y₁ * y₂ := by
  sorry

end NUMINAMATH_CALUDE_circle_intersections_l1060_106011
