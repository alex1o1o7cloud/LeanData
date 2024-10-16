import Mathlib

namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l248_24826

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 4)) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l248_24826


namespace NUMINAMATH_CALUDE_point_guard_footage_l248_24852

/-- Represents the number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- Represents the number of players on the basketball team -/
def num_players : ℕ := 5

/-- Represents the average number of minutes each player should get in the highlight film -/
def avg_minutes_per_player : ℕ := 2

/-- Represents the total seconds of footage for the shooting guard, small forward, power forward, and center -/
def other_players_footage : ℕ := 470

/-- Theorem stating that the point guard's footage is 130 seconds -/
theorem point_guard_footage : 
  (num_players * avg_minutes_per_player * seconds_per_minute) - other_players_footage = 130 := by
sorry

end NUMINAMATH_CALUDE_point_guard_footage_l248_24852


namespace NUMINAMATH_CALUDE_expression_value_l248_24891

theorem expression_value (a b : ℚ) (ha : a = -1) (hb : b = 1/4) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by sorry

end NUMINAMATH_CALUDE_expression_value_l248_24891


namespace NUMINAMATH_CALUDE_girls_boys_seating_arrangements_l248_24878

theorem girls_boys_seating_arrangements (n : ℕ) (h : n = 5) : 
  (n.factorial * n.factorial : ℕ) = 14400 := by
  sorry

end NUMINAMATH_CALUDE_girls_boys_seating_arrangements_l248_24878


namespace NUMINAMATH_CALUDE_tv_cost_l248_24807

def original_savings : ℚ := 600

def furniture_fraction : ℚ := 2/4

theorem tv_cost (savings : ℚ) (frac : ℚ) (h1 : savings = original_savings) (h2 : frac = furniture_fraction) : 
  savings * (1 - frac) = 300 := by
  sorry

end NUMINAMATH_CALUDE_tv_cost_l248_24807


namespace NUMINAMATH_CALUDE_women_who_bought_apples_l248_24860

/-- The number of women who bought apples -/
def num_women : ℕ := 3

/-- The number of men who bought apples -/
def num_men : ℕ := 2

/-- The number of apples each man bought -/
def apples_per_man : ℕ := 30

/-- The additional number of apples each woman bought compared to each man -/
def additional_apples_per_woman : ℕ := 20

/-- The total number of apples bought -/
def total_apples : ℕ := 210

theorem women_who_bought_apples :
  num_women * (apples_per_man + additional_apples_per_woman) +
  num_men * apples_per_man = total_apples :=
by sorry

end NUMINAMATH_CALUDE_women_who_bought_apples_l248_24860


namespace NUMINAMATH_CALUDE_ali_baba_max_coins_l248_24837

/-- Represents the state of the coin distribution game -/
structure GameState :=
  (piles : List Nat)
  (total_coins : Nat)

/-- Represents a move in the game -/
structure Move :=
  (chosen_piles : List Nat)
  (coins_removed : List Nat)

/-- Ali Baba's strategy -/
def aliBabaStrategy (state : GameState) : Move :=
  sorry

/-- Thief's strategy -/
def thiefStrategy (state : GameState) (move : Move) : List Nat :=
  sorry

/-- Simulate one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- Check if the game should end -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Calculate Ali Baba's final score -/
def calculateScore (state : GameState) : Nat :=
  sorry

/-- Main theorem: Ali Baba can secure at most 72 coins -/
theorem ali_baba_max_coins :
  ∀ (initial_state : GameState),
    initial_state.total_coins = 100 ∧ 
    initial_state.piles.length = 10 ∧ 
    (∀ pile ∈ initial_state.piles, pile = 10) →
    calculateScore (playRound initial_state) ≤ 72 :=
  sorry

end NUMINAMATH_CALUDE_ali_baba_max_coins_l248_24837


namespace NUMINAMATH_CALUDE_f_geq_g_l248_24802

/-- Given positive real numbers a, b, c, and a real number α, 
    we define functions f and g as follows:
    f(α) = abc(a^α + b^α + c^α)
    g(α) = a^(α+2)(b+c-a) + b^(α+2)(a-b+c) + c^(α+2)(a+b-c)
    This theorem states that f(α) ≥ g(α) for all real α. -/
theorem f_geq_g (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let f := fun (α : ℝ) ↦ a * b * c * (a^α + b^α + c^α)
  let g := fun (α : ℝ) ↦ a^(α+2)*(b+c-a) + b^(α+2)*(a-b+c) + c^(α+2)*(a+b-c)
  ∀ α, f α ≥ g α :=
by sorry

end NUMINAMATH_CALUDE_f_geq_g_l248_24802


namespace NUMINAMATH_CALUDE_greg_books_multiple_l248_24811

/-- The number of books Megan has read -/
def megan_books : ℕ := 32

/-- The number of books Kelcie has read -/
def kelcie_books : ℕ := megan_books / 4

/-- The total number of books read by all three people -/
def total_books : ℕ := 65

/-- The multiple of Kelcie's books that Greg has read -/
def greg_multiple : ℕ := 2

theorem greg_books_multiple : 
  megan_books + kelcie_books + (greg_multiple * kelcie_books + 9) = total_books :=
sorry

end NUMINAMATH_CALUDE_greg_books_multiple_l248_24811


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l248_24885

theorem coefficient_x_squared_in_expansion :
  (Finset.range 6).sum (fun k => (Nat.choose 5 k) * 2^k * (if k = 2 then 1 else 0)) = 40 :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l248_24885


namespace NUMINAMATH_CALUDE_smallest_multiple_of_5_and_711_l248_24867

theorem smallest_multiple_of_5_and_711 :
  ∀ n : ℕ, n > 0 ∧ 5 ∣ n ∧ 711 ∣ n → n ≥ 3555 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_5_and_711_l248_24867


namespace NUMINAMATH_CALUDE_quadratic_polynomial_proof_l248_24816

/-- A quadratic polynomial M in terms of x -/
def M (a : ℝ) (x : ℝ) : ℝ := (a + 4) * x^3 + 6 * x^2 - 2 * x + 5

/-- The coefficient of the quadratic term -/
def b : ℝ := 6

/-- Point A on the number line -/
def A : ℝ := -4

/-- Point B on the number line -/
def B : ℝ := 6

/-- Position of P after t seconds -/
def P (t : ℝ) : ℝ := A + 2 * t

/-- Position of Q after t seconds (starting 2 seconds after P) -/
def Q (t : ℝ) : ℝ := B - 2 * (t - 2)

/-- Distance between two points -/
def distance (x y : ℝ) : ℝ := |x - y|

theorem quadratic_polynomial_proof :
  (∀ x, M A x = 6 * x^2 - 2 * x + 5) ∧
  (∃ t, t > 0 ∧ (distance (P t) B = (1/2) * distance (P t) A)) ∧
  (∃ m, m > 2 ∧ distance (P m) (Q m) = 8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_proof_l248_24816


namespace NUMINAMATH_CALUDE_expand_and_simplify_l248_24865

theorem expand_and_simplify (x : ℝ) : (1 - x^2) * (1 + x^4 + x^6) = 1 - x^2 + x^4 - x^8 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l248_24865


namespace NUMINAMATH_CALUDE_subset_of_intersection_eq_union_l248_24832

theorem subset_of_intersection_eq_union {A B C : Set α} 
  (hA : A.Nonempty) (hB : B.Nonempty) (hC : C.Nonempty) 
  (h : A ∩ B = B ∪ C) : C ⊆ B := by
  sorry

end NUMINAMATH_CALUDE_subset_of_intersection_eq_union_l248_24832


namespace NUMINAMATH_CALUDE_cubic_equation_sum_of_cubes_l248_24818

theorem cubic_equation_sum_of_cubes :
  ∃ (u v w : ℝ),
    (u - Real.rpow 7 (1/3 : ℝ)) * (u - Real.rpow 29 (1/3 : ℝ)) * (u - Real.rpow 61 (1/3 : ℝ)) = 1/5 ∧
    (v - Real.rpow 7 (1/3 : ℝ)) * (v - Real.rpow 29 (1/3 : ℝ)) * (v - Real.rpow 61 (1/3 : ℝ)) = 1/5 ∧
    (w - Real.rpow 7 (1/3 : ℝ)) * (w - Real.rpow 29 (1/3 : ℝ)) * (w - Real.rpow 61 (1/3 : ℝ)) = 1/5 ∧
    u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
    u^3 + v^3 + w^3 = 97.6 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_sum_of_cubes_l248_24818


namespace NUMINAMATH_CALUDE_linear_function_property_l248_24805

-- Define a linear function g
def g (x : ℝ) : ℝ := sorry

-- State the theorem
theorem linear_function_property :
  (∀ x y a b : ℝ, g (a * x + b * y) = a * g x + b * g y) →  -- g is linear
  (∀ x : ℝ, g x = 3 * g⁻¹ x + 5) →  -- g(x) = 3g^(-1)(x) + 5
  g 0 = 3 →  -- g(0) = 3
  g (-1) = 3 - Real.sqrt 3 :=  -- g(-1) = 3 - √3
by sorry

end NUMINAMATH_CALUDE_linear_function_property_l248_24805


namespace NUMINAMATH_CALUDE_josephs_speed_josephs_speed_proof_l248_24895

/-- Joseph's driving problem -/
theorem josephs_speed : ℝ → Prop :=
  fun speed : ℝ =>
    let kyle_distance : ℝ := 62 * 2
    let joseph_distance : ℝ := kyle_distance + 1
    let joseph_time : ℝ := 2.5
    speed * joseph_time = joseph_distance → speed = 50

/-- Proof of Joseph's speed -/
theorem josephs_speed_proof : ∃ (speed : ℝ), josephs_speed speed := by
  sorry

end NUMINAMATH_CALUDE_josephs_speed_josephs_speed_proof_l248_24895


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l248_24822

/-- The trajectory of point M satisfying the given conditions is an ellipse -/
theorem trajectory_is_ellipse (x y : ℝ) : 
  let F : ℝ × ℝ := (0, 2)
  let line_y : ℝ := 8
  let distance_to_F := Real.sqrt ((x - F.1)^2 + (y - F.2)^2)
  let distance_to_line := |y - line_y|
  distance_to_F / distance_to_line = 1 / 2 → x^2 / 12 + y^2 / 16 = 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l248_24822


namespace NUMINAMATH_CALUDE_focaccia_price_is_four_l248_24846

/-- The price of a focaccia loaf given Sean's Sunday purchases -/
def focaccia_price : ℝ :=
  let almond_croissant : ℝ := 4.50
  let salami_cheese_croissant : ℝ := 4.50
  let plain_croissant : ℝ := 3.00
  let latte : ℝ := 2.50
  let total_spent : ℝ := 21.00
  total_spent - (almond_croissant + salami_cheese_croissant + plain_croissant + 2 * latte)

theorem focaccia_price_is_four : focaccia_price = 4.00 := by
  sorry

end NUMINAMATH_CALUDE_focaccia_price_is_four_l248_24846


namespace NUMINAMATH_CALUDE_geometric_sequence_tangent_l248_24803

open Real

theorem geometric_sequence_tangent (x : ℝ) : 
  (∃ (r : ℝ), (tan (π/12 - x) = tan (π/12) * r ∧ tan (π/12) = tan (π/12 + x) * r) ∨
               (tan (π/12 - x) = tan (π/12 + x) * r ∧ tan (π/12) = tan (π/12 - x) * r) ∨
               (tan (π/12) = tan (π/12 - x) * r ∧ tan (π/12 + x) = tan (π/12) * r)) ↔ 
  (∃ (ε : ℤ) (n : ℤ), ε ∈ ({-1, 0, 1} : Set ℤ) ∧ x = ε * (π/3) + n * π) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_tangent_l248_24803


namespace NUMINAMATH_CALUDE_product_of_squares_and_products_l248_24828

theorem product_of_squares_and_products (a b c : ℝ) 
  (sum_eq : a + b + c = 5)
  (sum_squares_eq : a^2 + b^2 + c^2 = 15)
  (sum_cubes_eq : a^3 + b^3 + c^3 = 47) :
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) = 625 := by
  sorry

end NUMINAMATH_CALUDE_product_of_squares_and_products_l248_24828


namespace NUMINAMATH_CALUDE_larger_number_l248_24873

theorem larger_number (a b : ℝ) (h1 : a - b = 6) (h2 : a + b = 40) : max a b = 23 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_l248_24873


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l248_24854

/-- An equilateral triangle ABC divided into 9 smaller equilateral triangles -/
structure TriangleABC where
  /-- The side length of the large equilateral triangle ABC -/
  side : ℝ
  /-- The side length of each smaller equilateral triangle -/
  small_side : ℝ
  /-- The number of smaller triangles that make up triangle ABC -/
  num_small_triangles : ℕ
  /-- The number of smaller triangles that are half shaded -/
  num_half_shaded : ℕ
  /-- Condition: The large triangle is divided into 9 smaller triangles -/
  h_num_small : num_small_triangles = 9
  /-- Condition: Two smaller triangles are half shaded -/
  h_num_half : num_half_shaded = 2
  /-- Condition: The side length of the large triangle is 3 times the small triangle -/
  h_side : side = 3 * small_side

/-- The shaded area is 2/9 of the total area of triangle ABC -/
theorem shaded_area_fraction (t : TriangleABC) : 
  (t.num_half_shaded : ℝ) / 2 / t.num_small_triangles = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l248_24854


namespace NUMINAMATH_CALUDE_cupcake_frosting_l248_24814

def cagney_rate : ℚ := 1 / 15
def lacey_rate : ℚ := 1 / 25
def lacey_delay : ℕ := 30
def total_time : ℕ := 600

def total_cupcakes : ℕ := 62

theorem cupcake_frosting :
  (cagney_rate * total_time).floor +
  (lacey_rate * (total_time - lacey_delay)).floor = total_cupcakes :=
sorry

end NUMINAMATH_CALUDE_cupcake_frosting_l248_24814


namespace NUMINAMATH_CALUDE_average_age_increase_l248_24833

/-- Proves that adding a 28-year-old student to a class of 9 students with an average age of 8 years increases the overall average age by 2 years -/
theorem average_age_increase (total_students : ℕ) (initial_students : ℕ) (initial_average : ℝ) (new_student_age : ℕ) :
  total_students = 10 →
  initial_students = 9 →
  initial_average = 8 →
  new_student_age = 28 →
  (initial_students * initial_average + new_student_age) / total_students - initial_average = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_age_increase_l248_24833


namespace NUMINAMATH_CALUDE_coordinates_of_point_B_l248_24863

/-- Given two points A and B in 2D space, this theorem proves that if the coordinates of A are (-2, -1) and the vector from A to B is (3, 4), then the coordinates of B are (1, 3). -/
theorem coordinates_of_point_B (A B : ℝ × ℝ) : 
  A = (-2, -1) → (B.1 - A.1, B.2 - A.2) = (3, 4) → B = (1, 3) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_of_point_B_l248_24863


namespace NUMINAMATH_CALUDE_k_value_proof_l248_24897

theorem k_value_proof (k : ℝ) (h1 : k ≠ 0) 
  (h2 : ∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 12)) : 
  k = 12 := by
  sorry

end NUMINAMATH_CALUDE_k_value_proof_l248_24897


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l248_24843

theorem absolute_value_inequality (x : ℝ) :
  |((x^2 - 5*x + 4) / 3)| < 1 ↔ (5 - Real.sqrt 21) / 2 < x ∧ x < (5 + Real.sqrt 21) / 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l248_24843


namespace NUMINAMATH_CALUDE_misha_earnings_l248_24853

theorem misha_earnings (current_amount target_amount : ℕ) 
  (h1 : current_amount = 34) 
  (h2 : target_amount = 47) : 
  target_amount - current_amount = 13 := by
  sorry

end NUMINAMATH_CALUDE_misha_earnings_l248_24853


namespace NUMINAMATH_CALUDE_complex_number_product_l248_24813

theorem complex_number_product (a b c d : ℂ) : 
  (a + b + c + d = 5) →
  ((5 - a)^4 + (5 - b)^4 + (5 - c)^4 + (5 - d)^4 = 125) →
  ((a + b)^4 + (b + c)^4 + (c + d)^4 + (d + a)^4 + (a + c)^4 + (b + d)^4 = 1205) →
  (a^4 + b^4 + c^4 + d^4 = 25) →
  a * b * c * d = 70 := by
sorry

end NUMINAMATH_CALUDE_complex_number_product_l248_24813


namespace NUMINAMATH_CALUDE_no_perfect_square_sum_of_prime_powers_l248_24876

theorem no_perfect_square_sum_of_prime_powers (p k m : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) :
  ¬∃ x : ℕ, p^k + p^m = x^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_sum_of_prime_powers_l248_24876


namespace NUMINAMATH_CALUDE_kat_boxing_hours_l248_24849

/-- Represents Kat's weekly training schedule -/
structure TrainingSchedule where
  strength_sessions : ℕ
  strength_hours_per_session : ℚ
  boxing_sessions : ℕ
  total_hours : ℚ

/-- Calculates the number of hours Kat trains at the boxing gym each time -/
def boxing_hours_per_session (schedule : TrainingSchedule) : ℚ :=
  (schedule.total_hours - schedule.strength_sessions * schedule.strength_hours_per_session) / schedule.boxing_sessions

/-- Theorem stating that Kat trains 1.5 hours at the boxing gym each time -/
theorem kat_boxing_hours (schedule : TrainingSchedule) 
  (h1 : schedule.strength_sessions = 3)
  (h2 : schedule.strength_hours_per_session = 1)
  (h3 : schedule.boxing_sessions = 4)
  (h4 : schedule.total_hours = 9) :
  boxing_hours_per_session schedule = 3/2 := by
  sorry

#eval boxing_hours_per_session { strength_sessions := 3, strength_hours_per_session := 1, boxing_sessions := 4, total_hours := 9 }

end NUMINAMATH_CALUDE_kat_boxing_hours_l248_24849


namespace NUMINAMATH_CALUDE_sequence_product_l248_24810

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_product (a b : ℕ → ℝ) :
  (∀ n, a n ≠ 0) →
  arithmetic_sequence a →
  geometric_sequence b →
  a 4 - 2 * (a 7)^2 + 3 * a 8 = 0 →
  b 7 = a 7 →
  b 2 * b 8 * b 11 = 8 := by
sorry

end NUMINAMATH_CALUDE_sequence_product_l248_24810


namespace NUMINAMATH_CALUDE_second_class_average_l248_24820

/-- Given two classes of students, this theorem proves that the average mark of the second class
    is 80, based on the given conditions. -/
theorem second_class_average (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) : 
  n₁ = 30 →
  n₂ = 50 →
  avg₁ = 40 →
  avg_total = 65 →
  let total_students : ℕ := n₁ + n₂
  let total_marks : ℚ := avg_total * total_students
  let first_class_marks : ℚ := avg₁ * n₁
  let second_class_marks : ℚ := total_marks - first_class_marks
  let avg₂ : ℚ := second_class_marks / n₂
  avg₂ = 80 := by sorry

end NUMINAMATH_CALUDE_second_class_average_l248_24820


namespace NUMINAMATH_CALUDE_lcm_9_12_15_l248_24806

theorem lcm_9_12_15 : Nat.lcm 9 (Nat.lcm 12 15) = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_9_12_15_l248_24806


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l248_24855

/-- The volume of a cube inscribed in a cylinder, which is inscribed in a larger cube --/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let cylinder_radius : ℝ := outer_cube_edge / 2
  let cylinder_diameter : ℝ := outer_cube_edge
  let inscribed_cube_face_diagonal : ℝ := cylinder_diameter
  let inscribed_cube_edge : ℝ := inscribed_cube_face_diagonal / Real.sqrt 2
  let inscribed_cube_volume : ℝ := inscribed_cube_edge ^ 3
  inscribed_cube_volume = 432 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l248_24855


namespace NUMINAMATH_CALUDE_binary_arithmetic_equality_l248_24804

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (λ (i, bit) acc => acc + if bit then 2^i else 0) 0

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec to_binary_aux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
    (to_binary_aux n).reverse

theorem binary_arithmetic_equality :
  let a := binary_to_decimal [true, false, true, true]  -- 1101₂
  let b := binary_to_decimal [false, true, true]        -- 110₂
  let c := binary_to_decimal [false, true, true, true]  -- 1110₂
  let d := binary_to_decimal [true, true, true, true]   -- 1111₂
  let result := decimal_to_binary (a + b - c + d)
  result = [false, true, false, false, false, true]     -- 100010₂
:= by sorry

end NUMINAMATH_CALUDE_binary_arithmetic_equality_l248_24804


namespace NUMINAMATH_CALUDE_show_episodes_count_l248_24889

/-- The number of episodes watched on Mondays each week -/
def monday_episodes : ℕ := 1

/-- The number of episodes watched on Wednesdays each week -/
def wednesday_episodes : ℕ := 2

/-- The number of weeks it takes to watch the whole series -/
def total_weeks : ℕ := 67

/-- The total number of episodes in the show -/
def total_episodes : ℕ := 201

theorem show_episodes_count : 
  monday_episodes + wednesday_episodes * total_weeks = total_episodes := by
  sorry

end NUMINAMATH_CALUDE_show_episodes_count_l248_24889


namespace NUMINAMATH_CALUDE_problem_statement_l248_24809

theorem problem_statement (x y : ℝ) 
  (h1 : (4 : ℝ) ^ x = 16 ^ (y + 2))
  (h2 : (27 : ℝ) ^ y = 9 ^ (x - 8)) :
  x + y = 40 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l248_24809


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l248_24845

def point : ℝ × ℝ := (-3, -2)

def in_third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

theorem point_in_third_quadrant : in_third_quadrant point := by
  sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l248_24845


namespace NUMINAMATH_CALUDE_range_of_expression_l248_24800

theorem range_of_expression (x y : ℝ) (h1 : x * y = 1) (h2 : 3 ≥ x) (h3 : x ≥ 4 * y) (h4 : y > 0) :
  ∃ (a b : ℝ), a = 4 ∧ b = 5 ∧
  (∀ z, (z = (x^2 + 4*y^2) / (x - 2*y)) → a ≤ z ∧ z ≤ b) ∧
  (∃ z1 z2, z1 = (x^2 + 4*y^2) / (x - 2*y) ∧ z2 = (x^2 + 4*y^2) / (x - 2*y) ∧ z1 = a ∧ z2 = b) :=
by sorry

end NUMINAMATH_CALUDE_range_of_expression_l248_24800


namespace NUMINAMATH_CALUDE_determinant_of_cubic_roots_l248_24859

/-- Given a, b, c are roots of x^3 - 2px + q = 0, 
    prove that the determinant of the matrix is 5 - 6p + q -/
theorem determinant_of_cubic_roots (p q a b c : ℝ) : 
  a^3 - 2*p*a + q = 0 → 
  b^3 - 2*p*b + q = 0 → 
  c^3 - 2*p*c + q = 0 → 
  Matrix.det !![2 + a, 1, 1; 1, 2 + b, 1; 1, 1, 2 + c] = 5 - 6*p + q := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_cubic_roots_l248_24859


namespace NUMINAMATH_CALUDE_calculate_expression_l248_24888

theorem calculate_expression : 15 * 30 + 45 * 15 = 1125 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l248_24888


namespace NUMINAMATH_CALUDE_journey_time_calculation_l248_24830

theorem journey_time_calculation (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) : 
  total_distance = 112 ∧ speed1 = 21 ∧ speed2 = 24 →
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_calculation_l248_24830


namespace NUMINAMATH_CALUDE_julia_played_with_33_kids_l248_24892

/-- The number of kids Julia played with on Monday and Tuesday combined -/
def total_kids_monday_tuesday (monday : ℕ) (tuesday : ℕ) : ℕ :=
  monday + tuesday

/-- Proof that Julia played with 33 kids on Monday and Tuesday combined -/
theorem julia_played_with_33_kids : 
  total_kids_monday_tuesday 15 18 = 33 := by
  sorry

end NUMINAMATH_CALUDE_julia_played_with_33_kids_l248_24892


namespace NUMINAMATH_CALUDE_problem_solution_l248_24801

theorem problem_solution : ∃ x : ℝ, 0.75 * x = x / 3 + 110 ∧ x = 264 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l248_24801


namespace NUMINAMATH_CALUDE_average_of_first_group_l248_24880

theorem average_of_first_group (total_average : ℝ) (second_group_average : ℝ) (third_group_average : ℝ)
  (h1 : total_average = 2.80)
  (h2 : second_group_average = 2.3)
  (h3 : third_group_average = 3.7) :
  let total_sum := 6 * total_average
  let second_group_sum := 2 * second_group_average
  let third_group_sum := 2 * third_group_average
  let first_group_sum := total_sum - second_group_sum - third_group_sum
  first_group_sum / 2 = 2.4 := by
sorry

end NUMINAMATH_CALUDE_average_of_first_group_l248_24880


namespace NUMINAMATH_CALUDE_fraction_subtraction_l248_24872

theorem fraction_subtraction : 
  (3 + 5 + 7) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 5 + 7) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l248_24872


namespace NUMINAMATH_CALUDE_sector_area_l248_24870

theorem sector_area (centralAngle : Real) (radius : Real) : 
  centralAngle = 72 → radius = 20 → 
  (centralAngle / 360) * Real.pi * radius^2 = 80 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l248_24870


namespace NUMINAMATH_CALUDE_no_prime_of_form_3811_l248_24899

def a (n : ℕ) : ℕ := 38 * 10^n + (10^n - 1)

theorem no_prime_of_form_3811 : ∀ n : ℕ, ¬ Nat.Prime (a n) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_of_form_3811_l248_24899


namespace NUMINAMATH_CALUDE_system_solution_l248_24864

theorem system_solution :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^3 + y₁^3) * (x₁^2 + y₁^2) = 64 ∧
    x₁ + y₁ = 2 ∧
    x₁ = 1 + Real.sqrt (5/3) ∧
    y₁ = 1 - Real.sqrt (5/3) ∧
    (x₂^3 + y₂^3) * (x₂^2 + y₂^2) = 64 ∧
    x₂ + y₂ = 2 ∧
    x₂ = 1 - Real.sqrt (5/3) ∧
    y₂ = 1 + Real.sqrt (5/3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l248_24864


namespace NUMINAMATH_CALUDE_quadratic_equation_with_given_root_properties_l248_24821

theorem quadratic_equation_with_given_root_properties :
  ∀ (a b c p q : ℝ),
    a ≠ 0 →
    (∀ x, a * x^2 + b * x + c = 0 ↔ x = p ∨ x = q) →
    p + q = 12 →
    |p - q| = 4 →
    a * x^2 + b * x + c = x^2 - 12 * x + 32 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_given_root_properties_l248_24821


namespace NUMINAMATH_CALUDE_range_of_a_l248_24825

-- Define the propositions P and Q
def P (a : ℝ) : Prop := ∀ x, x^2 + 2*a*x + 4 > 0

def Q (a : ℝ) : Prop := ∀ x y, x < y → (5-2*a)^x < (5-2*a)^y

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (P a ∨ Q a) ∧ ¬(P a ∧ Q a) → a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l248_24825


namespace NUMINAMATH_CALUDE_magician_tricks_conversion_l248_24819

/-- Converts a base-9 number represented as a list of digits to its base-10 equivalent -/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- The given number of tricks in base 9 -/
def tricksBase9 : List Nat := [2, 3, 4, 5]

theorem magician_tricks_conversion :
  base9ToBase10 tricksBase9 = 3998 := by
  sorry

end NUMINAMATH_CALUDE_magician_tricks_conversion_l248_24819


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l248_24893

-- Define the quadratic function f(x)
def f (x : ℝ) : ℝ := -x^2 + 2*x + 15

-- Define g(x) in terms of f(x) and a
def g (a x : ℝ) : ℝ := (2 - 2*a)*x - f x

-- Theorem statement
theorem quadratic_function_properties :
  -- f(x) has vertex (1, 16)
  (f 1 = 16 ∧ ∀ x, f x ≤ f 1) ∧
  -- The roots of f(x) are 8 units apart
  (∃ x₁ x₂, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₂ - x₁ = 8) →
  -- 1. f(x) = -x^2 + 2x + 15
  (∀ x, f x = -x^2 + 2*x + 15) ∧
  -- 2. g(x) is monotonically increasing on [0, 2] iff a ≤ 0
  (∀ a, (∀ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 → g a x₁ < g a x₂) ↔ a ≤ 0) ∧
  -- 3. Minimum value of g(x) on [0, 2]
  (∀ a, (∃ m, ∀ x, 0 ≤ x ∧ x ≤ 2 → m ≤ g a x ∧
    ((a > 2 → m = -4*a - 11) ∧
     (a < 0 → m = -15) ∧
     (0 ≤ a ∧ a ≤ 2 → m = -a^2 - 15)))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l248_24893


namespace NUMINAMATH_CALUDE_max_product_sum_300_l248_24894

theorem max_product_sum_300 : 
  (∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500) ∧ 
  (∀ (x y : ℤ), x + y = 300 → x * y ≤ 22500) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l248_24894


namespace NUMINAMATH_CALUDE_range_of_a_l248_24871

-- Define the sets A and B
def A : Set ℝ := {x | x < 3}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∪ B a = Set.univ → a < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l248_24871


namespace NUMINAMATH_CALUDE_basketball_games_played_l248_24856

theorem basketball_games_played (team_a_win_ratio : Rat) (team_b_win_ratio : Rat)
  (team_b_more_wins : ℕ) (team_b_more_losses : ℕ) :
  team_a_win_ratio = 3/4 →
  team_b_win_ratio = 2/3 →
  team_b_more_wins = 9 →
  team_b_more_losses = 9 →
  ∃ (team_a_games : ℕ),
    team_a_games = 36 ∧
    (team_a_games : Rat) * team_a_win_ratio + (team_a_games : Rat) * (1 - team_a_win_ratio) = team_a_games ∧
    ((team_a_games : Rat) + (team_b_more_wins + team_b_more_losses : Rat)) * team_b_win_ratio = 
      team_a_games * team_a_win_ratio + team_b_more_wins :=
by sorry

end NUMINAMATH_CALUDE_basketball_games_played_l248_24856


namespace NUMINAMATH_CALUDE_arithmetic_progression_squares_l248_24869

theorem arithmetic_progression_squares (a b c : ℝ) :
  (∃ d : ℝ, b = a + d ∧ c = a + 2*d) →
  ∃ q : ℝ, (a^2 + a*c + c^2) - (a^2 + a*b + b^2) = q ∧
           (b^2 + b*c + c^2) - (a^2 + a*c + c^2) = q :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_squares_l248_24869


namespace NUMINAMATH_CALUDE_nested_sqrt_bounds_l248_24866

theorem nested_sqrt_bounds : 
  ∃ x : ℝ, x = Real.sqrt (1 + x) ∧ 1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_bounds_l248_24866


namespace NUMINAMATH_CALUDE_expression_always_defined_l248_24862

theorem expression_always_defined (x : ℝ) (h : x > 12) : x^2 - 24*x + 144 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_always_defined_l248_24862


namespace NUMINAMATH_CALUDE_polygon_area_is_400_l248_24875

-- Define the polygon vertices
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (20, 0)
def C : ℝ × ℝ := (30, 10)
def D : ℝ × ℝ := (20, 20)
def E : ℝ × ℝ := (10, 10)
def F : ℝ × ℝ := (0, 20)

-- Define the polygon as a list of vertices
def polygon : List (ℝ × ℝ) := [A, B, C, D, E, F]

-- Function to calculate the area of a polygon given its vertices
def polygonArea (vertices : List (ℝ × ℝ)) : ℝ :=
  sorry -- Implementation not required for this task

-- Theorem statement
theorem polygon_area_is_400 : polygonArea polygon = 400 := by
  sorry -- Proof not required for this task

end NUMINAMATH_CALUDE_polygon_area_is_400_l248_24875


namespace NUMINAMATH_CALUDE_total_earnings_is_5800_l248_24858

/-- Represents the investment and return information for three investors -/
structure InvestmentInfo where
  investment_ratio : Fin 3 → ℕ
  return_ratio : Fin 3 → ℕ
  earnings_difference : ℕ

/-- Calculates the total earnings based on the given investment information -/
def calculate_total_earnings (info : InvestmentInfo) : ℕ :=
  sorry

/-- Theorem stating that the total earnings are 5800 given the specified conditions -/
theorem total_earnings_is_5800 (info : InvestmentInfo) 
  (h1 : info.investment_ratio = ![3, 4, 5])
  (h2 : info.return_ratio = ![6, 5, 4])
  (h3 : info.earnings_difference = 200) :
  calculate_total_earnings info = 5800 :=
sorry

end NUMINAMATH_CALUDE_total_earnings_is_5800_l248_24858


namespace NUMINAMATH_CALUDE_marble_problem_l248_24898

/-- The number of marbles Doug lost at the playground -/
def marbles_lost (ed_initial : ℕ) (doug_initial : ℕ) (ed_final : ℕ) (doug_final : ℕ) : ℕ :=
  doug_initial - doug_final

theorem marble_problem (ed_initial : ℕ) (doug_initial : ℕ) (ed_final : ℕ) (doug_final : ℕ) :
  ed_initial = doug_initial + 10 →
  ed_initial = 45 →
  ed_final = doug_final + 21 →
  ed_initial = ed_final →
  marbles_lost ed_initial doug_initial ed_final doug_final = 11 := by
  sorry

#check marble_problem

end NUMINAMATH_CALUDE_marble_problem_l248_24898


namespace NUMINAMATH_CALUDE_parabola_properties_l248_24881

-- Define the parabola
def parabola (x : ℝ) : ℝ := -2 * (x + 1)^2 - 3

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := -2 * (x + 3)^2 + 1

-- Theorem statement
theorem parabola_properties :
  (∀ x : ℝ, parabola x ≤ parabola (-1)) ∧
  (parabola (-1) = -3) ∧
  (∀ x : ℝ, shifted_parabola x = parabola (x + 2) + 4) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l248_24881


namespace NUMINAMATH_CALUDE_lattice_point_proximity_probability_l248_24848

theorem lattice_point_proximity_probability (d : ℝ) : 
  (d > 0) → 
  (π * d^2 = 1/3) → 
  (d = Real.sqrt (1 / (3 * π))) :=
by sorry

end NUMINAMATH_CALUDE_lattice_point_proximity_probability_l248_24848


namespace NUMINAMATH_CALUDE_travelers_checks_average_l248_24808

theorem travelers_checks_average (total_checks : ℕ) (total_worth : ℕ) 
  (spent_checks : ℕ) (h1 : total_checks = 30) (h2 : total_worth = 1800) 
  (h3 : spent_checks = 18) :
  let fifty_checks := (2 * total_worth - 100 * total_checks) / 50
  let hundred_checks := total_checks - fifty_checks
  let remaining_fifty := fifty_checks - spent_checks
  let remaining_total := remaining_fifty + hundred_checks
  let remaining_worth := 50 * remaining_fifty + 100 * hundred_checks
  remaining_worth / remaining_total = 75 := by
sorry

end NUMINAMATH_CALUDE_travelers_checks_average_l248_24808


namespace NUMINAMATH_CALUDE_mika_stickers_l248_24890

/-- The number of stickers Mika has left after a series of transactions -/
def stickers_left (initial : Float) (bought : Float) (birthday : Float) (from_friend : Float)
  (to_sister : Float) (used : Float) (sold : Float) : Float :=
  initial + bought + birthday + from_friend - to_sister - used - sold

/-- Theorem stating that Mika has 6 stickers left after the given transactions -/
theorem mika_stickers :
  stickers_left 20.5 26.25 19.75 7.5 6.3 58.5 3.2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_mika_stickers_l248_24890


namespace NUMINAMATH_CALUDE_batsman_average_l248_24836

theorem batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℕ) :
  total_innings = 25 →
  last_innings_score = 175 →
  average_increase = 6 →
  (∃ (previous_average : ℕ),
    (previous_average * (total_innings - 1) + last_innings_score) / total_innings =
    previous_average + average_increase) →
  (((total_innings - 1) * ((last_innings_score / average_increase) - total_innings) +
    last_innings_score) / total_innings) = 31 :=
by sorry

end NUMINAMATH_CALUDE_batsman_average_l248_24836


namespace NUMINAMATH_CALUDE_jackson_decorations_given_l248_24835

/-- The number of decorations given to the neighbor -/
def decorations_given_to_neighbor (num_boxes : ℕ) (decorations_per_box : ℕ) (decorations_used : ℕ) : ℕ :=
  num_boxes * decorations_per_box - decorations_used

/-- Theorem: Mrs. Jackson gave 92 decorations to her neighbor -/
theorem jackson_decorations_given :
  decorations_given_to_neighbor 6 25 58 = 92 := by
  sorry

end NUMINAMATH_CALUDE_jackson_decorations_given_l248_24835


namespace NUMINAMATH_CALUDE_symmetry_axis_of_sine_function_l248_24851

/-- Given that cos(2π/3 - φ) = cosφ, prove that x = 5π/6 is a symmetry axis of f(x) = sin(x - φ) -/
theorem symmetry_axis_of_sine_function (φ : ℝ) 
  (h : Real.cos (2 * Real.pi / 3 - φ) = Real.cos φ) :
  ∀ x : ℝ, Real.sin (x - φ) = Real.sin ((5 * Real.pi / 3 - x) - φ) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_axis_of_sine_function_l248_24851


namespace NUMINAMATH_CALUDE_perpendicular_lines_relationship_l248_24844

-- Define a type for lines in 3D space
def Line3D := ℝ × ℝ × ℝ → Prop

-- Define perpendicularity of lines
def perpendicular (l₁ l₂ : Line3D) : Prop := sorry

-- Define parallel lines
def parallel (l₁ l₂ : Line3D) : Prop := sorry

-- Define skew lines
def skew (l₁ l₂ : Line3D) : Prop := sorry

theorem perpendicular_lines_relationship (a b c : Line3D) 
  (h1 : perpendicular a b) (h2 : perpendicular b c) :
  ¬ (parallel a c ∨ perpendicular a c ∨ skew a c) → False := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_relationship_l248_24844


namespace NUMINAMATH_CALUDE_corrected_mean_l248_24823

theorem corrected_mean (n : ℕ) (incorrect_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) :
  n = 50 ∧ incorrect_mean = 36 ∧ incorrect_value = 23 ∧ correct_value = 45 →
  (n : ℚ) * incorrect_mean - incorrect_value + correct_value = 36.44 * n :=
by sorry

end NUMINAMATH_CALUDE_corrected_mean_l248_24823


namespace NUMINAMATH_CALUDE_point_A_final_position_l248_24884

-- Define the initial position of point A
def initial_position : Set ℤ := {-5, 5}

-- Define the movement function
def move (start : ℤ) (left : ℤ) (right : ℤ) : ℤ := start - left + right

-- Theorem statement
theorem point_A_final_position :
  ∀ start ∈ initial_position,
  move start 2 6 = -1 ∨ move start 2 6 = 9 := by
sorry

end NUMINAMATH_CALUDE_point_A_final_position_l248_24884


namespace NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l248_24842

theorem smaller_solution_of_quadratic (x : ℝ) : 
  x^2 + 7*x - 30 = 0 ∧ (∀ y : ℝ, y^2 + 7*y - 30 = 0 → y ≥ x) → x = -10 :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l248_24842


namespace NUMINAMATH_CALUDE_ellipse_foci_coordinates_l248_24850

/-- The coordinates of the foci of an ellipse given by the equation mx^2 + ny^2 + mn = 0,
    where m < n < 0 -/
theorem ellipse_foci_coordinates (m n : ℝ) (h1 : m < n) (h2 : n < 0) :
  let equation := fun (x y : ℝ) => m * x^2 + n * y^2 + m * n
  ∃ c : ℝ, c > 0 ∧ 
    (∀ x y : ℝ, equation x y = 0 → 
      ((x = 0 ∧ y = c) ∨ (x = 0 ∧ y = -c)) ↔ 
      (x, y) ∈ {p : ℝ × ℝ | p.1^2 / (-n) + p.2^2 / (-m) = 1 ∧ p.1^2 + p.2^2 > 1}) ∧
    c^2 = n - m :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_coordinates_l248_24850


namespace NUMINAMATH_CALUDE_continued_fraction_solution_l248_24883

theorem continued_fraction_solution :
  ∃ y : ℝ, y = 3 + 6 / (2 + 6 / y) ∧ y = 2 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_solution_l248_24883


namespace NUMINAMATH_CALUDE_min_value_of_expression_l248_24887

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 + 2*a*b - 3 = 0) :
  ∃ (k : ℝ), k = 2*a + b ∧ k ≥ 4 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x^2 + 2*x*y - 3 = 0 → 2*x + y ≥ k :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l248_24887


namespace NUMINAMATH_CALUDE_circle_radii_sum_l248_24834

theorem circle_radii_sum : 
  ∀ r : ℝ, 
  (r > 0) →
  ((r - 5)^2 + r^2 = (r + 2)^2) →
  (∃ r₁ r₂ : ℝ, (r = r₁ ∨ r = r₂) ∧ r₁ + r₂ = 14) :=
by sorry

end NUMINAMATH_CALUDE_circle_radii_sum_l248_24834


namespace NUMINAMATH_CALUDE_diagonals_bisect_if_equal_areas_l248_24829

/-- A quadrilateral in a 2D plane. -/
structure Quadrilateral (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D : V)

/-- The area of a triangle given its vertices. -/
noncomputable def triangleArea {V : Type*} [AddCommGroup V] [Module ℝ V] (A B C : V) : ℝ := sorry

/-- Statement that a line segment divides a quadrilateral into two equal areas. -/
def dividesEquallyBy {V : Type*} [AddCommGroup V] [Module ℝ V] (q : Quadrilateral V) (P Q : V) : Prop :=
  triangleArea q.A P Q + triangleArea Q P q.D = triangleArea q.B P Q + triangleArea Q P q.C

/-- The intersection point of the diagonals of a quadrilateral. -/
noncomputable def diagonalIntersection {V : Type*} [AddCommGroup V] [Module ℝ V] (q : Quadrilateral V) : V := sorry

/-- Statement that a point is the midpoint of a line segment. -/
def isMidpoint {V : Type*} [AddCommGroup V] [Module ℝ V] (M A B : V) : Prop :=
  2 • M = A + B

theorem diagonals_bisect_if_equal_areas {V : Type*} [AddCommGroup V] [Module ℝ V] (q : Quadrilateral V) :
  dividesEquallyBy q q.A q.C → dividesEquallyBy q q.B q.D →
  let E := diagonalIntersection q
  isMidpoint E q.A q.C ∧ isMidpoint E q.B q.D :=
sorry

end NUMINAMATH_CALUDE_diagonals_bisect_if_equal_areas_l248_24829


namespace NUMINAMATH_CALUDE_zoo_animals_l248_24824

theorem zoo_animals (X : ℕ) : 
  X - 6 + 1 + 3 + 8 + 16 = 90 → X = 68 := by
sorry

end NUMINAMATH_CALUDE_zoo_animals_l248_24824


namespace NUMINAMATH_CALUDE_exists_triangle_area_not_greater_than_two_l248_24815

-- Define a lattice point type
structure LatticePoint where
  x : Int
  y : Int

-- Define the condition for a lattice point to be within the 5x5 grid
def isWithinGrid (p : LatticePoint) : Prop :=
  abs p.x ≤ 2 ∧ abs p.y ≤ 2

-- Define a function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : LatticePoint) : ℚ :=
  let x1 := p1.x
  let y1 := p1.y
  let x2 := p2.x
  let y2 := p2.y
  let x3 := p3.x
  let y3 := p3.y
  abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

-- Define the condition for three points to be non-collinear
def nonCollinear (p1 p2 p3 : LatticePoint) : Prop :=
  triangleArea p1 p2 p3 ≠ 0

-- Main theorem
theorem exists_triangle_area_not_greater_than_two 
  (points : Fin 6 → LatticePoint)
  (h_within_grid : ∀ i, isWithinGrid (points i))
  (h_non_collinear : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → nonCollinear (points i) (points j) (points k)) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ triangleArea (points i) (points j) (points k) ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_exists_triangle_area_not_greater_than_two_l248_24815


namespace NUMINAMATH_CALUDE_system_solution_l248_24882

theorem system_solution (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧ 
  (3 * y - Real.sqrt (y / x) - 6 * Real.sqrt (x * y) + 2 = 0) ∧
  (x^2 + 81 * x^2 * y^4 = 2 * y^2) →
  ((x = 1/3 ∧ y = 1/3) ∨ (x = Real.rpow 31 (1/4) / 12 ∧ y = Real.rpow 31 (1/4) / 3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l248_24882


namespace NUMINAMATH_CALUDE_choose_books_different_languages_l248_24896

theorem choose_books_different_languages (chinese english japanese : ℕ) :
  chinese = 5 → english = 4 → japanese = 3 →
  chinese + english + japanese = 12 :=
by sorry

end NUMINAMATH_CALUDE_choose_books_different_languages_l248_24896


namespace NUMINAMATH_CALUDE_sector_arc_length_l248_24817

theorem sector_arc_length (θ : Real) (A : Real) (l : Real) : 
  θ = 120 → A = π → l = (2 * Real.sqrt 3 * π) / 3 → 
  l = (θ * Real.sqrt (3 * A / θ) * π) / 180 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l248_24817


namespace NUMINAMATH_CALUDE_class_size_proof_l248_24827

/-- The number of students in a class with English and German courses -/
def class_size (english_only german_only both : ℕ) : ℕ :=
  english_only + german_only + both

theorem class_size_proof (english_only german_only both : ℕ) 
  (h1 : both = 12)
  (h2 : german_only + both = 22)
  (h3 : english_only = 30) :
  class_size english_only german_only both = 52 := by
  sorry

#check class_size_proof

end NUMINAMATH_CALUDE_class_size_proof_l248_24827


namespace NUMINAMATH_CALUDE_net_salary_proof_l248_24874

/-- Represents a person's monthly financial situation -/
structure MonthlySalary where
  net : ℝ
  discretionary : ℝ
  remaining : ℝ

/-- Calculates the net monthly salary given the conditions -/
def calculate_net_salary (m : MonthlySalary) : Prop :=
  m.discretionary = m.net / 5 ∧
  m.remaining = m.discretionary * 0.1 ∧
  m.remaining = 105 ∧
  m.net = 5250

theorem net_salary_proof (m : MonthlySalary) :
  calculate_net_salary m → m.net = 5250 := by
  sorry

end NUMINAMATH_CALUDE_net_salary_proof_l248_24874


namespace NUMINAMATH_CALUDE_r_daily_earnings_l248_24838

/-- Represents the daily earnings of individuals p, q, and r -/
structure Earnings where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The conditions given in the problem -/
def problem_conditions (e : Earnings) : Prop :=
  9 * (e.p + e.q + e.r) = 1620 ∧
  5 * (e.p + e.r) = 600 ∧
  7 * (e.q + e.r) = 910

/-- The theorem stating that given the problem conditions, r's daily earnings are 70 -/
theorem r_daily_earnings (e : Earnings) : 
  problem_conditions e → e.r = 70 := by
  sorry

#check r_daily_earnings

end NUMINAMATH_CALUDE_r_daily_earnings_l248_24838


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l248_24861

theorem matrix_equation_solution (A B : Matrix (Fin 2) (Fin 2) ℝ) :
  A * B = A - B →
  A * B = ![![7, -2], ![3, -1]] →
  B * A = ![![8, -2], ![3, 0]] := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l248_24861


namespace NUMINAMATH_CALUDE_even_6digit_integers_count_l248_24868

/-- The count of even 6-digit positive integers -/
def count_even_6digit_integers : ℕ :=
  9 * 10^4 * 5

/-- Theorem: The count of even 6-digit positive integers is 450,000 -/
theorem even_6digit_integers_count : count_even_6digit_integers = 450000 := by
  sorry

end NUMINAMATH_CALUDE_even_6digit_integers_count_l248_24868


namespace NUMINAMATH_CALUDE_parabola_properties_l248_24877

-- Define the parabola
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 4 * a * x - 5 * a

theorem parabola_properties :
  ∀ a : ℝ, a ≠ 0 →
  -- 1. Intersections with x-axis
  (∃ x : ℝ, parabola a x = 0 ↔ x = -1 ∨ x = 5) ∧
  -- 2. Conditions for a = 1
  (a > 0 → (∀ m n : ℝ, parabola a m = n → m ≥ 0 → n ≥ -9) → a = 1 ∧ 
    ∀ x : ℝ, parabola 1 x = x^2 - 4*x - 5) ∧
  -- 3. Range of m for shifted parabola
  (∀ m : ℝ, m > 0 → 
    (∃ t : ℝ, -1/2 < t ∧ t < 5/2 ∧ parabola 1 t + m = 0) →
    11/4 < m ∧ m ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l248_24877


namespace NUMINAMATH_CALUDE_min_value_theorem_l248_24847

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 4) :
  (x + 1) * (2*y + 1) / (x*y) ≥ 9/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l248_24847


namespace NUMINAMATH_CALUDE_exactly_one_shot_probability_l248_24831

/-- The probability that exactly one person makes a shot given the probabilities of A and B making shots. -/
theorem exactly_one_shot_probability (p_a p_b : ℝ) (h_a : p_a = 0.8) (h_b : p_b = 0.6) :
  p_a * (1 - p_b) + (1 - p_a) * p_b = 0.44 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_shot_probability_l248_24831


namespace NUMINAMATH_CALUDE_rhombus_count_in_divided_equilateral_triangle_l248_24839

/-- Given an equilateral triangle ABC with each side divided into n equal parts,
    and parallel lines drawn through each division point to form a grid of smaller
    equilateral triangles, the number of rhombuses with side length 1/n in this grid
    is equal to 3 * C(n,2), where C(n,2) is the binomial coefficient. -/
theorem rhombus_count_in_divided_equilateral_triangle (n : ℕ) :
  let num_rhombuses := 3 * (n.choose 2)
  num_rhombuses = 3 * (n * (n - 1)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_count_in_divided_equilateral_triangle_l248_24839


namespace NUMINAMATH_CALUDE_arithmetic_sequence_min_sum_l248_24840

/-- Arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

/-- Sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1 : ℤ) * d) / 2

/-- The value of n that minimizes the sum of the first n terms -/
def minimizing_n (a₁ d : ℤ) : Set ℕ :=
  {n : ℕ | ∀ m : ℕ, arithmetic_sum a₁ d n ≤ arithmetic_sum a₁ d m}

theorem arithmetic_sequence_min_sum :
  minimizing_n (-28) 4 = {7, 8} := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_min_sum_l248_24840


namespace NUMINAMATH_CALUDE_number_reversal_property_l248_24812

/-- Reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Counts the number of zero digits in a natural number -/
def countZeroDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is of the form 1099...989 with k repetitions of 99 -/
def isSpecialForm (n : ℕ) : Prop := ∃ k : ℕ, n = 10^(2*k+1) + 9 * (10^(2*k) - 1) / 99

theorem number_reversal_property (N : ℕ) :
  (9 * N = reverseDigits N) ∧ (countZeroDigits N ≤ 1) ↔ N = 0 ∨ isSpecialForm N :=
sorry

end NUMINAMATH_CALUDE_number_reversal_property_l248_24812


namespace NUMINAMATH_CALUDE_imaginary_unit_power_sum_l248_24857

theorem imaginary_unit_power_sum : ∀ i : ℂ, i^2 = -1 → i^45 + i^205 + i^365 = 3*i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_sum_l248_24857


namespace NUMINAMATH_CALUDE_exists_divisor_friendly_bijection_l248_24879

/-- The number of positive divisors of a positive integer n -/
def d (n : ℕ+) : ℕ := sorry

/-- A bijection is divisor-friendly if it satisfies the given property -/
def divisor_friendly (F : ℕ+ → ℕ+) : Prop :=
  Function.Bijective F ∧ ∀ m n : ℕ+, d (F (m * n)) = d (F m) * d (F n)

/-- There exists a divisor-friendly bijection -/
theorem exists_divisor_friendly_bijection : ∃ F : ℕ+ → ℕ+, divisor_friendly F := by
  sorry

end NUMINAMATH_CALUDE_exists_divisor_friendly_bijection_l248_24879


namespace NUMINAMATH_CALUDE_fourth_largest_divisor_of_n_l248_24841

def n : ℕ := 1000800000

def fourth_largest_divisor (m : ℕ) : ℕ := sorry

theorem fourth_largest_divisor_of_n :
  fourth_largest_divisor n = 62550000 := by sorry

end NUMINAMATH_CALUDE_fourth_largest_divisor_of_n_l248_24841


namespace NUMINAMATH_CALUDE_long_furred_dogs_count_l248_24886

/-- Represents the characteristics of dogs in a kennel --/
structure KennelData where
  total_dogs : ℕ
  brown_dogs : ℕ
  neither_long_furred_nor_brown : ℕ
  long_furred_brown : ℕ

/-- Calculates the number of dogs with long fur in the kennel --/
def long_furred_dogs (data : KennelData) : ℕ :=
  data.long_furred_brown + (data.total_dogs - data.brown_dogs - data.neither_long_furred_nor_brown)

/-- Theorem stating the number of dogs with long fur in the specific kennel scenario --/
theorem long_furred_dogs_count (data : KennelData) 
  (h1 : data.total_dogs = 45)
  (h2 : data.brown_dogs = 30)
  (h3 : data.neither_long_furred_nor_brown = 8)
  (h4 : data.long_furred_brown = 19) :
  long_furred_dogs data = 26 := by
  sorry

#eval long_furred_dogs { total_dogs := 45, brown_dogs := 30, neither_long_furred_nor_brown := 8, long_furred_brown := 19 }

end NUMINAMATH_CALUDE_long_furred_dogs_count_l248_24886
