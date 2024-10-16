import Mathlib

namespace NUMINAMATH_CALUDE_cube_face_sum_l1095_109559

/-- Represents the six positive integers on the faces of a cube -/
structure CubeFaces where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+

/-- Calculates the sum of vertex labels given the face values -/
def vertexSum (faces : CubeFaces) : ℕ :=
  (faces.a * faces.b * faces.c) + (faces.a * faces.e * faces.c) +
  (faces.a * faces.b * faces.f) + (faces.a * faces.e * faces.f) +
  (faces.d * faces.b * faces.c) + (faces.d * faces.e * faces.c) +
  (faces.d * faces.b * faces.f) + (faces.d * faces.e * faces.f)

/-- Calculates the sum of all face values -/
def faceSum (faces : CubeFaces) : ℕ :=
  faces.a + faces.b + faces.c + faces.d + faces.e + faces.f

/-- Theorem: If the vertex sum is 1452, then the face sum is 47 -/
theorem cube_face_sum (faces : CubeFaces) :
  vertexSum faces = 1452 → faceSum faces = 47 := by
  sorry

end NUMINAMATH_CALUDE_cube_face_sum_l1095_109559


namespace NUMINAMATH_CALUDE_sock_pair_count_l1095_109523

/-- The number of ways to choose a pair of socks from a drawer with specific conditions. -/
def sock_pairs (white brown blue : ℕ) : ℕ :=
  let total := white + brown + blue
  let same_color := Nat.choose white 2 + Nat.choose brown 2 + Nat.choose blue 2
  let not_blue := Nat.choose (white + brown) 2
  not_blue

/-- Theorem stating the number of valid sock pairs for the given problem. -/
theorem sock_pair_count :
  sock_pairs 5 5 2 = 45 := by
  sorry

#eval sock_pairs 5 5 2

end NUMINAMATH_CALUDE_sock_pair_count_l1095_109523


namespace NUMINAMATH_CALUDE_print_shop_price_differences_l1095_109590

/-- Represents a print shop with its pricing structure -/
structure PrintShop where
  base_price : ℝ
  discount_threshold : ℕ
  discount_rate : ℝ
  flat_discount : ℝ

/-- Calculates the price for a given number of copies at a print shop -/
def calculate_price (shop : PrintShop) (copies : ℕ) : ℝ :=
  let base_total := shop.base_price * copies
  if copies ≥ shop.discount_threshold then
    base_total * (1 - shop.discount_rate) - shop.flat_discount
  else
    base_total

/-- Theorem stating the price differences between print shops for 60 copies -/
theorem print_shop_price_differences
  (shop_x shop_y shop_z shop_w : PrintShop)
  (hx : shop_x = { base_price := 1.25, discount_threshold := 0, discount_rate := 0, flat_discount := 0 })
  (hy : shop_y = { base_price := 2.75, discount_threshold := 0, discount_rate := 0, flat_discount := 0 })
  (hz : shop_z = { base_price := 3.00, discount_threshold := 50, discount_rate := 0.1, flat_discount := 0 })
  (hw : shop_w = { base_price := 2.00, discount_threshold := 60, discount_rate := 0, flat_discount := 5 }) :
  let copies := 60
  let min_price := min (min (min (calculate_price shop_x copies) (calculate_price shop_y copies))
                            (calculate_price shop_z copies))
                       (calculate_price shop_w copies)
  (calculate_price shop_y copies - min_price = 90) ∧
  (calculate_price shop_z copies - min_price = 87) ∧
  (calculate_price shop_w copies - min_price = 40) := by
  sorry

end NUMINAMATH_CALUDE_print_shop_price_differences_l1095_109590


namespace NUMINAMATH_CALUDE_triangle_inradius_l1095_109502

/-- Given a triangle with perimeter 36 and area 45, prove that its inradius is 2.5 -/
theorem triangle_inradius (P A r : ℝ) (h1 : P = 36) (h2 : A = 45) (h3 : A = r * (P / 2)) : r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l1095_109502


namespace NUMINAMATH_CALUDE_quadratic_root_k_value_l1095_109514

theorem quadratic_root_k_value : ∃ k : ℝ, 3^2 - k*3 - 6 = 0 ∧ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_k_value_l1095_109514


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1095_109538

theorem algebraic_expression_value (a b : ℝ) (h : 2 * a - b = 5) :
  2 * b - 4 * a + 8 = -2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1095_109538


namespace NUMINAMATH_CALUDE_sum_of_24_numbers_l1095_109520

theorem sum_of_24_numbers (numbers : List ℤ) : 
  numbers.length = 24 → numbers.sum = 576 → 
  (∀ n ∈ numbers, Even n) ∨ 
  (∃ (evens odds : List ℤ), 
    numbers = evens ++ odds ∧ 
    (∀ n ∈ evens, Even n) ∧ 
    (∀ n ∈ odds, Odd n) ∧ 
    Even (odds.length)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_24_numbers_l1095_109520


namespace NUMINAMATH_CALUDE_deposit_withdrawal_amount_l1095_109544

/-- Proves that the total amount withdrawn after 4 years of annual deposits
    with compound interest is equal to (a/p) * ((1+p)^5 - (1+p)),
    where a is the annual deposit amount and p is the interest rate. -/
theorem deposit_withdrawal_amount (a p : ℝ) (h₁ : a > 0) (h₂ : p > 0) :
  a * (1 + p)^4 + a * (1 + p)^3 + a * (1 + p)^2 + a * (1 + p) + a = 
  (a / p) * ((1 + p)^5 - (1 + p)) :=
sorry

end NUMINAMATH_CALUDE_deposit_withdrawal_amount_l1095_109544


namespace NUMINAMATH_CALUDE_premium_rate_calculation_l1095_109528

theorem premium_rate_calculation (total_investment dividend_rate share_face_value total_dividend : ℚ)
  (h1 : total_investment = 14400)
  (h2 : dividend_rate = 5 / 100)
  (h3 : share_face_value = 100)
  (h4 : total_dividend = 576) :
  ∃ premium_rate : ℚ,
    premium_rate = 25 ∧
    total_dividend = dividend_rate * share_face_value * (total_investment / (share_face_value + premium_rate)) :=
by sorry


end NUMINAMATH_CALUDE_premium_rate_calculation_l1095_109528


namespace NUMINAMATH_CALUDE_one_nonnegative_solution_for_quadratic_l1095_109534

theorem one_nonnegative_solution_for_quadratic :
  ∃! (x : ℝ), x ≥ 0 ∧ x^2 = -5*x := by sorry

end NUMINAMATH_CALUDE_one_nonnegative_solution_for_quadratic_l1095_109534


namespace NUMINAMATH_CALUDE_perfect_square_coefficient_l1095_109595

theorem perfect_square_coefficient (x : ℝ) : ∃ (r s : ℝ), 
  (81/16) * x^2 + 18 * x + 16 = (r * x + s)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_coefficient_l1095_109595


namespace NUMINAMATH_CALUDE_mary_max_earnings_l1095_109511

/-- Calculates the maximum weekly earnings for Mary given her work conditions -/
theorem mary_max_earnings :
  let max_hours : ℕ := 60
  let regular_hours : ℕ := 20
  let regular_rate : ℚ := 8
  let overtime_rate_increase : ℚ := 0.25
  let overtime_rate : ℚ := regular_rate * (1 + overtime_rate_increase)
  let overtime_hours : ℕ := max_hours - regular_hours
  let regular_earnings : ℚ := regular_hours * regular_rate
  let overtime_earnings : ℚ := overtime_hours * overtime_rate
  regular_earnings + overtime_earnings = 560 := by
sorry

end NUMINAMATH_CALUDE_mary_max_earnings_l1095_109511


namespace NUMINAMATH_CALUDE_complement_of_intersection_l1095_109577

def U : Finset ℕ := {1, 2, 3, 4, 6}
def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {2, 3, 4}

theorem complement_of_intersection (U A B : Finset ℕ) 
  (hU : U = {1, 2, 3, 4, 6})
  (hA : A = {1, 2, 3})
  (hB : B = {2, 3, 4}) :
  U \ (A ∩ B) = {1, 4, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l1095_109577


namespace NUMINAMATH_CALUDE_triangle_side_and_area_l1095_109583

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  cosC : ℝ

/-- Theorem about the side length and area of a specific triangle -/
theorem triangle_side_and_area (t : Triangle) 
  (h1 : t.a = 1)
  (h2 : t.b = 2)
  (h3 : t.cosC = 1/4) :
  t.c = 2 ∧ (1/2 * t.a * t.b * Real.sqrt (1 - t.cosC^2)) = Real.sqrt 15 / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_and_area_l1095_109583


namespace NUMINAMATH_CALUDE_power_six_mod_fifty_l1095_109501

theorem power_six_mod_fifty : 6^2040 ≡ 26 [ZMOD 50] := by sorry

end NUMINAMATH_CALUDE_power_six_mod_fifty_l1095_109501


namespace NUMINAMATH_CALUDE_exists_segment_with_sum_455_l1095_109588

/-- Represents a 10x10 table filled with numbers 1 to 100 as described in the problem -/
def Table := Matrix (Fin 10) (Fin 10) Nat

/-- Defines how the table is filled -/
def fillTable : Table :=
  fun i j => i.val * 10 + j.val + 1

/-- Represents a 7-cell segment in the specified form -/
structure Segment where
  center : Fin 10 × Fin 10
  direction : Bool  -- True for vertical, False for horizontal

/-- Calculates the sum of a segment -/
def segmentSum (t : Table) (s : Segment) : Nat :=
  let (i, j) := s.center
  if s.direction then
    t i j + t (i-1) j + t (i+1) j +
    t (i-1) (j-1) + t (i-1) (j+1) +
    t (i+1) (j-1) + t (i+1) (j+1)
  else
    t i j + t i (j-1) + t i (j+1) +
    t (i-1) (j-1) + t (i+1) (j-1) +
    t (i-1) (j+1) + t (i+1) (j+1)

/-- The main theorem to prove -/
theorem exists_segment_with_sum_455 :
  ∃ s : Segment, segmentSum fillTable s = 455 := by
  sorry

end NUMINAMATH_CALUDE_exists_segment_with_sum_455_l1095_109588


namespace NUMINAMATH_CALUDE_steves_speed_ratio_l1095_109512

/-- Proves the ratio of Steve's speeds given the problem conditions -/
theorem steves_speed_ratio :
  let distance : ℝ := 10 -- km
  let total_time : ℝ := 6 -- hours
  let speed_back : ℝ := 5 -- km/h
  let speed_to_work : ℝ := distance / (total_time - distance / speed_back)
  speed_back / speed_to_work = 2
  := by sorry

end NUMINAMATH_CALUDE_steves_speed_ratio_l1095_109512


namespace NUMINAMATH_CALUDE_triangle_properties_l1095_109556

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sin t.C * Real.sin (t.A - t.B) = Real.sin t.B * Real.sin (t.C - t.A)) 
  (h2 : t.a = 5)
  (h3 : Real.cos t.A = 25 / 31) :
  (2 * t.a^2 = t.b^2 + t.c^2) ∧ 
  (t.a + t.b + t.c = 14) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1095_109556


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l1095_109545

def selling_price : ℝ := 900
def profit : ℝ := 300

theorem profit_percentage_calculation : 
  (profit / (selling_price - profit)) * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l1095_109545


namespace NUMINAMATH_CALUDE_count_odd_numbers_between_150_and_350_l1095_109560

theorem count_odd_numbers_between_150_and_350 : 
  (Finset.filter (fun n => n % 2 = 1 ∧ 150 < n ∧ n < 350) (Finset.range 350)).card = 100 :=
by sorry

end NUMINAMATH_CALUDE_count_odd_numbers_between_150_and_350_l1095_109560


namespace NUMINAMATH_CALUDE_triangle_properties_l1095_109599

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states properties of a specific triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : (2 * t.c - t.a) * Real.cos t.B - t.b * Real.cos t.A = 0)
  (h2 : t.a + t.c = 6)
  (h3 : t.b = 2 * Real.sqrt 3) :
  t.B = π / 3 ∧ 
  (1 / 2) * t.a * t.c * Real.sin t.B = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1095_109599


namespace NUMINAMATH_CALUDE_prob_three_white_correct_prob_two_yellow_one_white_correct_total_earnings_correct_l1095_109569

-- Define the number of yellow and white balls
def yellow_balls : ℕ := 3
def white_balls : ℕ := 3

-- Define the total number of balls
def total_balls : ℕ := yellow_balls + white_balls

-- Define the number of balls drawn
def balls_drawn : ℕ := 3

-- Define the probability of drawing 3 white balls
def prob_three_white : ℚ := 1 / 20

-- Define the probability of drawing 2 yellow and 1 white ball
def prob_two_yellow_one_white : ℚ := 9 / 20

-- Define the number of draws per day
def draws_per_day : ℕ := 100

-- Define the number of days in a month
def days_in_month : ℕ := 30

-- Define the earnings for non-matching draws
def earn_non_matching : ℤ := 1

-- Define the loss for matching draws
def loss_matching : ℤ := 5

-- Theorem for the probability of drawing 3 white balls
theorem prob_three_white_correct :
  prob_three_white = 1 / 20 := by sorry

-- Theorem for the probability of drawing 2 yellow and 1 white ball
theorem prob_two_yellow_one_white_correct :
  prob_two_yellow_one_white = 9 / 20 := by sorry

-- Theorem for the total earnings in a month
theorem total_earnings_correct :
  (draws_per_day * days_in_month * 
   (earn_non_matching * (1 - (prob_three_white + prob_two_yellow_one_white)) - 
    loss_matching * (prob_three_white + prob_two_yellow_one_white))) = 1200 := by sorry

end NUMINAMATH_CALUDE_prob_three_white_correct_prob_two_yellow_one_white_correct_total_earnings_correct_l1095_109569


namespace NUMINAMATH_CALUDE_equation_solutions_l1095_109575

/-- The integer part of a real number -/
noncomputable def intPart (x : ℝ) : ℤ := Int.floor x

/-- The fractional part of a real number -/
noncomputable def fracPart (x : ℝ) : ℝ := x - Int.floor x

/-- The main theorem stating the solutions to the equation -/
theorem equation_solutions :
  ∀ x : ℝ, 
  (intPart x : ℝ) * fracPart x + x = 2 * fracPart x + 9 →
  (x = 9 ∨ x = 8 + 1/7 ∨ x = 7 + 1/3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1095_109575


namespace NUMINAMATH_CALUDE_reflection_of_M_l1095_109567

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The point M -/
def M : ℝ × ℝ := (5, 2)

/-- Theorem: The reflection of M(5, 2) across the x-axis is (5, -2) -/
theorem reflection_of_M : reflect_x M = (5, -2) := by sorry

end NUMINAMATH_CALUDE_reflection_of_M_l1095_109567


namespace NUMINAMATH_CALUDE_playground_area_l1095_109579

/-- Represents a rectangular landscape with a playground -/
structure Landscape where
  breadth : ℝ
  length : ℝ
  playgroundArea : ℝ

/-- Theorem: The area of the playground in a rectangular landscape -/
theorem playground_area (l : Landscape) : 
  l.length = 4 * l.breadth → 
  l.length = 120 → 
  l.playgroundArea = (1/3) * (l.length * l.breadth) → 
  l.playgroundArea = 1200 := by
  sorry

/-- The main result -/
def main_result : ℝ := 1200

#check playground_area
#check main_result

end NUMINAMATH_CALUDE_playground_area_l1095_109579


namespace NUMINAMATH_CALUDE_state_return_cost_l1095_109589

/-- The cost of a federal tax return -/
def federal_cost : ℕ := 50

/-- The cost of quarterly business taxes -/
def quarterly_cost : ℕ := 80

/-- The number of federal returns sold -/
def federal_sold : ℕ := 60

/-- The number of state returns sold -/
def state_sold : ℕ := 20

/-- The number of quarterly returns sold -/
def quarterly_sold : ℕ := 10

/-- The total revenue -/
def total_revenue : ℕ := 4400

/-- The cost of a state return -/
def state_cost : ℕ := 30

theorem state_return_cost :
  federal_cost * federal_sold + state_cost * state_sold + quarterly_cost * quarterly_sold = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_state_return_cost_l1095_109589


namespace NUMINAMATH_CALUDE_sqrt_b_minus_a_l1095_109519

theorem sqrt_b_minus_a (a b : ℝ) 
  (h1 : (2 * a - 1).sqrt = 3)
  (h2 : (3 * a + b - 1)^(1/3) = 3) :
  (b - a).sqrt = 2 * Real.sqrt 2 ∨ (b - a).sqrt = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_b_minus_a_l1095_109519


namespace NUMINAMATH_CALUDE_remaining_problems_to_grade_l1095_109530

theorem remaining_problems_to_grade
  (total_worksheets : ℕ)
  (graded_worksheets : ℕ)
  (problems_per_worksheet : ℕ)
  (h1 : total_worksheets = 17)
  (h2 : graded_worksheets = 8)
  (h3 : problems_per_worksheet = 7)
  : (total_worksheets - graded_worksheets) * problems_per_worksheet = 63 := by
  sorry

end NUMINAMATH_CALUDE_remaining_problems_to_grade_l1095_109530


namespace NUMINAMATH_CALUDE_sauce_per_pulled_pork_sandwich_l1095_109561

/-- The amount of sauce each pulled pork sandwich takes -/
def pulled_pork_sauce : ℚ :=
  1 / 6

theorem sauce_per_pulled_pork_sandwich 
  (total_sauce : ℚ) 
  (burger_sauce : ℚ) 
  (num_burgers : ℕ) 
  (num_pulled_pork : ℕ) 
  (h1 : total_sauce = 5)
  (h2 : burger_sauce = 1 / 4)
  (h3 : num_burgers = 8)
  (h4 : num_pulled_pork = 18)
  (h5 : num_burgers * burger_sauce + num_pulled_pork * pulled_pork_sauce = total_sauce) :
  pulled_pork_sauce = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sauce_per_pulled_pork_sandwich_l1095_109561


namespace NUMINAMATH_CALUDE_simplify_roots_l1095_109526

theorem simplify_roots : (256 : ℝ)^(1/4) * (625 : ℝ)^(1/2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_simplify_roots_l1095_109526


namespace NUMINAMATH_CALUDE_binary_to_septal_conversion_l1095_109516

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its septal (base 7) representation -/
def decimal_to_septal (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem binary_to_septal_conversion :
  let binary := [true, false, true, false, true, true]
  let decimal := binary_to_decimal binary
  let septal := decimal_to_septal decimal
  decimal = 53 ∧ septal = [1, 0, 4] :=
by sorry

end NUMINAMATH_CALUDE_binary_to_septal_conversion_l1095_109516


namespace NUMINAMATH_CALUDE_min_value_a_l1095_109541

theorem min_value_a (a : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x^2 + 2*x*y ≤ a*(x^2 + y^2)) ↔ 
  a ≥ (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_l1095_109541


namespace NUMINAMATH_CALUDE_wedding_guest_ratio_l1095_109539

def wedding_guests (bridgette_guests : ℕ) (extra_plates : ℕ) (spears_per_plate : ℕ) (total_spears : ℕ) : Prop :=
  ∃ (alex_guests : ℕ),
    (bridgette_guests + alex_guests + extra_plates) * spears_per_plate = total_spears ∧
    alex_guests * 3 = bridgette_guests * 2

theorem wedding_guest_ratio :
  wedding_guests 84 10 8 1200 :=
sorry

end NUMINAMATH_CALUDE_wedding_guest_ratio_l1095_109539


namespace NUMINAMATH_CALUDE_cans_purchased_theorem_l1095_109518

/-- The number of cans that can be purchased given the conditions -/
def cans_purchased (N P T : ℚ) : ℚ :=
  5 * N * (T - 1) / P

/-- Theorem stating the number of cans that can be purchased under given conditions -/
theorem cans_purchased_theorem (N P T : ℚ) 
  (h_positive : N > 0 ∧ P > 0 ∧ T > 1) 
  (h_N_P : N / P > 0) -- N cans can be purchased for P quarters
  (h_dollar_worth : (1 : ℚ) = 5 / 4) -- 1 dollar is worth 5 quarters
  (h_fee : (1 : ℚ) > 0) -- There is a 1 dollar fee per transaction
  : cans_purchased N P T = 5 * N * (T - 1) / P :=
sorry

end NUMINAMATH_CALUDE_cans_purchased_theorem_l1095_109518


namespace NUMINAMATH_CALUDE_sin_squared_sum_l1095_109510

theorem sin_squared_sum (α : ℝ) : 
  Real.sin α ^ 2 + Real.sin (α + Real.pi / 3) ^ 2 + Real.sin (α + 2 * Real.pi / 3) ^ 2 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_sum_l1095_109510


namespace NUMINAMATH_CALUDE_integral_sum_equals_pi_over_four_plus_ln_two_l1095_109592

theorem integral_sum_equals_pi_over_four_plus_ln_two :
  ∫ (x : ℝ) in (0)..(1), Real.sqrt (1 - x^2) + ∫ (x : ℝ) in (1)..(2), 1/x = π/4 + Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sum_equals_pi_over_four_plus_ln_two_l1095_109592


namespace NUMINAMATH_CALUDE_points_on_line_procedure_l1095_109576

theorem points_on_line_procedure (n : ℕ) : ∃ n, 9 * n - 8 = 82 := by
  sorry

end NUMINAMATH_CALUDE_points_on_line_procedure_l1095_109576


namespace NUMINAMATH_CALUDE_simplify_sqrt_one_third_l1095_109537

theorem simplify_sqrt_one_third : Real.sqrt (1/3) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_one_third_l1095_109537


namespace NUMINAMATH_CALUDE_step_count_problem_l1095_109581

theorem step_count_problem (x y : Nat) : 
  x < 10 ∧ y < 10 ∧ x ≠ 0 ∧ 
  (100 * y + 10 + x) - (10 * x + y) = 586 → 
  10 * x + y = 26 := by
  sorry

end NUMINAMATH_CALUDE_step_count_problem_l1095_109581


namespace NUMINAMATH_CALUDE_spinner_final_direction_l1095_109573

-- Define the possible directions
inductive Direction
| North
| East
| South
| West

-- Define the rotation type
inductive RotationType
| Clockwise
| Counterclockwise

-- Define a function to represent a rotation
def rotate (initial : Direction) (amount : Rat) (type : RotationType) : Direction :=
  sorry

-- Define the problem statement
theorem spinner_final_direction 
  (initial : Direction)
  (rotation1 : Rat)
  (type1 : RotationType)
  (rotation2 : Rat)
  (type2 : RotationType)
  (h1 : initial = Direction.South)
  (h2 : rotation1 = 19/4)
  (h3 : type1 = RotationType.Clockwise)
  (h4 : rotation2 = 13/2)
  (h5 : type2 = RotationType.Counterclockwise) :
  rotate (rotate initial rotation1 type1) rotation2 type2 = Direction.East :=
sorry

end NUMINAMATH_CALUDE_spinner_final_direction_l1095_109573


namespace NUMINAMATH_CALUDE_binomial_coefficient_equation_l1095_109532

theorem binomial_coefficient_equation : 
  ∀ n : ℤ, (Nat.choose 25 n.toNat + Nat.choose 25 12 = Nat.choose 26 13) ↔ (n = 11 ∨ n = 13) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equation_l1095_109532


namespace NUMINAMATH_CALUDE_certain_number_divisibility_l1095_109546

theorem certain_number_divisibility (m : ℕ+) 
  (h1 : ∃ (k : ℕ+), m = 8 * k) 
  (h2 : ∀ (d : ℕ+), d ∣ m → d ≤ 8) : 
  64 ∣ m^2 ∧ ∀ (n : ℕ+), (∀ (k : ℕ+), n ∣ (8*k)^2) → n ≤ 64 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_divisibility_l1095_109546


namespace NUMINAMATH_CALUDE_max_product_constrained_sum_l1095_109552

theorem max_product_constrained_sum (x y : ℝ) (h1 : x + y = 40) (h2 : x > 0) (h3 : y > 0) :
  x * y ≤ 400 ∧ ∃ (a b : ℝ), a + b = 40 ∧ a > 0 ∧ b > 0 ∧ a * b = 400 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_sum_l1095_109552


namespace NUMINAMATH_CALUDE_train_passing_time_l1095_109513

/-- The time taken for a faster train to catch and pass a slower train -/
theorem train_passing_time (train_length : ℝ) (speed_fast speed_slow : ℝ) : 
  train_length = 25 →
  speed_fast = 46 * (1000 / 3600) →
  speed_slow = 36 * (1000 / 3600) →
  speed_fast > speed_slow →
  (2 * train_length) / (speed_fast - speed_slow) = 18 := by
  sorry

#eval (2 * 25) / ((46 - 36) * (1000 / 3600))

end NUMINAMATH_CALUDE_train_passing_time_l1095_109513


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1095_109580

theorem quadratic_factorization (b : ℤ) :
  (∃ m n p q : ℤ, ∀ x : ℝ, 35 * x^2 + b * x + 35 = (m * x + n) * (p * x + q)) →
  (∃ k : ℤ, b = 2 * k) ∧
  ¬(∀ k : ℤ, ∃ m n p q : ℤ, ∀ x : ℝ, 35 * x^2 + (2 * k) * x + 35 = (m * x + n) * (p * x + q)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1095_109580


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1095_109543

theorem arithmetic_calculation : 2354 + 240 / 60 - 354 * 2 = 1650 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1095_109543


namespace NUMINAMATH_CALUDE_sunglasses_cost_price_l1095_109591

-- Define the given variables
def selling_price : ℝ := 30
def pairs_sold : ℕ := 10
def sign_cost : ℝ := 20

-- Define the theorem
theorem sunglasses_cost_price : 
  ∃ (cost_price : ℝ),
    cost_price = selling_price - (selling_price * pairs_sold - cost_price * pairs_sold - 2 * sign_cost) / pairs_sold :=
by sorry

end NUMINAMATH_CALUDE_sunglasses_cost_price_l1095_109591


namespace NUMINAMATH_CALUDE_cube_edge_length_l1095_109563

theorem cube_edge_length (surface_area : ℝ) (h : surface_area = 16 * Real.pi) :
  ∃ (a : ℝ), a > 0 ∧ a = (4 * Real.sqrt 3) / 3 ∧ 
  surface_area = 4 * Real.pi * ((Real.sqrt 3 * a) / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l1095_109563


namespace NUMINAMATH_CALUDE_class_size_l1095_109500

/-- The number of boys in the class -/
def n : ℕ := sorry

/-- The initial (incorrect) average weight -/
def initial_avg : ℚ := 584/10

/-- The correct average weight -/
def correct_avg : ℚ := 587/10

/-- The difference between the correct and misread weight -/
def weight_diff : ℚ := 62 - 56

theorem class_size :
  (n : ℚ) * initial_avg + weight_diff = n * correct_avg ∧ n = 20 := by sorry

end NUMINAMATH_CALUDE_class_size_l1095_109500


namespace NUMINAMATH_CALUDE_odd_cube_plus_multiple_l1095_109550

theorem odd_cube_plus_multiple (p m : ℤ) (hp : Odd p) :
  Odd (p^3 + m*p) ↔ Even m :=
sorry

end NUMINAMATH_CALUDE_odd_cube_plus_multiple_l1095_109550


namespace NUMINAMATH_CALUDE_sum_base4_equals_1232_l1095_109506

/-- Converts a base 4 number represented as a list of digits to its decimal equivalent -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 4 * acc + d) 0

/-- The sum of 111₄, 323₄, and 132₄ is equal to 1232₄ in base 4 -/
theorem sum_base4_equals_1232 :
  let a := base4ToDecimal [1, 1, 1]
  let b := base4ToDecimal [3, 2, 3]
  let c := base4ToDecimal [1, 3, 2]
  let sum := base4ToDecimal [1, 2, 3, 2]
  a + b + c = sum := by
  sorry

end NUMINAMATH_CALUDE_sum_base4_equals_1232_l1095_109506


namespace NUMINAMATH_CALUDE_prime_pairs_theorem_l1095_109542

def is_valid_pair (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ (7 * p + 1) % q = 0 ∧ (7 * q + 1) % p = 0

theorem prime_pairs_theorem : 
  ∀ p q : ℕ, is_valid_pair p q ↔ (p = 2 ∧ q = 3) ∨ (p = 2 ∧ q = 5) ∨ (p = 3 ∧ q = 11) :=
sorry

end NUMINAMATH_CALUDE_prime_pairs_theorem_l1095_109542


namespace NUMINAMATH_CALUDE_girls_in_class_l1095_109529

theorem girls_in_class (boys girls : ℕ) : 
  girls = boys + 3 →
  girls + boys = 41 →
  girls = 22 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l1095_109529


namespace NUMINAMATH_CALUDE_factorial_difference_quotient_l1095_109508

theorem factorial_difference_quotient (n : ℕ) (h : n ≥ 8) :
  (Nat.factorial (n + 3) - Nat.factorial (n + 2)) / Nat.factorial n = n^2 + 3*n + 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_quotient_l1095_109508


namespace NUMINAMATH_CALUDE_find_A_in_rounding_l1095_109582

theorem find_A_in_rounding : ∃ A : ℕ, 
  (A < 10) ∧ 
  (6000 + A * 100 + 35 ≥ 6100) ∧ 
  (6000 + (A + 1) * 100 + 35 > 6100) → 
  A = 1 := by
sorry

end NUMINAMATH_CALUDE_find_A_in_rounding_l1095_109582


namespace NUMINAMATH_CALUDE_function_properties_l1095_109564

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the derivatives of f and g
variable (f' g' : ℝ → ℝ)

-- State the given conditions
variable (h1 : ∀ x, f (x + 3) = g (-x) + 2)
variable (h2 : ∀ x, f' (x - 1) = g' x)
variable (h3 : ∀ x, g (-x + 1) = -g (x + 1))

-- State the properties to be proved
theorem function_properties :
  (g 1 = 0) ∧
  (∀ x, g' (x + 1) = -g' (3 - x)) ∧
  (∀ x, g (x + 1) = g (3 - x)) ∧
  (∀ x, g (x + 4) = g x) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1095_109564


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1095_109548

theorem negation_of_proposition :
  (¬ ∀ (a b : ℤ), a = 0 → a * b = 0) ↔ (∃ (a b : ℤ), a = 0 ∧ a * b ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1095_109548


namespace NUMINAMATH_CALUDE_castle_extension_l1095_109585

theorem castle_extension (a : ℝ) (ha : a > 0) :
  let original_perimeter := 4 * a
  let new_perimeter := 4 * a + 2 * (0.2 * a)
  let original_area := a ^ 2
  let new_area := a ^ 2 + (0.2 * a) ^ 2
  (new_perimeter = 1.1 * original_perimeter) →
  ((new_area - original_area) / original_area = 0.04) :=
by sorry

end NUMINAMATH_CALUDE_castle_extension_l1095_109585


namespace NUMINAMATH_CALUDE_max_bouquets_sara_l1095_109578

def red_flowers : ℕ := 47
def yellow_flowers : ℕ := 63
def blue_flowers : ℕ := 54
def orange_flowers : ℕ := 29
def pink_flowers : ℕ := 36

theorem max_bouquets_sara :
  ∀ n : ℕ,
    n ≤ red_flowers ∧
    n ≤ yellow_flowers ∧
    n ≤ blue_flowers ∧
    n ≤ orange_flowers ∧
    n ≤ pink_flowers →
    n ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_max_bouquets_sara_l1095_109578


namespace NUMINAMATH_CALUDE_midpoint_ordinate_l1095_109533

theorem midpoint_ordinate (a : Real) (h1 : 0 < a) (h2 : a < π / 2) :
  let P : Real × Real := (a, Real.sin a)
  let Q : Real × Real := (a, Real.cos a)
  let distance := |P.2 - Q.2|
  let midpoint_y := (P.2 + Q.2) / 2
  distance = 1/4 → midpoint_y = Real.sqrt 31 / 8 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_ordinate_l1095_109533


namespace NUMINAMATH_CALUDE_pam_withdrawal_l1095_109524

def initial_balance : ℕ := 400
def current_balance : ℕ := 950

def tripled_balance : ℕ := initial_balance * 3

def withdrawn_amount : ℕ := tripled_balance - current_balance

theorem pam_withdrawal : withdrawn_amount = 250 := by
  sorry

end NUMINAMATH_CALUDE_pam_withdrawal_l1095_109524


namespace NUMINAMATH_CALUDE_chord_inclination_range_l1095_109558

/-- The range of inclination angles for a chord through the focus of a parabola -/
theorem chord_inclination_range (x y : ℝ) (α : ℝ) : 
  (y^2 = 4*x) →                             -- Parabola equation
  (3*x^2 + 2*y^2 = 2) →                     -- Ellipse equation
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    y = (x - 1)*Real.tan α ∧               -- Chord passes through focus (1, 0)
    y^2 = 4*x ∧                            -- Chord intersects parabola
    (x₂ - x₁)^2 + ((x₂ - 1)*Real.tan α - (x₁ - 1)*Real.tan α)^2 ≤ 64) → -- Chord length ≤ 8
  (α ∈ Set.Icc (Real.pi/4) (Real.pi/3) ∪ Set.Icc (2*Real.pi/3) (3*Real.pi/4)) :=
by sorry

end NUMINAMATH_CALUDE_chord_inclination_range_l1095_109558


namespace NUMINAMATH_CALUDE_kylies_coins_l1095_109554

theorem kylies_coins (piggy_bank : ℕ) (brother : ℕ) (father : ℕ) (gave_away : ℕ) (left : ℕ) : 
  piggy_bank = 15 → 
  brother = 13 → 
  gave_away = 21 → 
  left = 15 → 
  piggy_bank + brother + father - gave_away = left → 
  father = 8 := by
sorry

end NUMINAMATH_CALUDE_kylies_coins_l1095_109554


namespace NUMINAMATH_CALUDE_total_tickets_sold_l1095_109584

def student_tickets : ℕ := 90
def non_student_tickets : ℕ := 60

theorem total_tickets_sold : student_tickets + non_student_tickets = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l1095_109584


namespace NUMINAMATH_CALUDE_strawberry_jam_money_l1095_109521

/-- Calculates the total money made from selling strawberry jam given the number of strawberries picked by Betty, Matthew, and Natalie, and the jam-making and selling conditions. -/
theorem strawberry_jam_money (betty_strawberries : ℕ) (matthew_extra : ℕ) (jam_strawberries : ℕ) (jar_price : ℕ) : 
  betty_strawberries = 25 →
  matthew_extra = 30 →
  jam_strawberries = 12 →
  jar_price = 6 →
  let matthew_strawberries := betty_strawberries + matthew_extra
  let natalie_strawberries := matthew_strawberries / 3
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let jars := total_strawberries / jam_strawberries
  let total_money := jars * jar_price
  total_money = 48 := by
sorry

end NUMINAMATH_CALUDE_strawberry_jam_money_l1095_109521


namespace NUMINAMATH_CALUDE_son_work_time_l1095_109562

-- Define the work rates
def man_rate : ℚ := 1 / 6
def combined_rate : ℚ := 1 / 3

-- Define the son's work rate
def son_rate : ℚ := combined_rate - man_rate

-- Theorem to prove
theorem son_work_time : (1 : ℚ) / son_rate = 6 := by sorry

end NUMINAMATH_CALUDE_son_work_time_l1095_109562


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1095_109509

theorem quadratic_roots_sum_product (m n : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x + y = 9 ∧ x * y = 20) →
  m + n = 87 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1095_109509


namespace NUMINAMATH_CALUDE_lcm_gcd_sum_problem_l1095_109522

theorem lcm_gcd_sum_problem (a b : ℕ) (ha : a = 12) (hb : b = 20) :
  (Nat.lcm a b * Nat.gcd a b) + (a + b) = 272 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_sum_problem_l1095_109522


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1095_109587

theorem min_value_of_expression (a : ℝ) (h1 : 1 < a) (h2 : a < 4) :
  a / (4 - a) + 1 / (a - 1) ≥ 2 ∧
  (a / (4 - a) + 1 / (a - 1) = 2 ↔ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1095_109587


namespace NUMINAMATH_CALUDE_multiplication_mistake_difference_l1095_109574

theorem multiplication_mistake_difference : 
  let correct_multiplicand : Nat := 136
  let correct_multiplier : Nat := 43
  let mistaken_multiplier : Nat := 34
  (correct_multiplicand * correct_multiplier) - (correct_multiplicand * mistaken_multiplier) = 1224 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_mistake_difference_l1095_109574


namespace NUMINAMATH_CALUDE_square_side_length_l1095_109504

theorem square_side_length (x : ℝ) (h : x > 0) : 4 * x = 2 * (x ^ 2) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1095_109504


namespace NUMINAMATH_CALUDE_seven_minus_three_times_number_l1095_109540

theorem seven_minus_three_times_number (n : ℝ) (c : ℝ) : 
  n = 3 → 7 * n = 3 * n + c → 7 * n - 3 * n = 12 := by
  sorry

end NUMINAMATH_CALUDE_seven_minus_three_times_number_l1095_109540


namespace NUMINAMATH_CALUDE_kareem_son_age_ratio_l1095_109531

/-- Proves that the ratio of Kareem's age to his son's age is 3:1 --/
theorem kareem_son_age_ratio :
  let kareem_age : ℕ := 42
  let son_age : ℕ := 14
  let future_sum : ℕ := 76
  let future_years : ℕ := 10
  (kareem_age + future_years) + (son_age + future_years) = future_sum →
  (kareem_age : ℚ) / son_age = 3 / 1 := by
sorry

end NUMINAMATH_CALUDE_kareem_son_age_ratio_l1095_109531


namespace NUMINAMATH_CALUDE_f_is_linear_l1095_109515

def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

def f (x : ℝ) : ℝ := -2 * x

theorem f_is_linear : is_linear f := by sorry

end NUMINAMATH_CALUDE_f_is_linear_l1095_109515


namespace NUMINAMATH_CALUDE_definite_integral_sine_cosine_l1095_109551

theorem definite_integral_sine_cosine : 
  ∫ x in (0)..(Real.pi / 2), (4 * Real.sin x + Real.cos x) = 5 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_sine_cosine_l1095_109551


namespace NUMINAMATH_CALUDE_max_product_vertical_multiplication_l1095_109593

theorem max_product_vertical_multiplication :
  ∀ a b : ℕ,
  50 ≤ a ∧ a < 100 →
  100 ≤ b ∧ b < 1000 →
  ∃ c d e f g : ℕ,
  a * b = 10000 * c + 1000 * d + 100 * e + 10 * f + g ∧
  c = 2 ∧ d = 0 ∧ e = 1 ∧ f = 5 →
  a * b ≤ 19864 :=
by sorry

end NUMINAMATH_CALUDE_max_product_vertical_multiplication_l1095_109593


namespace NUMINAMATH_CALUDE_metal_waste_l1095_109553

/-- Given a rectangle with sides a and b (a < b), calculate the total metal wasted
    after cutting out a maximum circular piece and then a maximum square piece from the circle. -/
theorem metal_waste (a b : ℝ) (h : 0 < a ∧ a < b) :
  let circle_area := Real.pi * (a / 2)^2
  let square_side := a / Real.sqrt 2
  let square_area := square_side^2
  ab - square_area = ab - a^2 / 2 := by sorry

end NUMINAMATH_CALUDE_metal_waste_l1095_109553


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1095_109570

/-- Given two vectors OA and OB in 2D space, where OA is perpendicular to AB, prove that m = 4 -/
theorem perpendicular_vectors (OA OB : ℝ × ℝ) (m : ℝ) : 
  OA = (-1, 2) → 
  OB = (3, m) → 
  OA.1 * (OB.1 - OA.1) + OA.2 * (OB.2 - OA.2) = 0 → 
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1095_109570


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_103_l1095_109565

theorem last_three_digits_of_7_to_103 : 7^103 % 1000 = 60 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_103_l1095_109565


namespace NUMINAMATH_CALUDE_function_always_positive_implies_x_range_l1095_109566

theorem function_always_positive_implies_x_range 
  (x : ℝ) 
  (h : ∀ a : ℝ, a ∈ Set.Icc (-1) 1 → x^2 + (a - 4)*x + 4 - 2*a > 0) : 
  x < 1 ∨ x > 3 := by
sorry

end NUMINAMATH_CALUDE_function_always_positive_implies_x_range_l1095_109566


namespace NUMINAMATH_CALUDE_complement_A_in_U_l1095_109571

def U : Set ℕ := {x | x ≥ 3}
def A : Set ℕ := {x | x^2 ≥ 10}

theorem complement_A_in_U : U \ A = {3} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l1095_109571


namespace NUMINAMATH_CALUDE_complex_multiplication_l1095_109536

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 - i) = 1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1095_109536


namespace NUMINAMATH_CALUDE_complex_sum_simplification_l1095_109597

theorem complex_sum_simplification :
  let z₁ : ℂ := (-1 + Complex.I * Real.sqrt 7) / 2
  let z₂ : ℂ := (-1 - Complex.I * Real.sqrt 7) / 2
  z₁^8 + z₂^8 = -7.375 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_simplification_l1095_109597


namespace NUMINAMATH_CALUDE_block_running_difference_l1095_109572

theorem block_running_difference (inner_side_length outer_side_length : ℝ) 
  (h1 : inner_side_length = 450)
  (h2 : outer_side_length = inner_side_length + 50) : 
  4 * outer_side_length - 4 * inner_side_length = 200 :=
by sorry

end NUMINAMATH_CALUDE_block_running_difference_l1095_109572


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l1095_109525

theorem ceiling_neg_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l1095_109525


namespace NUMINAMATH_CALUDE_max_a_value_l1095_109594

/-- Given a quadratic trinomial f(x) = x^2 + ax + b, if for any real x there exists a real y 
    such that f(y) = f(x) + y, then the maximum possible value of a is 1/2. -/
theorem max_a_value (a b : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, (y^2 + a*y + b) = (x^2 + a*x + b) + y) → 
  a ≤ (1/2 : ℝ) ∧ ∃ a₀ : ℝ, a₀ ≤ (1/2 : ℝ) ∧ 
    (∀ x : ℝ, ∃ y : ℝ, (y^2 + a₀*y + b) = (x^2 + a₀*x + b) + y) :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l1095_109594


namespace NUMINAMATH_CALUDE_magician_card_decks_l1095_109586

/-- A problem about a magician selling magic card decks. -/
theorem magician_card_decks (price : ℕ) (decks_left : ℕ) (earnings : ℕ) (initial_decks : ℕ) : 
  price = 2 →
  decks_left = 3 →
  earnings = 4 →
  initial_decks = earnings / price + decks_left →
  initial_decks = 5 := by
  sorry

end NUMINAMATH_CALUDE_magician_card_decks_l1095_109586


namespace NUMINAMATH_CALUDE_equation_is_quadratic_l1095_109555

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := 3 - 5 * x^2 - x

theorem equation_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_equation_is_quadratic_l1095_109555


namespace NUMINAMATH_CALUDE_hexagon_diagonal_intersection_probability_l1095_109507

/-- A convex hexagon -/
structure ConvexHexagon where
  -- Add necessary fields if needed

/-- A diagonal in a convex hexagon -/
structure Diagonal (H : ConvexHexagon) where
  -- Add necessary fields if needed

/-- Predicate to check if two diagonals intersect inside the hexagon -/
def intersect_inside (H : ConvexHexagon) (d1 d2 : Diagonal H) : Prop :=
  sorry

/-- The set of all diagonals in a hexagon -/
def all_diagonals (H : ConvexHexagon) : Set (Diagonal H) :=
  sorry

/-- The number of diagonals in a hexagon -/
def num_diagonals (H : ConvexHexagon) : ℕ :=
  9

/-- The number of pairs of diagonals that intersect inside the hexagon -/
def num_intersecting_pairs (H : ConvexHexagon) : ℕ :=
  15

/-- The probability of two randomly chosen diagonals intersecting inside the hexagon -/
def prob_intersect (H : ConvexHexagon) : ℚ :=
  15 / 36

theorem hexagon_diagonal_intersection_probability (H : ConvexHexagon) :
  prob_intersect H = 5 / 12 :=
sorry

end NUMINAMATH_CALUDE_hexagon_diagonal_intersection_probability_l1095_109507


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_conditions_l1095_109505

theorem smallest_integer_satisfying_conditions : ∃ (N x y : ℕ), 
  N > 0 ∧ 
  (N : ℚ) = 1.2 * x ∧ 
  (N : ℚ) = 0.81 * y ∧ 
  (∀ (M z w : ℕ), M > 0 → (M : ℚ) = 1.2 * z → (M : ℚ) = 0.81 * w → M ≥ N) ∧
  N = 162 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_conditions_l1095_109505


namespace NUMINAMATH_CALUDE_class_average_height_l1095_109596

theorem class_average_height (total_girls : ℕ) (group1_girls : ℕ) (group2_girls : ℕ) 
  (group1_avg_height : ℝ) (group2_avg_height : ℝ) :
  total_girls = group1_girls + group2_girls →
  group1_girls = 30 →
  group2_girls = 10 →
  group1_avg_height = 160 →
  group2_avg_height = 156 →
  (group1_girls * group1_avg_height + group2_girls * group2_avg_height) / total_girls = 159 := by
sorry

end NUMINAMATH_CALUDE_class_average_height_l1095_109596


namespace NUMINAMATH_CALUDE_unequal_gender_probability_l1095_109557

theorem unequal_gender_probability :
  let n : ℕ := 8  -- number of children
  let p : ℚ := 1/2  -- probability of each gender
  let total_outcomes : ℕ := 2^n
  let equal_outcomes : ℕ := n.choose (n/2)
  let unequal_outcomes : ℕ := total_outcomes - equal_outcomes
  (unequal_outcomes : ℚ) / total_outcomes = 93/128 := by
sorry

end NUMINAMATH_CALUDE_unequal_gender_probability_l1095_109557


namespace NUMINAMATH_CALUDE_driver_speed_driver_speed_proof_l1095_109503

/-- The actual average speed of a driver, given that increasing the speed by 12 miles per hour
would have reduced the travel time by 1/3. -/
theorem driver_speed : ℝ → Prop :=
  fun v : ℝ =>
    ∀ t d : ℝ,
      t > 0 → d > 0 →
      d = v * t →
      d = (v + 12) * (2/3 * t) →
      v = 24

-- The proof is omitted
theorem driver_speed_proof : driver_speed 24 := by sorry

end NUMINAMATH_CALUDE_driver_speed_driver_speed_proof_l1095_109503


namespace NUMINAMATH_CALUDE_vector_properties_l1095_109598

def a : ℝ × ℝ := (4, 2)
def b : ℝ × ℝ := (1, 2)

theorem vector_properties :
  let cos_angle := (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))
  let projection := ((a.1 + b.1) * a.1 + (a.2 + b.2) * a.2) / Real.sqrt (a.1^2 + a.2^2)
  cos_angle = 4/5 ∧ projection = 14 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l1095_109598


namespace NUMINAMATH_CALUDE_tangent_circles_area_l1095_109517

/-- The area of the region outside a circle of radius 2 and inside two circles of radius 3
    that are internally tangent to the smaller circle at opposite ends of its diameter -/
theorem tangent_circles_area : Real :=
  let r₁ : Real := 2  -- radius of smaller circle
  let r₂ : Real := 3  -- radius of larger circles
  let total_area : Real := (5 * Real.pi) / 2 - 4 * Real.sqrt 5
  total_area

#check tangent_circles_area

end NUMINAMATH_CALUDE_tangent_circles_area_l1095_109517


namespace NUMINAMATH_CALUDE_coupon_value_l1095_109549

def total_price : ℕ := 67
def num_people : ℕ := 3
def individual_contribution : ℕ := 21

theorem coupon_value :
  total_price - (num_people * individual_contribution) = 4 := by
  sorry

end NUMINAMATH_CALUDE_coupon_value_l1095_109549


namespace NUMINAMATH_CALUDE_mother_twice_lucy_age_year_l1095_109535

def lucy_age_2006 : ℕ := 10
def mother_age_2006 : ℕ := 5 * lucy_age_2006

def year_mother_twice_lucy (y : ℕ) : Prop :=
  mother_age_2006 + (y - 2006) = 2 * (lucy_age_2006 + (y - 2006))

theorem mother_twice_lucy_age_year :
  ∃ y : ℕ, y = 2036 ∧ year_mother_twice_lucy y := by sorry

end NUMINAMATH_CALUDE_mother_twice_lucy_age_year_l1095_109535


namespace NUMINAMATH_CALUDE_pencils_in_drawer_l1095_109568

theorem pencils_in_drawer (initial_pencils final_pencils : ℕ) 
  (h1 : initial_pencils = 2)
  (h2 : final_pencils = 5) :
  final_pencils - initial_pencils = 3 := by
  sorry

end NUMINAMATH_CALUDE_pencils_in_drawer_l1095_109568


namespace NUMINAMATH_CALUDE_number_ratio_l1095_109527

theorem number_ratio (x y z : ℝ) (k : ℝ) : 
  y = 2 * x →
  z = k * y →
  (x + y + z) / 3 = 165 →
  x = 45 →
  z / y = 4 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l1095_109527


namespace NUMINAMATH_CALUDE_leibniz_recursive_relation_leibniz_boundary_condition_pascal_leibniz_relation_l1095_109547

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Element in Leibniz's Triangle -/
def leibniz (n k : ℕ) : ℚ := 1 / ((n + 1 : ℚ) * (binomial n k))

/-- Theorem stating the recursive relationship in Leibniz's Triangle -/
theorem leibniz_recursive_relation (n k : ℕ) (h : 0 < k ∧ k ≤ n) :
  leibniz n (k - 1) + leibniz n k = leibniz (n - 1) (k - 1) := by sorry

/-- Theorem stating that the formula for Leibniz's Triangle satisfies its boundary condition -/
theorem leibniz_boundary_condition (n : ℕ) :
  leibniz n 0 = 1 / (n + 1 : ℚ) ∧ leibniz n n = 1 / (n + 1 : ℚ) := by sorry

/-- Main theorem relating Pascal's Triangle to Leibniz's Triangle -/
theorem pascal_leibniz_relation (n k : ℕ) (h : k ≤ n) :
  leibniz n k = 1 / ((n + 1 : ℚ) * (binomial n k : ℚ)) := by sorry

end NUMINAMATH_CALUDE_leibniz_recursive_relation_leibniz_boundary_condition_pascal_leibniz_relation_l1095_109547
