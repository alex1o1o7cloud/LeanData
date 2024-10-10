import Mathlib

namespace fence_width_is_ten_l388_38884

/-- A rectangular fence with specific properties -/
structure RectangularFence where
  circumference : ℝ
  length : ℝ
  width : ℝ
  circ_eq : circumference = 2 * (length + width)
  width_eq : width = 2 * length

/-- The width of a rectangular fence with circumference 30m and width twice the length is 10m -/
theorem fence_width_is_ten (fence : RectangularFence) 
    (h_circ : fence.circumference = 30) : fence.width = 10 := by
  sorry

end fence_width_is_ten_l388_38884


namespace greatest_power_of_two_factor_l388_38869

theorem greatest_power_of_two_factor (n : ℕ) : 
  (∃ k : ℕ, 12^600 - 8^400 = 2^1204 * k ∧ k % 2 ≠ 0) ∧
  (∀ m : ℕ, m > 1204 → ¬(∃ l : ℕ, 12^600 - 8^400 = 2^m * l)) :=
by sorry

end greatest_power_of_two_factor_l388_38869


namespace smallest_multiples_sum_l388_38840

theorem smallest_multiples_sum : ∃ (a b : ℕ),
  (a ≥ 10 ∧ a < 100 ∧ a % 5 = 0 ∧ ∀ x : ℕ, (x ≥ 10 ∧ x < 100 ∧ x % 5 = 0) → a ≤ x) ∧
  (b ≥ 100 ∧ b < 1000 ∧ b % 6 = 0 ∧ ∀ y : ℕ, (y ≥ 100 ∧ y < 1000 ∧ y % 6 = 0) → b ≤ y) ∧
  a + b = 112 :=
by sorry

end smallest_multiples_sum_l388_38840


namespace roots_of_polynomial_l388_38858

def p (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 2) :=
by sorry

end roots_of_polynomial_l388_38858


namespace product_equals_simplified_fraction_l388_38841

/-- The repeating decimal 0.456̅ as a rational number -/
def repeating_decimal : ℚ := 456 / 999

/-- The product of 0.456̅ and 8 -/
def product : ℚ := repeating_decimal * 8

/-- Theorem stating that the product of 0.456̅ and 8 is equal to 1216/333 -/
theorem product_equals_simplified_fraction : product = 1216 / 333 := by
  sorry

end product_equals_simplified_fraction_l388_38841


namespace vector_CQ_equals_2p_l388_38802

-- Define the space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define points
variable (A B C P Q : V)

-- Define vector p
variable (p : V)

-- Conditions
variable (h1 : P ∈ interior (triangle A B C))
variable (h2 : A - P + 2 • (B - P) + 3 • (C - P) = 0)
variable (h3 : ∃ t : ℝ, Q = C + t • (P - C) ∧ Q ∈ line_through A B)
variable (h4 : C - P = p)

-- Theorem to prove
theorem vector_CQ_equals_2p : C - Q = 2 • p := by sorry

end vector_CQ_equals_2p_l388_38802


namespace total_food_for_three_months_l388_38872

-- Define the number of days in each month
def december_days : ℕ := 31
def january_days : ℕ := 31
def february_days : ℕ := 28

-- Define the amount of food per feeding
def food_per_feeding : ℚ := 1/2

-- Define the number of feedings per day
def feedings_per_day : ℕ := 2

-- Theorem statement
theorem total_food_for_three_months :
  let total_days := december_days + january_days + february_days
  let daily_food := food_per_feeding * feedings_per_day
  total_days * daily_food = 90 := by sorry

end total_food_for_three_months_l388_38872


namespace max_skip_percentage_is_five_percent_l388_38896

/-- The maximum percentage of school days a senior can miss and still skip final exams -/
def max_skip_percentage (total_days : ℕ) (max_skip_days : ℕ) : ℚ :=
  (max_skip_days : ℚ) / (total_days : ℚ) * 100

/-- Theorem stating the maximum percentage of school days a senior can miss in the given scenario -/
theorem max_skip_percentage_is_five_percent :
  max_skip_percentage 180 9 = 5 := by
  sorry

end max_skip_percentage_is_five_percent_l388_38896


namespace program_output_is_66_l388_38859

/-- A simplified representation of the program output -/
def program_output : ℕ := 66

/-- The theorem stating that the program output is 66 -/
theorem program_output_is_66 : program_output = 66 := by sorry

end program_output_is_66_l388_38859


namespace aliyah_vivienne_phone_difference_l388_38833

theorem aliyah_vivienne_phone_difference :
  ∀ (aliyah_phones : ℕ) (vivienne_phones : ℕ),
    vivienne_phones = 40 →
    (aliyah_phones + vivienne_phones) * 400 = 36000 →
    aliyah_phones - vivienne_phones = 10 := by
  sorry

end aliyah_vivienne_phone_difference_l388_38833


namespace min_value_a_l388_38824

theorem min_value_a (a : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x^2 + 2*x*y ≤ a*(x^2 + y^2)) ↔ 
  a ≥ (Real.sqrt 5 + 1) / 2 := by
  sorry

end min_value_a_l388_38824


namespace marks_wage_proof_l388_38873

/-- Mark's hourly wage before the raise -/
def pre_raise_wage : ℝ := 40

/-- Mark's weekly work hours -/
def weekly_hours : ℝ := 40

/-- Mark's raise percentage -/
def raise_percentage : ℝ := 0.05

/-- Mark's weekly expenses -/
def weekly_expenses : ℝ := 700

/-- Mark's leftover money per week -/
def weekly_leftover : ℝ := 980

theorem marks_wage_proof :
  pre_raise_wage * weekly_hours * (1 + raise_percentage) = weekly_expenses + weekly_leftover :=
by sorry

end marks_wage_proof_l388_38873


namespace quiz_competition_arrangements_l388_38834

/-- The number of permutations of k items chosen from n distinct items -/
def permutations (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

/-- Theorem: There are 24 ways to arrange 3 out of 4 distinct items in order -/
theorem quiz_competition_arrangements : permutations 4 3 = 24 := by
  sorry

end quiz_competition_arrangements_l388_38834


namespace cloth_loss_per_meter_l388_38820

/-- Calculates the loss per meter of cloth given the total meters sold, total selling price, and cost price per meter. -/
def loss_per_meter (total_meters : ℕ) (total_selling_price : ℕ) (cost_price_per_meter : ℕ) : ℕ :=
  let total_cost_price := total_meters * cost_price_per_meter
  let total_loss := total_cost_price - total_selling_price
  total_loss / total_meters

/-- Theorem stating that for 400 meters of cloth sold at Rs. 18,000 with a cost price of Rs. 50 per meter, the loss per meter is Rs. 5. -/
theorem cloth_loss_per_meter :
  loss_per_meter 400 18000 50 = 5 := by
  sorry

end cloth_loss_per_meter_l388_38820


namespace beta_values_l388_38893

theorem beta_values (β : ℂ) (h1 : β ≠ 1) 
  (h2 : Complex.abs (β^3 - 1) = 3 * Complex.abs (β - 1))
  (h3 : Complex.abs (β^6 - 1) = 6 * Complex.abs (β - 1)) :
  β = Complex.I * Real.sqrt 2 ∨ β = -Complex.I * Real.sqrt 2 := by
sorry

end beta_values_l388_38893


namespace segment_length_to_reflection_segment_length_F_to_F_l388_38836

/-- The length of a segment from a point to its reflection over the x-axis -/
theorem segment_length_to_reflection (x y : ℝ) : 
  let F : ℝ × ℝ := (x, y)
  let F' : ℝ × ℝ := (x, -y)
  Real.sqrt ((F'.1 - F.1)^2 + (F'.2 - F.2)^2) = 2 * abs y :=
by sorry

/-- The specific case for F(-4, 3) -/
theorem segment_length_F_to_F'_is_6 : 
  let F : ℝ × ℝ := (-4, 3)
  let F' : ℝ × ℝ := (-4, -3)
  Real.sqrt ((F'.1 - F.1)^2 + (F'.2 - F.2)^2) = 6 :=
by sorry

end segment_length_to_reflection_segment_length_F_to_F_l388_38836


namespace cos_2alpha_plus_3pi_over_5_l388_38887

theorem cos_2alpha_plus_3pi_over_5 (α : ℝ) (h : Real.sin (π / 5 - α) = 1 / 4) :
  Real.cos (2 * α + 3 * π / 5) = -7 / 8 := by
  sorry

end cos_2alpha_plus_3pi_over_5_l388_38887


namespace pam_withdrawal_l388_38805

def initial_balance : ℕ := 400
def current_balance : ℕ := 950

def tripled_balance : ℕ := initial_balance * 3

def withdrawn_amount : ℕ := tripled_balance - current_balance

theorem pam_withdrawal : withdrawn_amount = 250 := by
  sorry

end pam_withdrawal_l388_38805


namespace bank_transfer_theorem_l388_38890

def calculate_final_balance (initial_balance : ℚ) (transfer1 : ℚ) (transfer2 : ℚ) (service_charge_rate : ℚ) : ℚ :=
  let service_charge1 := transfer1 * service_charge_rate
  let service_charge2 := transfer2 * service_charge_rate
  initial_balance - (transfer1 + service_charge1) - service_charge2

theorem bank_transfer_theorem (initial_balance : ℚ) (transfer1 : ℚ) (transfer2 : ℚ) (service_charge_rate : ℚ) 
  (h1 : initial_balance = 400)
  (h2 : transfer1 = 90)
  (h3 : transfer2 = 60)
  (h4 : service_charge_rate = 2/100) :
  calculate_final_balance initial_balance transfer1 transfer2 service_charge_rate = 307 := by
  sorry

end bank_transfer_theorem_l388_38890


namespace max_dominos_with_room_for_one_l388_38877

/-- Represents a chessboard -/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a domino -/
structure Domino :=
  (width : Nat)
  (height : Nat)

/-- Represents a placement of dominos on a chessboard -/
def DominoPlacement := List (Nat × Nat)

/-- Function to check if a domino placement is valid -/
def isValidPlacement (board : Chessboard) (domino : Domino) (placement : DominoPlacement) : Prop :=
  sorry

/-- Function to check if there's room for one more domino -/
def hasRoomForOne (board : Chessboard) (domino : Domino) (placement : DominoPlacement) : Prop :=
  sorry

/-- The main theorem -/
theorem max_dominos_with_room_for_one (board : Chessboard) (domino : Domino) :
  board.rows = 6 →
  board.cols = 6 →
  domino.width = 1 →
  domino.height = 2 →
  (∃ (n : Nat) (placement : DominoPlacement),
    n = 11 ∧
    isValidPlacement board domino placement ∧
    placement.length = n ∧
    hasRoomForOne board domino placement) ∧
  (∀ (m : Nat) (placement : DominoPlacement),
    m > 11 →
    isValidPlacement board domino placement →
    placement.length = m →
    ¬hasRoomForOne board domino placement) :=
  by sorry

end max_dominos_with_room_for_one_l388_38877


namespace chocolate_count_l388_38851

/-- The number of boxes of chocolates -/
def num_boxes : ℕ := 6

/-- The number of pieces in each box -/
def pieces_per_box : ℕ := 500

/-- The total number of chocolate pieces -/
def total_pieces : ℕ := num_boxes * pieces_per_box

theorem chocolate_count : total_pieces = 3000 := by
  sorry

end chocolate_count_l388_38851


namespace small_parallelogram_area_l388_38838

/-- Given a parallelogram ABCD with area 1, where sides AB and CD are divided into n equal parts,
    and sides AD and BC are divided into m equal parts, the area of each smaller parallelogram
    formed by connecting the division points is 1 / (mn - 1). -/
theorem small_parallelogram_area (n m : ℕ) (h1 : n > 0) (h2 : m > 0) :
  let total_area : ℝ := 1
  let num_small_parallelograms : ℕ := n * m - 1
  let small_parallelogram_area : ℝ := total_area / num_small_parallelograms
  small_parallelogram_area = 1 / (n * m - 1) := by
  sorry

end small_parallelogram_area_l388_38838


namespace division_problem_l388_38875

theorem division_problem (dividend : Nat) (divisor : Nat) (remainder : Nat) (quotient : Nat) : 
  dividend = 127 → divisor = 14 → remainder = 1 → 
  dividend = divisor * quotient + remainder → quotient = 9 := by
sorry

end division_problem_l388_38875


namespace solve_for_s_l388_38889

theorem solve_for_s (s t : ℚ) 
  (eq1 : 8 * s + 6 * t = 120)
  (eq2 : t - 3 = s) : 
  s = 51 / 7 := by
sorry

end solve_for_s_l388_38889


namespace vowels_on_board_l388_38871

/-- The number of vowels in the alphabet -/
def num_vowels : ℕ := 5

/-- The number of times each vowel is written -/
def times_written : ℕ := 3

/-- The total number of alphabets written on the board -/
def total_written : ℕ := num_vowels * times_written

theorem vowels_on_board : total_written = 15 := by
  sorry

end vowels_on_board_l388_38871


namespace shaded_fraction_of_rectangle_l388_38818

theorem shaded_fraction_of_rectangle : ∀ (length width : ℕ) (shaded_fraction : ℚ),
  length = 15 →
  width = 20 →
  shaded_fraction = 1/4 →
  (shaded_fraction * (1/2 : ℚ)) * (length * width : ℚ) = (1/8 : ℚ) * (length * width : ℚ) :=
by
  sorry

end shaded_fraction_of_rectangle_l388_38818


namespace larger_circle_radius_l388_38827

/-- A system of two circles with specific properties -/
structure CircleSystem where
  R : ℝ  -- Radius of the larger circle
  r : ℝ  -- Radius of the smaller circle
  longest_chord : ℝ  -- Length of the longest chord in the larger circle

/-- Properties of the circle system -/
def circle_system_properties (cs : CircleSystem) : Prop :=
  cs.longest_chord = 24 ∧  -- The longest chord of the larger circle is 24
  cs.r = cs.R / 2 ∧  -- The radius of the smaller circle is half the radius of the larger circle
  cs.R > 0 ∧  -- The radius of the larger circle is positive
  cs.r > 0  -- The radius of the smaller circle is positive

/-- Theorem stating that the radius of the larger circle is 12 -/
theorem larger_circle_radius (cs : CircleSystem) 
  (h : circle_system_properties cs) : cs.R = 12 := by
  sorry

end larger_circle_radius_l388_38827


namespace good_carrots_count_l388_38882

/-- Given that Carol picked 29 carrots, her mother picked 16 carrots, and they had 7 bad carrots,
    prove that the number of good carrots is 38. -/
theorem good_carrots_count (carol_carrots : ℕ) (mom_carrots : ℕ) (bad_carrots : ℕ)
    (h1 : carol_carrots = 29)
    (h2 : mom_carrots = 16)
    (h3 : bad_carrots = 7) :
    carol_carrots + mom_carrots - bad_carrots = 38 := by
  sorry

end good_carrots_count_l388_38882


namespace infinite_even_k_composite_sum_l388_38847

theorem infinite_even_k_composite_sum (t : ℕ+) (p : ℕ) :
  let k := 30 * t + 26
  (∃ n : ℕ+, k = 2 * n) ∧ 
  (Nat.Prime p → ∃ (m n : ℕ+), p^2 + k = m * n ∧ m ≠ 1 ∧ n ≠ 1) :=
by sorry

end infinite_even_k_composite_sum_l388_38847


namespace cindy_pens_l388_38845

theorem cindy_pens (initial_pens : ℕ) (mike_gives : ℕ) (sharon_receives : ℕ) (final_pens : ℕ)
  (h1 : initial_pens = 5)
  (h2 : mike_gives = 20)
  (h3 : sharon_receives = 19)
  (h4 : final_pens = 31) :
  final_pens = initial_pens + mike_gives - sharon_receives + 25 :=
by sorry

end cindy_pens_l388_38845


namespace planted_fraction_is_seven_tenths_l388_38811

/-- Represents a right triangular field with an unplanted square at the right angle -/
structure RightTriangleField where
  leg1 : ℝ
  leg2 : ℝ
  square_to_hypotenuse : ℝ

/-- Calculates the fraction of the field that is planted -/
def planted_fraction (field : RightTriangleField) : ℝ :=
  sorry

/-- Theorem stating that the planted fraction is 7/10 for the given field -/
theorem planted_fraction_is_seven_tenths :
  let field : RightTriangleField := {
    leg1 := 5,
    leg2 := 12,
    square_to_hypotenuse := 3
  }
  planted_fraction field = 7/10 := by sorry

end planted_fraction_is_seven_tenths_l388_38811


namespace count_not_divisible_9999_l388_38864

def count_not_divisible (n : ℕ) : ℕ :=
  n + 1 - (
    (n / 3 + 1) + (n / 5 + 1) + (n / 7 + 1) -
    (n / 15 + 1) - (n / 21 + 1) - (n / 35 + 1) +
    (n / 105 + 1)
  )

theorem count_not_divisible_9999 :
  count_not_divisible 9999 = 4571 := by
sorry

end count_not_divisible_9999_l388_38864


namespace equal_roots_quadratic_l388_38860

/-- For a quadratic equation with two equal real roots, the value of k is ±6 --/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - k*x + 9 = 0 ∧ 
   ∀ y : ℝ, y^2 - k*y + 9 = 0 → y = x) →
  k = 6 ∨ k = -6 := by
sorry

end equal_roots_quadratic_l388_38860


namespace sequence_contains_24_l388_38867

theorem sequence_contains_24 : ∃ n : ℕ+, n * (n + 2) = 24 := by
  sorry

end sequence_contains_24_l388_38867


namespace sock_pair_count_l388_38804

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

end sock_pair_count_l388_38804


namespace min_value_expression_l388_38863

theorem min_value_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x^2 - y^2 = 1) :
  ∃ (min : ℝ), min = 1 ∧ ∀ z, z = 2*x^2 + 3*y^2 - 4*x*y → z ≥ min :=
by sorry

end min_value_expression_l388_38863


namespace total_income_is_139_80_l388_38885

/-- Represents a pastry item with its original price, discount rate, and quantity sold. -/
structure Pastry where
  name : String
  originalPrice : Float
  discountRate : Float
  quantitySold : Nat

/-- Calculates the total income generated from selling pastries after applying discounts. -/
def calculateTotalIncome (pastries : List Pastry) : Float :=
  pastries.foldl (fun acc pastry =>
    let discountedPrice := pastry.originalPrice * (1 - pastry.discountRate)
    acc + discountedPrice * pastry.quantitySold.toFloat
  ) 0

/-- Theorem stating that the total income from the given pastries is $139.80. -/
theorem total_income_is_139_80 : 
  let pastries : List Pastry := [
    { name := "Cupcakes", originalPrice := 3.00, discountRate := 0.30, quantitySold := 25 },
    { name := "Cookies", originalPrice := 2.00, discountRate := 0.45, quantitySold := 18 },
    { name := "Brownies", originalPrice := 4.00, discountRate := 0.25, quantitySold := 15 },
    { name := "Macarons", originalPrice := 1.50, discountRate := 0.50, quantitySold := 30 }
  ]
  calculateTotalIncome pastries = 139.80 := by
  sorry

end total_income_is_139_80_l388_38885


namespace songcheng_visitors_l388_38870

/-- Calculates the total number of visitors to Hangzhou Songcheng on Sunday -/
def total_visitors (morning_visitors : ℕ) (noon_departures : ℕ) (afternoon_increase : ℕ) : ℕ :=
  morning_visitors + (noon_departures + afternoon_increase)

/-- Theorem stating the total number of visitors to Hangzhou Songcheng on Sunday -/
theorem songcheng_visitors :
  total_visitors 500 119 138 = 757 := by
  sorry

end songcheng_visitors_l388_38870


namespace no_chess_tournament_with_804_games_l388_38852

theorem no_chess_tournament_with_804_games : ¬∃ (n : ℕ), n > 0 ∧ n * (n - 4) = 1608 := by
  sorry

end no_chess_tournament_with_804_games_l388_38852


namespace remainder_problem_l388_38843

theorem remainder_problem (n : ℤ) (h : n % 22 = 12) : (2 * n) % 11 = 2 := by
  sorry

end remainder_problem_l388_38843


namespace michaels_class_size_l388_38801

theorem michaels_class_size (b : ℕ) : 
  (100 < b ∧ b < 200) ∧ 
  (∃ k : ℕ, b = 4 * k - 2) ∧ 
  (∃ l : ℕ, b = 5 * l - 3) ∧ 
  (∃ m : ℕ, b = 6 * m - 4) →
  (b = 122 ∨ b = 182) := by
sorry

end michaels_class_size_l388_38801


namespace calculate_interest_rate_l388_38812

/-- Calculates the interest rate at which B lends money to C -/
theorem calculate_interest_rate (principal : ℝ) (rate_ab : ℝ) (time : ℝ) (gain : ℝ) : 
  principal = 4000 →
  rate_ab = 10 →
  time = 3 →
  gain = 180 →
  ∃ (rate_bc : ℝ), rate_bc = 11.5 ∧ 
    principal * (rate_bc / 100) * time = principal * (rate_ab / 100) * time + gain :=
by sorry


end calculate_interest_rate_l388_38812


namespace complex_equation_sum_l388_38848

theorem complex_equation_sum (a t : ℝ) (i : ℂ) : 
  i * i = -1 → a + i = (1 + 2*i) * t*i → t + a = -1 := by
  sorry

end complex_equation_sum_l388_38848


namespace age_puzzle_l388_38865

theorem age_puzzle (A : ℕ) (x : ℕ) (h1 : A = 32) (h2 : 4 * (A + x) - 4 * (A - 4) = A) : x = 4 := by
  sorry

end age_puzzle_l388_38865


namespace A_completes_in_15_days_l388_38815

-- Define the total work as a positive real number
variable (W : ℝ) (hW : W > 0)

-- Define the rate at which A and B work
variable (A_rate B_rate : ℝ)

-- Define the time it takes for A and B to complete the work alone
variable (A_time B_time : ℝ)

-- Conditions from the problem
axiom B_time_18 : B_time = 18
axiom B_rate_def : B_rate = W / B_time
axiom work_split : A_rate * 5 + B_rate * 12 = W
axiom A_rate_def : A_rate = W / A_time

-- Theorem to prove
theorem A_completes_in_15_days : A_time = 15 := by
  sorry

end A_completes_in_15_days_l388_38815


namespace division_problem_l388_38846

theorem division_problem : (-1/24) / (1/3 - 1/6 + 3/8) = -1/13 := by
  sorry

end division_problem_l388_38846


namespace range_of_m_l388_38800

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x - 6

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ∈ Set.Icc (-10) (-6)) ∧
  (∀ y ∈ Set.Icc (-10) (-6), ∃ x ∈ Set.Icc 0 m, f x = y) →
  m ∈ Set.Icc 2 4 :=
sorry

end range_of_m_l388_38800


namespace gcd_lcm_problem_l388_38821

def diamond (x y : ℕ) : ℕ := Nat.gcd x y

def oplus (x y : ℕ) : ℕ := Nat.lcm x y

theorem gcd_lcm_problem : 
  (oplus (oplus (diamond 24 36) (diamond 54 24)) (diamond (48 * 60) (72 * 48))) = 576 := by
  sorry

end gcd_lcm_problem_l388_38821


namespace price_before_increase_l388_38881

/-- Proves that the total price before the increase was 25 pounds, given the original prices and percentage increases. -/
theorem price_before_increase 
  (candy_price : ℝ) 
  (soda_price : ℝ) 
  (candy_increase : ℝ) 
  (soda_increase : ℝ) 
  (h1 : candy_price = 10)
  (h2 : soda_price = 15)
  (h3 : candy_increase = 0.25)
  (h4 : soda_increase = 0.50) :
  candy_price + soda_price = 25 := by
  sorry

#check price_before_increase

end price_before_increase_l388_38881


namespace rectangle_area_l388_38891

/-- Given a rectangle where the sum of width and length is half of 28, and the width is 6,
    prove that its area is 48 square units. -/
theorem rectangle_area (w l : ℝ) : w = 6 → w + l = 28 / 2 → w * l = 48 := by sorry

end rectangle_area_l388_38891


namespace selection_schemes_count_l388_38835

-- Define the number of individuals and cities
def total_individuals : ℕ := 6
def total_cities : ℕ := 4

-- Define the number of restricted individuals (A and B)
def restricted_individuals : ℕ := 2

-- Function to calculate permutations
def permutations (n k : ℕ) : ℕ := (n.factorial) / (n - k).factorial

-- Theorem statement
theorem selection_schemes_count :
  (permutations total_individuals total_cities) -
  (restricted_individuals * permutations (total_individuals - 1) (total_cities - 1)) = 240 :=
sorry

end selection_schemes_count_l388_38835


namespace linear_decreasing_iff_negative_slope_l388_38817

/-- A linear function from ℝ to ℝ -/
def LinearFunction (m b : ℝ) : ℝ → ℝ := fun x ↦ m * x + b

/-- A function is decreasing if for any x₁ < x₂, f(x₁) > f(x₂) -/
def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

theorem linear_decreasing_iff_negative_slope (m b : ℝ) :
  IsDecreasing (LinearFunction m b) ↔ m < 0 := by
  sorry

end linear_decreasing_iff_negative_slope_l388_38817


namespace trivia_game_score_l388_38898

/-- Represents the score distribution in a trivia game --/
structure TriviaGame where
  total_members : Float
  absent_members : Float
  total_points : Float

/-- Calculates the score per member for a given trivia game --/
def score_per_member (game : TriviaGame) : Float :=
  game.total_points / (game.total_members - game.absent_members)

/-- Theorem: In the given trivia game scenario, each member scores 2.0 points --/
theorem trivia_game_score :
  let game := TriviaGame.mk 5.0 2.0 6.0
  score_per_member game = 2.0 := by
  sorry

end trivia_game_score_l388_38898


namespace investment_scientific_notation_l388_38874

/-- Represents the total investment in yuan -/
def total_investment : ℝ := 82000000000

/-- The scientific notation representation of the total investment -/
def scientific_notation : ℝ := 8.2 * (10 ^ 10)

/-- Theorem stating that the total investment equals its scientific notation representation -/
theorem investment_scientific_notation : total_investment = scientific_notation := by
  sorry

end investment_scientific_notation_l388_38874


namespace parabola_perpendicular_range_l388_38825

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Parabola equation y^2 = x + 4 -/
def OnParabola (p : Point) : Prop :=
  p.y^2 = p.x + 4

/-- Perpendicular lines have product of slopes equal to -1 -/
def Perpendicular (a b c : Point) : Prop :=
  (b.y - a.y) * (c.y - b.y) = -(b.x - a.x) * (c.x - b.x)

/-- The main theorem -/
theorem parabola_perpendicular_range :
  ∀ (b c : Point),
    OnParabola b → OnParabola c →
    Perpendicular ⟨0, 2⟩ b c →
    c.y ≤ 0 ∨ c.y ≥ 4 :=
sorry

end parabola_perpendicular_range_l388_38825


namespace tan_equality_proof_l388_38894

theorem tan_equality_proof (n : ℤ) : 
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (225 * π / 180) → n = 45 := by
sorry

end tan_equality_proof_l388_38894


namespace bill_face_value_l388_38813

/-- Calculates the face value of a bill given the true discount, interest rate, and time until due. -/
def face_value (true_discount : ℝ) (interest_rate : ℝ) (time : ℝ) : ℝ :=
  true_discount * (1 + interest_rate * time)

/-- Theorem: The face value of a bill with a true discount of 210, interest rate of 16% per annum, 
    and due in 9 months is 235.20. -/
theorem bill_face_value : 
  face_value 210 0.16 (9 / 12) = 235.20 := by
  sorry

end bill_face_value_l388_38813


namespace min_value_expression_l388_38862

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (b^2 + 2) / (a + b) + a^2 / (a * b + 1) ≥ 2 := by sorry

end min_value_expression_l388_38862


namespace square_root_equation_l388_38888

theorem square_root_equation (n : ℝ) : 
  Real.sqrt (8 + n) = 9 → n = 73 := by
sorry

end square_root_equation_l388_38888


namespace sqrt_equation_solution_l388_38861

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (x - 3) + Real.sqrt (x - 8) = 10 → x = 30.5625 := by
  sorry

end sqrt_equation_solution_l388_38861


namespace percent_relation_l388_38868

theorem percent_relation (x y : ℝ) (h : 0.2 * (x - y) = 0.14 * (x + y)) :
  y = (3 / 17) * x := by
  sorry

end percent_relation_l388_38868


namespace complex_modulus_l388_38828

theorem complex_modulus (z : ℂ) (h : z * (3 - 4*I) = 1) : Complex.abs z = 1/5 := by
  sorry

end complex_modulus_l388_38828


namespace line_BC_equation_l388_38876

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  angle_bisector_B : ℝ → ℝ → Prop
  angle_bisector_C : ℝ → ℝ → Prop

-- Define the specific triangle from the problem
def triangle_ABC : Triangle where
  A := (1, 4)
  angle_bisector_B := λ x y => x - 2*y = 0
  angle_bisector_C := λ x y => x + y - 1 = 0

-- Define the equation of line BC
def line_BC (x y : ℝ) : Prop := 4*x + 17*y + 12 = 0

-- Theorem statement
theorem line_BC_equation (t : Triangle) (h1 : t = triangle_ABC) :
  ∀ x y, t.angle_bisector_B x y ∧ t.angle_bisector_C x y → line_BC x y :=
by sorry

end line_BC_equation_l388_38876


namespace perpendicular_bisector_of_intersection_points_l388_38842

open Real

/-- Represents a curve in polar coordinates -/
structure PolarCurve where
  equation : ℝ → ℝ → Prop

/-- The first curve: ρ = 2sin θ -/
def C₁ : PolarCurve :=
  ⟨λ ρ θ ↦ ρ = 2 * sin θ⟩

/-- The second curve: ρ = 2cos θ -/
def C₂ : PolarCurve :=
  ⟨λ ρ θ ↦ ρ = 2 * cos θ⟩

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Finds the intersection points of two polar curves -/
def intersectionPoints (c₁ c₂ : PolarCurve) : Set PolarPoint :=
  {p | c₁.equation p.ρ p.θ ∧ c₂.equation p.ρ p.θ}

/-- The perpendicular bisector equation -/
def perpendicularBisector (ρ θ : ℝ) : Prop :=
  ρ = 1 / (sin θ + cos θ)

theorem perpendicular_bisector_of_intersection_points :
  ∀ (A B : PolarPoint), A ∈ intersectionPoints C₁ C₂ → B ∈ intersectionPoints C₁ C₂ → A ≠ B →
  ∀ ρ θ, perpendicularBisector ρ θ ↔ 
    (∃ t, ρ * cos θ = A.ρ * cos A.θ + t * (B.ρ * cos B.θ - A.ρ * cos A.θ) ∧
          ρ * sin θ = A.ρ * sin A.θ + t * (B.ρ * sin B.θ - A.ρ * sin A.θ) ∧
          0 < t ∧ t < 1) :=
sorry

end perpendicular_bisector_of_intersection_points_l388_38842


namespace difference_of_squares_l388_38830

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end difference_of_squares_l388_38830


namespace solve_for_y_l388_38844

theorem solve_for_y (x y : ℝ) (h1 : x^(2*y) = 4) (h2 : x = 2) : y = 1 := by
  sorry

end solve_for_y_l388_38844


namespace two_valid_arrangements_l388_38850

/-- Represents an arrangement of people in rows. -/
structure Arrangement where
  rows : ℕ
  front : ℕ

/-- Checks if an arrangement is valid according to the problem conditions. -/
def isValidArrangement (a : Arrangement) : Prop :=
  a.rows ≥ 3 ∧
  a.front * a.rows + a.rows * (a.rows - 1) / 2 = 100

/-- The main theorem stating that there are exactly two valid arrangements. -/
theorem two_valid_arrangements :
  ∃! (s : Finset Arrangement), (∀ a ∈ s, isValidArrangement a) ∧ s.card = 2 := by
  sorry

end two_valid_arrangements_l388_38850


namespace third_side_length_l388_38883

/-- A scalene triangle with integer side lengths satisfying certain conditions -/
structure ScaleneTriangle where
  a : ℤ
  b : ℤ
  c : ℤ
  scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c
  condition : (a - 3)^2 + (b - 2)^2 = 0

/-- The third side of the triangle is either 2, 3, or 4 -/
theorem third_side_length (t : ScaleneTriangle) : t.c = 2 ∨ t.c = 3 ∨ t.c = 4 :=
  sorry

end third_side_length_l388_38883


namespace factorial_fraction_equality_l388_38899

theorem factorial_fraction_equality : (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 4 / 3 := by
  sorry

end factorial_fraction_equality_l388_38899


namespace magician_card_decks_l388_38807

/-- A problem about a magician selling magic card decks. -/
theorem magician_card_decks (price : ℕ) (decks_left : ℕ) (earnings : ℕ) (initial_decks : ℕ) : 
  price = 2 →
  decks_left = 3 →
  earnings = 4 →
  initial_decks = earnings / price + decks_left →
  initial_decks = 5 := by
  sorry

end magician_card_decks_l388_38807


namespace inequality_solution_l388_38857

theorem inequality_solution (x : ℝ) : (1/2)^x - x + 1/2 > 0 → x < 1 := by
  sorry

end inequality_solution_l388_38857


namespace complement_intersection_theorem_l388_38803

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3}
def B : Set Nat := {3, 4}

theorem complement_intersection_theorem : 
  (U \ A) ∩ (U \ B) = {2, 5} := by sorry

end complement_intersection_theorem_l388_38803


namespace interest_equality_problem_l388_38832

theorem interest_equality_problem (total : ℚ) (x : ℚ) : 
  total = 2743 →
  (x * 3 * 8) / 100 = ((total - x) * 5 * 3) / 100 →
  total - x = 1688 := by
  sorry

end interest_equality_problem_l388_38832


namespace sector_arc_length_l388_38855

/-- Given a circular sector with area 4 cm² and central angle 2 radians, 
    the length of its arc is 4 cm. -/
theorem sector_arc_length (area : ℝ) (central_angle : ℝ) (arc_length : ℝ) : 
  area = 4 → central_angle = 2 → arc_length = area / central_angle * 2 := by
  sorry

end sector_arc_length_l388_38855


namespace alex_cake_slices_l388_38886

theorem alex_cake_slices (total_slices : ℕ) (cakes : ℕ) : 
  cakes = 2 →
  (total_slices / 4 : ℚ) + (3 * total_slices / 4 / 3 : ℚ) + 3 + 5 = total_slices →
  total_slices / cakes = 8 := by
sorry

end alex_cake_slices_l388_38886


namespace hyperbola_eccentricity_range_l388_38879

theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_intersect : ∃ x y : ℝ, y = 3*x ∧ x^2/a^2 - y^2/b^2 = 1) :
  let e := Real.sqrt (1 + (b/a)^2)
  e > Real.sqrt 10 := by sorry

end hyperbola_eccentricity_range_l388_38879


namespace last_disc_is_blue_l388_38880

/-- Represents the color of a disc --/
inductive Color
  | Red
  | Blue
  | Yellow

/-- Represents the state of the bag --/
structure BagState where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Initial state of the bag --/
def initial_state : BagState :=
  { red := 7, blue := 8, yellow := 9 }

/-- Represents the rules for drawing and replacing discs --/
def draw_and_replace (state : BagState) : BagState :=
  sorry

/-- Represents the process of repeatedly drawing and replacing discs until the end condition is met --/
def process (state : BagState) : BagState :=
  sorry

/-- Theorem stating that the last remaining disc(s) will be blue --/
theorem last_disc_is_blue :
  ∃ (final_state : BagState), process initial_state = final_state ∧ 
  final_state.blue > 0 ∧ final_state.red = 0 ∧ final_state.yellow = 0 :=
sorry

end last_disc_is_blue_l388_38880


namespace cosine_sine_equation_l388_38837

theorem cosine_sine_equation (x : ℝ) :
  2 * Real.cos x - 3 * Real.sin x = 4 →
  3 * Real.sin x + 2 * Real.cos x = 0 ∨ 3 * Real.sin x + 2 * Real.cos x = 8/13 :=
by sorry

end cosine_sine_equation_l388_38837


namespace function_always_positive_implies_x_range_l388_38819

theorem function_always_positive_implies_x_range 
  (x : ℝ) 
  (h : ∀ a : ℝ, a ∈ Set.Icc (-1) 1 → x^2 + (a - 4)*x + 4 - 2*a > 0) : 
  x < 1 ∨ x > 3 := by
sorry

end function_always_positive_implies_x_range_l388_38819


namespace riding_mower_rate_riding_mower_rate_is_two_l388_38829

theorem riding_mower_rate (total_area : ℝ) (riding_mower_fraction : ℝ) 
  (push_mower_rate : ℝ) (total_time : ℝ) : ℝ :=
by
  -- Define the conditions
  have h1 : total_area = 8 := by sorry
  have h2 : riding_mower_fraction = 3/4 := by sorry
  have h3 : push_mower_rate = 1 := by sorry
  have h4 : total_time = 5 := by sorry

  -- Calculate the area mowed by each mower
  let riding_mower_area := total_area * riding_mower_fraction
  let push_mower_area := total_area * (1 - riding_mower_fraction)

  -- Calculate the time spent with the push mower
  let push_mower_time := push_mower_area / push_mower_rate

  -- Calculate the time spent with the riding mower
  let riding_mower_time := total_time - push_mower_time

  -- Calculate and return the riding mower rate
  exact riding_mower_area / riding_mower_time
  
-- The theorem statement proves that the riding mower rate is 2 acres per hour
theorem riding_mower_rate_is_two : 
  riding_mower_rate 8 (3/4) 1 5 = 2 := by sorry

end riding_mower_rate_riding_mower_rate_is_two_l388_38829


namespace function_inequality_l388_38839

-- Define the function f
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem function_inequality (b c : ℝ) 
  (h : ∀ x : ℝ, f b c (-x) = f b c x) : 
  f b c 1 < f b c (-2) ∧ f b c (-2) < f b c 3 := by
  sorry

end function_inequality_l388_38839


namespace smallest_value_complex_expression_l388_38856

theorem smallest_value_complex_expression (a b c : ℤ) (ω : ℂ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_omega_power : ω^4 = 1) 
  (h_omega_neq_one : ω ≠ 1) :
  ∃ (m : ℝ), m = Real.sqrt 3 ∧ 
    ∀ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z → 
      m ≤ Complex.abs (x + y*ω + z*ω^3) :=
sorry

end smallest_value_complex_expression_l388_38856


namespace symmetry_example_l388_38897

/-- Given two points in a 2D plane, this function checks if they are symmetric with respect to the origin. -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

/-- Theorem stating that the point (2,-3) is symmetric to (-2,3) with respect to the origin. -/
theorem symmetry_example : symmetric_wrt_origin (-2, 3) (2, -3) := by
  sorry

end symmetry_example_l388_38897


namespace reeyas_average_score_l388_38831

def scores : List ℕ := [55, 67, 76, 82, 55]

theorem reeyas_average_score :
  (scores.sum : ℚ) / scores.length = 67 := by sorry

end reeyas_average_score_l388_38831


namespace geometric_problem_l388_38814

/-- Given a parabola and an ellipse with specific properties, prove the coordinates of intersection points, 
    the equation of a hyperbola, and the maximum area of a triangle. -/
theorem geometric_problem (a t : ℝ) (h_a_pos : a > 0) (h_a_range : a ∈ Set.Icc 1 2) (h_t : t > 4) :
  let C₁ := {(x, y) : ℝ × ℝ | y^2 = 4*a*x}
  let C := {(x, y) : ℝ × ℝ | x^2/(2*a^2) + y^2/a^2 = 1}
  let l := {(x, y) : ℝ × ℝ | y = x - a}
  let P := (4*a/3, a/3)
  let Q := ((3 - 2*Real.sqrt 2)*a, (2 - 2*Real.sqrt 2)*a)
  let A := (t, 0)
  let H := {(x, y) : ℝ × ℝ | 7*x^2 - 13*y^2 = 11*a^2}
  (P ∈ C ∧ P ∈ l) ∧
  (Q ∈ C₁ ∧ Q ∈ l) ∧
  (∃ Q' ∈ H, ∃ d : ℝ, d = 4*a ∧ (Q'.1 - Q.1)^2 + (Q'.2 - Q.2)^2 = d^2) ∧
  (∀ a' ∈ Set.Icc 1 2, 
    let S := abs ((P.1 - A.1)*(Q.2 - A.2) - (Q.1 - A.1)*(P.2 - A.2)) / 2
    S ≤ (Real.sqrt 2 - 5/6)*(2*t - 4)) ∧
  (∃ S : ℝ, S = (Real.sqrt 2 - 5/6)*(2*t - 4) ∧
    S = abs ((P.1 - A.1)*(Q.2 - A.2) - (Q.1 - A.1)*(P.2 - A.2)) / 2 ∧
    a = 2) :=
by sorry


end geometric_problem_l388_38814


namespace tangent_triangle_area_l388_38895

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Define the tangent line at (1, -1)
def tangent_line (x : ℝ) : ℝ := -3*x + 2

-- Theorem statement
theorem tangent_triangle_area : 
  let x_intercept : ℝ := 2/3
  let y_intercept : ℝ := tangent_line 0
  let area : ℝ := (1/2) * x_intercept * y_intercept
  (f 1 = -1) ∧ (f' 1 = -3) → area = 2/3 := by
  sorry


end tangent_triangle_area_l388_38895


namespace right_focus_of_hyperbola_l388_38854

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

/-- The right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (2, 0)

/-- Theorem: The right focus of the hyperbola x^2/3 - y^2 = 1 is (2, 0) -/
theorem right_focus_of_hyperbola :
  ∀ (x y : ℝ), hyperbola x y → right_focus = (2, 0) := by
  sorry

end right_focus_of_hyperbola_l388_38854


namespace parabola_directrix_l388_38808

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := x^2 = 4*y

/-- The directrix equation -/
def directrix_equation (y : ℝ) : Prop := y = -1

/-- Theorem: The directrix equation of the parabola x^2 = 4y is y = -1 -/
theorem parabola_directrix : 
  ∀ x y : ℝ, parabola_equation x y → directrix_equation y := by
  sorry

end parabola_directrix_l388_38808


namespace ernie_circles_problem_l388_38810

/-- Given a total number of boxes, the number of boxes Ali uses per circle,
    the number of circles Ali makes, and the number of boxes Ernie uses per circle,
    calculate the number of circles Ernie can make with the remaining boxes. -/
def ernie_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ali_circles : ℕ) (ernie_boxes_per_circle : ℕ) : ℕ :=
  (total_boxes - ali_boxes_per_circle * ali_circles) / ernie_boxes_per_circle

/-- Theorem stating that given 80 boxes, if Ali uses 8 boxes per circle and makes 5 circles,
    and Ernie uses 10 boxes per circle, then Ernie can make 4 circles with the remaining boxes. -/
theorem ernie_circles_problem :
  ernie_circles 80 8 5 10 = 4 := by
  sorry

end ernie_circles_problem_l388_38810


namespace weight_loss_problem_l388_38826

/-- Given four people who lost weight, prove that the last two people each lost 28 kg. -/
theorem weight_loss_problem (total_loss weight_loss1 weight_loss2 weight_loss3 weight_loss4 : ℕ) :
  total_loss = 103 →
  weight_loss1 = 27 →
  weight_loss2 = weight_loss1 - 7 →
  weight_loss3 = weight_loss4 →
  total_loss = weight_loss1 + weight_loss2 + weight_loss3 + weight_loss4 →
  weight_loss3 = 28 ∧ weight_loss4 = 28 := by
  sorry


end weight_loss_problem_l388_38826


namespace intersection_of_A_and_B_l388_38866

def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-2, 2}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end intersection_of_A_and_B_l388_38866


namespace geometric_sequence_product_l388_38853

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  a 3 * a 5 = 4 →
  a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7 = 128 := by
  sorry

end geometric_sequence_product_l388_38853


namespace castle_extension_l388_38806

theorem castle_extension (a : ℝ) (ha : a > 0) :
  let original_perimeter := 4 * a
  let new_perimeter := 4 * a + 2 * (0.2 * a)
  let original_area := a ^ 2
  let new_area := a ^ 2 + (0.2 * a) ^ 2
  (new_perimeter = 1.1 * original_perimeter) →
  ((new_area - original_area) / original_area = 0.04) :=
by sorry

end castle_extension_l388_38806


namespace geometric_sequence_property_l388_38878

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) (h_geom : is_geometric_sequence a) 
  (h1 : a 3 * a 6 = 9) (h2 : a 2 * a 4 * a 5 = 27) : a 2 = 3 := by
  sorry

end geometric_sequence_property_l388_38878


namespace seven_minus_three_times_number_l388_38823

theorem seven_minus_three_times_number (n : ℝ) (c : ℝ) : 
  n = 3 → 7 * n = 3 * n + c → 7 * n - 3 * n = 12 := by
  sorry

end seven_minus_three_times_number_l388_38823


namespace complex_sum_squares_l388_38849

theorem complex_sum_squares (z : ℂ) (h : Complex.abs (z - (3 - 2*I)) = 3) :
  Complex.abs (z + (1 - I))^2 + Complex.abs (z - (7 - 3*I))^2 = 94 := by
  sorry

end complex_sum_squares_l388_38849


namespace quadratic_solution_and_max_product_l388_38892

-- Define the quadratic inequality
def quadratic_inequality (x m : ℝ) : Prop := x^2 - 3*x + m < 0

-- Define the solution set
def solution_set (x n : ℝ) : Prop := 1 < x ∧ x < n

-- Define the constraint for a and b
def constraint (m n a b : ℝ) : Prop := m*a + 2*n*b = 3

-- Theorem statement
theorem quadratic_solution_and_max_product :
  ∃ (m n : ℝ),
    (∀ x, quadratic_inequality x m ↔ solution_set x n) ∧
    (m = 2 ∧ n = 2) ∧
    (∀ a b : ℝ, a > 0 → b > 0 → constraint m n a b →
      a * b ≤ 9/32 ∧ ∃ a₀ b₀, a₀ > 0 ∧ b₀ > 0 ∧ constraint m n a₀ b₀ ∧ a₀ * b₀ = 9/32) :=
by sorry

end quadratic_solution_and_max_product_l388_38892


namespace smallest_bob_number_l388_38809

def alice_number : ℕ := 36

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → (p ∣ n → p ∣ m)

theorem smallest_bob_number :
  ∃ (bob_number : ℕ), 
    has_all_prime_factors alice_number bob_number ∧
    (∀ m : ℕ, has_all_prime_factors alice_number m → bob_number ≤ m) ∧
    bob_number = 6 :=
sorry

end smallest_bob_number_l388_38809


namespace derivative_tangent_line_existence_no_derivative_no_tangent_slope_l388_38822

-- Define a real-valued function f
variable (f : ℝ → ℝ)
-- Define a point x₀
variable (x₀ : ℝ)

-- Define the existence of a derivative at x₀
def has_derivative_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ L, ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → |f x - f x₀ - L * (x - x₀)| ≤ ε * |x - x₀|

-- Define the existence of a tangent line at x₀
def has_tangent_line_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ m b, ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → |f x - (m * x + b)| < ε * |x - x₀|

-- Define the existence of a slope of the tangent line at x₀
def has_tangent_slope_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ m, ∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → |f x - f x₀ - m * (x - x₀)| < ε * |x - x₀|

-- Theorem 1: Non-existence of derivative doesn't imply non-existence of tangent line
theorem derivative_tangent_line_existence (f : ℝ → ℝ) (x₀ : ℝ) :
  ¬(has_derivative_at f x₀) → (has_tangent_line_at f x₀ ∨ ¬(has_tangent_line_at f x₀)) :=
sorry

-- Theorem 2: Non-existence of derivative implies non-existence of tangent slope
theorem no_derivative_no_tangent_slope (f : ℝ → ℝ) (x₀ : ℝ) :
  ¬(has_derivative_at f x₀) → ¬(has_tangent_slope_at f x₀) :=
sorry

end derivative_tangent_line_existence_no_derivative_no_tangent_slope_l388_38822


namespace amy_work_hours_l388_38816

/-- Calculates the required weekly hours for a given total earnings, number of weeks, and hourly rate -/
def required_weekly_hours (total_earnings : ℚ) (num_weeks : ℚ) (hourly_rate : ℚ) : ℚ :=
  total_earnings / (num_weeks * hourly_rate)

/-- Represents Amy's work scenario -/
theorem amy_work_hours 
  (summer_weekly_hours : ℚ) 
  (summer_weeks : ℚ) 
  (summer_earnings : ℚ) 
  (school_weeks : ℚ) 
  (school_earnings : ℚ)
  (h1 : summer_weekly_hours = 45)
  (h2 : summer_weeks = 8)
  (h3 : summer_earnings = 3600)
  (h4 : school_weeks = 24)
  (h5 : school_earnings = 3600) :
  required_weekly_hours school_earnings school_weeks 
    (summer_earnings / (summer_weekly_hours * summer_weeks)) = 15 := by
  sorry

#eval required_weekly_hours 3600 24 (3600 / (45 * 8))

end amy_work_hours_l388_38816
