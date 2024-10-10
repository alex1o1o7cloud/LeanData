import Mathlib

namespace distribute_5_3_l1908_190887

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to distribute 5 distinguishable objects into 3 distinguishable containers is 3^5 -/
theorem distribute_5_3 : distribute 5 3 = 3^5 := by
  sorry

end distribute_5_3_l1908_190887


namespace least_five_digit_congruent_to_seven_mod_twelve_l1908_190897

theorem least_five_digit_congruent_to_seven_mod_twelve :
  ∃ n : ℕ, 
    (n ≥ 10000 ∧ n < 100000) ∧  -- n is a five-digit number
    n % 12 = 7 ∧               -- n is congruent to 7 (mod 12)
    (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 12 = 7 → m ≥ n) ∧  -- n is the least such number
    n = 10003 :=               -- n equals 10003
by sorry

end least_five_digit_congruent_to_seven_mod_twelve_l1908_190897


namespace b_value_for_decreasing_increasing_cubic_l1908_190891

theorem b_value_for_decreasing_increasing_cubic (a c : ℝ) :
  let f : ℝ → ℝ := λ x ↦ -x^3 + a*x^2 + b*x + c
  (∀ x < 0, ∀ y < 0, x < y → f x > f y) →
  (∀ x ∈ Set.Ioo 0 1, ∀ y ∈ Set.Ioo 0 1, x < y → f x < f y) →
  b = 0 := by
sorry

end b_value_for_decreasing_increasing_cubic_l1908_190891


namespace smallest_number_is_10011_binary_l1908_190850

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem smallest_number_is_10011_binary :
  let a := 25
  let b := 111
  let c := binary_to_decimal [false, true, true, false, true]
  let d := binary_to_decimal [true, true, false, false, true]
  d = min a (min b (min c d)) :=
by sorry

end smallest_number_is_10011_binary_l1908_190850


namespace ivan_journey_time_l1908_190886

/-- Represents the journey details of Ivan and Peter --/
structure Journey where
  distance : ℝ
  ivan_speed : ℝ
  peter_speed : ℝ
  peter_wait_time : ℝ
  cafe_time : ℝ

/-- The theorem stating Ivan's total journey time --/
theorem ivan_journey_time (j : Journey) 
  (h1 : j.distance > 0)
  (h2 : j.ivan_speed > 0)
  (h3 : j.peter_speed > 0)
  (h4 : j.peter_wait_time = 10)
  (h5 : j.cafe_time = 30)
  (h6 : j.distance / (3 * j.ivan_speed) = j.distance / j.peter_speed + j.peter_wait_time)
  (h7 : j.distance / j.ivan_speed = 2 * (j.distance / j.peter_speed + j.peter_wait_time + j.cafe_time))
  : j.distance / j.ivan_speed = 75 := by
  sorry


end ivan_journey_time_l1908_190886


namespace prob_log_is_integer_l1908_190864

/-- A four-digit positive integer -/
def FourDigitInt := {n : ℕ // 1000 ≤ n ∧ n ≤ 9999}

/-- The count of four-digit positive integers -/
def countFourDigitInts : ℕ := 9000

/-- Predicate for N being a power of 10 -/
def isPowerOfTen (N : FourDigitInt) : Prop :=
  ∃ k : ℕ, N.val = 10^k

/-- The count of four-digit numbers that are powers of 10 -/
def countPowersOfTen : ℕ := 1

/-- The probability of a randomly chosen four-digit number being a power of 10 -/
def probPowerOfTen : ℚ :=
  countPowersOfTen / countFourDigitInts

theorem prob_log_is_integer :
  probPowerOfTen = 1 / 9000 := by sorry

end prob_log_is_integer_l1908_190864


namespace rectangle_equation_l1908_190888

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.length

theorem rectangle_equation (r : Rectangle) :
  r.length = r.width + 12 →
  r.area = 864 →
  r.width * (r.width + 12) = 864 := by
  sorry

end rectangle_equation_l1908_190888


namespace cookies_per_sheet_l1908_190852

theorem cookies_per_sheet (members : ℕ) (sheets_per_member : ℕ) (total_cookies : ℕ) :
  members = 100 →
  sheets_per_member = 10 →
  total_cookies = 16000 →
  total_cookies / (members * sheets_per_member) = 16 :=
by sorry

end cookies_per_sheet_l1908_190852


namespace two_composites_in_sequence_l1908_190868

/-- A sequence where each term is formed by appending a digit (other than 9) to the preceding term -/
def AppendDigitSequence (a : ℕ → ℕ) : Prop :=
  ∀ n, ∃ d, 0 ≤ d ∧ d ≤ 8 ∧ a (n + 1) = 10 * a n + d

/-- Proposition: In an infinite sequence where each term is formed by appending a digit (other than 9)
    to the preceding term, and the first term is any two-digit number, 
    there are at least two composite numbers in the sequence. -/
theorem two_composites_in_sequence (a : ℕ → ℕ) 
    (h_seq : AppendDigitSequence a) 
    (h_start : 10 ≤ a 0 ∧ a 0 < 100) : 
  ∃ i j, i ≠ j ∧ ¬ Nat.Prime (a i) ∧ ¬ Nat.Prime (a j) := by
  sorry

end two_composites_in_sequence_l1908_190868


namespace system_solution_l1908_190815

theorem system_solution (x y z : ℝ) : 
  (5*x + 7*y) / (x + y) = 6 ∧
  3*(z - x) / (x - y + z) = 1 ∧
  (2*x + 3*y - z) / (x/2 + 3) = 4 →
  x = 8 ∧ y = 8 ∧ z = 12 := by
sorry

end system_solution_l1908_190815


namespace simplify_fraction_l1908_190880

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) :
  (3 / (x - 1) - x - 1) / ((x - 2) / (x^2 - 2*x + 1)) = -x^2 - x + 2 := by
  sorry

end simplify_fraction_l1908_190880


namespace smallest_battleship_board_l1908_190844

/-- Represents a ship in the Battleship game -/
structure Ship :=
  (size : Nat)

/-- Represents the Battleship game board -/
structure Board :=
  (size : Nat)
  (ships : List Ship)

/-- Checks if the given board configuration is valid -/
def isValidBoard (board : Board) : Prop :=
  board.size ≥ 7 ∧
  board.ships.length = 10 ∧
  (board.ships.filter (λ s => s.size = 4)).length = 1 ∧
  (board.ships.filter (λ s => s.size = 3)).length = 2 ∧
  (board.ships.filter (λ s => s.size = 2)).length = 3 ∧
  (board.ships.filter (λ s => s.size = 1)).length = 4

/-- Theorem: The smallest valid square board for Battleship is 7x7 -/
theorem smallest_battleship_board :
  ∀ (board : Board), isValidBoard board →
    ∃ (minBoard : Board), isValidBoard minBoard ∧ minBoard.size = 7 ∧
      ∀ (b : Board), isValidBoard b → b.size ≥ minBoard.size :=
by sorry

end smallest_battleship_board_l1908_190844


namespace march_birthdays_march_birthdays_value_l1908_190837

/-- The number of Santana's brothers -/
def total_brothers : ℕ := 7

/-- The number of brothers with birthdays in October -/
def october_birthdays : ℕ := 1

/-- The number of brothers with birthdays in November -/
def november_birthdays : ℕ := 1

/-- The number of brothers with birthdays in December -/
def december_birthdays : ℕ := 2

/-- The number of presents bought in the second half of the year -/
def presents_second_half : ℕ := october_birthdays + november_birthdays + december_birthdays + total_brothers

/-- The difference in presents between the second and first half of the year -/
def present_difference : ℕ := 8

theorem march_birthdays : ℕ :=
  presents_second_half - present_difference
  
theorem march_birthdays_value : march_birthdays = 3 := by
  sorry

end march_birthdays_march_birthdays_value_l1908_190837


namespace impossibility_of_transformation_l1908_190876

/-- Represents a triplet of integers -/
structure Triplet where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Represents the allowed operation on a triplet -/
inductive Operation
  | inc_a_dec_bc : Operation
  | inc_b_dec_ac : Operation
  | inc_c_dec_ab : Operation

/-- Applies an operation to a triplet -/
def apply_operation (t : Triplet) (op : Operation) : Triplet :=
  match op with
  | Operation.inc_a_dec_bc => ⟨t.a + 2, t.b - 1, t.c - 1⟩
  | Operation.inc_b_dec_ac => ⟨t.a - 1, t.b + 2, t.c - 1⟩
  | Operation.inc_c_dec_ab => ⟨t.a - 1, t.b - 1, t.c + 2⟩

/-- Checks if a triplet has two zeros -/
def has_two_zeros (t : Triplet) : Prop :=
  (t.a = 0 ∧ t.b = 0) ∨ (t.a = 0 ∧ t.c = 0) ∨ (t.b = 0 ∧ t.c = 0)

/-- Defines a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to a triplet -/
def apply_sequence (t : Triplet) : OperationSequence → Triplet
  | [] => t
  | op :: ops => apply_sequence (apply_operation t op) ops

theorem impossibility_of_transformation : 
  ∀ (ops : OperationSequence), ¬(has_two_zeros (apply_sequence ⟨13, 15, 17⟩ ops)) :=
by
  sorry

end impossibility_of_transformation_l1908_190876


namespace pen_pencil_difference_l1908_190843

theorem pen_pencil_difference (pen_count : ℕ) (pencil_count : ℕ) : 
  pencil_count = 48 →
  pen_count * 6 = pencil_count * 5 →
  pencil_count > pen_count →
  pencil_count - pen_count = 8 := by
sorry

end pen_pencil_difference_l1908_190843


namespace f_is_even_and_increasing_l1908_190869

def f (x : ℝ) : ℝ := |x| + 1

theorem f_is_even_and_increasing :
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 < x → x < y → f x < f y) :=
sorry

end f_is_even_and_increasing_l1908_190869


namespace hope_project_donation_proof_l1908_190846

theorem hope_project_donation_proof 
  (total_donation_A total_donation_B : ℝ)
  (donation_difference : ℝ)
  (people_ratio : ℝ) :
  total_donation_A = 20000 →
  total_donation_B = 20000 →
  donation_difference = 20 →
  people_ratio = 4/5 →
  ∃ (people_A : ℝ) (donation_A donation_B : ℝ),
    people_A > 0 ∧
    donation_A > 0 ∧
    donation_B > 0 ∧
    people_A * donation_A = total_donation_A ∧
    (people_ratio * people_A) * donation_B = total_donation_B ∧
    donation_B = donation_A + donation_difference ∧
    donation_A = 80 ∧
    donation_B = 100 :=
by sorry

end hope_project_donation_proof_l1908_190846


namespace function_maximum_ratio_l1908_190856

/-- Given a function f(x) = x³ + ax² + bx - a² - 7a, if f(x) attains a maximum
    value of 10 at x = 1, then b/a = -3/2 -/
theorem function_maximum_ratio (a b : ℝ) : 
  let f := fun (x : ℝ) => x^3 + a*x^2 + b*x - a^2 - 7*a
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1 → |x - 1| < ε → f x < f 1) ∧ 
  f 1 = 10 → 
  b / a = -3/2 :=
sorry

end function_maximum_ratio_l1908_190856


namespace inequality_implication_l1908_190871

theorem inequality_implication (a b c : ℝ) : a * c^2 > b * c^2 → a > b := by
  sorry

end inequality_implication_l1908_190871


namespace time_after_1450_minutes_l1908_190820

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

def midnight : Time where
  hours := 0
  minutes := 0
  h_valid := by simp
  m_valid := by simp

def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := (totalMinutes / 60) % 24,
    minutes := totalMinutes % 60,
    h_valid := by sorry
    m_valid := by sorry }

theorem time_after_1450_minutes :
  addMinutes midnight 1450 = { hours := 0, minutes := 10, h_valid := by simp, m_valid := by simp } :=
by sorry

end time_after_1450_minutes_l1908_190820


namespace viewing_spot_coordinate_l1908_190838

/-- Given two landmarks and a viewing spot in a park, this theorem proves the coordinate of the viewing spot. -/
theorem viewing_spot_coordinate 
  (landmark1 landmark2 : ℝ) 
  (h1 : landmark1 = 150)
  (h2 : landmark2 = 450)
  (h3 : landmark2 > landmark1) :
  let distance := landmark2 - landmark1
  let viewing_spot := landmark1 + (2/3 * distance)
  viewing_spot = 350 := by
sorry

end viewing_spot_coordinate_l1908_190838


namespace total_dimes_proof_l1908_190847

/-- Calculates the total number of dimes Tom has after receiving more from his dad. -/
def total_dimes (initial_dimes : ℕ) (dimes_from_dad : ℕ) : ℕ :=
  initial_dimes + dimes_from_dad

/-- Proves that the total number of dimes Tom has is the sum of his initial dimes and those given by his dad. -/
theorem total_dimes_proof (initial_dimes : ℕ) (dimes_from_dad : ℕ) :
  total_dimes initial_dimes dimes_from_dad = initial_dimes + dimes_from_dad := by
  sorry

#eval total_dimes 15 33  -- Should output 48

end total_dimes_proof_l1908_190847


namespace product_of_ten_fractions_is_one_tenth_l1908_190814

theorem product_of_ten_fractions_is_one_tenth : 
  ∃ (a b c d e f g h i j : ℚ), 
    (0 < a ∧ a < 1) ∧ 
    (0 < b ∧ b < 1) ∧ 
    (0 < c ∧ c < 1) ∧ 
    (0 < d ∧ d < 1) ∧ 
    (0 < e ∧ e < 1) ∧ 
    (0 < f ∧ f < 1) ∧ 
    (0 < g ∧ g < 1) ∧ 
    (0 < h ∧ h < 1) ∧ 
    (0 < i ∧ i < 1) ∧ 
    (0 < j ∧ j < 1) ∧ 
    a * b * c * d * e * f * g * h * i * j = 1 / 10 :=
by sorry


end product_of_ten_fractions_is_one_tenth_l1908_190814


namespace new_cost_percentage_l1908_190823

variable (t b : ℝ)
variable (cost : ℝ → ℝ)

/-- The cost function is defined as tb^4 --/
def cost_function (t b : ℝ) : ℝ := t * b^4

/-- The original cost --/
def original_cost : ℝ := cost_function t b

/-- The new cost when b is doubled --/
def new_cost : ℝ := cost_function t (2*b)

/-- The theorem stating that the new cost is 1600% of the original cost --/
theorem new_cost_percentage : new_cost = 16 * original_cost := by sorry

end new_cost_percentage_l1908_190823


namespace x_plus_y_value_l1908_190822

theorem x_plus_y_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x / 3 = y^2) (h2 : x / 9 = 9*y) : x + y = 2214 := by
  sorry

end x_plus_y_value_l1908_190822


namespace scientific_notation_of_2720000_l1908_190857

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_2720000 :
  toScientificNotation 2720000 = ScientificNotation.mk 2.72 6 (by norm_num) := by
  sorry

end scientific_notation_of_2720000_l1908_190857


namespace symmetric_point_coordinates_l1908_190812

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the original point A
def A : Point := (-2, 3)

-- Define the symmetry operation about the x-axis
def symmetry_x_axis (p : Point) : Point :=
  (p.1, -p.2)

-- Theorem statement
theorem symmetric_point_coordinates :
  symmetry_x_axis A = (-2, -3) := by
  sorry

end symmetric_point_coordinates_l1908_190812


namespace divisor_product_256_l1908_190874

def divisor_product (n : ℕ+) : ℕ :=
  (List.range n.val).filter (λ i => i > 0 ∧ n.val % i = 0)
    |>.map (λ i => i + 1)
    |>.prod

theorem divisor_product_256 (n : ℕ+) :
  divisor_product n = 256 → n = 16 := by
  sorry

end divisor_product_256_l1908_190874


namespace binary_to_base4_conversion_l1908_190834

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldr (λ bit acc => 2 * acc + if bit then 1 else 0) 0

/-- Converts a decimal number to its base-4 representation -/
def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The binary representation of 1101010101₂ -/
def binary_num : List Bool :=
  [true, true, false, true, false, true, false, true, false, true]

theorem binary_to_base4_conversion :
  decimal_to_base4 (binary_to_decimal binary_num) = [3, 1, 1, 1, 1] := by
  sorry

#eval decimal_to_base4 (binary_to_decimal binary_num)

end binary_to_base4_conversion_l1908_190834


namespace total_letters_in_seven_hours_l1908_190842

def letters_per_hour (nathan jacob emily : ℕ) : ℕ :=
  nathan + jacob + emily

theorem total_letters_in_seven_hours 
  (nathan_speed : ℕ) 
  (h1 : nathan_speed = 25)
  (jacob_speed : ℕ) 
  (h2 : jacob_speed = 2 * nathan_speed)
  (emily_speed : ℕ) 
  (h3 : emily_speed = 3 * nathan_speed)
  : letters_per_hour nathan_speed jacob_speed emily_speed * 7 = 1050 :=
by
  sorry

end total_letters_in_seven_hours_l1908_190842


namespace prob_same_color_24_sided_die_l1908_190813

/-- Represents a 24-sided die with colored sides -/
structure ColoredDie :=
  (purple : Nat)
  (blue : Nat)
  (red : Nat)
  (gold : Nat)
  (total : Nat)
  (h_total : purple + blue + red + gold = total)

/-- The probability of rolling a specific color on a single die -/
def prob_color (d : ColoredDie) (color : Nat) : Rat :=
  color / d.total

/-- The probability of rolling the same color on two identical dice -/
def prob_same_color (d : ColoredDie) : Rat :=
  (prob_color d d.purple)^2 + (prob_color d d.blue)^2 +
  (prob_color d d.red)^2 + (prob_color d d.gold)^2

/-- The specific 24-sided die configuration from the problem -/
def problem_die : ColoredDie :=
  { purple := 5
    blue := 8
    red := 10
    gold := 1
    total := 24
    h_total := by simp }

theorem prob_same_color_24_sided_die :
  prob_same_color problem_die = 95 / 288 := by
  sorry


end prob_same_color_24_sided_die_l1908_190813


namespace min_parts_for_triangle_flip_l1908_190854

/-- Represents a triangle in a 2D plane -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a part of a triangle that can be flipped -/
structure TrianglePart where
  vertices : List Point

/-- A function that determines if a list of triangle parts can recreate the original triangle when flipped -/
def canReconstructTriangle (t : Triangle) (parts : List TrianglePart) : Prop :=
  sorry

/-- The theorem stating that the minimum number of parts to divide a triangle for flipping reconstruction is 3 -/
theorem min_parts_for_triangle_flip (t : Triangle) :
  ∃ (parts : List TrianglePart),
    parts.length = 3 ∧
    canReconstructTriangle t parts ∧
    ∀ (smaller_parts : List TrianglePart),
      smaller_parts.length < 3 →
      ¬(canReconstructTriangle t smaller_parts) :=
by sorry

end min_parts_for_triangle_flip_l1908_190854


namespace tangent_value_l1908_190816

theorem tangent_value (α : Real) 
  (h : (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 2) : 
  Real.tan α = 1 := by
  sorry

end tangent_value_l1908_190816


namespace last_four_digits_of_5_pow_2013_l1908_190831

/-- The last four digits of a natural number -/
def lastFourDigits (n : ℕ) : ℕ := n % 10000

/-- The sequence of last four digits of powers of 5 -/
def lastFourDigitsSequence : List ℕ := [3125, 5625, 8125, 0625]

theorem last_four_digits_of_5_pow_2013 :
  lastFourDigits (5^2013) = 3125 := by
  sorry

end last_four_digits_of_5_pow_2013_l1908_190831


namespace james_coin_sale_l1908_190862

/-- Proves the number of coins James needs to sell to recoup his investment -/
theorem james_coin_sale (initial_price : ℝ) (num_coins : ℕ) (price_increase_ratio : ℝ) :
  initial_price = 15 →
  num_coins = 20 →
  price_increase_ratio = 2/3 →
  let total_investment := initial_price * num_coins
  let new_price := initial_price * (1 + price_increase_ratio)
  let coins_to_sell := total_investment / new_price
  ⌊coins_to_sell⌋ = 12 := by
  sorry

end james_coin_sale_l1908_190862


namespace range_of_F_l1908_190836

-- Define the function F
def F (x : ℝ) : ℝ := |2*x + 2| - |2*x - 2|

-- State the theorem about the range of F
theorem range_of_F :
  ∀ y : ℝ, (∃ x : ℝ, F x = y) ↔ y ∈ Set.Icc (-4) 4 :=
sorry

end range_of_F_l1908_190836


namespace sixth_root_of_107918163081_l1908_190870

theorem sixth_root_of_107918163081 :
  let n : ℕ := 107918163081
  let expansion : ℕ := 1 * 101^6 + 6 * 101^5 + 15 * 101^4 + 20 * 101^3 + 15 * 101^2 + 6 * 101 + 1
  n = expansion → (n : ℝ) ^ (1/6 : ℝ) = 102 := by sorry

end sixth_root_of_107918163081_l1908_190870


namespace fraction_of_girls_at_joint_event_l1908_190817

/-- Represents a school with a given number of students and boy-to-girl ratio -/
structure School where
  total_students : ℕ
  boy_ratio : ℕ
  girl_ratio : ℕ

/-- Calculates the number of girls in a school -/
def girls_in_school (s : School) : ℚ :=
  (s.total_students : ℚ) * s.girl_ratio / (s.boy_ratio + s.girl_ratio)

/-- Theorem stating that the fraction of girls at the joint event is 5/7 -/
theorem fraction_of_girls_at_joint_event 
  (school_a : School) 
  (school_b : School) 
  (ha : school_a = ⟨300, 3, 2⟩) 
  (hb : school_b = ⟨240, 3, 4⟩) : 
  (girls_in_school school_a + girls_in_school school_b) / 
  (school_a.total_students + school_b.total_students : ℚ) = 5/7 := by
  sorry

end fraction_of_girls_at_joint_event_l1908_190817


namespace wage_change_equation_l1908_190894

/-- Represents the number of employees in each education category -/
structure EmployeeCount where
  illiterate : ℕ := 20
  primary : ℕ
  college : ℕ

/-- Represents the daily wages before and after the change -/
structure DailyWages where
  illiterate : (ℕ × ℕ) := (25, 10)
  primary : (ℕ × ℕ) := (40, 25)
  college : (ℕ × ℕ) := (50, 60)

/-- The main theorem stating the relationship between employee counts and total employees -/
theorem wage_change_equation (N : ℕ) (emp : EmployeeCount) (wages : DailyWages) :
  N = emp.illiterate + emp.primary + emp.college →
  15 * emp.primary - 10 * emp.college = 10 * N - 300 := by
  sorry

#check wage_change_equation

end wage_change_equation_l1908_190894


namespace slope_of_midpoint_line_l1908_190866

/-- The slope of a line containing the midpoints of two segments -/
theorem slope_of_midpoint_line : 
  let midpoint1 := ((3 + 7) / 2, (5 + 12) / 2)
  let midpoint2 := ((4 + 9) / 2, (1 + 6) / 2)
  let slope := (midpoint1.2 - midpoint2.2) / (midpoint1.1 - midpoint2.1)
  slope = -10 / 3 := by sorry

end slope_of_midpoint_line_l1908_190866


namespace symmetric_function_value_l1908_190872

/-- A function is symmetric to 2^(x-a) about y=-x -/
def SymmetricAboutNegativeX (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, f x = y ↔ 2^(-y - a) = -x

theorem symmetric_function_value (f : ℝ → ℝ) (a : ℝ) 
  (h_sym : SymmetricAboutNegativeX f a) 
  (h_sum : f (-2) + f (-4) = 1) : 
  a = -2 := by
sorry

end symmetric_function_value_l1908_190872


namespace total_cost_is_183_l1908_190804

/-- Represents the jewelry store inventory and pricing --/
structure JewelryStore where
  necklaceCapacity : ℕ
  currentNecklaces : ℕ
  ringCapacity : ℕ
  currentRings : ℕ
  braceletCapacity : ℕ
  currentBracelets : ℕ
  necklacePrice : ℕ
  ringPrice : ℕ
  braceletPrice : ℕ

/-- Calculates the total cost to fill the displays --/
def totalCost (store : JewelryStore) : ℕ :=
  (store.necklaceCapacity - store.currentNecklaces) * store.necklacePrice +
  (store.ringCapacity - store.currentRings) * store.ringPrice +
  (store.braceletCapacity - store.currentBracelets) * store.braceletPrice

/-- The specific jewelry store in the problem --/
def problemStore : JewelryStore := {
  necklaceCapacity := 12
  currentNecklaces := 5
  ringCapacity := 30
  currentRings := 18
  braceletCapacity := 15
  currentBracelets := 8
  necklacePrice := 4
  ringPrice := 10
  braceletPrice := 5
}

/-- Theorem stating that the total cost to fill the displays is $183 --/
theorem total_cost_is_183 : totalCost problemStore = 183 := by
  sorry

end total_cost_is_183_l1908_190804


namespace cubic_sum_from_sixth_power_l1908_190832

theorem cubic_sum_from_sixth_power (x : ℝ) (h : 34 = x^6 + 1/x^6) :
  x^3 + 1/x^3 = 6 ∨ x^3 + 1/x^3 = -6 := by
  sorry

end cubic_sum_from_sixth_power_l1908_190832


namespace zero_not_in_range_of_g_l1908_190803

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (2 / (x + 3))
  else if x < -3 then Int.floor (2 / (x + 3))
  else 0  -- This value doesn't matter as g is undefined at x = -3

-- Theorem statement
theorem zero_not_in_range_of_g :
  ∀ x : ℝ, x ≠ -3 → g x ≠ 0 := by sorry

end zero_not_in_range_of_g_l1908_190803


namespace min_processed_area_l1908_190895

/-- Represents the dimensions of a rectangular sheet -/
structure SheetDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular sheet -/
def area (d : SheetDimensions) : ℝ := d.length * d.width

/-- Applies the processing shrinkage to the sheet dimensions -/
def applyProcessing (d : SheetDimensions) (shrinkFactor : ℝ) : SheetDimensions :=
  { length := d.length * shrinkFactor,
    width := d.width * shrinkFactor }

/-- Theorem: The minimum possible area of the processed sheet is 12.15 square inches -/
theorem min_processed_area (reportedLength reportedWidth errorMargin shrinkFactor : ℝ) 
    (hLength : reportedLength = 6)
    (hWidth : reportedWidth = 4)
    (hError : errorMargin = 1)
    (hShrink : shrinkFactor = 0.9)
    : ∃ (d : SheetDimensions),
      d.length ≥ reportedLength - errorMargin ∧
      d.length ≤ reportedLength + errorMargin ∧
      d.width ≥ reportedWidth - errorMargin ∧
      d.width ≤ reportedWidth + errorMargin ∧
      area (applyProcessing d shrinkFactor) ≥ 12.15 ∧
      ∀ (d' : SheetDimensions),
        d'.length ≥ reportedLength - errorMargin →
        d'.length ≤ reportedLength + errorMargin →
        d'.width ≥ reportedWidth - errorMargin →
        d'.width ≤ reportedWidth + errorMargin →
        area (applyProcessing d' shrinkFactor) ≥ area (applyProcessing d shrinkFactor) :=
by sorry

end min_processed_area_l1908_190895


namespace fixed_points_of_square_minus_two_range_of_a_for_two_fixed_points_odd_function_fixed_points_l1908_190859

-- Definition of a fixed point
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

-- Statement 1
theorem fixed_points_of_square_minus_two :
  ∃ (x y : ℝ), x ≠ y ∧ 
    is_fixed_point (fun x => x^2 - 2) x ∧
    is_fixed_point (fun x => x^2 - 2) y ∧
    ∀ z, is_fixed_point (fun x => x^2 - 2) z → (z = x ∨ z = y) :=
sorry

-- Statement 2
theorem range_of_a_for_two_fixed_points (a b : ℝ) :
  (∀ b : ℝ, ∃ (x y : ℝ), x ≠ y ∧
    is_fixed_point (fun x => a*x^2 + b*x - b) x ∧
    is_fixed_point (fun x => a*x^2 + b*x - b) y) →
  (0 < a ∧ a < 1) :=
sorry

-- Statement 3
theorem odd_function_fixed_points (f : ℝ → ℝ) (K : ℕ) :
  (∀ x, f (-x) = -f x) →
  (∃ (S : Finset ℝ), S.card = K ∧ ∀ x ∈ S, is_fixed_point f x) →
  Odd K :=
sorry

end fixed_points_of_square_minus_two_range_of_a_for_two_fixed_points_odd_function_fixed_points_l1908_190859


namespace find_b_value_l1908_190878

theorem find_b_value (x y b : ℝ) (h1 : y ≠ 0) (h2 : x / (2 * y) = 3 / 2) (h3 : (7 * x + b * y) / (x - 2 * y) = 27) : b = 6 := by
  sorry

end find_b_value_l1908_190878


namespace cricket_team_size_l1908_190889

theorem cricket_team_size :
  ∀ (n : ℕ) (team_avg : ℝ) (keeper_age : ℝ) (remaining_avg : ℝ),
    team_avg = 24 →
    keeper_age = team_avg + 3 →
    remaining_avg = team_avg - 1 →
    (n : ℝ) * team_avg = (n - 2 : ℝ) * remaining_avg + team_avg + keeper_age →
    n = 5 := by
  sorry

end cricket_team_size_l1908_190889


namespace geometric_sequence_problem_l1908_190809

/-- A geometric sequence is a sequence where the ratio of any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The property of geometric sequences: if m + n = p + q, then a_m * a_n = a_p * a_q -/
axiom geometric_sequence_property {a : ℕ → ℝ} (h : IsGeometricSequence a) :
  ∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q

theorem geometric_sequence_problem (a : ℕ → ℝ) (h : IsGeometricSequence a) 
  (h1 : a 5 * a 14 = 5) : a 8 * a 9 * a 10 * a 11 = 25 := by
  sorry

end geometric_sequence_problem_l1908_190809


namespace max_value_of_objective_function_l1908_190896

theorem max_value_of_objective_function (x y : ℤ) : 
  x + 2*y - 5 ≤ 0 → 
  x - y - 2 ≤ 0 → 
  x ≥ 0 → 
  2*x + 3*y + 1 ≤ 10 :=
by sorry

end max_value_of_objective_function_l1908_190896


namespace octagon_area_l1908_190801

theorem octagon_area (r : ℝ) (h : r = 4) : 
  let octagon_area := 8 * (1/2 * r * r * Real.sin (π/4))
  octagon_area = 32 * Real.sqrt 2 := by
sorry

end octagon_area_l1908_190801


namespace total_cost_is_100_l1908_190893

-- Define the number of shirts
def num_shirts : ℕ := 10

-- Define the number of pants as half the number of shirts
def num_pants : ℕ := num_shirts / 2

-- Define the cost of each shirt
def cost_per_shirt : ℕ := 6

-- Define the cost of each pair of pants
def cost_per_pants : ℕ := 8

-- Theorem to prove the total cost
theorem total_cost_is_100 :
  num_shirts * cost_per_shirt + num_pants * cost_per_pants = 100 := by
  sorry

end total_cost_is_100_l1908_190893


namespace xy_not_6_sufficient_not_necessary_l1908_190800

theorem xy_not_6_sufficient_not_necessary :
  (∃ x y : ℝ, (x ≠ 2 ∨ y ≠ 3) ∧ x * y = 6) ∧
  (∀ x y : ℝ, x * y ≠ 6 → (x ≠ 2 ∨ y ≠ 3)) :=
by sorry

end xy_not_6_sufficient_not_necessary_l1908_190800


namespace problem_solution_l1908_190821

theorem problem_solution (x y : ℝ) 
  (eq1 : x + 2 * y = 1) 
  (eq2 : 2 * x - 3 * y = 2) : 
  (2 * x + 4 * y - 2) / 2 + (6 * x - 9 * y) / 3 = 2 := by
sorry

end problem_solution_l1908_190821


namespace thursday_miles_proof_l1908_190867

/-- Calculates the number of miles driven on Thursday given the rental conditions and total cost --/
def miles_driven_thursday (fixed_cost per_mile_cost monday_miles total_cost : ℚ) : ℚ :=
  (total_cost - fixed_cost - (per_mile_cost * monday_miles)) / per_mile_cost

/-- Theorem stating that given the specific rental conditions, the miles driven on Thursday is 744 --/
theorem thursday_miles_proof :
  miles_driven_thursday 150 0.5 620 832 = 744 := by
  sorry

end thursday_miles_proof_l1908_190867


namespace cubic_inequality_l1908_190899

theorem cubic_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * (a^3 + b^3 + c^3) ≥ a*b^2 + a^2*b + b*c^2 + b^2*c + a*c^2 + a^2*c := by
  sorry

end cubic_inequality_l1908_190899


namespace winning_configurations_l1908_190819

/-- Calculates the nim-value of a wall with given number of bricks -/
def nimValue (bricks : Nat) : Nat :=
  match bricks with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 0
  | 5 => 2
  | 6 => 1
  | 7 => 3
  | 8 => 1
  | 9 => 2
  | _ => sorry  -- For simplicity, we don't define beyond 9

/-- Calculates the nim-sum (XOR) of a list of natural numbers -/
def nimSum (list : List Nat) : Nat :=
  list.foldl Nat.xor 0

/-- Represents a configuration of walls -/
structure WallConfiguration where
  walls : List Nat

/-- Checks if a configuration is a winning position for the second player -/
def isWinningForSecondPlayer (config : WallConfiguration) : Prop :=
  nimSum (config.walls.map nimValue) = 0

/-- Theorem: The given configurations are winning positions for the second player -/
theorem winning_configurations :
  (isWinningForSecondPlayer ⟨[8, 2, 3]⟩) ∧
  (isWinningForSecondPlayer ⟨[9, 3, 3]⟩) ∧
  (isWinningForSecondPlayer ⟨[9, 5, 2]⟩) :=
by sorry

end winning_configurations_l1908_190819


namespace schlaf_flachs_divisibility_l1908_190881

def SCHLAF (S C H L A F : ℕ) : ℕ := S * 10^5 + C * 10^4 + H * 10^3 + L * 10^2 + A * 10 + F

def FLACHS (F L A C H S : ℕ) : ℕ := F * 10^5 + L * 10^4 + A * 10^3 + C * 10^2 + H * 10 + S

theorem schlaf_flachs_divisibility 
  (S C H L A F : ℕ) 
  (hS : S ∈ Finset.range 10) 
  (hC : C ∈ Finset.range 10) 
  (hH : H ∈ Finset.range 10) 
  (hL : L ∈ Finset.range 10) 
  (hA : A ∈ Finset.range 10) 
  (hF : F ∈ Finset.range 10) 
  (hSnonzero : S ≠ 0) 
  (hFnonzero : F ≠ 0) : 
  (271 ∣ (SCHLAF S C H L A F - FLACHS F L A C H S)) ↔ (C = L ∧ H = A) :=
sorry

end schlaf_flachs_divisibility_l1908_190881


namespace average_headcount_l1908_190808

def fall_02_03 : ℕ := 11700
def fall_03_04 : ℕ := 11500
def fall_04_05 : ℕ := 11600

theorem average_headcount : 
  (fall_02_03 + fall_03_04 + fall_04_05) / 3 = 11600 := by
  sorry

end average_headcount_l1908_190808


namespace nonagon_intersection_points_l1908_190877

/-- A regular nonagon is a 9-sided polygon -/
def regular_nonagon : ℕ := 9

/-- The number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of distinct intersection points of diagonals within a regular nonagon -/
def intersection_points (n : ℕ) : ℕ := choose n 4

theorem nonagon_intersection_points :
  intersection_points regular_nonagon = 126 := by
  sorry

end nonagon_intersection_points_l1908_190877


namespace fraction_equality_l1908_190828

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) (h1 : a / b = 1 / 3) :
  (2 * a + b) / (a - b) = -5 / 2 := by
  sorry

end fraction_equality_l1908_190828


namespace bicycle_sale_profit_percentage_l1908_190827

/-- Calculates the profit percentage of seller A given the conditions of the bicycle sale problem -/
theorem bicycle_sale_profit_percentage 
  (cp_a : ℝ)     -- Cost price for A
  (sp_c : ℝ)     -- Selling price for C
  (profit_b : ℝ) -- Profit percentage for B
  (h1 : cp_a = 120)
  (h2 : sp_c = 225)
  (h3 : profit_b = 25) :
  (((sp_c / (1 + profit_b / 100) - cp_a) / cp_a) * 100 = 50) := by
  sorry

#check bicycle_sale_profit_percentage

end bicycle_sale_profit_percentage_l1908_190827


namespace regular_polygon_144_degree_interior_angle_l1908_190879

/-- A regular polygon with an interior angle of 144° has 10 sides -/
theorem regular_polygon_144_degree_interior_angle :
  ∀ n : ℕ,
  n > 2 →
  (144 : ℝ) = (n - 2 : ℝ) * 180 / n →
  n = 10 :=
by sorry

end regular_polygon_144_degree_interior_angle_l1908_190879


namespace correct_factorization_l1908_190833

theorem correct_factorization (x : ℝ) : x^2 - x + (1/4) = (x - 1/2)^2 := by
  sorry

end correct_factorization_l1908_190833


namespace fencing_calculation_l1908_190873

/-- Calculates the required fencing for a rectangular field -/
theorem fencing_calculation (area : ℝ) (uncovered_side : ℝ) (h1 : area = 680) (h2 : uncovered_side = 20) :
  let width := area / uncovered_side
  2 * width + uncovered_side = 88 :=
sorry

end fencing_calculation_l1908_190873


namespace perimeter_bounds_l1908_190849

/-- A unit square with 100 segments drawn from its center to its sides, 
    dividing it into 100 parts of equal perimeter -/
structure SegmentedSquare where
  /-- The perimeter of each part -/
  p : ℝ
  /-- The square is a unit square -/
  is_unit_square : True
  /-- There are 100 segments -/
  segment_count : Nat
  segment_count_eq : segment_count = 100
  /-- The square is divided into 100 parts -/
  part_count : Nat
  part_count_eq : part_count = 100
  /-- All parts have equal perimeter -/
  equal_perimeter : True

/-- The perimeter of each part in a segmented unit square satisfies 14/10 < p < 15/10 -/
theorem perimeter_bounds (s : SegmentedSquare) : 14/10 < s.p ∧ s.p < 15/10 := by
  sorry

end perimeter_bounds_l1908_190849


namespace not_right_triangle_only_234_l1908_190863

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem not_right_triangle_only_234 :
  (is_right_triangle 1 1 (Real.sqrt 2)) ∧
  ¬(is_right_triangle 2 3 4) ∧
  (is_right_triangle 1 (Real.sqrt 3) 2) ∧
  (is_right_triangle 3 4 (Real.sqrt 7)) := by
  sorry

end not_right_triangle_only_234_l1908_190863


namespace min_value_on_line_l1908_190882

/-- The minimum value of 9^x + 3^y where (x, y) is on the line y = 4 - 2x -/
theorem min_value_on_line : ∃ (min : ℝ),
  (∀ (x y : ℝ), y = 4 - 2*x → 9^x + 3^y ≥ min) ∧
  (∃ (x y : ℝ), y = 4 - 2*x ∧ 9^x + 3^y = min) ∧
  min = 18 := by
  sorry

end min_value_on_line_l1908_190882


namespace min_value_theorem_l1908_190858

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 4) :
  (1 / x + 4 / y + 9 / z) ≥ 9 ∧ ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ 
  x' + y' + z' = 4 ∧ 1 / x' + 4 / y' + 9 / z' = 9 :=
sorry

end min_value_theorem_l1908_190858


namespace paise_to_rupees_l1908_190802

/-- Proves that if 0.5% of a equals 80 paise, then a equals 160 rupees. -/
theorem paise_to_rupees (a : ℝ) : (0.005 * a = 0.8) → a = 160 := by
  sorry

end paise_to_rupees_l1908_190802


namespace arrangement_count_l1908_190851

/-- The number of ways to divide teachers and students into groups -/
def divide_groups (num_teachers num_students num_groups : ℕ) : ℕ :=
  if num_teachers = 2 ∧ num_students = 4 ∧ num_groups = 2 then
    12
  else
    0

/-- Theorem stating that the number of arrangements is 12 -/
theorem arrangement_count :
  divide_groups 2 4 2 = 12 := by
  sorry

end arrangement_count_l1908_190851


namespace combined_average_score_l1908_190861

/-- Combined average score of two groups given their individual averages and size ratio -/
theorem combined_average_score 
  (morning_avg : ℝ) 
  (evening_avg : ℝ) 
  (morning_students : ℝ) 
  (evening_students : ℝ) 
  (h1 : morning_avg = 82)
  (h2 : evening_avg = 68)
  (h3 : morning_students / evening_students = 5 / 7) :
  (morning_avg * morning_students + evening_avg * evening_students) / (morning_students + evening_students) = 72 :=
by sorry

end combined_average_score_l1908_190861


namespace smallest_positive_integer_with_remainders_l1908_190845

theorem smallest_positive_integer_with_remainders : ∃! x : ℕ, 
  x > 0 ∧ 
  x % 4 = 1 ∧ 
  x % 5 = 2 ∧ 
  x % 6 = 3 ∧
  ∀ y : ℕ, y > 0 ∧ y % 4 = 1 ∧ y % 5 = 2 ∧ y % 6 = 3 → x ≤ y :=
by
  use 117
  sorry

end smallest_positive_integer_with_remainders_l1908_190845


namespace circle_center_l1908_190883

/-- The equation of a circle in the form (x - h)² + (y - k)² = r², 
    where (h, k) is the center and r is the radius. -/
def CircleEquation (h k r : ℝ) : ℝ → ℝ → Prop :=
  fun x y => (x - h)^2 + (y - k)^2 = r^2

/-- The given equation of the circle -/
def GivenCircle : ℝ → ℝ → Prop :=
  fun x y => x^2 + y^2 - 4*x + 6*y = 0

theorem circle_center : 
  ∃ r : ℝ, ∀ x y : ℝ, GivenCircle x y ↔ CircleEquation 2 (-3) r x y :=
sorry

end circle_center_l1908_190883


namespace least_addition_for_divisibility_l1908_190835

theorem least_addition_for_divisibility : 
  ∃ x : ℕ, x = 25 ∧ 
  (∀ y : ℕ, (27306 + y) % 151 = 0 → y ≥ x) ∧ 
  (27306 + x) % 151 = 0 := by
  sorry

end least_addition_for_divisibility_l1908_190835


namespace sheela_income_proof_l1908_190884

/-- Sheela's monthly income in Rupees -/
def monthly_income : ℝ := 22666.67

/-- The amount Sheela deposited in the bank in Rupees -/
def deposit : ℝ := 3400

/-- The percentage of monthly income that was deposited -/
def deposit_percentage : ℝ := 0.15

theorem sheela_income_proof :
  deposit = deposit_percentage * monthly_income :=
by sorry

end sheela_income_proof_l1908_190884


namespace equation_solution_l1908_190875

theorem equation_solution : ∃ (x₁ x₂ : ℝ), 
  x₁ ≠ 2/3 ∧ x₂ ≠ 2/3 ∧
  (3 * x₁ + 2) / (3 * x₁^2 + 7 * x₁ - 6) = (3 * x₁) / (3 * x₁ - 2) ∧
  (3 * x₂ + 2) / (3 * x₂^2 + 7 * x₂ - 6) = (3 * x₂) / (3 * x₂ - 2) ∧
  x₁ = -2 ∧ x₂ = 1/3 := by
sorry

end equation_solution_l1908_190875


namespace count_valid_house_numbers_l1908_190885

/-- A two-digit prime number less than 50 -/
def TwoDigitPrime : Type := { n : ℕ // n ≥ 10 ∧ n < 50 ∧ Nat.Prime n }

/-- The set of all two-digit primes less than 50 -/
def TwoDigitPrimes : Finset TwoDigitPrime := sorry

/-- A four-digit house number ABCD where AB and CD are distinct two-digit primes less than 50 -/
structure HouseNumber where
  ab : TwoDigitPrime
  cd : TwoDigitPrime
  distinct : ab ≠ cd

/-- The set of all valid house numbers -/
def ValidHouseNumbers : Finset HouseNumber := sorry

theorem count_valid_house_numbers : Finset.card ValidHouseNumbers = 110 := by sorry

end count_valid_house_numbers_l1908_190885


namespace pairing_natural_numbers_to_perfect_squares_l1908_190825

theorem pairing_natural_numbers_to_perfect_squares :
  ∃ f : ℕ → (ℕ × ℕ), 
    (∀ n : ℕ, ∃ m : ℕ, (f n).1 + (f n).2 = m^2) ∧ 
    (∀ x : ℕ, ∃! n : ℕ, x = (f n).1 ∨ x = (f n).2) := by
  sorry

end pairing_natural_numbers_to_perfect_squares_l1908_190825


namespace infinite_points_in_region_l1908_190898

/-- The set of points with positive rational coordinates satisfying x + 2y ≤ 6 is infinite -/
theorem infinite_points_in_region : 
  Set.Infinite {p : ℚ × ℚ | 0 < p.1 ∧ 0 < p.2 ∧ p.1 + 2 * p.2 ≤ 6} := by sorry

end infinite_points_in_region_l1908_190898


namespace incorrect_expression_l1908_190853

theorem incorrect_expression (x y : ℚ) (h : x / y = 5 / 6) :
  ¬((3 * x + 3 * y) / x = 18 / 5) := by
sorry

end incorrect_expression_l1908_190853


namespace slower_rider_speed_l1908_190810

/-- The speed of the slower rider in miles per hour -/
def slower_speed : ℚ := 5/3

/-- The speed of the faster rider in miles per hour -/
def faster_speed : ℚ := 2 * slower_speed

/-- The distance between the cyclists in miles -/
def distance : ℚ := 20

/-- The time it takes for the cyclists to meet when riding towards each other in hours -/
def time_towards : ℚ := 4

/-- The time it takes for the faster rider to catch up when riding in the same direction in hours -/
def time_same_direction : ℚ := 10

theorem slower_rider_speed :
  (distance = (faster_speed + slower_speed) * time_towards) ∧
  (distance = (faster_speed - slower_speed) * time_same_direction) ∧
  (faster_speed = 2 * slower_speed) →
  slower_speed = 5/3 := by sorry

end slower_rider_speed_l1908_190810


namespace dara_employment_age_l1908_190807

/-- Proves that Dara will reach the minimum employment age in 14 years -/
theorem dara_employment_age (min_age : ℕ) (jane_age : ℕ) (years_until_half : ℕ) : 
  min_age = 25 →
  jane_age = 28 →
  years_until_half = 6 →
  min_age - (jane_age + years_until_half) / 2 + years_until_half = 14 :=
by sorry

end dara_employment_age_l1908_190807


namespace horizontal_line_slope_intercept_product_l1908_190826

/-- Given two distinct points on a horizontal line with y-coordinate 20,
    the product of the slope and y-intercept of the line is 0. -/
theorem horizontal_line_slope_intercept_product (C D : ℝ × ℝ) : 
  C.1 ≠ D.1 → C.2 = 20 → D.2 = 20 → 
  let m := (D.2 - C.2) / (D.1 - C.1)
  let b := C.2
  m * b = 0 := by sorry

end horizontal_line_slope_intercept_product_l1908_190826


namespace parking_rate_proof_l1908_190829

/-- Proves that the monthly parking rate is $35 given the conditions --/
theorem parking_rate_proof (weekly_rate : ℕ) (weeks_per_year : ℕ) (months_per_year : ℕ) 
  (yearly_savings : ℕ) (monthly_rate : ℕ) : 
  weekly_rate = 10 →
  weeks_per_year = 52 →
  months_per_year = 12 →
  yearly_savings = 100 →
  (weeks_per_year * weekly_rate) - (months_per_year * monthly_rate) = yearly_savings →
  monthly_rate = 35 := by
sorry

end parking_rate_proof_l1908_190829


namespace ticket_sales_ratio_l1908_190865

/-- Proves that the ratio of full price tickets to reduced price tickets
    sold during the remaining weeks is 1:1 -/
theorem ticket_sales_ratio
  (total_tickets : ℕ)
  (first_week_reduced : ℕ)
  (total_full_price : ℕ)
  (h1 : total_tickets = 25200)
  (h2 : first_week_reduced = 5400)
  (h3 : total_full_price = 16500)
  (h4 : total_full_price = total_tickets - first_week_reduced - total_full_price) :
  total_full_price = total_tickets - first_week_reduced - total_full_price :=
by sorry

end ticket_sales_ratio_l1908_190865


namespace inequality_comparison_l1908_190806

theorem inequality_comparison :
  (-14 ≤ 0) ∧
  (-2.1 ≤ -2.01) ∧
  (1/2 ≥ -1/3) ∧
  (-0.6 > -4/5) :=
by sorry

end inequality_comparison_l1908_190806


namespace midpoint_trajectory_l1908_190841

/-- The trajectory of the midpoint between a fixed point and a point on a parabola -/
theorem midpoint_trajectory (B : ℝ × ℝ) :
  (B.2^2 = 2 * B.1) →  -- B is on the parabola y^2 = 2x
  let A : ℝ × ℝ := (2, 4)  -- Fixed point A(2, 4)
  let P : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)  -- P is the midpoint of AB
  (P.2 - 2)^2 = P.1 - 1 :=
by sorry

end midpoint_trajectory_l1908_190841


namespace parabola_vertex_l1908_190811

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 2

-- State the theorem
theorem parabola_vertex :
  ∃ (x y : ℝ), (x = 0 ∧ y = -2) ∧
  (∀ (x' : ℝ), parabola x' ≥ parabola x) :=
sorry

end parabola_vertex_l1908_190811


namespace not_perfect_square_l1908_190830

theorem not_perfect_square (x y : ℤ) : ∃ (z : ℤ), (x^2 + 3*x + 1)^2 + (y^2 + 3*y + 1)^2 ≠ z^2 := by
  sorry

end not_perfect_square_l1908_190830


namespace parabola_intersection_distance_l1908_190824

/-- The parabola y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- A line passing through the focus -/
def line_through_focus (m : ℝ) (x y : ℝ) : Prop :=
  y = m * (x - focus.1) + focus.2

/-- Theorem: For the parabola y^2 = 4x, if a line passing through its focus
    intersects the parabola at points A(x₁, y₁) and B(x₂, y₂), and x₁ + x₂ = 6,
    then the distance between A and B is 8. -/
theorem parabola_intersection_distance
  (x₁ y₁ x₂ y₂ m : ℝ)
  (h₁ : parabola x₁ y₁)
  (h₂ : parabola x₂ y₂)
  (h₃ : line_through_focus m x₁ y₁)
  (h₄ : line_through_focus m x₂ y₂)
  (h₅ : x₁ + x₂ = 6) :
  ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 64 :=
sorry

end parabola_intersection_distance_l1908_190824


namespace unique_solution_l1908_190839

/-- The set A of solutions to the quadratic equation (a^2 - 1)x^2 + (a + 1)x + 1 = 0 -/
def A (a : ℝ) : Set ℝ :=
  {x | (a^2 - 1) * x^2 + (a + 1) * x + 1 = 0}

/-- The set A contains exactly one element if and only if a = 1 or a = 5/3 -/
theorem unique_solution (a : ℝ) : (∃! x, x ∈ A a) ↔ (a = 1 ∨ a = 5/3) := by
  sorry

end unique_solution_l1908_190839


namespace round_0_9247_to_hundredth_l1908_190855

def round_to_hundredth (x : ℚ) : ℚ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem round_0_9247_to_hundredth :
  round_to_hundredth 0.9247 = 0.92 := by
  sorry

end round_0_9247_to_hundredth_l1908_190855


namespace square_side_lengths_l1908_190805

theorem square_side_lengths (s t : ℝ) : 
  (4 * s = 5 * (4 * t)) → -- perimeter of one square is 5 times the other
  (s + t = 60) →          -- sum of side lengths is 60
  ((s = 50 ∧ t = 10) ∨ (s = 10 ∧ t = 50)) := by
sorry

end square_side_lengths_l1908_190805


namespace area_between_concentric_circles_l1908_190860

theorem area_between_concentric_circles : 
  let r₁ : ℝ := 12
  let r₂ : ℝ := 7
  (π * r₁^2 - π * r₂^2) = 95 * π :=
by sorry

end area_between_concentric_circles_l1908_190860


namespace range_of_m_l1908_190890

theorem range_of_m (m : ℝ) : 
  (¬(∃ x : ℝ, m * x^2 + 1 ≤ 0) ∨ ¬(∀ x : ℝ, x^2 + m * x + 1 > 0)) → m ≥ 2 :=
by sorry

end range_of_m_l1908_190890


namespace bicycle_course_remaining_distance_l1908_190818

/-- Calculates the remaining distance of a bicycle course after Yoongi's travel -/
def remaining_distance (total_length : ℝ) (first_distance : ℝ) (second_distance : ℝ) : ℝ :=
  total_length * 1000 - (first_distance * 1000 + second_distance)

/-- Theorem: Given a bicycle course of 10.5 km, if Yoongi travels 1.5 km and then 3730 m, 
    the remaining distance is 5270 m -/
theorem bicycle_course_remaining_distance :
  remaining_distance 10.5 1.5 3730 = 5270 := by
  sorry

end bicycle_course_remaining_distance_l1908_190818


namespace james_score_problem_l1908_190892

theorem james_score_problem (field_goals : ℕ) (shots : ℕ) (total_points : ℕ) 
  (h1 : field_goals = 13)
  (h2 : shots = 20)
  (h3 : total_points = 79) :
  ∃ (points_per_shot : ℕ), 
    field_goals * 3 + shots * points_per_shot = total_points ∧ 
    points_per_shot = 2 := by
  sorry

end james_score_problem_l1908_190892


namespace max_value_problem_l1908_190848

theorem max_value_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x^2 * y) / (x + y) + (y^2 * z) / (y + z) + (z^2 * x) / (z + x) ≤ 3/2 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 ∧
    (a^2 * b) / (a + b) + (b^2 * c) / (b + c) + (c^2 * a) / (c + a) = 3/2 :=
by sorry

end max_value_problem_l1908_190848


namespace no_real_intersection_l1908_190840

theorem no_real_intersection : ¬∃ (x y : ℝ), y = 8 / (x^3 + 4*x + 3) ∧ x + y = 5 := by
  sorry

end no_real_intersection_l1908_190840
