import Mathlib

namespace complex_equation_solution_l2172_217285

theorem complex_equation_solution (z : ℂ) (h : z * (2 - Complex.I) = 5 * Complex.I) :
  z = -1 + 2 * Complex.I := by
  sorry

end complex_equation_solution_l2172_217285


namespace greatest_divisor_four_consecutive_integers_l2172_217269

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 → 
  (∃ m : ℕ, m > 12 ∧ ∀ k : ℕ, k > 0 → m ∣ (k * (k + 1) * (k + 2) * (k + 3))) → False :=
by sorry

end greatest_divisor_four_consecutive_integers_l2172_217269


namespace lg_sum_equals_three_l2172_217258

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_sum_equals_three : lg 8 + 3 * lg 5 = 3 := by
  sorry

end lg_sum_equals_three_l2172_217258


namespace sum_three_digit_integers_eq_385550_l2172_217292

/-- The sum of all three-digit positive integers from 200 to 900 -/
def sum_three_digit_integers : ℕ :=
  let first_term := 200
  let last_term := 900
  let common_difference := 1
  let num_terms := (last_term - first_term) / common_difference + 1
  (num_terms * (first_term + last_term)) / 2

theorem sum_three_digit_integers_eq_385550 : 
  sum_three_digit_integers = 385550 := by
  sorry

#eval sum_three_digit_integers

end sum_three_digit_integers_eq_385550_l2172_217292


namespace cube_space_division_l2172_217221

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the specifics of a cube for this problem

/-- A plane is a flat, two-dimensional surface -/
structure Plane where
  -- We don't need to define the specifics of a plane for this problem

/-- The number of parts that a cube and its face planes divide space into -/
def space_division (c : Cube) : Nat :=
  sorry

/-- Theorem stating that a cube and its face planes divide space into 27 parts -/
theorem cube_space_division (c : Cube) : space_division c = 27 := by
  sorry

end cube_space_division_l2172_217221


namespace quarters_remaining_l2172_217238

-- Define the initial number of quarters
def initial_quarters : ℕ := 375

-- Define the cost of the dress in cents
def dress_cost_cents : ℕ := 4263

-- Define the value of a quarter in cents
def quarter_value_cents : ℕ := 25

-- Theorem to prove
theorem quarters_remaining :
  initial_quarters - (dress_cost_cents / quarter_value_cents) = 205 := by
  sorry

end quarters_remaining_l2172_217238


namespace hexagon_sixth_angle_measure_l2172_217298

/-- The measure of the sixth angle in a hexagon, given the other five angles -/
theorem hexagon_sixth_angle_measure (a b c d e : ℝ) 
  (ha : a = 130)
  (hb : b = 95)
  (hc : c = 122)
  (hd : d = 108)
  (he : e = 114) :
  720 - (a + b + c + d + e) = 151 := by
  sorry

end hexagon_sixth_angle_measure_l2172_217298


namespace quaternary_1320_to_binary_l2172_217268

/-- Converts a quaternary (base 4) number to decimal (base 10) --/
def quaternary_to_decimal (q : List Nat) : Nat :=
  q.enum.foldl (λ sum (i, digit) => sum + digit * (4 ^ i)) 0

/-- Converts a decimal (base 10) number to binary (base 2) --/
def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec to_binary_aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else to_binary_aux (m / 2) ((m % 2) :: acc)
  to_binary_aux n []

/-- The main theorem stating that 1320₄ in binary is 1111000₂ --/
theorem quaternary_1320_to_binary :
  decimal_to_binary (quaternary_to_decimal [0, 2, 3, 1]) = [1, 1, 1, 1, 0, 0, 0] := by
  sorry


end quaternary_1320_to_binary_l2172_217268


namespace max_students_l2172_217271

/-- Represents the relationship between students -/
def knows (n : ℕ) := Fin n → Fin n → Prop

/-- At least two out of any three students know each other -/
def three_two_know (n : ℕ) (k : knows n) : Prop :=
  ∀ a b c : Fin n, a ≠ b ∧ b ≠ c ∧ a ≠ c →
    k a b ∨ k b c ∨ k a c

/-- At least two out of any four students do not know each other -/
def four_two_dont_know (n : ℕ) (k : knows n) : Prop :=
  ∀ a b c d : Fin n, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d →
    ¬(k a b) ∨ ¬(k a c) ∨ ¬(k a d) ∨ ¬(k b c) ∨ ¬(k b d) ∨ ¬(k c d)

/-- The maximum number of students satisfying the conditions is 8 -/
theorem max_students : 
  (∃ (k : knows 8), three_two_know 8 k ∧ four_two_dont_know 8 k) ∧
  (∀ n > 8, ¬∃ (k : knows n), three_two_know n k ∧ four_two_dont_know n k) :=
sorry

end max_students_l2172_217271


namespace chef_leftover_potatoes_l2172_217212

theorem chef_leftover_potatoes 
  (fries_per_potato : ℕ) 
  (total_potatoes : ℕ) 
  (required_fries : ℕ) 
  (h1 : fries_per_potato = 25)
  (h2 : total_potatoes = 15)
  (h3 : required_fries = 200) :
  total_potatoes - (required_fries / fries_per_potato) = 7 :=
by
  sorry

end chef_leftover_potatoes_l2172_217212


namespace min_value_zero_l2172_217240

/-- The quadratic form representing the expression -/
def Q (x y : ℝ) : ℝ := 5 * x^2 - 8 * x * y + 7 * y^2 - 6 * x - 6 * y + 9

/-- The theorem stating that the minimum value of Q is 0 -/
theorem min_value_zero : 
  ∀ x y : ℝ, Q x y ≥ 0 ∧ ∃ x₀ y₀ : ℝ, Q x₀ y₀ = 0 := by sorry

end min_value_zero_l2172_217240


namespace divisibility_criterion_37_l2172_217286

/-- Represents a function that divides a positive integer into three-digit segments from right to left -/
def segmentNumber (n : ℕ+) : List ℕ :=
  sorry

/-- Theorem: A positive integer is divisible by 37 if and only if the sum of its three-digit segments is divisible by 37 -/
theorem divisibility_criterion_37 (n : ℕ+) :
  37 ∣ n ↔ 37 ∣ (segmentNumber n).sum :=
by sorry

end divisibility_criterion_37_l2172_217286


namespace line_segments_in_proportion_l2172_217211

theorem line_segments_in_proportion (a b c d : ℝ) : 
  a = 5 ∧ b = 15 ∧ c = 3 ∧ d = 9 → a * d = b * c := by
  sorry

end line_segments_in_proportion_l2172_217211


namespace log_inequality_l2172_217294

theorem log_inequality (a b c : ℝ) (h1 : 0 < c) (h2 : c < 1) (h3 : b > 1) (h4 : a > b) :
  Real.log c / Real.log a > Real.log c / Real.log b :=
sorry

end log_inequality_l2172_217294


namespace bertha_family_childless_l2172_217260

/-- Represents the family structure of Bertha and her descendants -/
structure BerthaFamily where
  daughters : ℕ
  granddaughters : ℕ
  daughters_with_children : ℕ

/-- The properties of Bertha's family -/
def bertha_family_properties (f : BerthaFamily) : Prop :=
  f.daughters = 6 ∧
  f.granddaughters = 6 * f.daughters_with_children ∧
  f.daughters + f.granddaughters = 30

/-- The theorem stating the number of Bertha's daughters and granddaughters without children -/
theorem bertha_family_childless (f : BerthaFamily) 
  (h : bertha_family_properties f) : 
  f.daughters + f.granddaughters - f.daughters_with_children = 26 := by
  sorry


end bertha_family_childless_l2172_217260


namespace wolf_and_nobel_count_l2172_217256

/-- Represents the number of scientists in various categories at a workshop --/
structure WorkshopAttendees where
  total : ℕ
  wolf : ℕ
  nobel : ℕ
  wolf_and_nobel : ℕ

/-- The conditions of the workshop --/
def workshop : WorkshopAttendees where
  total := 50
  wolf := 31
  nobel := 25
  wolf_and_nobel := 0  -- This is what we need to prove

/-- Theorem stating the number of scientists who were both Wolf and Nobel laureates --/
theorem wolf_and_nobel_count (w : WorkshopAttendees) 
  (h1 : w.total = 50)
  (h2 : w.wolf = 31)
  (h3 : w.nobel = 25)
  (h4 : w.nobel - w.wolf = 3 + (w.total - w.nobel - (w.wolf - w.wolf_and_nobel))) :
  w.wolf_and_nobel = 3 := by
  sorry

end wolf_and_nobel_count_l2172_217256


namespace non_union_women_percent_is_75_percent_l2172_217266

/-- Represents the composition of employees in a company -/
structure CompanyComposition where
  total : ℕ
  men_percent : ℚ
  union_percent : ℚ
  union_men_percent : ℚ

/-- Calculates the percentage of women among non-union employees -/
def non_union_women_percent (c : CompanyComposition) : ℚ :=
  let total_men := c.men_percent * c.total
  let total_women := c.total - total_men
  let union_employees := c.union_percent * c.total
  let union_men := c.union_men_percent * union_employees
  let non_union_men := total_men - union_men
  let non_union_total := c.total - union_employees
  let non_union_women := non_union_total - non_union_men
  non_union_women / non_union_total

/-- Theorem stating that given the company composition, 
    the percentage of women among non-union employees is 75% -/
theorem non_union_women_percent_is_75_percent 
  (c : CompanyComposition) 
  (h1 : c.men_percent = 52/100)
  (h2 : c.union_percent = 60/100)
  (h3 : c.union_men_percent = 70/100) :
  non_union_women_percent c = 75/100 := by
  sorry

end non_union_women_percent_is_75_percent_l2172_217266


namespace inequality_holds_l2172_217219

theorem inequality_holds (x y : ℝ) (h : 2 * y + 5 * x = 10) : 3 * x * y - x^2 - y^2 < 7 := by
  sorry

end inequality_holds_l2172_217219


namespace supermarket_spending_l2172_217254

theorem supermarket_spending (total : ℚ) :
  (1/2 : ℚ) * total +
  (1/3 : ℚ) * total +
  (1/10 : ℚ) * total +
  8 = total →
  total = 120 := by
sorry

end supermarket_spending_l2172_217254


namespace inequality_multiplication_l2172_217237

theorem inequality_multiplication (x y : ℝ) : y > x → 2 * y > 2 * x := by
  sorry

end inequality_multiplication_l2172_217237


namespace average_of_three_numbers_l2172_217287

theorem average_of_three_numbers : 
  let x : ℤ := -63
  let numbers : List ℤ := [2, 76, x]
  (numbers.sum : ℚ) / numbers.length = 5 := by sorry

end average_of_three_numbers_l2172_217287


namespace odd_function_inequality_l2172_217290

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

theorem odd_function_inequality (f : ℝ → ℝ) (m : ℝ) :
  is_odd f →
  (∀ x, x ∈ Set.Icc (-2) 2 → f x ≠ 0) →
  is_decreasing_on f (-2) 0 →
  f (1 - m) + f (1 - m^2) < 0 →
  -1 ≤ m ∧ m < 1 := by
sorry

end odd_function_inequality_l2172_217290


namespace parabola_focus_and_tangent_point_l2172_217262

noncomputable section

/-- Parabola C with parameter p -/
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y

/-- Line passing through the focus -/
def focus_line (x y : ℝ) : Prop := x + 2*y - 2 = 0

/-- Directrix of the parabola -/
def directrix (p : ℝ) (y : ℝ) : Prop := y = -p/2

/-- Tangent line to the parabola from point (m, -p/2) -/
def tangent_line (p m k x y : ℝ) : Prop := y = -p/2 + k*(x - m)

/-- Area of triangle AMN -/
def triangle_area (m : ℝ) : ℝ := (1/2) * Real.sqrt (m^2 + 4)

theorem parabola_focus_and_tangent_point (p : ℝ) :
  p > 0 →
  (∃ x y : ℝ, parabola p x y ∧ focus_line x y) →
  (∃ m : ℝ, directrix p (-p/2) ∧ triangle_area m = Real.sqrt 5 / 2) →
  (∃ m : ℝ, m = 1 ∨ m = -1) :=
sorry

end parabola_focus_and_tangent_point_l2172_217262


namespace infinitely_many_n_divisible_by_sqrt3_d_l2172_217227

def d (n : ℕ+) : ℕ := (Nat.divisors n.val).card

theorem infinitely_many_n_divisible_by_sqrt3_d :
  Set.Infinite {n : ℕ+ | ∃ k : ℕ+, n = k * ⌊Real.sqrt 3 * d n⌋} := by
  sorry

end infinitely_many_n_divisible_by_sqrt3_d_l2172_217227


namespace contrapositive_example_l2172_217276

theorem contrapositive_example :
  (∀ x : ℝ, x > 1 → x^2 + x > 2) ↔ (∀ x : ℝ, x^2 + x ≤ 2 → x ≤ 1) :=
by sorry

end contrapositive_example_l2172_217276


namespace recipe_reduction_recipe_reduction_mixed_numbers_l2172_217291

-- Define the original recipe quantities
def flour_original : Rat := 31/4  -- 7 3/4 cups
def sugar_original : Rat := 5/2   -- 2 1/2 cups

-- Define the reduced recipe quantities
def flour_reduced : Rat := 31/12  -- 2 7/12 cups
def sugar_reduced : Rat := 5/6    -- 5/6 cups

-- Theorem to prove the correct reduced quantities
theorem recipe_reduction :
  flour_reduced = (1/3) * flour_original ∧
  sugar_reduced = (1/3) * sugar_original :=
by sorry

-- Helper function to convert rational to mixed number string representation
noncomputable def rat_to_mixed_string (r : Rat) : String :=
  let whole := Int.floor r
  let frac := r - whole
  if frac = 0 then
    s!"{whole}"
  else
    let num := (frac.num : Int)
    let den := (frac.den : Int)
    if whole = 0 then
      s!"{num}/{den}"
    else
      s!"{whole} {num}/{den}"

-- Theorem to prove the correct string representations
theorem recipe_reduction_mixed_numbers :
  rat_to_mixed_string flour_reduced = "2 7/12" ∧
  rat_to_mixed_string sugar_reduced = "5/6" :=
by sorry

end recipe_reduction_recipe_reduction_mixed_numbers_l2172_217291


namespace square_room_carpet_area_l2172_217226

theorem square_room_carpet_area (room_side : ℝ) (sq_yard_to_sq_feet : ℝ) : 
  room_side = 9 → sq_yard_to_sq_feet = 9 → (room_side * room_side) / sq_yard_to_sq_feet = 9 := by
  sorry

end square_room_carpet_area_l2172_217226


namespace father_age_and_pen_cost_l2172_217253

/-- Xiao Ming's age -/
def xiao_ming_age : ℕ := 9

/-- The factor by which Xiao Ming's father's age is greater than Xiao Ming's -/
def father_age_factor : ℕ := 5

/-- The cost of one pen in yuan -/
def pen_cost : ℕ := 2

/-- The number of pens to be purchased -/
def pen_quantity : ℕ := 60

theorem father_age_and_pen_cost :
  (xiao_ming_age * father_age_factor = 45) ∧
  (pen_cost * pen_quantity = 120) := by
  sorry


end father_age_and_pen_cost_l2172_217253


namespace cube_space_diagonal_length_l2172_217252

/-- The length of a space diagonal in a cube with side length 15 -/
theorem cube_space_diagonal_length :
  ∀ (s : ℝ), s = 15 →
  ∃ (d : ℝ), d = s * Real.sqrt 3 ∧ d^2 = 3 * s^2 := by
  sorry

end cube_space_diagonal_length_l2172_217252


namespace solve_linear_equation_l2172_217283

theorem solve_linear_equation (x : ℝ) :
  3 * x - 8 = 4 * x + 5 → x = -13 := by
sorry

end solve_linear_equation_l2172_217283


namespace odd_function_sum_l2172_217257

def f (x : ℝ) (b : ℝ) : ℝ := 2016 * x^3 - 5 * x + b + 2

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (a b : ℝ) :
  (∃ f : ℝ → ℝ, is_odd f ∧ (∀ x, f x = 2016 * x^3 - 5 * x + b + 2) ∧
   (∃ c d : ℝ, c = a - 4 ∧ d = 2 * a - 2 ∧ Set.Icc c d = Set.range f)) →
  f a + f b = 0 :=
sorry

end odd_function_sum_l2172_217257


namespace solution_days_is_forty_l2172_217231

/-- The number of days required to solve all problems given the conditions -/
def solution_days (a b c : ℕ) : ℕ :=
  let total_problems := 5 * (11 * a + 7 * b + 9 * c)
  40

/-- The theorem stating that the solution_days function returns 40 given the problem conditions -/
theorem solution_days_is_forty (a b c : ℕ) :
  (5 * (11 * a + 7 * b + 9 * c) = 16 * (4 * a + 2 * b + 3 * c)) →
  solution_days a b c = 40 := by
  sorry

#check solution_days_is_forty

end solution_days_is_forty_l2172_217231


namespace pants_price_is_54_l2172_217215

/-- The price of a pair of pants Laura bought -/
def price_of_pants : ℕ := sorry

/-- The number of pairs of pants Laura bought -/
def num_pants : ℕ := 2

/-- The number of shirts Laura bought -/
def num_shirts : ℕ := 4

/-- The price of each shirt -/
def price_of_shirt : ℕ := 33

/-- The amount Laura gave to the cashier -/
def amount_given : ℕ := 250

/-- The change Laura received -/
def change_received : ℕ := 10

theorem pants_price_is_54 : price_of_pants = 54 := by
  sorry

end pants_price_is_54_l2172_217215


namespace existence_of_divisor_l2172_217293

def f : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 23 * f (n + 1) + f n

theorem existence_of_divisor (m : ℕ) : ∃ d : ℕ, ∀ n : ℕ, m ∣ f (f n) ↔ d ∣ n := by
  sorry

end existence_of_divisor_l2172_217293


namespace unique_three_digit_number_l2172_217251

theorem unique_three_digit_number : ∃! abc : ℕ,
  (abc ≥ 100 ∧ abc < 1000) ∧
  (abc % 100 = (abc / 100) ^ 2) ∧
  (abc % 9 = 4) :=
by
  sorry

end unique_three_digit_number_l2172_217251


namespace fractional_equation_solution_range_l2172_217261

theorem fractional_equation_solution_range (x m : ℝ) :
  (3 * x) / (x - 1) = m / (x - 1) + 2 →
  x ≥ 0 →
  x ≠ 1 →
  m ≥ 2 ∧ m ≠ 3 :=
by sorry

end fractional_equation_solution_range_l2172_217261


namespace switch_pairs_bound_l2172_217206

/-- Represents a row in Pascal's Triangle --/
def PascalRow (n : ℕ) := List ℕ

/-- Counts the number of odd entries in a Pascal's Triangle row --/
def countOddEntries (row : PascalRow n) : ℕ := sorry

/-- Counts the number of switch pairs in a Pascal's Triangle row --/
def countSwitchPairs (row : PascalRow n) : ℕ := sorry

/-- Theorem: The number of switch pairs in a Pascal's Triangle row is at most twice the number of odd entries --/
theorem switch_pairs_bound (n : ℕ) (row : PascalRow n) :
  countSwitchPairs row ≤ 2 * countOddEntries row := by sorry

end switch_pairs_bound_l2172_217206


namespace arithmetic_expression_equality_l2172_217220

theorem arithmetic_expression_equality : 5 + 2 * (8 - 3) = 15 := by
  sorry

end arithmetic_expression_equality_l2172_217220


namespace equal_intercepts_iff_specific_equation_not_in_second_quadrant_iff_a_leq_neg_one_l2172_217217

-- Define the line equation
def line_equation (a x y : ℝ) : Prop := (a + 1) * x + y + 2 - a = 0

-- Define the condition for equal intercepts
def equal_intercepts (a : ℝ) : Prop := ∃ (k : ℝ), line_equation a k 0 ∧ line_equation a 0 k

-- Define the condition for not passing through the second quadrant
def not_in_second_quadrant (a : ℝ) : Prop := ∀ (x y : ℝ), line_equation a x y → (x > 0 → y ≤ 0)

-- Theorem 1: Equal intercepts condition
theorem equal_intercepts_iff_specific_equation :
  ∀ (a : ℝ), equal_intercepts a ↔ (∀ (x y : ℝ), x + y + 4 = 0 ↔ line_equation a x y) :=
sorry

-- Theorem 2: Not passing through second quadrant condition
theorem not_in_second_quadrant_iff_a_leq_neg_one :
  ∀ (a : ℝ), not_in_second_quadrant a ↔ a ≤ -1 :=
sorry

end equal_intercepts_iff_specific_equation_not_in_second_quadrant_iff_a_leq_neg_one_l2172_217217


namespace share_ratio_l2172_217209

/-- Represents the shares of money for three individuals -/
structure Shares where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The problem setup -/
def problem_setup (s : Shares) : Prop :=
  s.a = 80 ∧                          -- a's share is $80
  s.a + s.b + s.c = 200 ∧             -- Total amount is $200
  s.a = (2/3) * (s.b + s.c) ∧         -- a gets 2/3 as much as b and c together
  ∃ x, s.b = x * (s.a + s.c)          -- b gets some fraction of a and c together

/-- The theorem to be proved -/
theorem share_ratio (s : Shares) (h : problem_setup s) : 
  s.b / (s.a + s.c) = 2/3 := by sorry

end share_ratio_l2172_217209


namespace four_digit_divisible_by_9_l2172_217249

def is_divisible_by_9 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 9 * k

def digit_sum (a : ℕ) : ℕ :=
  3 + a + a + 1

theorem four_digit_divisible_by_9 :
  ∃ A : ℕ, A < 10 ∧ is_divisible_by_9 (3000 + 100 * A + 10 * A + 1) ∧ A = 7 :=
sorry

end four_digit_divisible_by_9_l2172_217249


namespace negative_two_exponent_sum_l2172_217204

theorem negative_two_exponent_sum : (-2)^2023 + (-2)^2024 = 2^2023 := by
  sorry

end negative_two_exponent_sum_l2172_217204


namespace congruence_solution_l2172_217208

theorem congruence_solution (n : Int) : n ≡ 26 [ZMOD 47] ↔ 13 * n ≡ 9 [ZMOD 47] ∧ 0 ≤ n ∧ n < 47 := by
  sorry

end congruence_solution_l2172_217208


namespace strawberry_problem_l2172_217241

theorem strawberry_problem (initial : Float) (eaten : Float) (remaining : Float) : 
  initial = 78.0 → eaten = 42.0 → remaining = initial - eaten → remaining = 36.0 := by
  sorry

end strawberry_problem_l2172_217241


namespace sum_of_interior_angles_in_special_pentagon_l2172_217242

/-- A pentagon with two interior lines -/
structure PentagonWithInteriorLines where
  -- Exterior angles
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ
  -- Interior angles formed by the lines
  angle_x : ℝ
  angle_y : ℝ

/-- Theorem: Sum of interior angles in special pentagon configuration -/
theorem sum_of_interior_angles_in_special_pentagon
  (p : PentagonWithInteriorLines)
  (h_A : p.angle_A = 35)
  (h_B : p.angle_B = 65)
  (h_C : p.angle_C = 40) :
  p.angle_x + p.angle_y = 140 := by
  sorry

#check sum_of_interior_angles_in_special_pentagon

end sum_of_interior_angles_in_special_pentagon_l2172_217242


namespace worker_weekly_pay_l2172_217200

/-- Worker's weekly pay calculation --/
theorem worker_weekly_pay (regular_rate : ℝ) (total_surveys : ℕ) (cellphone_rate_increase : ℝ) (cellphone_surveys : ℕ) :
  regular_rate = 10 →
  total_surveys = 100 →
  cellphone_rate_increase = 0.3 →
  cellphone_surveys = 60 →
  let non_cellphone_surveys := total_surveys - cellphone_surveys
  let cellphone_rate := regular_rate * (1 + cellphone_rate_increase)
  let non_cellphone_pay := non_cellphone_surveys * regular_rate
  let cellphone_pay := cellphone_surveys * cellphone_rate
  let total_pay := non_cellphone_pay + cellphone_pay
  total_pay = 1180 := by
sorry

end worker_weekly_pay_l2172_217200


namespace midpoint_implies_equation_ratio_implies_equation_l2172_217216

/-- A line passing through point M(-2,1) and intersecting x and y axes at A and B respectively -/
structure Line :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (h_A : A.2 = 0)
  (h_B : B.1 = 0)

/-- The point M(-2,1) -/
def M : ℝ × ℝ := (-2, 1)

/-- M is the midpoint of AB -/
def is_midpoint (l : Line) : Prop :=
  M = ((l.A.1 + l.B.1) / 2, (l.A.2 + l.B.2) / 2)

/-- M divides AB in the ratio of 2:1 or 1:2 -/
def divides_in_ratio (l : Line) : Prop :=
  (M.1 - l.A.1, M.2 - l.A.2) = (2 * (l.B.1 - M.1), 2 * (l.B.2 - M.2)) ∨
  (M.1 - l.A.1, M.2 - l.A.2) = (-2 * (l.B.1 - M.1), -2 * (l.B.2 - M.2))

/-- The equation of the line in the form ax + by + c = 0 -/
structure LineEquation :=
  (a b c : ℝ)

theorem midpoint_implies_equation (l : Line) (h : is_midpoint l) :
  ∃ (eq : LineEquation), eq.a * l.A.1 + eq.b * l.A.2 + eq.c = 0 ∧
                         eq.a * l.B.1 + eq.b * l.B.2 + eq.c = 0 ∧
                         eq.a * M.1 + eq.b * M.2 + eq.c = 0 ∧
                         eq.a = 1 ∧ eq.b = -2 ∧ eq.c = 4 := by sorry

theorem ratio_implies_equation (l : Line) (h : divides_in_ratio l) :
  ∃ (eq1 eq2 : LineEquation),
    (eq1.a * l.A.1 + eq1.b * l.A.2 + eq1.c = 0 ∧
     eq1.a * l.B.1 + eq1.b * l.B.2 + eq1.c = 0 ∧
     eq1.a * M.1 + eq1.b * M.2 + eq1.c = 0 ∧
     eq1.a = 1 ∧ eq1.b = -4 ∧ eq1.c = 6) ∨
    (eq2.a * l.A.1 + eq2.b * l.A.2 + eq2.c = 0 ∧
     eq2.a * l.B.1 + eq2.b * l.B.2 + eq2.c = 0 ∧
     eq2.a * M.1 + eq2.b * M.2 + eq2.c = 0 ∧
     eq2.a = 1 ∧ eq2.b = 4 ∧ eq2.c = -2) := by sorry

end midpoint_implies_equation_ratio_implies_equation_l2172_217216


namespace delivery_driver_boxes_l2172_217299

/-- Theorem: A delivery driver with 3 stops and 9 boxes per stop has 27 boxes in total. -/
theorem delivery_driver_boxes (stops : ℕ) (boxes_per_stop : ℕ) (h1 : stops = 3) (h2 : boxes_per_stop = 9) :
  stops * boxes_per_stop = 27 := by
  sorry

end delivery_driver_boxes_l2172_217299


namespace smallest_sum_20_consecutive_twice_square_l2172_217210

/-- The sum of 20 consecutive integers starting from n -/
def sum_20_consecutive (n : ℕ) : ℕ := 10 * (2 * n + 19)

/-- Predicate to check if a number is twice a perfect square -/
def is_twice_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = 2 * k^2

/-- The smallest sum of 20 consecutive positive integers that is twice a perfect square -/
theorem smallest_sum_20_consecutive_twice_square : 
  (∃ n : ℕ, sum_20_consecutive n = 450 ∧ 
    is_twice_perfect_square (sum_20_consecutive n) ∧
    ∀ m : ℕ, m < n → ¬(is_twice_perfect_square (sum_20_consecutive m))) :=
sorry

end smallest_sum_20_consecutive_twice_square_l2172_217210


namespace symmetric_points_y_axis_l2172_217278

theorem symmetric_points_y_axis (m n : ℝ) : 
  (m - 1 = -2 ∧ 4 = n + 2) → n^m = 1/2 := by
  sorry

end symmetric_points_y_axis_l2172_217278


namespace power_of_eight_sum_equals_power_of_two_l2172_217205

theorem power_of_eight_sum_equals_power_of_two : 8^18 + 8^18 + 8^18 = 2^56 := by
  sorry

end power_of_eight_sum_equals_power_of_two_l2172_217205


namespace tangential_quadrilateral_theorem_l2172_217272

/-- A tangential quadrilateral with circumscribed and inscribed circles -/
structure TangentialQuadrilateral where
  /-- The radius of the circumscribed circle -/
  R : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The distance between the centers of the circumscribed and inscribed circles -/
  d : ℝ
  /-- R is positive -/
  R_pos : R > 0
  /-- r is positive -/
  r_pos : r > 0
  /-- d is non-negative and less than R -/
  d_bounds : 0 ≤ d ∧ d < R

/-- The main theorem about the relationship between R, r, and d in a tangential quadrilateral -/
theorem tangential_quadrilateral_theorem (q : TangentialQuadrilateral) :
  1 / (q.R + q.d)^2 + 1 / (q.R - q.d)^2 = 1 / q.r^2 := by
  sorry

end tangential_quadrilateral_theorem_l2172_217272


namespace area_AEHF_is_twelve_l2172_217288

/-- Rectangle ABCD with dimensions 5x6 -/
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)

/-- Point on a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Definition of rectangle ABCD -/
def rect_ABCD : Rectangle :=
  { width := 5, height := 6 }

/-- Point A at (0,0) -/
def point_A : Point :=
  { x := 0, y := 0 }

/-- Point E on CD, 3 units from D -/
def point_E : Point :=
  { x := 3, y := rect_ABCD.height }

/-- Point F on AB, 2 units from A -/
def point_F : Point :=
  { x := 2, y := 0 }

/-- Area of rectangle AEHF -/
def area_AEHF : ℝ :=
  (point_E.x - point_A.x) * (point_E.y - point_A.y)

/-- Theorem stating that the area of rectangle AEHF is 12 square units -/
theorem area_AEHF_is_twelve : area_AEHF = 12 := by
  sorry

end area_AEHF_is_twelve_l2172_217288


namespace range_of_M_l2172_217247

theorem range_of_M (x y : ℝ) (h : x^2 + x*y + y^2 = 2) : 
  let M := x^2 - x*y + y^2
  2/3 ≤ M ∧ M ≤ 6 := by
sorry

end range_of_M_l2172_217247


namespace soccer_field_kids_l2172_217275

/-- The number of kids on a soccer field after more kids join -/
def total_kids (initial : ℕ) (joined : ℕ) : ℕ :=
  initial + joined

/-- Theorem: The total number of kids on the soccer field is 36 -/
theorem soccer_field_kids : total_kids 14 22 = 36 := by
  sorry

end soccer_field_kids_l2172_217275


namespace sin_double_angle_l2172_217236

theorem sin_double_angle (x : Real) (h : Real.sin (x + π/4) = 4/5) : 
  Real.sin (2*x) = 7/25 := by
  sorry

end sin_double_angle_l2172_217236


namespace smallest_natural_divisible_l2172_217277

theorem smallest_natural_divisible (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m + 1 = 4 * k)) ∨
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m + 1 = 6 * k)) ∨
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m + 1 = 10 * k)) ∨
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m + 1 = 12 * k)) →
  (∃ k1 k2 k3 k4 : ℕ, n + 1 = 4 * k1 ∧ n + 1 = 6 * k2 ∧ n + 1 = 10 * k3 ∧ n + 1 = 12 * k4) →
  n = 59 := by
sorry

end smallest_natural_divisible_l2172_217277


namespace max_leftover_apples_l2172_217281

theorem max_leftover_apples (n : ℕ) (students : ℕ) (h : students = 8) :
  ∃ (apples_per_student : ℕ) (leftover : ℕ),
    n = students * apples_per_student + leftover ∧
    leftover < students ∧
    leftover ≤ 7 ∧
    (∀ k, k > leftover → ¬(∃ m, n = students * m + k)) :=
by sorry

end max_leftover_apples_l2172_217281


namespace remainder_sum_l2172_217289

theorem remainder_sum (n : ℤ) : n % 20 = 11 → (n % 4 + n % 5 = 4) := by
  sorry

end remainder_sum_l2172_217289


namespace product_equals_32_l2172_217279

theorem product_equals_32 : 
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 := by
  sorry

end product_equals_32_l2172_217279


namespace pi_only_irrational_l2172_217270

-- Define a function to check if a number is rational
def is_rational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- State the theorem
theorem pi_only_irrational : 
  is_rational (1/7) ∧ 
  ¬(is_rational Real.pi) ∧ 
  is_rational (-1) ∧ 
  is_rational 0 :=
sorry

end pi_only_irrational_l2172_217270


namespace complement_union_A_B_l2172_217203

open Set

def U : Set ℝ := univ

def A : Set ℝ := {x | (x - 2) * (x + 1) ≤ 0}

def B : Set ℝ := {x | 0 ≤ x ∧ x < 3}

theorem complement_union_A_B : 
  (Uᶜ ∩ (A ∪ B)ᶜ) = {x | x < -1 ∨ x ≥ 3} := by sorry

end complement_union_A_B_l2172_217203


namespace calculation_proof_l2172_217225

theorem calculation_proof :
  ((-4)^2 * ((-3/4) + (-5/8)) = -22) ∧
  (-2^2 - (1 - 0.5) * (1/3) * (2 - (-4)^2) = -5/3) := by
  sorry

end calculation_proof_l2172_217225


namespace math_city_intersections_l2172_217273

/-- Represents a city with a given number of streets -/
structure City where
  num_streets : ℕ
  no_parallel : Bool
  no_triple_intersections : Bool

/-- Calculates the number of intersections in a city -/
def num_intersections (c : City) : ℕ :=
  (c.num_streets * (c.num_streets - 1)) / 2

/-- Theorem: A city with 10 streets, no parallel streets, and no triple intersections has 45 intersections -/
theorem math_city_intersections :
  ∀ (c : City), c.num_streets = 10 → c.no_parallel = true → c.no_triple_intersections = true →
  num_intersections c = 45 :=
by sorry

end math_city_intersections_l2172_217273


namespace event_X_6_equivalent_to_draw_6_and_two_others_l2172_217245

/-- Represents a ball with a number -/
structure Ball :=
  (number : Nat)

/-- The set of all balls in the bag -/
def bag : Finset Ball := sorry

/-- The number of balls to be drawn -/
def numDrawn : Nat := 3

/-- X represents the highest number on the drawn balls -/
def X (drawn : Finset Ball) : Nat := sorry

/-- The event where X equals 6 -/
def event_X_equals_6 (drawn : Finset Ball) : Prop :=
  X drawn = 6

/-- The event of drawing 3 balls with one numbered 6 and two others from 1 to 5 -/
def event_draw_6_and_two_others (drawn : Finset Ball) : Prop := sorry

theorem event_X_6_equivalent_to_draw_6_and_two_others :
  ∀ drawn : Finset Ball,
  drawn.card = numDrawn →
  (event_X_equals_6 drawn ↔ event_draw_6_and_two_others drawn) :=
sorry

end event_X_6_equivalent_to_draw_6_and_two_others_l2172_217245


namespace emelya_balls_count_l2172_217213

def total_balls : ℕ := 10
def broken_balls : ℕ := 3
def lost_balls : ℕ := 3

theorem emelya_balls_count :
  ∀ (M : ℝ),
  M > 0 →
  (broken_balls : ℝ) * M * (35/100) = (7/20) * M →
  ∃ (remaining_balls : ℕ),
  remaining_balls > 0 ∧
  (remaining_balls : ℝ) * M * (8/13) = (2/5) * M ∧
  total_balls = remaining_balls + broken_balls + lost_balls :=
by sorry

end emelya_balls_count_l2172_217213


namespace positive_real_inequalities_l2172_217248

theorem positive_real_inequalities (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a * b + a + b + 1) * (a * b + a * c + b * c + c^2) ≥ 16 * a * b * c) ∧
  ((b + c - a) / a + (c + a - b) / b + (a + b - c) / c ≥ 3) := by
  sorry

end positive_real_inequalities_l2172_217248


namespace binomial_2023_2_l2172_217243

theorem binomial_2023_2 : Nat.choose 2023 2 = 2045323 := by
  sorry

end binomial_2023_2_l2172_217243


namespace election_percentage_l2172_217228

theorem election_percentage (total_votes : ℕ) (second_candidate_votes : ℕ) :
  total_votes = 1200 →
  second_candidate_votes = 240 →
  (total_votes - second_candidate_votes : ℝ) / total_votes * 100 = 80 := by
sorry

end election_percentage_l2172_217228


namespace another_divisor_of_44404_l2172_217280

theorem another_divisor_of_44404 (n : Nat) (h1 : n = 44404) 
  (h2 : 12 ∣ n) (h3 : 48 ∣ n) (h4 : 74 ∣ n) (h5 : 100 ∣ n) : 
  199 ∣ n := by
  sorry

end another_divisor_of_44404_l2172_217280


namespace absolute_value_calculation_system_of_inequalities_l2172_217263

-- Part 1
theorem absolute_value_calculation : |(-2 : ℝ)| + Real.sqrt 4 - 2^(0 : ℕ) = 3 := by sorry

-- Part 2
theorem system_of_inequalities (x : ℝ) : 
  (2 * x < 6 ∧ 3 * x > -2 * x + 5) ↔ (1 < x ∧ x < 3) := by sorry

end absolute_value_calculation_system_of_inequalities_l2172_217263


namespace min_distance_between_curves_l2172_217250

/-- The minimum distance between curves C₁ and C₂ is 0 -/
theorem min_distance_between_curves (x y : ℝ) : 
  let C₁ := {(x, y) | x^2/8 + y^2/4 = 1}
  let C₂ := {(x, y) | x - Real.sqrt 2 * y - 4 = 0}
  ∃ (p q : ℝ × ℝ), p ∈ C₁ ∧ q ∈ C₂ ∧ 
    ∀ (p' q' : ℝ × ℝ), p' ∈ C₁ → q' ∈ C₂ → 
      Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) ≥ 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ∧
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 0 :=
by sorry

end min_distance_between_curves_l2172_217250


namespace charts_brought_is_eleven_l2172_217218

/-- The number of charts brought to a committee meeting --/
def charts_brought (associate_profs assistant_profs : ℕ) : ℕ :=
  associate_profs + 2 * assistant_profs

/-- Proof that 11 charts were brought to the meeting --/
theorem charts_brought_is_eleven :
  ∃ (associate_profs assistant_profs : ℕ),
    associate_profs + assistant_profs = 7 ∧
    2 * associate_profs + assistant_profs = 10 ∧
    charts_brought associate_profs assistant_profs = 11 :=
by
  sorry

#check charts_brought_is_eleven

end charts_brought_is_eleven_l2172_217218


namespace petya_vasya_game_l2172_217274

theorem petya_vasya_game (k : ℚ) : 
  ∃ (a b c : ℚ), ∃ (x y : ℚ), 
    x^3 + a*x^2 + b*x + c = 0 ∧ 
    y^3 + a*y^2 + b*y + c = 0 ∧ 
    y - x = 2014 :=
by sorry

end petya_vasya_game_l2172_217274


namespace max_value_x_minus_2y_l2172_217224

theorem max_value_x_minus_2y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + 2*y) :
  ∃ (max : ℝ), max = 2/3 ∧ x - 2*y ≤ max :=
sorry

end max_value_x_minus_2y_l2172_217224


namespace mrs_blue_garden_yield_l2172_217233

/-- Represents the dimensions of a rectangular garden in steps -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the expected tomato yield from a garden -/
def expectedTomatoYield (garden : GardenDimensions) (stepLength : ℚ) (yieldPerSqFt : ℚ) : ℚ :=
  (garden.length : ℚ) * stepLength * (garden.width : ℚ) * stepLength * yieldPerSqFt

/-- Theorem stating the expected tomato yield for Mrs. Blue's garden -/
theorem mrs_blue_garden_yield :
  let garden : GardenDimensions := { length := 18, width := 24 }
  let stepLength : ℚ := 3/2
  let yieldPerSqFt : ℚ := 2/3
  expectedTomatoYield garden stepLength yieldPerSqFt = 648 := by
  sorry

end mrs_blue_garden_yield_l2172_217233


namespace expression_value_l2172_217201

theorem expression_value (x : ℝ) (h : 2 * x^2 - x - 1 = 5) : 6 * x^2 - 3 * x - 9 = 9 := by
  sorry

end expression_value_l2172_217201


namespace expression_evaluation_l2172_217296

theorem expression_evaluation : ((-3)^2)^4 * (-3)^8 * 2 = 86093442 := by
  sorry

end expression_evaluation_l2172_217296


namespace prob_at_least_two_tails_is_half_l2172_217214

/-- The probability of getting at least 2 tails when tossing 3 fair coins -/
def prob_at_least_two_tails : ℚ := 1/2

/-- The number of possible outcomes when tossing 3 coins -/
def total_outcomes : ℕ := 2^3

/-- The number of favorable outcomes (at least 2 tails) -/
def favorable_outcomes : ℕ := 4

theorem prob_at_least_two_tails_is_half :
  prob_at_least_two_tails = (favorable_outcomes : ℚ) / total_outcomes :=
by sorry

end prob_at_least_two_tails_is_half_l2172_217214


namespace additional_cards_proof_l2172_217222

/-- The number of cards in the original deck -/
def original_deck : ℕ := 52

/-- The number of players -/
def num_players : ℕ := 3

/-- The number of cards each player has after splitting the deck -/
def cards_per_player : ℕ := 18

/-- The number of additional cards added to the deck -/
def additional_cards : ℕ := (num_players * cards_per_player) - original_deck

theorem additional_cards_proof :
  additional_cards = 2 := by sorry

end additional_cards_proof_l2172_217222


namespace profit_percentage_calculation_l2172_217239

theorem profit_percentage_calculation 
  (tv_cost dvd_cost selling_price : ℕ) : 
  tv_cost = 16000 → 
  dvd_cost = 6250 → 
  selling_price = 35600 → 
  (selling_price - (tv_cost + dvd_cost)) * 100 / (tv_cost + dvd_cost) = 60 := by
  sorry

end profit_percentage_calculation_l2172_217239


namespace smallest_k_for_f_divides_l2172_217282

/-- The polynomial z^12 + z^11 + z^7 + z^6 + z^5 + z + 1 -/
def f (z : ℂ) : ℂ := z^12 + z^11 + z^7 + z^6 + z^5 + z + 1

/-- Proposition: 91 is the smallest positive integer k such that f(z) divides z^k - 1 -/
theorem smallest_k_for_f_divides : ∀ z : ℂ, z ≠ 0 →
  (∀ k : ℕ, k > 0 → k < 91 → ¬(f z ∣ z^k - 1)) ∧
  (f z ∣ z^91 - 1) := by
  sorry

#check smallest_k_for_f_divides

end smallest_k_for_f_divides_l2172_217282


namespace dice_probability_l2172_217229

/-- The number of dice -/
def num_dice : ℕ := 4

/-- The number of sides on each die -/
def sides_per_die : ℕ := 8

/-- The probability of all dice showing the same number -/
def prob_all_same : ℚ := 1 / (sides_per_die ^ (num_dice - 1))

theorem dice_probability :
  prob_all_same = 1 / 512 := by sorry

end dice_probability_l2172_217229


namespace fathers_age_l2172_217255

theorem fathers_age (man_age father_age : ℚ) : 
  man_age = (2 / 5) * father_age →
  man_age + 10 = (1 / 2) * (father_age + 10) →
  father_age = 50 := by
sorry

end fathers_age_l2172_217255


namespace phi_value_l2172_217232

/-- Given a function f(x) = 2sin(ωx + φ) with the following properties:
    - ω > 0
    - |φ| < π/2
    - x = 5π/8 is an axis of symmetry for y = f(x)
    - x = 11π/8 is a zero of f(x)
    - The smallest positive period of f(x) is greater than 2π
    Prove that φ = π/12 -/
theorem phi_value (ω φ : Real) (h1 : ω > 0) (h2 : |φ| < π/2)
  (h3 : ∀ x, 2 * Real.sin (ω * (5*π/4 - (x - 5*π/8)) + φ) = 2 * Real.sin (ω * x + φ))
  (h4 : 2 * Real.sin (ω * 11*π/8 + φ) = 0)
  (h5 : 2*π / ω > 2*π) : φ = π/12 := by
  sorry

end phi_value_l2172_217232


namespace substitution_ways_mod_1000_l2172_217267

/-- Represents the number of players in a soccer team --/
def total_players : ℕ := 22

/-- Represents the number of starting players --/
def starting_players : ℕ := 11

/-- Represents the maximum number of substitutions allowed --/
def max_substitutions : ℕ := 4

/-- Calculates the number of ways to make substitutions in a soccer game --/
def substitution_ways : ℕ := 
  1 + 
  (starting_players * starting_players) + 
  (starting_players^3 * (starting_players - 1)) + 
  (starting_players^5 * (starting_players - 1) * (starting_players - 2)) + 
  (starting_players^7 * (starting_players - 1) * (starting_players - 2) * (starting_players - 3))

/-- Theorem stating that the number of substitution ways modulo 1000 is 712 --/
theorem substitution_ways_mod_1000 : 
  substitution_ways % 1000 = 712 := by sorry

end substitution_ways_mod_1000_l2172_217267


namespace half_dollar_percentage_l2172_217207

def nickel_value : ℚ := 5
def half_dollar_value : ℚ := 50
def num_nickels : ℕ := 75
def num_half_dollars : ℕ := 30

theorem half_dollar_percentage :
  (num_half_dollars * half_dollar_value) / 
  (num_nickels * nickel_value + num_half_dollars * half_dollar_value) = 4/5 := by
  sorry

end half_dollar_percentage_l2172_217207


namespace terminal_zeros_fifty_times_three_sixty_l2172_217244

-- Define the prime factorizations of 50 and 360
def fifty : ℕ := 2 * 5^2
def three_sixty : ℕ := 2^3 * 3^2 * 5

-- Function to count terminal zeros
def count_terminal_zeros (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem terminal_zeros_fifty_times_three_sixty : 
  count_terminal_zeros (fifty * three_sixty) = 3 := by sorry

end terminal_zeros_fifty_times_three_sixty_l2172_217244


namespace negative_fraction_comparison_l2172_217265

theorem negative_fraction_comparison : -4/5 < -2/3 := by
  sorry

end negative_fraction_comparison_l2172_217265


namespace class_payment_problem_l2172_217284

theorem class_payment_problem (total_students : ℕ) (full_payment half_payment total_collected : ℚ) 
  (h1 : total_students = 25)
  (h2 : full_payment = 50)
  (h3 : half_payment = 25)
  (h4 : total_collected = 1150)
  (h5 : ∃ (full_payers half_payers : ℕ), 
    full_payers + half_payers = total_students ∧ 
    full_payers * full_payment + half_payers * half_payment = total_collected) :
  ∃ (half_payers : ℕ), half_payers = 4 := by
sorry

end class_payment_problem_l2172_217284


namespace quadratic_roots_property_l2172_217264

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 - 7 * p - 6 = 0) → 
  (3 * q^2 - 7 * q - 6 = 0) → 
  p ≠ q →
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 7 := by
sorry

end quadratic_roots_property_l2172_217264


namespace problem_solution_l2172_217234

theorem problem_solution (x y : ℝ) (h : x^2 + y^2 = 12*x - 4*y - 40) :
  x * Real.cos (-23/3 * Real.pi) + y * Real.tan (-15/4 * Real.pi) = 1 := by
  sorry

end problem_solution_l2172_217234


namespace oil_price_reduction_l2172_217235

/-- Proves that the percentage reduction in oil price is 40% given the problem conditions -/
theorem oil_price_reduction (original_price reduced_price : ℝ) : 
  reduced_price = 120 →
  2400 / reduced_price - 2400 / original_price = 8 →
  (original_price - reduced_price) / original_price * 100 = 40 := by
  sorry

#check oil_price_reduction

end oil_price_reduction_l2172_217235


namespace square_sum_of_difference_and_product_l2172_217230

theorem square_sum_of_difference_and_product (a b : ℝ) 
  (h1 : a - b = 4) 
  (h2 : a * b = 1) : 
  a^2 + b^2 = 18 := by
sorry

end square_sum_of_difference_and_product_l2172_217230


namespace circles_tangent_radius_l2172_217246

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 8*y - 5 = 0
def circle_O2 (x y r : ℝ) : Prop := (x+2)^2 + y^2 = r^2

-- Theorem statement
theorem circles_tangent_radius (r : ℝ) :
  (r > 0) →
  (∃! p : ℝ × ℝ, circle_O1 p.1 p.2 ∧ circle_O2 p.1 p.2 r) →
  r = 1 ∨ r = 9 := by
  sorry

end circles_tangent_radius_l2172_217246


namespace jeans_to_shirt_cost_ratio_l2172_217202

/-- The ratio of the cost of a pair of jeans to the cost of a shirt is 2:1 -/
theorem jeans_to_shirt_cost_ratio :
  ∀ (jeans_cost : ℚ),
  20 * 10 + 10 * jeans_cost = 400 →
  jeans_cost / 10 = 2 / 1 := by
sorry

end jeans_to_shirt_cost_ratio_l2172_217202


namespace bag_pieces_problem_l2172_217259

theorem bag_pieces_problem (w b n : ℕ) : 
  b = 2 * w →                 -- The number of black pieces is twice the number of white pieces
  w - 2 * n = 1 →             -- After n rounds, 1 white piece is left
  b - 3 * n = 31 →            -- After n rounds, 31 black pieces are left
  b = 118 :=                  -- The initial number of black pieces was 118
by sorry

end bag_pieces_problem_l2172_217259


namespace min_sum_complementary_events_l2172_217295

theorem min_sum_complementary_events (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hcomp : 4/x + 1/y = 1) : 
  x + y ≥ 9 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 4/x + 1/y = 1 ∧ x + y = 9 :=
by sorry

end min_sum_complementary_events_l2172_217295


namespace line_through_point_at_distance_l2172_217297

/-- A line passing through a point (x₀, y₀) and at a distance d from the origin -/
structure DistanceLine where
  x₀ : ℝ
  y₀ : ℝ
  d : ℝ

/-- Check if a line equation ax + by + c = 0 passes through a point (x₀, y₀) -/
def passesThrough (a b c x₀ y₀ : ℝ) : Prop :=
  a * x₀ + b * y₀ + c = 0

/-- Check if a line equation ax + by + c = 0 is at a distance d from the origin -/
def distanceFromOrigin (a b c d : ℝ) : Prop :=
  |c| / Real.sqrt (a^2 + b^2) = d

theorem line_through_point_at_distance (l : DistanceLine) :
  (passesThrough 1 0 (-3) l.x₀ l.y₀ ∧ distanceFromOrigin 1 0 (-3) l.d) ∨
  (passesThrough 8 (-15) 51 l.x₀ l.y₀ ∧ distanceFromOrigin 8 (-15) 51 l.d) :=
by sorry

#check line_through_point_at_distance

end line_through_point_at_distance_l2172_217297


namespace intersection_point_on_fixed_line_l2172_217223

/-- Hyperbola C with given properties -/
structure Hyperbola where
  center : ℝ × ℝ
  left_focus : ℝ × ℝ
  eccentricity : ℝ
  left_vertex : ℝ × ℝ
  right_vertex : ℝ × ℝ

/-- Line passing through a point and intersecting the hyperbola -/
structure IntersectingLine where
  passing_point : ℝ × ℝ
  intersection_point1 : ℝ × ℝ
  intersection_point2 : ℝ × ℝ

/-- Theorem stating that the intersection point P lies on a fixed line -/
theorem intersection_point_on_fixed_line (C : Hyperbola) (L : IntersectingLine) : 
  C.center = (0, 0) →
  C.left_focus = (-2 * Real.sqrt 5, 0) →
  C.eccentricity = Real.sqrt 5 →
  C.left_vertex = (-2, 0) →
  C.right_vertex = (2, 0) →
  L.passing_point = (-4, 0) →
  L.intersection_point1.1 < 0 ∧ L.intersection_point1.2 > 0 → -- M in second quadrant
  ∃ (P : ℝ × ℝ), P.1 = -1 := by sorry

end intersection_point_on_fixed_line_l2172_217223
