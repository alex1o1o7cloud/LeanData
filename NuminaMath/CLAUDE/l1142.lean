import Mathlib

namespace cricket_bat_cost_price_l1142_114295

theorem cricket_bat_cost_price (profit_A_to_B : Real) (profit_B_to_C : Real) (price_C : Real) :
  profit_A_to_B = 0.20 →
  profit_B_to_C = 0.25 →
  price_C = 231 →
  ∃ (cost_price_A : Real), cost_price_A = 154 ∧
    price_C = cost_price_A * (1 + profit_A_to_B) * (1 + profit_B_to_C) := by
  sorry

end cricket_bat_cost_price_l1142_114295


namespace expression_equals_zero_l1142_114267

theorem expression_equals_zero (θ : Real) (h : Real.tan θ = 5) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := by
sorry

end expression_equals_zero_l1142_114267


namespace bottle_caps_distribution_l1142_114236

theorem bottle_caps_distribution (num_children : ℕ) (total_caps : ℕ) (caps_per_child : ℕ) :
  num_children = 9 →
  total_caps = 45 →
  total_caps = num_children * caps_per_child →
  caps_per_child = 5 := by
sorry

end bottle_caps_distribution_l1142_114236


namespace total_legs_in_collection_l1142_114214

/-- The number of legs for a spider -/
def spider_legs : ℕ := 8

/-- The number of legs for an ant -/
def ant_legs : ℕ := 6

/-- The number of spiders in the collection -/
def num_spiders : ℕ := 8

/-- The number of ants in the collection -/
def num_ants : ℕ := 12

/-- Theorem stating that the total number of legs in the collection is 136 -/
theorem total_legs_in_collection : 
  num_spiders * spider_legs + num_ants * ant_legs = 136 := by
  sorry

end total_legs_in_collection_l1142_114214


namespace imaginary_part_of_z_l1142_114230

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = 1 + 2 * Complex.I) : 
  z.im = -1 := by sorry

end imaginary_part_of_z_l1142_114230


namespace blue_tetrahedron_volume_l1142_114296

/-- The volume of a tetrahedron formed by alternately colored vertices of a cube -/
theorem blue_tetrahedron_volume (s : ℝ) (h : s = 8) :
  let cube_volume := s^3
  let small_tetrahedron_volume := (1/6) * s^3
  cube_volume - 4 * small_tetrahedron_volume = (512:ℝ)/3 :=
by
  sorry

end blue_tetrahedron_volume_l1142_114296


namespace tree_height_after_two_years_l1142_114289

def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

theorem tree_height_after_two_years 
  (h : tree_height (tree_height h0 2) 2 = 81) : tree_height h0 2 = 9 :=
by
  sorry

#check tree_height_after_two_years

end tree_height_after_two_years_l1142_114289


namespace count_valid_sequences_l1142_114215

/-- Represents a die throw result -/
inductive DieThrow
  | even (n : Nat)
  | odd (n : Nat)

/-- Represents a point in 2D space -/
structure Point where
  x : Nat
  y : Nat

/-- Defines how a point moves based on a die throw -/
def move (p : Point) (t : DieThrow) : Point :=
  match t with
  | DieThrow.even n => Point.mk (p.x + n) p.y
  | DieThrow.odd n => Point.mk p.x (p.y + n)

/-- Defines a valid sequence of die throws -/
def validSequence (seq : List DieThrow) : Prop :=
  let finalPoint := seq.foldl move (Point.mk 0 0)
  finalPoint.x = 4 ∧ finalPoint.y = 4

/-- The main theorem to prove -/
theorem count_valid_sequences : 
  (∃ (seqs : List (List DieThrow)), 
    (∀ seq ∈ seqs, validSequence seq) ∧ 
    (∀ seq, validSequence seq → seq ∈ seqs) ∧
    seqs.length = 38) := by
  sorry

end count_valid_sequences_l1142_114215


namespace purely_imaginary_complex_number_l1142_114259

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I : ℂ) * (((a - Complex.I) / (1 - Complex.I)).im) = ((a - Complex.I) / (1 - Complex.I)) → 
  a = -1 := by
  sorry

end purely_imaginary_complex_number_l1142_114259


namespace cost_of_treats_treats_cost_is_twelve_l1142_114202

/-- Calculates the cost of a bag of treats given the total spent and other expenses --/
theorem cost_of_treats (puppy_cost : ℝ) (dog_food : ℝ) (toys : ℝ) (crate : ℝ) (bed : ℝ) (collar_leash : ℝ) 
  (discount_rate : ℝ) (total_spent : ℝ) : ℝ :=
  let other_items := dog_food + toys + crate + bed + collar_leash
  let discounted_other_items := other_items * (1 - discount_rate)
  let treats_total := total_spent - puppy_cost - discounted_other_items
  treats_total / 2

/-- Proves that the cost of a bag of treats is $12.00 --/
theorem treats_cost_is_twelve : 
  cost_of_treats 20 20 15 20 20 15 0.2 96 = 12 := by
  sorry

end cost_of_treats_treats_cost_is_twelve_l1142_114202


namespace function_equality_up_to_constant_l1142_114246

theorem function_equality_up_to_constant 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g) 
  (h : ∀ x, deriv f x = deriv g x) : 
  ∃ C, ∀ x, f x = g x + C :=
sorry

end function_equality_up_to_constant_l1142_114246


namespace a_in_M_necessary_not_sufficient_for_a_in_N_l1142_114247

-- Define sets M and N
def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Theorem stating that "a ∈ M" is a necessary but not sufficient condition for "a ∈ N"
theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) := by
  sorry


end a_in_M_necessary_not_sufficient_for_a_in_N_l1142_114247


namespace min_value_of_function_l1142_114227

theorem min_value_of_function (x : ℝ) (h : x > 10) : x^2 / (x - 10) ≥ 40 ∧ ∃ y > 10, y^2 / (y - 10) = 40 := by
  sorry

end min_value_of_function_l1142_114227


namespace project_completion_days_l1142_114277

/-- Represents the work rates and schedule for a project completed by three persons. -/
structure ProjectSchedule where
  rate_A : ℚ  -- Work rate of person A (fraction of work completed per day)
  rate_B : ℚ  -- Work rate of person B
  rate_C : ℚ  -- Work rate of person C
  days_A : ℕ  -- Number of days A works alone
  days_BC : ℕ  -- Number of days B and C work together

/-- Calculates the total number of days needed to complete the project. -/
def totalDays (p : ProjectSchedule) : ℚ :=
  let work_A := p.rate_A * p.days_A
  let rate_BC := p.rate_B + p.rate_C
  let work_BC := rate_BC * p.days_BC
  let remaining_work := 1 - (work_A + work_BC)
  p.days_A + p.days_BC + remaining_work / p.rate_C

/-- Theorem stating that for the given project schedule, the total days needed is 9. -/
theorem project_completion_days :
  let p := ProjectSchedule.mk (1/10) (1/12) (1/15) 2 4
  totalDays p = 9 := by sorry

end project_completion_days_l1142_114277


namespace balanced_quadruple_theorem_l1142_114207

def is_balanced (a b c d : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ a + b + c + d = a^2 + b^2 + c^2 + d^2

theorem balanced_quadruple_theorem :
  ∀ x : ℝ, x > 0 →
  (∀ a b c d : ℝ, is_balanced a b c d → (x - a) * (x - b) * (x - c) * (x - d) ≥ 0) ↔
  x ≥ 3/2 := by sorry

end balanced_quadruple_theorem_l1142_114207


namespace remainder_of_large_number_div_16_l1142_114235

theorem remainder_of_large_number_div_16 :
  65985241545898754582556898522454889 % 16 = 9 := by
  sorry

end remainder_of_large_number_div_16_l1142_114235


namespace units_digit_of_M_M8_l1142_114221

-- Define the Lucas-like sequence M_n
def M : ℕ → ℕ
  | 0 => 3
  | 1 => 2
  | n + 2 => 2 * M (n + 1) + M n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_M_M8 : unitsDigit (M (M 8)) = 6 := by
  sorry

end units_digit_of_M_M8_l1142_114221


namespace quadratic_root_l1142_114244

theorem quadratic_root (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - (m + 3) * x + m = 0 ∧ x = 1) → 
  (∃ y : ℝ, 2 * y^2 - (m + 3) * y + m = 0 ∧ y = (-m - 5) / 2) :=
by sorry

end quadratic_root_l1142_114244


namespace sophie_germain_prime_units_digit_l1142_114228

/-- A positive prime number p is a Sophie Germain prime if 2p + 1 is also prime. -/
def SophieGermainPrime (p : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime (2 * p + 1)

/-- The units digit of a natural number. -/
def unitsDigit (n : ℕ) : ℕ :=
  n % 10

theorem sophie_germain_prime_units_digit (p : ℕ) (h : SophieGermainPrime p) (h_gt : p > 6) :
  unitsDigit p = 1 ∨ unitsDigit p = 3 :=
sorry

end sophie_germain_prime_units_digit_l1142_114228


namespace prob_two_heads_with_second_tail_l1142_114254

/-- A fair coin flip sequence that ends with either two heads or two tails in a row -/
inductive CoinFlipSequence
| TH : CoinFlipSequence → CoinFlipSequence
| TT : CoinFlipSequence
| HH : CoinFlipSequence

/-- The probability of a specific coin flip sequence -/
def probability (seq : CoinFlipSequence) : ℚ :=
  match seq with
  | CoinFlipSequence.TH s => (1/2) * probability s
  | CoinFlipSequence.TT => (1/2) * (1/2)
  | CoinFlipSequence.HH => (1/2) * (1/2)

/-- The probability of getting two heads in a row while seeing a second tail before seeing a second head -/
def probTwoHeadsWithSecondTail : ℚ :=
  (1/2) * (1/2) * (1/2) * (1/3)

theorem prob_two_heads_with_second_tail :
  probTwoHeadsWithSecondTail = 1/24 :=
sorry

end prob_two_heads_with_second_tail_l1142_114254


namespace divisibility_implication_l1142_114299

theorem divisibility_implication (m n : ℤ) : 
  (11 ∣ (5*m + 3*n)) → (11 ∣ (9*m + n)) := by
sorry

end divisibility_implication_l1142_114299


namespace decimal_multiplication_l1142_114255

theorem decimal_multiplication : (0.25 : ℝ) * 0.75 * 0.1 = 0.01875 := by sorry

end decimal_multiplication_l1142_114255


namespace giants_playoff_wins_l1142_114263

theorem giants_playoff_wins (total_games : ℕ) (games_to_win : ℕ) (more_wins_needed : ℕ) : 
  total_games = 30 →
  games_to_win = (2 * total_games) / 3 →
  more_wins_needed = 8 →
  games_to_win - more_wins_needed = 12 :=
by sorry

end giants_playoff_wins_l1142_114263


namespace max_saturdays_is_five_l1142_114261

/-- Represents the possible number of days in a month -/
inductive MonthLength
  | Days28
  | Days29
  | Days30
  | Days31

/-- Represents the day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the number of Saturdays in a month -/
def saturdays_in_month (length : MonthLength) (start : DayOfWeek) : Nat :=
  sorry

/-- The maximum number of Saturdays in any month -/
def max_saturdays : Nat := 5

/-- Theorem: The maximum number of Saturdays in any month is 5 -/
theorem max_saturdays_is_five :
  ∀ (length : MonthLength) (start : DayOfWeek),
    saturdays_in_month length start ≤ max_saturdays :=
  sorry

end max_saturdays_is_five_l1142_114261


namespace eight_fourth_equals_sixteen_n_l1142_114210

theorem eight_fourth_equals_sixteen_n (n : ℕ) : 8^4 = 16^n → n = 3 := by
  sorry

end eight_fourth_equals_sixteen_n_l1142_114210


namespace divisibility_condition_l1142_114281

theorem divisibility_condition (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  n ∣ (1 + m^(3^n) + m^(2*3^n)) ↔ ∃ t : ℕ+, n = 3 ∧ m = 3 * t - 2 :=
by sorry

end divisibility_condition_l1142_114281


namespace divisibility_implies_multiple_of_three_l1142_114229

theorem divisibility_implies_multiple_of_three (a b : ℤ) :
  (9 : ℤ) ∣ (a^2 + a*b + b^2) → (3 : ℤ) ∣ a ∧ (3 : ℤ) ∣ b := by
  sorry

end divisibility_implies_multiple_of_three_l1142_114229


namespace calculate_expression_l1142_114266

theorem calculate_expression : -Real.sqrt 4 + |Real.sqrt 2 - 2| - 202 * 3^0 = -Real.sqrt 2 - 1 := by
  sorry

end calculate_expression_l1142_114266


namespace problem_solution_l1142_114279

theorem problem_solution : (-1/2)⁻¹ - 4 * Real.cos (30 * π / 180) - (π + 2013)^0 + Real.sqrt 12 = -3 := by
  sorry

end problem_solution_l1142_114279


namespace distance_between_points_l1142_114270

def point1 : ℝ × ℝ := (2, -2)
def point2 : ℝ × ℝ := (8, 9)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 157 := by
  sorry

end distance_between_points_l1142_114270


namespace number_problem_l1142_114251

theorem number_problem : ∃ x : ℚ, x / 3 = x - 30 ∧ x = 45 := by
  sorry

end number_problem_l1142_114251


namespace area_between_circles_l1142_114294

/-- The area of the region between two concentric circles with given radii and a tangent chord --/
theorem area_between_circles (R r c : ℝ) (hR : R = 60) (hr : r = 40) (hc : c = 100)
  (h_concentric : R > r) (h_tangent : c^2 = 4 * (R^2 - r^2)) :
  π * (R^2 - r^2) = 2000 * π := by
sorry

end area_between_circles_l1142_114294


namespace square_difference_l1142_114250

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end square_difference_l1142_114250


namespace remaining_box_mass_l1142_114200

/-- Given a list of box masses, prove that the 20 kg box remains in the store when two companies buy five boxes, with one company taking twice the mass of the other. -/
theorem remaining_box_mass (boxes : List ℕ) : boxes = [15, 16, 18, 19, 20, 31] →
  ∃ (company1 company2 : List ℕ),
    (company1.sum + company2.sum = boxes.sum - 20) ∧
    (company2.sum = 2 * company1.sum) ∧
    (company1.length + company2.length = 5) ∧
    (∀ x ∈ company1, x ∈ boxes) ∧
    (∀ x ∈ company2, x ∈ boxes) :=
by sorry

end remaining_box_mass_l1142_114200


namespace friendship_theorem_l1142_114285

-- Define a type for people
def Person : Type := Nat

-- Define the friendship relation
def IsFriend (p q : Person) : Prop := sorry

-- State the theorem
theorem friendship_theorem :
  ∀ (group : Finset Person),
  (Finset.card group = 12) →
  ∃ (A B : Person),
    A ∈ group ∧ B ∈ group ∧ A ≠ B ∧
    ∃ (C D E F G : Person),
      C ∈ group ∧ D ∈ group ∧ E ∈ group ∧ F ∈ group ∧ G ∈ group ∧
      C ≠ A ∧ C ≠ B ∧ D ≠ A ∧ D ≠ B ∧ E ≠ A ∧ E ≠ B ∧ F ≠ A ∧ F ≠ B ∧ G ≠ A ∧ G ≠ B ∧
      C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ E ≠ F ∧ E ≠ G ∧ F ≠ G ∧
      ((IsFriend C A ∧ IsFriend C B) ∨ (¬IsFriend C A ∧ ¬IsFriend C B)) ∧
      ((IsFriend D A ∧ IsFriend D B) ∨ (¬IsFriend D A ∧ ¬IsFriend D B)) ∧
      ((IsFriend E A ∧ IsFriend E B) ∨ (¬IsFriend E A ∧ ¬IsFriend E B)) ∧
      ((IsFriend F A ∧ IsFriend F B) ∨ (¬IsFriend F A ∧ ¬IsFriend F B)) ∧
      ((IsFriend G A ∧ IsFriend G B) ∨ (¬IsFriend G A ∧ ¬IsFriend G B)) :=
by
  sorry


end friendship_theorem_l1142_114285


namespace tangent_line_sum_l1142_114291

/-- Given a function f: ℝ → ℝ with a tangent line y = -x + 6 at x=2, 
    prove that f(2) + f'(2) = 3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x, f 2 + (deriv f 2) * (x - 2) = -x + 6) : 
    f 2 + deriv f 2 = 3 := by
  sorry

end tangent_line_sum_l1142_114291


namespace minimal_fraction_sum_l1142_114297

theorem minimal_fraction_sum (a b : ℕ+) (h : (9:ℚ)/22 < (a:ℚ)/b ∧ (a:ℚ)/b < 5/11) :
  (∃ (c d : ℕ+), (9:ℚ)/22 < (c:ℚ)/d ∧ (c:ℚ)/d < 5/11 ∧ c.val + d.val < a.val + b.val) ∨ (a = 3 ∧ b = 7) :=
sorry

end minimal_fraction_sum_l1142_114297


namespace divisibility_property_l1142_114213

theorem divisibility_property (q : ℕ) (h1 : q > 1) (h2 : Odd q) :
  ∃ k : ℕ, (q + 1) ^ ((q + 1) / 2) = (q + 1) * k := by
  sorry

end divisibility_property_l1142_114213


namespace triangle_inequality_l1142_114212

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^3 + b^3 + c^3 ≤ (a + b + c) * (a*b + b*c + c*a) := by
  sorry

end triangle_inequality_l1142_114212


namespace donut_distribution_l1142_114293

/-- The number of ways to distribute n identical items among k distinct groups,
    where each group must receive at least one item -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- The problem statement -/
theorem donut_distribution : distribute 4 5 = 70 := by
  sorry

end donut_distribution_l1142_114293


namespace letters_with_dot_only_in_given_alphabet_l1142_114271

/-- Represents an alphabet with letters containing dots and/or straight lines -/
structure Alphabet where
  total : ℕ
  both : ℕ
  straight_only : ℕ
  dot_only : ℕ
  all_contain : both + straight_only + dot_only = total

/-- The number of letters containing only a dot in a specific alphabet -/
def letters_with_dot_only (a : Alphabet) : ℕ := a.dot_only

/-- Theorem stating the number of letters with only a dot in the given alphabet -/
theorem letters_with_dot_only_in_given_alphabet :
  ∃ (a : Alphabet), a.total = 60 ∧ a.both = 20 ∧ a.straight_only = 36 ∧ letters_with_dot_only a = 4 := by
  sorry

end letters_with_dot_only_in_given_alphabet_l1142_114271


namespace number_of_sevens_in_Q_l1142_114219

/-- Definition of R_k as an integer consisting of k repetitions of the digit 7 -/
def R (k : ℕ) : ℕ := 7 * ((10^k - 1) / 9)

/-- The quotient of R_16 divided by R_2 -/
def Q : ℕ := R 16 / R 2

/-- Count the number of sevens in a natural number -/
def count_sevens (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of sevens in Q is equal to 2 -/
theorem number_of_sevens_in_Q : count_sevens Q = 2 := by sorry

end number_of_sevens_in_Q_l1142_114219


namespace line_slope_l1142_114211

theorem line_slope (x y : ℝ) : 3 * y + 2 = -4 * x - 9 → (y - (-11/3)) / (x - 0) = -4/3 := by
  sorry

end line_slope_l1142_114211


namespace email_count_theorem_l1142_114278

/-- Calculates the total number of emails received in a month with changing email rates --/
def total_emails (days : ℕ) (initial_rate : ℕ) (increase : ℕ) : ℕ :=
  let half_days := days / 2
  let first_half := initial_rate * half_days
  let second_half := (initial_rate + increase) * (days - half_days)
  first_half + second_half

/-- Theorem stating that given the conditions, the total emails received is 675 --/
theorem email_count_theorem :
  total_emails 30 20 5 = 675 := by
  sorry

end email_count_theorem_l1142_114278


namespace phone_plan_cost_difference_l1142_114268

/-- Calculates the cost difference between Darnell's current phone plan and an alternative plan -/
theorem phone_plan_cost_difference :
  let current_plan_cost : ℚ := 12
  let texts_per_month : ℕ := 60
  let call_minutes_per_month : ℕ := 60
  let alt_plan_text_cost : ℚ := 1
  let alt_plan_text_limit : ℕ := 30
  let alt_plan_call_cost : ℚ := 3
  let alt_plan_call_limit : ℕ := 20
  let alt_plan_text_total : ℚ := (texts_per_month : ℚ) / alt_plan_text_limit * alt_plan_text_cost
  let alt_plan_call_total : ℚ := (call_minutes_per_month : ℚ) / alt_plan_call_limit * alt_plan_call_cost
  let alt_plan_total_cost : ℚ := alt_plan_text_total + alt_plan_call_total
  current_plan_cost - alt_plan_total_cost = 1 :=
by sorry

end phone_plan_cost_difference_l1142_114268


namespace at_least_one_equals_one_iff_sum_gt_product_l1142_114240

theorem at_least_one_equals_one_iff_sum_gt_product (m n : ℕ+) :
  (m = 1 ∨ n = 1) ↔ (m + n : ℝ) > m * n := by sorry

end at_least_one_equals_one_iff_sum_gt_product_l1142_114240


namespace intersection_complement_sets_l1142_114225

open Set

theorem intersection_complement_sets (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let M : Set ℝ := {x | b < x ∧ x < (a + b) / 2}
  let N : Set ℝ := {x | Real.sqrt (a * b) < x ∧ x < a}
  M ∩ (Nᶜ) = {x | b < x ∧ x ≤ Real.sqrt (a * b)} := by
  sorry

end intersection_complement_sets_l1142_114225


namespace union_of_A_and_B_l1142_114248

def A : Set ℕ := {x | (x + 1) * (x - 2) = 0}
def B : Set ℕ := {2, 4, 5}

theorem union_of_A_and_B : A ∪ B = {2, 4, 5} := by
  sorry

end union_of_A_and_B_l1142_114248


namespace quality_difference_confidence_l1142_114231

/-- Data for machine production --/
structure MachineData where
  first_class : ℕ
  second_class : ℕ

/-- Calculate K² statistic --/
def calculate_k_squared (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Theorem stating the confidence level in quality difference --/
theorem quality_difference_confidence
  (machine_a machine_b : MachineData)
  (h_total : machine_a.first_class + machine_a.second_class = 200)
  (h_total_b : machine_b.first_class + machine_b.second_class = 200)
  (h_a_first : machine_a.first_class = 150)
  (h_b_first : machine_b.first_class = 120) :
  calculate_k_squared machine_a.first_class machine_a.second_class
                      machine_b.first_class machine_b.second_class > 6635 / 1000 :=
sorry

end quality_difference_confidence_l1142_114231


namespace cable_cost_calculation_l1142_114205

/-- Calculates the total cost of cable for a neighborhood given the following parameters:
* Number of east-west and north-south streets
* Length of east-west and north-south streets
* Cable required per mile of street
* Cost of regular cable for east-west and north-south streets
* Number of intersections and cost per intersection
* Number of streets requiring higher grade cable and its cost
-/
def total_cable_cost (
  num_ew_streets : ℕ
  ) (num_ns_streets : ℕ
  ) (len_ew_street : ℝ
  ) (len_ns_street : ℝ
  ) (cable_per_mile : ℝ
  ) (cost_ew_cable : ℝ
  ) (cost_ns_cable : ℝ
  ) (num_intersections : ℕ
  ) (cost_per_intersection : ℝ
  ) (num_hg_ew_streets : ℕ
  ) (num_hg_ns_streets : ℕ
  ) (cost_hg_cable : ℝ
  ) : ℝ :=
  let regular_ew_cost := (num_ew_streets : ℝ) * len_ew_street * cable_per_mile * cost_ew_cable
  let regular_ns_cost := (num_ns_streets : ℝ) * len_ns_street * cable_per_mile * cost_ns_cable
  let hg_ew_cost := (num_hg_ew_streets : ℝ) * len_ew_street * cable_per_mile * cost_hg_cable
  let hg_ns_cost := (num_hg_ns_streets : ℝ) * len_ns_street * cable_per_mile * cost_hg_cable
  let intersection_cost := (num_intersections : ℝ) * cost_per_intersection
  regular_ew_cost + regular_ns_cost + hg_ew_cost + hg_ns_cost + intersection_cost

theorem cable_cost_calculation :
  total_cable_cost 18 10 2 4 5 2500 3500 20 5000 3 2 4000 = 1530000 := by
  sorry

end cable_cost_calculation_l1142_114205


namespace only_sunrise_certain_l1142_114245

-- Define the type for events
inductive Event
  | MovieTicket
  | TVAdvertisement
  | Rain
  | Sunrise

-- Define what it means for an event to be certain
def is_certain (e : Event) : Prop :=
  match e with
  | Event.Sunrise => true
  | _ => false

-- Theorem stating that only the sunrise event is certain
theorem only_sunrise_certain :
  ∀ (e : Event), is_certain e ↔ e = Event.Sunrise :=
by
  sorry

end only_sunrise_certain_l1142_114245


namespace total_cost_is_correct_l1142_114223

def tshirt_price : ℝ := 10
def sweater_price : ℝ := 25
def jacket_price : ℝ := 100
def jeans_price : ℝ := 40
def shoes_price : ℝ := 70

def tshirt_discount : ℝ := 0.20
def sweater_discount : ℝ := 0.10
def jacket_discount : ℝ := 0.15
def jeans_discount : ℝ := 0.05
def shoes_discount : ℝ := 0.25

def clothes_tax : ℝ := 0.06
def shoes_tax : ℝ := 0.09

def tshirt_quantity : ℕ := 8
def sweater_quantity : ℕ := 5
def jacket_quantity : ℕ := 3
def jeans_quantity : ℕ := 6
def shoes_quantity : ℕ := 4

def total_cost : ℝ :=
  (tshirt_price * tshirt_quantity * (1 - tshirt_discount) * (1 + clothes_tax)) +
  (sweater_price * sweater_quantity * (1 - sweater_discount) * (1 + clothes_tax)) +
  (jacket_price * jacket_quantity * (1 - jacket_discount) * (1 + clothes_tax)) +
  (jeans_price * jeans_quantity * (1 - jeans_discount) * (1 + clothes_tax)) +
  (shoes_price * shoes_quantity * (1 - shoes_discount) * (1 + shoes_tax))

theorem total_cost_is_correct : total_cost = 927.97 := by
  sorry

end total_cost_is_correct_l1142_114223


namespace paint_area_is_123_l1142_114242

/-- Calculates the area to be painted on a wall with given dimensions and window areas -/
def area_to_paint (wall_height wall_length window1_height window1_width window2_height window2_width : ℝ) : ℝ :=
  let wall_area := wall_height * wall_length
  let window1_area := window1_height * window1_width
  let window2_area := window2_height * window2_width
  wall_area - (window1_area + window2_area)

/-- Theorem: The area to be painted is 123 square feet -/
theorem paint_area_is_123 :
  area_to_paint 10 15 3 5 2 6 = 123 := by
  sorry

#eval area_to_paint 10 15 3 5 2 6

end paint_area_is_123_l1142_114242


namespace perpendicular_implies_parallel_skew_perpendicular_parallel_implies_perpendicular_l1142_114264

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the property of skew lines
variable (skew : Line → Line → Prop)

-- Theorem 1: If a line is perpendicular to two planes, then those planes are parallel
theorem perpendicular_implies_parallel
  (m : Line) (α β : Plane)
  (h1 : perpendicular m α)
  (h2 : perpendicular m β) :
  plane_parallel α β :=
sorry

-- Theorem 2: If two skew lines are each perpendicular to one plane and parallel to the other, 
-- then the planes are perpendicular
theorem skew_perpendicular_parallel_implies_perpendicular
  (m n : Line) (α β : Plane)
  (h1 : skew m n)
  (h2 : perpendicular m α)
  (h3 : parallel m β)
  (h4 : perpendicular n β)
  (h5 : parallel n α) :
  plane_perpendicular α β :=
sorry

end perpendicular_implies_parallel_skew_perpendicular_parallel_implies_perpendicular_l1142_114264


namespace nth_root_sum_theorem_l1142_114284

theorem nth_root_sum_theorem (a : ℝ) (n : ℕ) (hn : n > 1) :
  let f : ℝ → ℝ := λ x => (x^n - a^n)^(1/n) + (2*a^n - x^n)^(1/n)
  ∀ x, f x = a ↔ 
    (a ≠ 0 ∧ 
      ((n % 2 = 1 ∧ (x = a * (2^(1/n)) ∨ x = a)) ∨ 
       (n % 2 = 0 ∧ a > 0 ∧ (x = a * (2^(1/n)) ∨ x = -a * (2^(1/n)) ∨ x = a ∨ x = -a)))) ∨
    (a = 0 ∧ 
      ((n % 2 = 1 ∧ true) ∨ 
       (n % 2 = 0 ∧ x = 0))) :=
by sorry


end nth_root_sum_theorem_l1142_114284


namespace forty_ab_value_l1142_114265

theorem forty_ab_value (a b : ℝ) (h : 4 * a = 5 * b ∧ 5 * b = 30) : 40 * a * b = 1800 := by
  sorry

end forty_ab_value_l1142_114265


namespace sum_of_a_and_b_is_one_max_value_of_m_max_value_of_m_achievable_l1142_114275

-- Define the function f
def f (x a b : ℝ) : ℝ := |x - a| + 2 * |x + b|

-- Theorem 1
theorem sum_of_a_and_b_is_one 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hmin : ∀ x, f x a b ≥ 1) 
  (hmin_exists : ∃ x, f x a b = 1) : 
  a + b = 1 := 
sorry

-- Theorem 2
theorem max_value_of_m 
  (a b m : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a + b = 1) 
  (hm : m ≤ 1/a + 2/b) : 
  m ≤ 3 + 2 * Real.sqrt 2 := 
sorry

-- The maximum value is achievable
theorem max_value_of_m_achievable :
  ∃ m, m = 3 + 2 * Real.sqrt 2 ∧ 
  ∃ a b, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ m ≤ 1/a + 2/b :=
sorry

end sum_of_a_and_b_is_one_max_value_of_m_max_value_of_m_achievable_l1142_114275


namespace regular_polygon_30_degree_exterior_angle_l1142_114290

/-- A regular polygon with exterior angles measuring 30° has 12 sides -/
theorem regular_polygon_30_degree_exterior_angle (n : ℕ) :
  (n > 0) →
  (360 / n = 30) →
  n = 12 := by
  sorry

end regular_polygon_30_degree_exterior_angle_l1142_114290


namespace minimum_value_of_sum_l1142_114209

/-- A positive geometric sequence -/
def PositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ a 1 > 0 ∧ ∀ n, a (n + 1) = a n * q

theorem minimum_value_of_sum (a : ℕ → ℝ) :
  PositiveGeometricSequence a →
  a 4 + a 3 = a 2 + a 1 + 8 →
  ∀ x, a 6 + a 5 ≥ x →
  x ≤ 32 :=
sorry

end minimum_value_of_sum_l1142_114209


namespace sufficient_not_necessary_l1142_114282

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * a^2 < 0 → a < b) ∧
  (∃ a b : ℝ, a < b ∧ ¬((a - b) * a^2 < 0)) :=
by sorry

end sufficient_not_necessary_l1142_114282


namespace gathering_handshakes_l1142_114239

/-- Represents the number of handshakes in a gathering with specific group dynamics -/
def number_of_handshakes (total_people : ℕ) (group1_size : ℕ) (group2_size : ℕ) (group3_size : ℕ) (known_by_group3 : ℕ) : ℕ :=
  let group2_handshakes := group2_size * (total_people - group2_size)
  let group3_handshakes := group3_size * (group1_size - known_by_group3 + group2_size)
  (group2_handshakes + group3_handshakes) / 2

/-- The theorem states that for the given group sizes and dynamics, the number of handshakes is 210 -/
theorem gathering_handshakes :
  number_of_handshakes 35 25 5 5 18 = 210 := by
  sorry

end gathering_handshakes_l1142_114239


namespace exponential_inequality_l1142_114226

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  2^a + 2*a = 2^b + 3*b → a > b := by
  sorry

end exponential_inequality_l1142_114226


namespace curve_W_and_rectangle_perimeter_l1142_114298

-- Define the curve W
def W (x y : ℝ) : Prop := |y| = Real.sqrt (x^2 + (y - 1/2)^2)

-- Define a rectangle with three vertices on W
def RectangleOnW (A B C D : ℝ × ℝ) : Prop :=
  W A.1 A.2 ∧ W B.1 B.2 ∧ W C.1 C.2 ∧
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0 ∧
  (A.1 - D.1) * (C.1 - D.1) + (A.2 - D.2) * (C.2 - D.2) = 0

-- Theorem statement
theorem curve_W_and_rectangle_perimeter 
  (A B C D : ℝ × ℝ) (h : RectangleOnW A B C D) :
  (∀ x y : ℝ, W x y ↔ y = x^2 + 1/4) ∧ 
  2 * (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) + 
       Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)) > 3 * Real.sqrt 3 := by
  sorry

end curve_W_and_rectangle_perimeter_l1142_114298


namespace fraction_product_simplification_l1142_114203

theorem fraction_product_simplification :
  (20 : ℚ) / 21 * 35 / 48 * 84 / 55 * 11 / 40 = 1 / 6 := by
  sorry

end fraction_product_simplification_l1142_114203


namespace forty_ab_over_c_value_l1142_114238

theorem forty_ab_over_c_value (a b c : ℝ) 
  (eq1 : 4 * a = 5 * b)
  (eq2 : 5 * b = 30)
  (eq3 : a + b + c = 15) :
  40 * a * b / c = 1200 := by
  sorry

end forty_ab_over_c_value_l1142_114238


namespace list_number_fraction_l1142_114234

theorem list_number_fraction (n : ℕ) (S : ℝ) (h1 : n > 0) (h2 : S ≥ 0) : 
  n = 3 * (S / (n - 1)) → n / (S + n) = 3 / (n + 2) :=
by sorry

end list_number_fraction_l1142_114234


namespace parametric_eq_normal_l1142_114288

/-- The parametric equation of a plane -/
def plane_parametric (s t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 2*s - 3*t, 4 - s + 2*t, 1 - 3*s - t)

/-- The normal form equation of a plane -/
def plane_normal (x y z : ℝ) : Prop :=
  5*x + 11*y + 7*z - 61 = 0

/-- Theorem stating that the parametric and normal form equations represent the same plane -/
theorem parametric_eq_normal :
  ∀ (x y z : ℝ), (∃ (s t : ℝ), plane_parametric s t = (x, y, z)) ↔ plane_normal x y z :=
by sorry

end parametric_eq_normal_l1142_114288


namespace largest_negative_root_l1142_114249

noncomputable def α : ℝ := Real.arctan (4 / 13)
noncomputable def β : ℝ := Real.arctan (8 / 11)

def equation (x : ℝ) : Prop :=
  4 * Real.sin (3 * x) + 13 * Real.cos (3 * x) = 8 * Real.sin x + 11 * Real.cos x

theorem largest_negative_root :
  ∃ (x : ℝ), x < 0 ∧ equation x ∧ 
  ∀ (y : ℝ), y < 0 → equation y → y ≤ x ∧
  x = (α - β) / 2 :=
sorry

end largest_negative_root_l1142_114249


namespace sum_of_nth_row_sum_of_100th_row_l1142_114286

/-- The sum of numbers in the nth row of the triangular array -/
def f (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2 * f (n - 1) + 2 * n

/-- Theorem: The closed form of the sum of numbers in the nth row -/
theorem sum_of_nth_row (n : ℕ) : f n = 3 * 2^(n-1) - 2 * n := by
  sorry

/-- Corollary: The sum of numbers in the 100th row -/
theorem sum_of_100th_row : f 100 = 3 * 2^99 - 200 := by
  sorry

end sum_of_nth_row_sum_of_100th_row_l1142_114286


namespace probability_of_science_second_draw_l1142_114287

/-- Represents the type of questions --/
inductive QuestionType
| Science
| LiberalArts

/-- Represents the state of the questions after the first draw --/
structure QuestionState :=
  (total : Nat)
  (science : Nat)
  (liberal_arts : Nat)

/-- The initial state of questions --/
def initial_state : QuestionState :=
  ⟨5, 3, 2⟩

/-- The state after drawing a science question --/
def after_first_draw (s : QuestionState) : QuestionState :=
  ⟨s.total - 1, s.science - 1, s.liberal_arts⟩

/-- The probability of drawing a science question on the second draw --/
def prob_science_second_draw (s : QuestionState) : Rat :=
  s.science / s.total

theorem probability_of_science_second_draw :
  prob_science_second_draw (after_first_draw initial_state) = 1/2 := by
  sorry

end probability_of_science_second_draw_l1142_114287


namespace range_of_a_in_linear_program_l1142_114216

/-- The range of values for a given the specified constraints and maximum point -/
theorem range_of_a_in_linear_program (x y a : ℝ) : 
  (1 ≤ x + y) → (x + y ≤ 4) → 
  (-2 ≤ x - y) → (x - y ≤ 2) → 
  (a > 0) →
  (∀ x' y', (1 ≤ x' + y') → (x' + y' ≤ 4) → (-2 ≤ x' - y') → (x' - y' ≤ 2) → 
    (a * x' + y' ≤ a * x + y)) →
  (x = 3 ∧ y = 1) →
  a > 1 := by sorry

end range_of_a_in_linear_program_l1142_114216


namespace coefficient_x_squared_expansion_l1142_114257

/-- The coefficient of x^2 in the expansion of (1/x - √x)^10 is 45 -/
theorem coefficient_x_squared_expansion (x : ℝ) : 
  (Finset.range 11).sum (fun k => (-1)^k * (Nat.choose 10 k : ℝ) * x^((3*k:ℤ)/2 - 5)) = 45 := by
  sorry

end coefficient_x_squared_expansion_l1142_114257


namespace solution_set_inequality_l1142_114258

theorem solution_set_inequality (x : ℝ) : 
  x * (x + 3) ≥ 0 ↔ x ≥ 0 ∨ x ≤ -3 := by sorry

end solution_set_inequality_l1142_114258


namespace perfect_square_condition_l1142_114292

theorem perfect_square_condition (k : ℤ) : 
  (∀ x : ℤ, ∃ y : ℤ, x^2 - 2*(k+1)*x + 4 = y^2) → (k = -3 ∨ k = 1) :=
by sorry

end perfect_square_condition_l1142_114292


namespace central_angle_of_sector_l1142_114241

-- Define the sector
structure Sector where
  radius : ℝ
  area : ℝ

-- Define the theorem
theorem central_angle_of_sector (s : Sector) (h1 : s.radius = 2) (h2 : s.area = 8) :
  (2 * s.area) / (s.radius ^ 2) = 4 := by
  sorry

end central_angle_of_sector_l1142_114241


namespace jeffrey_steps_to_mailbox_l1142_114283

/-- Represents Jeffrey's walking pattern -/
structure WalkingPattern where
  forward : ℕ
  backward : ℕ

/-- Calculates the total steps taken given a walking pattern and distance -/
def totalSteps (pattern : WalkingPattern) (distance : ℕ) : ℕ :=
  let effectiveStep := pattern.forward - pattern.backward
  let cycles := distance / effectiveStep
  cycles * (pattern.forward + pattern.backward)

/-- Theorem: Jeffrey's total steps to the mailbox -/
theorem jeffrey_steps_to_mailbox :
  let pattern : WalkingPattern := { forward := 3, backward := 2 }
  let distance : ℕ := 66
  totalSteps pattern distance = 330 := by
  sorry


end jeffrey_steps_to_mailbox_l1142_114283


namespace minimum_marbles_to_add_proof_minimum_marbles_l1142_114252

theorem minimum_marbles_to_add (initial_marbles : Nat) (people : Nat) : Nat :=
  let additional_marbles := people - initial_marbles % people
  if additional_marbles = people then 0 else additional_marbles

theorem proof_minimum_marbles :
  minimum_marbles_to_add 62 8 = 2 ∧
  (62 + minimum_marbles_to_add 62 8) % 8 = 0 ∧
  ∀ x : Nat, x < minimum_marbles_to_add 62 8 → (62 + x) % 8 ≠ 0 :=
by sorry

end minimum_marbles_to_add_proof_minimum_marbles_l1142_114252


namespace smallest_sticker_count_l1142_114208

theorem smallest_sticker_count (S : ℕ) (h1 : S > 3) 
  (h2 : S % 5 = 3) (h3 : S % 11 = 3) (h4 : S % 13 = 3) : 
  S ≥ 718 ∧ ∃ (T : ℕ), T = 718 ∧ T % 5 = 3 ∧ T % 11 = 3 ∧ T % 13 = 3 := by
  sorry

end smallest_sticker_count_l1142_114208


namespace root_sum_squares_l1142_114243

theorem root_sum_squares (p q r : ℝ) : 
  (p^3 - 24*p^2 + 50*p - 8 = 0) →
  (q^3 - 24*q^2 + 50*q - 8 = 0) →
  (r^3 - 24*r^2 + 50*r - 8 = 0) →
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 1052 := by
sorry

end root_sum_squares_l1142_114243


namespace minimized_surface_area_sum_l1142_114276

/-- Represents a tetrahedron with specific properties -/
structure Tetrahedron where
  /-- One face has sides of length 3, 4, and 5 -/
  base_sides : Fin 3 → ℝ
  base_sides_values : base_sides = ![3, 4, 5]
  /-- The volume of the tetrahedron is 24 -/
  volume : ℝ
  volume_value : volume = 24

/-- Represents the surface area of the tetrahedron in the form a√b + c -/
structure SurfaceArea where
  a : ℕ
  b : ℕ
  c : ℕ
  /-- b is not divisible by the square of any prime -/
  b_squarefree : ∀ p : ℕ, Prime p → ¬(p^2 ∣ b)

/-- The main theorem stating the sum of a, b, and c for the minimized surface area -/
theorem minimized_surface_area_sum (t : Tetrahedron) :
  ∃ (sa : SurfaceArea), (∀ other_sa : SurfaceArea, 
    sa.a * Real.sqrt sa.b + sa.c ≤ other_sa.a * Real.sqrt other_sa.b + other_sa.c) → 
    sa.a + sa.b + sa.c = 157 := by
  sorry

end minimized_surface_area_sum_l1142_114276


namespace unique_solution_for_equation_l1142_114222

/-- Represents a three-digit number ABC --/
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- Represents a two-digit number AB --/
def two_digit_number (a b : ℕ) : ℕ := 10 * a + b

/-- Predicate to check if a number is a single digit --/
def is_single_digit (n : ℕ) : Prop := n < 10

/-- Predicate to check if four numbers are distinct --/
def are_distinct (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem unique_solution_for_equation :
  ∃! (a b c d : ℕ),
    is_single_digit a ∧
    is_single_digit b ∧
    is_single_digit c ∧
    is_single_digit d ∧
    are_distinct a b c d ∧
    three_digit_number a b c * two_digit_number a b + c * d = 2017 ∧
    two_digit_number a b = 14 :=
by sorry

end unique_solution_for_equation_l1142_114222


namespace cube_root_equal_self_l1142_114224

theorem cube_root_equal_self : {x : ℝ | x = x^(1/3)} = {-1, 0, 1} := by sorry

end cube_root_equal_self_l1142_114224


namespace product_of_sums_geq_product_l1142_114232

theorem product_of_sums_geq_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
by sorry

end product_of_sums_geq_product_l1142_114232


namespace two_numbers_difference_l1142_114206

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 391) :
  |x - y| = 6 := by
  sorry

end two_numbers_difference_l1142_114206


namespace gcd_factorial_eight_and_factorial_six_squared_l1142_114201

theorem gcd_factorial_eight_and_factorial_six_squared :
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 11520 := by
  sorry

end gcd_factorial_eight_and_factorial_six_squared_l1142_114201


namespace factorial_five_equals_120_l1142_114218

theorem factorial_five_equals_120 : 5 * 4 * 3 * 2 * 1 = 120 := by
  sorry

end factorial_five_equals_120_l1142_114218


namespace sin_cos_difference_equals_half_l1142_114272

theorem sin_cos_difference_equals_half : 
  Real.sin (43 * π / 180) * Real.cos (13 * π / 180) - 
  Real.cos (43 * π / 180) * Real.sin (13 * π / 180) = 1/2 := by
sorry

end sin_cos_difference_equals_half_l1142_114272


namespace angle_halving_l1142_114260

-- Define what it means for an angle to be in the fourth quadrant
def in_fourth_quadrant (θ : Real) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi - Real.pi / 2 < θ ∧ θ < 2 * k * Real.pi

-- Define what it means for an angle to be in the first or third quadrant
def in_first_or_third_quadrant (θ : Real) : Prop :=
  ∃ k : ℤ, k * Real.pi < θ ∧ θ < k * Real.pi + Real.pi / 2

theorem angle_halving (θ : Real) :
  in_fourth_quadrant θ → in_first_or_third_quadrant (-θ/2) :=
by sorry

end angle_halving_l1142_114260


namespace andrew_fruit_purchase_l1142_114253

/-- The total cost of a fruit purchase, including tax -/
def totalCost (grapeQuantity mangoQuantity grapePrice mangoPrice grapeTaxRate mangoTaxRate : ℚ) : ℚ :=
  let grapeCost := grapeQuantity * grapePrice
  let mangoCost := mangoQuantity * mangoPrice
  let grapeTax := grapeCost * grapeTaxRate
  let mangoTax := mangoCost * mangoTaxRate
  grapeCost + mangoCost + grapeTax + mangoTax

/-- The theorem stating the total cost of Andrew's fruit purchase -/
theorem andrew_fruit_purchase :
  totalCost 8 9 70 55 (8/100) (11/100) = 1154.25 := by
  sorry

end andrew_fruit_purchase_l1142_114253


namespace intersection_eq_union_implies_a_eq_3_intersection_eq_nonempty_implies_a_eq_neg_5_div_2_l1142_114220

-- Define the sets A, B, and C
def A (a : ℝ) := {x : ℝ | x^2 + (4 - a^2) * x + a + 3 = 0}
def B := {x : ℝ | x^2 - 5 * x + 6 = 0}
def C := {x : ℝ | 2 * x^2 - 5 * x + 2 = 0}

-- Theorem 1
theorem intersection_eq_union_implies_a_eq_3 :
  ∃ a : ℝ, (A a) ∩ B = (A a) ∪ B → a = 3 := by sorry

-- Theorem 2
theorem intersection_eq_nonempty_implies_a_eq_neg_5_div_2 :
  ∃ a : ℝ, (A a) ∩ B = (A a) ∩ C ∧ (A a) ∩ B ≠ ∅ → a = -5/2 := by sorry

end intersection_eq_union_implies_a_eq_3_intersection_eq_nonempty_implies_a_eq_neg_5_div_2_l1142_114220


namespace binomial_9_choose_5_l1142_114256

theorem binomial_9_choose_5 : Nat.choose 9 5 = 126 := by sorry

end binomial_9_choose_5_l1142_114256


namespace track_length_is_400_l1142_114233

/-- Represents a circular running track -/
structure Track :=
  (length : ℝ)

/-- Represents a runner on the track -/
structure Runner :=
  (speed : ℝ)
  (initialPosition : ℝ)

/-- Represents a meeting between two runners -/
structure Meeting :=
  (position : ℝ)
  (time : ℝ)

/-- The scenario of two runners on a circular track -/
def runningScenario (t : Track) (r1 r2 : Runner) (m1 m2 : Meeting) : Prop :=
  r1.initialPosition = 0 ∧
  r2.initialPosition = t.length / 2 ∧
  r1.speed > 0 ∧
  r2.speed < 0 ∧
  m1.position = 100 ∧
  m2.position - m1.position = 150 ∧
  m1.time * r1.speed = 100 ∧
  m1.time * r2.speed = t.length / 2 - 100 ∧
  m2.time * r1.speed = t.length / 2 - 50 ∧
  m2.time * r2.speed = t.length / 2 + 50

theorem track_length_is_400 (t : Track) (r1 r2 : Runner) (m1 m2 : Meeting) :
  runningScenario t r1 r2 m1 m2 → t.length = 400 :=
by sorry

end track_length_is_400_l1142_114233


namespace distribute_fraction_over_parentheses_l1142_114262

theorem distribute_fraction_over_parentheses (x : ℝ) : (1 / 3) * (6 * x - 3) = 2 * x - 1 := by
  sorry

end distribute_fraction_over_parentheses_l1142_114262


namespace quadratic_factorization_l1142_114274

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end quadratic_factorization_l1142_114274


namespace complex_equality_implies_sum_l1142_114217

theorem complex_equality_implies_sum (a b : ℝ) :
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (2 + Complex.I) * (1 - b * Complex.I) = a + Complex.I →
  a + b = 2 := by
  sorry

end complex_equality_implies_sum_l1142_114217


namespace solution_set_equivalence_l1142_114204

/-- Given that the solution set of ax² + 2x + c < 0 is (-∞, -1/3) ∪ (1/2, +∞),
    prove that the solution set of cx² + 2x + a ≤ 0 is [-3, 2]. -/
theorem solution_set_equivalence 
  (h : ∀ x : ℝ, (ax^2 + 2*x + c < 0) ↔ (x < -1/3 ∨ x > 1/2))
  (a c : ℝ) :
  ∀ x : ℝ, (c*x^2 + 2*x + a ≤ 0) ↔ (-3 ≤ x ∧ x ≤ 2) :=
by sorry

end solution_set_equivalence_l1142_114204


namespace perpendicular_vectors_l1142_114273

/-- Given vectors a and b in ℝ², if a + k * b is perpendicular to a - b, then k = 11/20 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (2, -1)) 
  (h2 : b = (-1, 4)) 
  (h3 : (a.1 + k * b.1, a.2 + k * b.2) • (a.1 - b.1, a.2 - b.2) = 0) : 
  k = 11/20 := by sorry

end perpendicular_vectors_l1142_114273


namespace ball_max_height_l1142_114280

theorem ball_max_height :
  let f : ℝ → ℝ := fun t ↦ -5 * t^2 + 20 * t + 10
  ∃ (max : ℝ), max = 30 ∧ ∀ t, f t ≤ max :=
by
  sorry

end ball_max_height_l1142_114280


namespace simplify_and_evaluate_l1142_114237

theorem simplify_and_evaluate (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (1 - x / (x + 1)) / ((x^2 - 2*x + 1) / (x^2 - 1)) = 1 / (x - 1) :=
by sorry

end simplify_and_evaluate_l1142_114237


namespace hexagon_sixth_angle_l1142_114269

/-- The sum of interior angles of a hexagon is 720 degrees -/
def hexagon_angle_sum : ℝ := 720

/-- The given angles of the hexagon -/
def given_angles : List ℝ := [135, 105, 87, 120, 78]

/-- Theorem: In a hexagon where five of the interior angles measure 135°, 105°, 87°, 120°, and 78°, the sixth angle measures 195°. -/
theorem hexagon_sixth_angle : 
  List.sum given_angles + 195 = hexagon_angle_sum := by
  sorry

end hexagon_sixth_angle_l1142_114269
