import Mathlib

namespace flagpole_height_l1781_178105

theorem flagpole_height (A B C D E : ℝ × ℝ) : 
  let AC : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let AD : ℝ := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  let DE : ℝ := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
  let AB : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  (A.2 = 0 ∧ B.2 > 0 ∧ C.2 = 0 ∧ D.2 = 0 ∧ E.2 > 0) → -- Points on x-axis or above
  (B.1 = A.1) → -- AB is vertical
  (AC = 4) → -- Wire length on ground
  (AD = 3) → -- Tom's distance from pole
  (DE = 1.8) → -- Tom's height
  ((E.1 - D.1) * (C.1 - A.1) + (E.2 - D.2) * (C.2 - A.2) = 0) → -- DE perpendicular to AC
  (AB = 7.2) -- Flagpole height
  := by sorry

end flagpole_height_l1781_178105


namespace magic_trick_result_l1781_178175

theorem magic_trick_result (x : ℚ) : ((2 * x + 8) / 4) - (x / 2) = 2 := by
  sorry

end magic_trick_result_l1781_178175


namespace prob_three_l_is_one_fiftyfifth_l1781_178145

/-- The number of cards in the deck -/
def total_cards : ℕ := 12

/-- The number of L cards in the deck -/
def l_cards : ℕ := 4

/-- The number of cards drawn -/
def cards_drawn : ℕ := 3

/-- The probability of drawing 3 L cards without replacement -/
def prob_three_l : ℚ := (l_cards.choose cards_drawn : ℚ) / (total_cards.choose cards_drawn : ℚ)

theorem prob_three_l_is_one_fiftyfifth : prob_three_l = 1 / 55 := by
  sorry

end prob_three_l_is_one_fiftyfifth_l1781_178145


namespace real_roots_iff_a_leq_two_l1781_178141

theorem real_roots_iff_a_leq_two (a : ℝ) : 
  (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ a ≤ 2 :=
sorry

end real_roots_iff_a_leq_two_l1781_178141


namespace probability_of_two_packages_l1781_178172

/-- The number of tablets in a new package -/
def n : ℕ := 10

/-- The probability of having exactly two packages of tablets -/
def probability_two_packages : ℚ := (2^n - 1) / (2^(n-1) * n)

/-- Theorem stating the probability of having exactly two packages of tablets -/
theorem probability_of_two_packages :
  probability_two_packages = (2^n - 1) / (2^(n-1) * n) := by sorry

end probability_of_two_packages_l1781_178172


namespace f_integer_iff_l1781_178160

def f (x : ℝ) : ℝ := (1 + x) ^ (1/3) + (3 - x) ^ (1/3)

theorem f_integer_iff (x : ℝ) : 
  ∃ (n : ℤ), f x = n ↔ 
  (x = 1 + Real.sqrt 5 ∨ 
   x = 1 - Real.sqrt 5 ∨ 
   x = 1 + (10/9) * Real.sqrt 3 ∨ 
   x = 1 - (10/9) * Real.sqrt 3) :=
sorry

end f_integer_iff_l1781_178160


namespace negation_of_existence_l1781_178168

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) :=
by sorry

end negation_of_existence_l1781_178168


namespace workday_end_time_l1781_178146

-- Define a custom time type
structure Time where
  hours : ℕ
  minutes : ℕ
  deriving Repr

def Time.toMinutes (t : Time) : ℕ :=
  t.hours * 60 + t.minutes

def Time.addMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.toMinutes + m
  { hours := totalMinutes / 60,
    minutes := totalMinutes % 60 }

def Time.subtractMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.toMinutes - m
  { hours := totalMinutes / 60,
    minutes := totalMinutes % 60 }

def Time.differenceInMinutes (t1 t2 : Time) : ℕ :=
  if t1.toMinutes ≥ t2.toMinutes then
    t1.toMinutes - t2.toMinutes
  else
    t2.toMinutes - t1.toMinutes

theorem workday_end_time 
  (total_work_time : ℕ)
  (lunch_break : ℕ)
  (start_time : Time)
  (lunch_time : Time)
  (h1 : total_work_time = 8 * 60)  -- 8 hours in minutes
  (h2 : lunch_break = 30)  -- 30 minutes
  (h3 : start_time = { hours := 7, minutes := 0 })  -- 7:00 AM
  (h4 : lunch_time = { hours := 11, minutes := 30 })  -- 11:30 AM
  : Time.addMinutes lunch_time (total_work_time - Time.differenceInMinutes lunch_time start_time + lunch_break) = { hours := 15, minutes := 30 } :=
by sorry

end workday_end_time_l1781_178146


namespace cave_door_weight_l1781_178179

/-- The weight already on the switch in pounds -/
def initial_weight : ℕ := 234

/-- The additional weight needed in pounds -/
def additional_weight : ℕ := 478

/-- The total weight needed to open the cave doors in pounds -/
def total_weight : ℕ := initial_weight + additional_weight

/-- Theorem stating that the total weight needed to open the cave doors is 712 pounds -/
theorem cave_door_weight : total_weight = 712 := by
  sorry

end cave_door_weight_l1781_178179


namespace least_subtraction_for_divisibility_l1781_178147

theorem least_subtraction_for_divisibility : 
  ∃ (n : ℕ), n = 1 ∧ 
  (∀ (m : ℕ), (13294 - m) % 97 = 0 → m ≥ n) ∧
  (13294 - n) % 97 = 0 := by
  sorry

end least_subtraction_for_divisibility_l1781_178147


namespace notebook_purchase_possible_l1781_178126

theorem notebook_purchase_possible : ∃ x y : ℤ, 16 * x + 27 * y = 1 := by
  sorry

end notebook_purchase_possible_l1781_178126


namespace age_difference_l1781_178184

/-- Given the ages of Taehyung, his father, and his mother, prove that the age difference
between the father and mother is equal to Taehyung's age. -/
theorem age_difference (taehyung_age : ℕ) (father_age : ℕ) (mother_age : ℕ)
  (h1 : taehyung_age = 9)
  (h2 : father_age = 5 * taehyung_age)
  (h3 : mother_age = 4 * taehyung_age) :
  father_age - mother_age = taehyung_age :=
by sorry

end age_difference_l1781_178184


namespace geometric_sequence_fifth_term_l1781_178167

/-- A geometric sequence with common ratio r -/
def geometricSequence (r : ℝ) : ℕ → ℝ := fun n => r^(n-1)

/-- The third term of a geometric sequence -/
def a₃ (r : ℝ) : ℝ := geometricSequence r 3

/-- The seventh term of a geometric sequence -/
def a₇ (r : ℝ) : ℝ := geometricSequence r 7

/-- The fifth term of a geometric sequence -/
def a₅ (r : ℝ) : ℝ := geometricSequence r 5

theorem geometric_sequence_fifth_term (r : ℝ) :
  (a₃ r)^2 - 4*(a₃ r) + 3 = 0 ∧ (a₇ r)^2 - 4*(a₇ r) + 3 = 0 → a₅ r = Real.sqrt 3 :=
by
  sorry

end geometric_sequence_fifth_term_l1781_178167


namespace quadratic_minimum_l1781_178194

/-- The quadratic function f(x) = x^2 - 12x + 36 -/
def f (x : ℝ) : ℝ := x^2 - 12*x + 36

theorem quadratic_minimum :
  ∃ (x_min : ℝ), f x_min = 0 ∧ ∀ (x : ℝ), f x ≥ 0 :=
by
  -- The proof goes here
  sorry

end quadratic_minimum_l1781_178194


namespace tree_space_calculation_l1781_178185

/-- The space taken up by each tree in square feet -/
def tree_space : ℝ := 1

/-- The total length of the road in feet -/
def road_length : ℝ := 148

/-- The number of trees to be planted -/
def num_trees : ℕ := 8

/-- The space between each tree in feet -/
def space_between : ℝ := 20

theorem tree_space_calculation :
  tree_space * num_trees + space_between * (num_trees - 1) = road_length := by
  sorry

end tree_space_calculation_l1781_178185


namespace sports_parade_children_count_l1781_178157

theorem sports_parade_children_count :
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 150 ∧ n % 8 = 5 ∧ n % 10 = 7 ∧ n = 125 := by
  sorry

end sports_parade_children_count_l1781_178157


namespace quadratic_inequality_solution_set_l1781_178123

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo 1 2 = {x | a * x^2 + b * x + c < 0}) : 
  Set.Ioo (1/2) 1 = {x | c * x^2 + b * x + a < 0} :=
sorry

end quadratic_inequality_solution_set_l1781_178123


namespace min_value_function_max_value_constraint_l1781_178115

-- Problem 1
theorem min_value_function (x : ℝ) (h : x > 1/2) :
  (∀ z, z > 1/2 → 2*z + 4/(2*z - 1) ≥ 2*x + 4/(2*x - 1)) →
  2*x + 4/(2*x - 1) = 5 :=
sorry

-- Problem 2
theorem max_value_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 3) :
  (∀ z w : ℝ, z > 0 → w > 0 → z + w = 3 → x*y + 2*x + y ≥ z*w + 2*z + w) →
  x*y + 2*x + y = 7 :=
sorry

end min_value_function_max_value_constraint_l1781_178115


namespace number_of_roots_l1781_178171

/-- The number of real roots of a quadratic equation (m-5)x^2 - 2(m+2)x + m = 0,
    given that mx^2 - 2(m+2)x + m + 5 = 0 has no real roots -/
theorem number_of_roots (m : ℝ) 
  (h : ∀ x : ℝ, m * x^2 - 2*(m+2)*x + m + 5 ≠ 0) :
  (∃! x : ℝ, (m-5) * x^2 - 2*(m+2)*x + m = 0) ∨ 
  (∃ x y : ℝ, x ≠ y ∧ (m-5) * x^2 - 2*(m+2)*x + m = 0 ∧ (m-5) * y^2 - 2*(m+2)*y + m = 0) :=
sorry

end number_of_roots_l1781_178171


namespace min_a_value_l1781_178199

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a - 1 ≤ 0}

-- Define the set B
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}

-- Theorem statement
theorem min_a_value (a : ℝ) : 
  (A a ∪ B = A a) → a ≥ 4 :=
sorry

end min_a_value_l1781_178199


namespace complement_of_event_A_l1781_178180

/-- The total number of products in the batch -/
def total_products : ℕ := 10

/-- Event A: There are at least 2 defective products -/
def event_A (defective : ℕ) : Prop := defective ≥ 2

/-- The complement of event A -/
def complement_A (defective : ℕ) : Prop := defective ≤ 1

/-- Theorem stating that the complement of event A is correctly defined -/
theorem complement_of_event_A :
  ∀ defective : ℕ, defective ≤ total_products →
    (¬ event_A defective ↔ complement_A defective) :=
by sorry

end complement_of_event_A_l1781_178180


namespace other_divisor_of_h_l1781_178149

def is_divisor (d n : ℕ) : Prop := n % d = 0

theorem other_divisor_of_h (h a b c : ℕ) : 
  h > 0 →
  is_divisor 225 h →
  h = 2^a * 3^b * 5^c →
  a > 0 →
  b > 0 →
  c > 0 →
  a + b + c ≥ 8 →
  (∀ a' b' c' : ℕ, a' > 0 → b' > 0 → c' > 0 → a' + b' + c' < a + b + c → ¬(h = 2^a' * 3^b' * 5^c')) →
  ∃ d : ℕ, d ≠ 225 ∧ is_divisor d h ∧ d = 16 :=
by sorry

end other_divisor_of_h_l1781_178149


namespace distance_between_points_l1781_178133

theorem distance_between_points : 
  ∀ (A B : ℝ), A = -4 ∧ B = 2 → |B - A| = |2 - (-4)| := by sorry

end distance_between_points_l1781_178133


namespace a_in_M_necessary_not_sufficient_for_a_in_N_l1781_178198

def M : Set ℝ := {x | 0 < x ∧ x ≤ 4}
def N : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) :=
by sorry

end a_in_M_necessary_not_sufficient_for_a_in_N_l1781_178198


namespace domain_of_composite_function_l1781_178142

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x > 1}

-- State the theorem
theorem domain_of_composite_function :
  {x : ℝ | ∃ y ∈ domain_f, y = 2*x + 1} = {x : ℝ | x > 0} := by sorry

end domain_of_composite_function_l1781_178142


namespace peas_corn_difference_l1781_178176

/-- The number of cans of peas Beth bought -/
def peas : ℕ := 35

/-- The number of cans of corn Beth bought -/
def corn : ℕ := 10

/-- The difference between the number of cans of peas and twice the number of cans of corn -/
def difference : ℕ := peas - 2 * corn

theorem peas_corn_difference : difference = 15 := by
  sorry

end peas_corn_difference_l1781_178176


namespace point_same_side_as_origin_l1781_178106

def same_side_as_origin (x y : ℝ) : Prop :=
  (3 * x + 2 * y + 5) * (3 * 0 + 2 * 0 + 5) > 0

theorem point_same_side_as_origin :
  same_side_as_origin (-3) 4 := by sorry

end point_same_side_as_origin_l1781_178106


namespace polar_coords_of_negative_one_negative_one_l1781_178173

/-- Prove that the polar coordinates of the point P(-1, -1) are (√2, 5π/4) -/
theorem polar_coords_of_negative_one_negative_one :
  let x : ℝ := -1
  let y : ℝ := -1
  let ρ : ℝ := Real.sqrt 2
  let θ : ℝ := 5 * Real.pi / 4
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ := by
  sorry

end polar_coords_of_negative_one_negative_one_l1781_178173


namespace complex_modulus_problem_l1781_178188

theorem complex_modulus_problem (z : ℂ) (h : (1 + z) / (1 - z) = Complex.I) : 
  Complex.abs z = Real.sqrt 5 / 2 := by
sorry

end complex_modulus_problem_l1781_178188


namespace remainder_problem_l1781_178189

theorem remainder_problem (L S R : ℕ) : 
  L - S = 2395 → 
  S = 476 → 
  L = 6 * S + R → 
  R < S → 
  R = 15 := by
sorry

end remainder_problem_l1781_178189


namespace unique_quadratic_with_prime_roots_l1781_178191

theorem unique_quadratic_with_prime_roots (a : ℝ) (ha : a > 0) :
  (∃! k : ℝ, ∃ p q : ℕ, 
    Prime p ∧ Prime q ∧ 
    (∀ x : ℝ, x^2 + (k^2 + a*k)*x + (1999 + k^2 + a*k) = 0 ↔ x = p ∨ x = q)) ↔ 
  a = 2 * Real.sqrt 502 :=
sorry

end unique_quadratic_with_prime_roots_l1781_178191


namespace power_sum_inequality_l1781_178136

theorem power_sum_inequality (a b c : ℝ) (n : ℕ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hn : 0 < n) :
  a^n + b^n + c^n ≥ a * b^(n-1) + b * c^(n-1) + c * a^(n-1) := by
  sorry

end power_sum_inequality_l1781_178136


namespace time_difference_to_halfway_point_l1781_178101

def danny_time : ℝ := 29

theorem time_difference_to_halfway_point :
  let steve_time := 2 * danny_time
  let danny_halfway_time := danny_time / 2
  let steve_halfway_time := steve_time / 2
  steve_halfway_time - danny_halfway_time = 14.5 := by
  sorry

end time_difference_to_halfway_point_l1781_178101


namespace bales_in_barn_l1781_178120

/-- The number of bales originally in the barn -/
def original_bales : ℕ := sorry

/-- The number of bales Keith stacked today -/
def keith_bales : ℕ := 67

/-- The total number of bales in the barn now -/
def total_bales : ℕ := 89

theorem bales_in_barn :
  original_bales + keith_bales = total_bales ∧ original_bales = 22 :=
by sorry

end bales_in_barn_l1781_178120


namespace ticket_price_is_25_l1781_178154

-- Define the number of attendees for the first show
def first_show_attendees : ℕ := 200

-- Define the number of attendees for the second show
def second_show_attendees : ℕ := 3 * first_show_attendees

-- Define the total revenue
def total_revenue : ℕ := 20000

-- Define the ticket price
def ticket_price : ℚ := total_revenue / (first_show_attendees + second_show_attendees)

-- Theorem statement
theorem ticket_price_is_25 : ticket_price = 25 := by
  sorry

end ticket_price_is_25_l1781_178154


namespace largest_divisor_is_24_l1781_178107

/-- The set of all integer tuples (a, b, c, d, e, f) satisfying a^2 + b^2 + c^2 + d^2 + e^2 = f^2 -/
def S : Set (ℤ × ℤ × ℤ × ℤ × ℤ × ℤ) :=
  {t | let (a, b, c, d, e, f) := t
       a^2 + b^2 + c^2 + d^2 + e^2 = f^2}

/-- The property that k divides the product of all elements in a tuple -/
def DividesTuple (k : ℤ) (t : ℤ × ℤ × ℤ × ℤ × ℤ × ℤ) : Prop :=
  let (a, b, c, d, e, f) := t
  k ∣ (a * b * c * d * e * f)

theorem largest_divisor_is_24 :
  ∃ (k : ℤ), k = 24 ∧ (∀ t ∈ S, DividesTuple k t) ∧
  (∀ m : ℤ, (∀ t ∈ S, DividesTuple m t) → m ≤ k) :=
sorry

end largest_divisor_is_24_l1781_178107


namespace g_one_equals_three_l1781_178113

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem g_one_equals_three (f g : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_even : is_even_function g) 
  (h1 : f (-1) + g 1 = 2) 
  (h2 : f 1 + g (-1) = 4) : 
  g 1 = 3 := by
  sorry

end g_one_equals_three_l1781_178113


namespace unique_fraction_l1781_178186

theorem unique_fraction : ∃! (m n : ℕ), 
  m < 10 ∧ n < 10 ∧ 
  n = m^2 - 1 ∧
  (m + 2 : ℚ) / (n + 2) > 1/3 ∧
  (m - 3 : ℚ) / (n - 3) < 1/10 ∧
  m = 3 ∧ n = 8 := by
sorry

end unique_fraction_l1781_178186


namespace apple_purchase_cost_l1781_178129

/-- The price of apples per kilogram before discount -/
def original_price : ℝ := 5

/-- The discount percentage as a decimal -/
def discount_rate : ℝ := 0.4

/-- The quantity of apples in kilograms -/
def quantity : ℝ := 10

/-- Calculates the discounted price per kilogram -/
def discounted_price : ℝ := original_price * (1 - discount_rate)

/-- Calculates the total cost for the given quantity of apples -/
def total_cost : ℝ := discounted_price * quantity

/-- Theorem stating that the total cost for 10 kilograms of apples with a 40% discount is $30 -/
theorem apple_purchase_cost : total_cost = 30 := by
  sorry

end apple_purchase_cost_l1781_178129


namespace final_student_count_l1781_178193

theorem final_student_count (initial_students leaving_students new_students : ℕ) :
  initial_students = 11 →
  leaving_students = 6 →
  new_students = 42 →
  initial_students - leaving_students + new_students = 47 :=
by
  sorry

end final_student_count_l1781_178193


namespace mary_remaining_sheep_l1781_178127

/-- Calculates the number of sheep Mary has left after distributing to her relatives --/
def remaining_sheep (initial : ℕ) : ℕ :=
  let after_sister := initial - (initial / 4)
  let after_brother := after_sister - (after_sister / 3)
  after_brother - (after_brother / 6)

/-- Theorem stating that Mary will have 500 sheep remaining --/
theorem mary_remaining_sheep :
  remaining_sheep 1200 = 500 := by
  sorry

end mary_remaining_sheep_l1781_178127


namespace point_symmetry_product_l1781_178138

/-- A point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given that:
    - Point A lies on the y-axis with coordinates (3a-8, -3)
    - Points A and B(0, b) are symmetric with respect to the x-axis
    Prove that ab = 8 -/
theorem point_symmetry_product (a b : ℝ) : 
  let A : Point := ⟨3*a - 8, -3⟩
  let B : Point := ⟨0, b⟩
  (A.x = 0) →  -- A lies on the y-axis
  (A.y = -B.y) →  -- A and B are symmetric with respect to the x-axis
  a * b = 8 := by
  sorry

end point_symmetry_product_l1781_178138


namespace sandy_age_l1781_178177

theorem sandy_age (sandy_age molly_age : ℕ) : 
  molly_age = sandy_age + 12 →
  sandy_age * 9 = molly_age * 7 →
  sandy_age = 42 := by
sorry

end sandy_age_l1781_178177


namespace pentagon_shaded_probability_l1781_178108

/-- A regular pentagon game board with shaded regions -/
structure PentagonBoard where
  /-- The total number of regions formed by the diagonals -/
  total_regions : ℕ
  /-- The number of shaded regions -/
  shaded_regions : ℕ
  /-- Proof that the number of shaded regions is less than or equal to the total regions -/
  h_shaded_le_total : shaded_regions ≤ total_regions

/-- The probability of landing in a shaded region -/
def shaded_probability (board : PentagonBoard) : ℚ :=
  board.shaded_regions / board.total_regions

/-- Theorem stating the probability of landing in a shaded region for the specific game board -/
theorem pentagon_shaded_probability :
  ∃ (board : PentagonBoard),
    board.total_regions = 10 ∧
    board.shaded_regions = 3 ∧
    shaded_probability board = 3 / 10 := by
  sorry

end pentagon_shaded_probability_l1781_178108


namespace problem_statement_l1781_178152

noncomputable section

def f (a b x : ℝ) : ℝ := Real.exp x - a * x^2 - b * x - 1

def g (a b : ℝ) : ℝ → ℝ := λ x ↦ Real.exp x - 2 * a * x - b

theorem problem_statement (a b : ℝ) :
  (∀ x, |x - a| ≥ f a b x) →
  (∀ x, (Real.exp 1 - 1) * x - 1 = (f a b x - f a b 1) / (x - 1) + f a b 1) →
  (a ≤ 1/2) ∧
  (a = 0 ∧ b = 1) ∧
  (∀ x ∈ Set.Icc 0 1,
    g a b x ≥ min (1 - b)
      (min (2*a - 2*a * Real.log (2*a) - b)
        (1 - 2*a - b))) :=
by sorry

end problem_statement_l1781_178152


namespace cars_sold_per_day_second_period_l1781_178178

def total_quota : ℕ := 50
def total_days : ℕ := 30
def first_period : ℕ := 3
def second_period : ℕ := 4
def cars_per_day_first_period : ℕ := 5
def remaining_cars : ℕ := 23

theorem cars_sold_per_day_second_period :
  let cars_sold_first_period := first_period * cars_per_day_first_period
  let remaining_after_first_period := total_quota - cars_sold_first_period
  let cars_to_sell_second_period := remaining_after_first_period - remaining_cars
  cars_to_sell_second_period / second_period = 3 := by sorry

end cars_sold_per_day_second_period_l1781_178178


namespace complex_product_quadrant_l1781_178164

theorem complex_product_quadrant : 
  let z : ℂ := (1 + 3*I) * (3 - I)
  (z.re > 0 ∧ z.im > 0) := by sorry

end complex_product_quadrant_l1781_178164


namespace y_squared_plus_7y_plus_12_range_l1781_178104

theorem y_squared_plus_7y_plus_12_range (y : ℝ) (h : y^2 - 7*y + 12 < 0) :
  42 < y^2 + 7*y + 12 ∧ y^2 + 7*y + 12 < 56 := by
  sorry

end y_squared_plus_7y_plus_12_range_l1781_178104


namespace special_function_inequality_l1781_178125

/-- A function that is increasing on (1,+∞) and has F(x) = f(x+1) symmetrical about the y-axis -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, 1 < x ∧ x < y → f x < f y) ∧
  (∀ x, f (-x + 1) = f (x + 1))

/-- Theorem: For a special function f, f(-1) > f(2) -/
theorem special_function_inequality (f : ℝ → ℝ) (h : SpecialFunction f) : f (-1) > f 2 := by
  sorry

end special_function_inequality_l1781_178125


namespace tangent_line_and_extrema_l1781_178114

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := 4 * x^3 + a * x^2 + b * x + 5

-- Define the derivative of f(x)
def f' (a b : ℝ) (x : ℝ) : ℝ := 12 * x^2 + 2 * a * x + b

theorem tangent_line_and_extrema 
  (a b : ℝ) 
  (h1 : f' a b 1 = -12)  -- Slope of tangent line at x=1 is -12
  (h2 : f a b 1 = -12)   -- Point (1, -12) lies on the graph of f(x)
  : 
  (a = -3 ∧ b = -18) ∧   -- Part 1: Coefficients a and b
  (∀ x ∈ Set.Icc (-3) 1, f (-3) (-18) x ≤ 16) ∧  -- Part 2: Maximum value
  (∀ x ∈ Set.Icc (-3) 1, f (-3) (-18) x ≥ -76)   -- Part 2: Minimum value
  := by sorry

end tangent_line_and_extrema_l1781_178114


namespace f_difference_l1781_178187

-- Define the function f
def f (x : ℝ) : ℝ := x^4 + x^2 + 7*x

-- State the theorem
theorem f_difference : f 3 - f (-3) = 42 := by
  sorry

end f_difference_l1781_178187


namespace coin_flip_probability_l1781_178159

/-- The number of coins being flipped -/
def num_coins : ℕ := 6

/-- The number of specific coins we're interested in -/
def num_specific_coins : ℕ := 3

/-- The number of possible outcomes for each coin (heads or tails) -/
def outcomes_per_coin : ℕ := 2

/-- The probability of three specific coins out of six showing the same face -/
def probability_same_face : ℚ := 1 / 4

theorem coin_flip_probability :
  (outcomes_per_coin ^ num_specific_coins * outcomes_per_coin ^ (num_coins - num_specific_coins)) /
  (outcomes_per_coin ^ num_coins) = probability_same_face :=
sorry

end coin_flip_probability_l1781_178159


namespace cubes_passed_in_specific_solid_l1781_178132

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_by_diagonal (l w h : ℕ) : ℕ :=
  l + w + h - (Nat.gcd l w + Nat.gcd w h + Nat.gcd h l) + Nat.gcd l (Nat.gcd w h)

/-- Theorem stating the number of cubes an internal diagonal passes through
    in a 105 × 140 × 195 rectangular solid -/
theorem cubes_passed_in_specific_solid :
  cubes_passed_by_diagonal 105 140 195 = 395 := by
  sorry

end cubes_passed_in_specific_solid_l1781_178132


namespace hyperbola_k_range_l1781_178182

/-- The equation of a hyperbola with parameter k -/
def hyperbola_equation (x y k : ℝ) : Prop :=
  x^2 / (1 - 2*k) - y^2 / (k - 2) = 1

/-- The condition that the hyperbola has foci on the y-axis -/
def foci_on_y_axis (k : ℝ) : Prop :=
  1 - 2*k < 0 ∧ k - 2 < 0

/-- Theorem: If the equation represents a hyperbola with foci on the y-axis,
    then k is in the open interval (1/2, 2) -/
theorem hyperbola_k_range (k : ℝ) :
  (∃ x y : ℝ, hyperbola_equation x y k) →
  foci_on_y_axis k →
  k ∈ Set.Ioo (1/2 : ℝ) 2 :=
sorry

end hyperbola_k_range_l1781_178182


namespace product_mod_30_l1781_178119

theorem product_mod_30 : ∃ m : ℕ, 0 ≤ m ∧ m < 30 ∧ (33 * 77 * 99) % 30 = m ∧ m = 9 := by
  sorry

end product_mod_30_l1781_178119


namespace nabla_calculation_l1781_178196

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- Theorem statement
theorem nabla_calculation : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end nabla_calculation_l1781_178196


namespace puppies_per_cage_l1781_178102

theorem puppies_per_cage 
  (initial_puppies : ℕ) 
  (sold_puppies : ℕ) 
  (num_cages : ℕ) 
  (h1 : initial_puppies = 13)
  (h2 : sold_puppies = 7)
  (h3 : num_cages = 3)
  (h4 : num_cages > 0)
  (h5 : initial_puppies > sold_puppies) :
  (initial_puppies - sold_puppies) / num_cages = 2 := by
  sorry

end puppies_per_cage_l1781_178102


namespace least_subtraction_for_divisibility_problem_solution_l1781_178111

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  ∃ (x : ℕ), x = 6 ∧ (7538 - x) % 14 = 0 ∧ ∀ (y : ℕ), y < x → (7538 - y) % 14 ≠ 0 :=
by
  sorry

end least_subtraction_for_divisibility_problem_solution_l1781_178111


namespace binomial_18_10_l1781_178134

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 47190 := by
  sorry

end binomial_18_10_l1781_178134


namespace square_roots_and_cube_root_l1781_178135

theorem square_roots_and_cube_root (x a : ℝ) (hx : x > 0) : 
  ((2*a - 1)^2 = x ∧ (-a + 2)^2 = x ∧ 2*a - 1 ≠ -a + 2) →
  (a = -1 ∧ x = 9 ∧ (4*x + 9*a)^(1/3) = 3) :=
by sorry

end square_roots_and_cube_root_l1781_178135


namespace sum_equality_l1781_178153

def sum_ascending (n : ℕ) : ℕ := (n * (n + 1)) / 2

def sum_descending (n : ℕ) : ℕ := 
  if n = 0 then 0 else n + sum_descending (n - 1)

theorem sum_equality : 
  sum_ascending 1000 = sum_descending 1000 :=
by sorry

end sum_equality_l1781_178153


namespace cone_surface_area_theorem_sphere_surface_area_theorem_cylinder_surface_area_theorem_l1781_178100

-- Define the necessary variables and functions
variable (R : ℝ)
variable (x y z : ℝ)

-- Define the equations for the surfaces
def cone_equation (x y z : ℝ) : Prop := z^2 = 2*x*y
def sphere_equation (x y z R : ℝ) : Prop := x^2 + y^2 + z^2 = R^2
def cylinder_equation (x y R : ℝ) : Prop := x^2 + y^2 = R*x

-- Define the surface area functions
noncomputable def cone_surface_area (x_max y_max : ℝ) : ℝ := 
  sorry

noncomputable def sphere_surface_area_in_cylinder (R : ℝ) : ℝ := 
  sorry

noncomputable def cylinder_surface_area_in_sphere (R : ℝ) : ℝ := 
  sorry

-- State the theorems to be proven
theorem cone_surface_area_theorem :
  cone_surface_area 2 4 = 16 :=
sorry

theorem sphere_surface_area_theorem :
  sphere_surface_area_in_cylinder R = 2 * R^2 * (Real.pi - 2) :=
sorry

theorem cylinder_surface_area_theorem :
  cylinder_surface_area_in_sphere R = 4 * R^2 :=
sorry

end cone_surface_area_theorem_sphere_surface_area_theorem_cylinder_surface_area_theorem_l1781_178100


namespace acute_triangle_properties_l1781_178150

theorem acute_triangle_properties (A B C a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle
  A + B + C = π ∧  -- Sum of angles in a triangle
  Real.sqrt ((1 - Real.cos (2 * C)) / 2) + Real.sin (B - A) = 2 * Real.sin (2 * A) ∧  -- Given equation
  c ≥ max a b ∧  -- AB is the longest side
  a = Real.sin A ∧ b = Real.sin B ∧ c = Real.sin C  -- Law of sines
  →
  a / b = 1 / 2 ∧ 0 < Real.cos C ∧ Real.cos C ≤ 1 / 4 :=
by sorry

end acute_triangle_properties_l1781_178150


namespace difference_of_squares_l1781_178130

theorem difference_of_squares (a b : ℝ) : (a - b) * (-b - a) = b^2 - a^2 := by
  sorry

end difference_of_squares_l1781_178130


namespace pet_insurance_cost_l1781_178156

/-- Calculates the monthly cost of pet insurance given the surgery cost, insurance duration,
    coverage percentage, and total savings. -/
def monthly_insurance_cost (surgery_cost : ℚ) (insurance_duration : ℕ) 
    (coverage_percent : ℚ) (total_savings : ℚ) : ℚ :=
  let insurance_payment := surgery_cost * coverage_percent
  let total_insurance_cost := insurance_payment - total_savings
  total_insurance_cost / insurance_duration

/-- Theorem stating that the monthly insurance cost is $20 given the specified conditions. -/
theorem pet_insurance_cost :
  monthly_insurance_cost 5000 24 (4/5) 3520 = 20 := by
  sorry

end pet_insurance_cost_l1781_178156


namespace circle_radii_in_square_l1781_178162

theorem circle_radii_in_square (r : ℝ) : 
  r > 0 →  -- radius is positive
  r < 1/4 →  -- each circle fits in a corner
  (∀ (i j : Fin 4), i ≠ j → 
    (∃ (x y : ℝ), x^2 + y^2 = (2*r)^2 ∧ 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1)) →  -- circles touch
  (∃ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    (∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ), 
      (x₁ - x₂)^2 + (y₁ - y₂)^2 = (2*r)^2 ∧
      (x₂ - x₃)^2 + (y₂ - y₃)^2 = (2*r)^2 ∧
      (x₃ - x₁)^2 + (y₃ - y₁)^2 > (2*r)^2)) →  -- only two circles touch each other
  1 - Real.sqrt 2 / 2 < r ∧ r < 2 - Real.sqrt 2 / 2 - Real.sqrt (4 - 2 * Real.sqrt 2) :=
by sorry

end circle_radii_in_square_l1781_178162


namespace history_not_statistics_l1781_178122

theorem history_not_statistics (total : ℕ) (history : ℕ) (statistics : ℕ) (history_or_statistics : ℕ)
  (h_total : total = 90)
  (h_history : history = 36)
  (h_statistics : statistics = 32)
  (h_history_or_statistics : history_or_statistics = 57) :
  history - (history + statistics - history_or_statistics) = 25 := by
  sorry

end history_not_statistics_l1781_178122


namespace initial_average_calculation_l1781_178117

theorem initial_average_calculation (n : ℕ) (correct_sum wrong_sum : ℝ) 
  (h1 : n = 10)
  (h2 : correct_sum / n = 24)
  (h3 : wrong_sum = correct_sum - 10) :
  wrong_sum / n = 23 := by
sorry

end initial_average_calculation_l1781_178117


namespace largest_angle_in_pentagon_l1781_178128

theorem largest_angle_in_pentagon (A B C D E : ℝ) : 
  A = 70 → 
  B = 120 → 
  C = D → 
  E = 3 * C - 30 → 
  A + B + C + D + E = 540 → 
  max A (max B (max C (max D E))) = 198 :=
by sorry

end largest_angle_in_pentagon_l1781_178128


namespace orange_probability_l1781_178169

/-- Given a box of fruit with apples and oranges, calculate the probability of selecting an orange -/
theorem orange_probability (apples oranges : ℕ) (h1 : apples = 20) (h2 : oranges = 10) :
  (oranges : ℚ) / ((apples : ℚ) + (oranges : ℚ)) = 1 / 3 := by
  sorry

end orange_probability_l1781_178169


namespace birdhouse_cost_theorem_l1781_178155

/-- Calculates the total cost of building birdhouses -/
def total_cost_birdhouses (small_count large_count : ℕ) 
  (small_plank_req large_plank_req : ℕ) 
  (small_nail_req large_nail_req : ℕ) 
  (small_plank_cost large_plank_cost nail_cost : ℚ) 
  (discount_threshold : ℕ) (discount_rate : ℚ) : ℚ :=
  let total_small_planks := small_count * small_plank_req
  let total_large_planks := large_count * large_plank_req
  let total_nails := small_count * small_nail_req + large_count * large_nail_req
  let plank_cost := total_small_planks * small_plank_cost + total_large_planks * large_plank_cost
  let nail_cost_before_discount := total_nails * nail_cost
  let nail_cost_after_discount := 
    if total_nails > discount_threshold
    then nail_cost_before_discount * (1 - discount_rate)
    else nail_cost_before_discount
  plank_cost + nail_cost_after_discount

theorem birdhouse_cost_theorem :
  total_cost_birdhouses 3 2 7 10 20 36 3 5 (5/100) 100 (1/10) = 16894/100 := by
  sorry

end birdhouse_cost_theorem_l1781_178155


namespace sum_of_cubes_l1781_178195

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = -3) : x^3 + y^3 = 26 := by
  sorry

end sum_of_cubes_l1781_178195


namespace arithmetic_mean_of_powers_of_three_l1781_178116

def powers_of_three : List ℕ := [3, 9, 27, 81, 243, 729, 2187, 6561, 19683]

theorem arithmetic_mean_of_powers_of_three :
  (List.sum powers_of_three) / powers_of_three.length = 2970 := by
  sorry

end arithmetic_mean_of_powers_of_three_l1781_178116


namespace toms_dog_age_l1781_178118

theorem toms_dog_age (brother_age dog_age : ℕ) : 
  brother_age = 4 * dog_age →
  brother_age + 6 = 30 →
  dog_age + 6 = 12 := by
sorry

end toms_dog_age_l1781_178118


namespace basketball_team_average_weight_l1781_178121

/-- Given a basketball team with boys and girls, calculate the average weight of all players. -/
theorem basketball_team_average_weight 
  (num_boys : ℕ) 
  (num_girls : ℕ) 
  (avg_weight_boys : ℚ) 
  (avg_weight_girls : ℚ) 
  (h_num_boys : num_boys = 8) 
  (h_num_girls : num_girls = 5) 
  (h_avg_weight_boys : avg_weight_boys = 160) 
  (h_avg_weight_girls : avg_weight_girls = 130) : 
  (num_boys * avg_weight_boys + num_girls * avg_weight_girls) / (num_boys + num_girls) = 148 := by
  sorry

end basketball_team_average_weight_l1781_178121


namespace unique_sums_count_l1781_178163

def bag_A : Finset ℕ := {1, 4, 5, 8}
def bag_B : Finset ℕ := {2, 3, 7, 9}

theorem unique_sums_count : 
  Finset.card ((bag_A.product bag_B).image (λ (p : ℕ × ℕ) => p.1 + p.2)) = 12 := by
  sorry

end unique_sums_count_l1781_178163


namespace fishbowl_count_l1781_178192

theorem fishbowl_count (total_fish : ℕ) (fish_per_bowl : ℕ) (h1 : total_fish = 6003) (h2 : fish_per_bowl = 23) :
  total_fish / fish_per_bowl = 261 := by
sorry

end fishbowl_count_l1781_178192


namespace total_matches_played_l1781_178110

theorem total_matches_played (home_wins rival_wins home_draws rival_draws : ℕ) : 
  home_wins = 3 →
  rival_wins = 2 * home_wins →
  home_draws = 4 →
  rival_draws = 4 →
  home_wins + rival_wins + home_draws + rival_draws = 17 :=
by
  sorry

end total_matches_played_l1781_178110


namespace intersection_implies_m_value_l1781_178151

def A (m : ℝ) : Set ℝ := {m + 1, -3}
def B (m : ℝ) : Set ℝ := {2*m + 1, m - 3}

theorem intersection_implies_m_value :
  ∀ m : ℝ, (A m ∩ B m = {-3}) → m = -2 := by
sorry

end intersection_implies_m_value_l1781_178151


namespace representative_selection_count_l1781_178103

def total_students : ℕ := 10
def female_students : ℕ := 4
def male_students : ℕ := 6
def representatives : ℕ := 3

theorem representative_selection_count : 
  (Nat.choose female_students 1 * Nat.choose male_students 2) + 
  (Nat.choose female_students 2 * Nat.choose male_students 1) + 
  (Nat.choose female_students 3) = 100 := by
  sorry

end representative_selection_count_l1781_178103


namespace square_sum_from_product_and_sum_l1781_178137

theorem square_sum_from_product_and_sum (x y : ℝ) 
  (h1 : x * y = 12) 
  (h2 : x + y = 10) : 
  x^2 + y^2 = 76 := by
sorry

end square_sum_from_product_and_sum_l1781_178137


namespace f_lower_bound_a_range_l1781_178165

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |x + 1| + 2 * |x + 2|

-- Theorem for part (I)
theorem f_lower_bound : ∀ x : ℝ, f x ≥ 5 := by sorry

-- Theorem for part (II)
theorem a_range (a : ℝ) : 
  (∀ x : ℝ, 15 - 2 * f x < a^2 + 9 / (a^2 + 1)) ↔ a ≠ Real.sqrt 2 ∧ a ≠ -Real.sqrt 2 := by sorry

end f_lower_bound_a_range_l1781_178165


namespace polynomial_bound_l1781_178170

open Complex

theorem polynomial_bound (a b c : ℂ) :
  (∀ z : ℂ, Complex.abs z ≤ 1 → Complex.abs (a * z^2 + b * z + c) ≤ 1) →
  (∀ z : ℂ, Complex.abs z ≤ 1 → 0 ≤ Complex.abs (a * z + b) ∧ Complex.abs (a * z + b) ≤ 2) :=
by sorry

end polynomial_bound_l1781_178170


namespace solve_quadratic_equation_l1781_178112

theorem solve_quadratic_equation (m n : ℤ) 
  (h : m^2 - 2*m*n + 2*n^2 - 8*n + 16 = 0) : 
  m = 4 ∧ n = 4 := by
sorry

end solve_quadratic_equation_l1781_178112


namespace max_pieces_is_72_l1781_178174

/-- Represents a rectangular cake with dimensions m and n -/
structure Cake where
  m : ℕ+
  n : ℕ+

/-- Calculates the number of pieces in the two central rows -/
def central_pieces (c : Cake) : ℕ := (c.m - 4) * (c.n - 4)

/-- Calculates the number of pieces on the perimeter -/
def perimeter_pieces (c : Cake) : ℕ := 2 * c.m + 2 * c.n - 4

/-- Checks if the cake satisfies the chef's condition -/
def satisfies_condition (c : Cake) : Prop :=
  central_pieces c = perimeter_pieces c

/-- Calculates the total number of pieces -/
def total_pieces (c : Cake) : ℕ := c.m * c.n

/-- States that the maximum number of pieces satisfying the condition is 72 -/
theorem max_pieces_is_72 :
  ∃ (c : Cake), satisfies_condition c ∧
    total_pieces c = 72 ∧
    ∀ (c' : Cake), satisfies_condition c' → total_pieces c' ≤ 72 := by
  sorry

end max_pieces_is_72_l1781_178174


namespace power_four_times_power_four_l1781_178109

theorem power_four_times_power_four (x : ℝ) : x^4 * x^4 = x^8 := by
  sorry

end power_four_times_power_four_l1781_178109


namespace game_a_more_likely_than_game_b_l1781_178161

def prob_heads : ℚ := 3/4
def prob_tails : ℚ := 1/4

def prob_game_a : ℚ := prob_heads^4

def prob_game_b : ℚ := (prob_heads * prob_tails)^3

theorem game_a_more_likely_than_game_b : prob_game_a > prob_game_b := by
  sorry

end game_a_more_likely_than_game_b_l1781_178161


namespace gingers_children_l1781_178139

/-- The number of cakes Ginger bakes for each child per year -/
def cakes_per_child : ℕ := 4

/-- The number of cakes Ginger bakes for her husband per year -/
def cakes_for_husband : ℕ := 6

/-- The number of cakes Ginger bakes for her parents per year -/
def cakes_for_parents : ℕ := 2

/-- The total number of cakes Ginger bakes in 10 years -/
def total_cakes : ℕ := 160

/-- The number of years over which the total cakes are counted -/
def years : ℕ := 10

/-- Ginger's number of children -/
def num_children : ℕ := 2

theorem gingers_children :
  num_children * cakes_per_child * years + cakes_for_husband * years + cakes_for_parents * years = total_cakes :=
by sorry

end gingers_children_l1781_178139


namespace third_equals_sixth_implies_seven_odd_terms_sum_128_implies_eight_and_max_term_l1781_178158

-- For the first part of the problem
theorem third_equals_sixth_implies_seven (n : ℕ) :
  (Nat.choose n 2 = Nat.choose n 5) → n = 7 := by sorry

-- For the second part of the problem
theorem odd_terms_sum_128_implies_eight_and_max_term (n : ℕ) (x : ℝ) :
  (2^(n-1) = 128) →
  n = 8 ∧
  (Nat.choose 8 4 * x^4 * x^(2/3) = 70 * x^4 * x^(2/3)) := by sorry

end third_equals_sixth_implies_seven_odd_terms_sum_128_implies_eight_and_max_term_l1781_178158


namespace sum_of_digits_of_triangular_array_rows_l1781_178183

/-- The number of coins in a triangular array with n rows -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The theorem stating that the sum of digits of N is 14, where N is the number of rows in a triangular array containing 3003 coins -/
theorem sum_of_digits_of_triangular_array_rows :
  ∃ N : ℕ, triangular_sum N = 3003 ∧ sum_of_digits N = 14 := by
  sorry

end sum_of_digits_of_triangular_array_rows_l1781_178183


namespace det_dilation_matrix_l1781_178148

/-- A 3x3 matrix representing a dilation with scale factor 5 -/
def E : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.diagonal (λ _ => 5)

/-- Theorem stating that the determinant of E is 125 -/
theorem det_dilation_matrix : Matrix.det E = 125 := by
  sorry

end det_dilation_matrix_l1781_178148


namespace square_with_inscribed_semicircles_l1781_178131

theorem square_with_inscribed_semicircles (square_side : ℝ) (semicircle_count : ℕ) : 
  square_side = 4 → 
  semicircle_count = 4 → 
  (square_side^2 - semicircle_count * (π * (square_side/2)^2 / 2)) = 16 - 8*π := by
sorry

end square_with_inscribed_semicircles_l1781_178131


namespace quadratic_root_coefficients_l1781_178140

theorem quadratic_root_coefficients (b c : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (1 - Complex.I * Real.sqrt 2) ^ 2 + b * (1 - Complex.I * Real.sqrt 2) + c = 0 →
  b = -2 ∧ c = 3 := by
  sorry

end quadratic_root_coefficients_l1781_178140


namespace store_clearance_sale_l1781_178144

/-- Calculates the amount owed to creditors after a store's clearance sale --/
theorem store_clearance_sale 
  (total_items : ℕ) 
  (original_price : ℝ) 
  (discount_percent : ℝ) 
  (sold_percent : ℝ) 
  (remaining_amount : ℝ) 
  (h1 : total_items = 2000)
  (h2 : original_price = 50)
  (h3 : discount_percent = 0.8)
  (h4 : sold_percent = 0.9)
  (h5 : remaining_amount = 3000) : 
  (total_items : ℝ) * sold_percent * (original_price * (1 - discount_percent)) - remaining_amount = 15000 := by
  sorry

end store_clearance_sale_l1781_178144


namespace cubic_function_min_value_l1781_178124

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 12*x + c

-- State the theorem
theorem cubic_function_min_value 
  (c : ℝ) 
  (h_max : ∃ x, f c x ≤ 28 ∧ ∀ y, f c y ≤ f c x) : 
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f c x ≤ f c y) ∧ 
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, f c x = -4) :=
by sorry

end cubic_function_min_value_l1781_178124


namespace bank_coin_value_l1781_178181

/-- Proves that the total value of coins in a bank is 555 cents -/
theorem bank_coin_value : 
  let total_coins : ℕ := 70
  let nickel_count : ℕ := 29
  let nickel_value : ℕ := 5
  let dime_value : ℕ := 10
  let dime_count : ℕ := total_coins - nickel_count
  total_coins = nickel_count + dime_count →
  nickel_count * nickel_value + dime_count * dime_value = 555 := by
  sorry

end bank_coin_value_l1781_178181


namespace circle_radius_zero_circle_equation_point_circle_radius_l1781_178190

theorem circle_radius_zero (x y : ℝ) :
  x^2 + 8*x + y^2 - 10*y + 41 = 0 → (x + 4)^2 + (y - 5)^2 = 0 :=
by sorry

theorem circle_equation_point (x y : ℝ) :
  (x + 4)^2 + (y - 5)^2 = 0 → x = -4 ∧ y = 5 :=
by sorry

theorem circle_radius (x y : ℝ) :
  x^2 + 8*x + y^2 - 10*y + 41 = 0 → ∃! (center : ℝ × ℝ), center = (-4, 5) ∧ (x - center.1)^2 + (y - center.2)^2 = 0 :=
by sorry

end circle_radius_zero_circle_equation_point_circle_radius_l1781_178190


namespace number_of_people_is_fifteen_l1781_178197

theorem number_of_people_is_fifteen (x : ℕ) (y : ℕ) : 
  (12 * x + 3 = y) → 
  (13 * x - 12 = y) → 
  x = 15 := by
sorry

end number_of_people_is_fifteen_l1781_178197


namespace enclosing_polygons_sides_l1781_178143

/-- The number of sides of the central polygon -/
def central_sides : ℕ := 12

/-- The number of polygons enclosing the central polygon -/
def enclosing_polygons : ℕ := 12

/-- The number of enclosing polygons meeting at each vertex of the central polygon -/
def polygons_at_vertex : ℕ := 4

/-- The number of sides of each enclosing polygon -/
def n : ℕ := 12

/-- Theorem stating that n must be 12 for the given configuration -/
theorem enclosing_polygons_sides (h1 : central_sides = 12)
                                 (h2 : enclosing_polygons = 12)
                                 (h3 : polygons_at_vertex = 4) :
  n = 12 := by sorry

end enclosing_polygons_sides_l1781_178143


namespace blue_candy_count_l1781_178166

theorem blue_candy_count (total : ℕ) (red : ℕ) (h1 : total = 11567) (h2 : red = 792) :
  total - red = 10775 := by
  sorry

end blue_candy_count_l1781_178166
