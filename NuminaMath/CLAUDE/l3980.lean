import Mathlib

namespace benjamin_steps_to_times_square_l3980_398041

/-- The number of steps Benjamin took to reach Rockefeller Center -/
def steps_to_rockefeller : ℕ := 354

/-- The number of steps Benjamin took from Rockefeller Center to Times Square -/
def steps_rockefeller_to_times_square : ℕ := 228

/-- The total number of steps Benjamin took before reaching Times Square -/
def total_steps : ℕ := steps_to_rockefeller + steps_rockefeller_to_times_square

theorem benjamin_steps_to_times_square : total_steps = 582 := by
  sorry

end benjamin_steps_to_times_square_l3980_398041


namespace birds_joined_birds_joined_fence_l3980_398011

theorem birds_joined (initial_birds : ℕ) (initial_storks : ℕ) (final_total : ℕ) : ℕ :=
  let initial_total := initial_birds + initial_storks
  let birds_joined := final_total - initial_total
  birds_joined

theorem birds_joined_fence : birds_joined 3 2 10 = 5 := by
  sorry

end birds_joined_birds_joined_fence_l3980_398011


namespace pages_remaining_l3980_398048

/-- Given a book with 93 pages, if Jerry reads 30 pages on Saturday and 20 pages on Sunday,
    then the number of pages remaining to finish the book is 43. -/
theorem pages_remaining (total_pages : Nat) (pages_read_saturday : Nat) (pages_read_sunday : Nat)
    (h1 : total_pages = 93)
    (h2 : pages_read_saturday = 30)
    (h3 : pages_read_sunday = 20) :
    total_pages - pages_read_saturday - pages_read_sunday = 43 := by
  sorry

end pages_remaining_l3980_398048


namespace square_sum_of_differences_l3980_398047

theorem square_sum_of_differences (x y z : ℤ) : 
  ∃ (σ₂ : ℤ), (1/2 : ℚ) * ((x - y)^4 + (y - z)^4 + (z - x)^4) = (σ₂^2 : ℚ) := by
  sorry

end square_sum_of_differences_l3980_398047


namespace copper_in_mixture_l3980_398050

/-- Given a mixture of zinc and copper in the ratio 9:11 with a total weight of 60 kg,
    the amount of copper in the mixture is 33 kg. -/
theorem copper_in_mixture (zinc_ratio : ℕ) (copper_ratio : ℕ) (total_weight : ℝ) :
  zinc_ratio = 9 →
  copper_ratio = 11 →
  total_weight = 60 →
  (copper_ratio : ℝ) / ((zinc_ratio : ℝ) + (copper_ratio : ℝ)) * total_weight = 33 :=
by sorry

end copper_in_mixture_l3980_398050


namespace expression_evaluation_l3980_398004

theorem expression_evaluation : 120 * (120 - 5) - (120 * 120 - 10 + 2) = -592 := by
  sorry

end expression_evaluation_l3980_398004


namespace committee_count_l3980_398045

/-- Represents a department in the division of sciences -/
inductive Department
| Mathematics
| Statistics
| ComputerScience
| Physics

/-- Represents the gender of a professor -/
inductive Gender
| Male
| Female

/-- Represents the number of professors in each department by gender -/
def professors_count (d : Department) (g : Gender) : Nat :=
  match d, g with
  | Department.Physics, _ => 1
  | _, _ => 3

/-- Represents the total number of professors to be selected from each department -/
def selection_count (d : Department) : Nat :=
  match d with
  | Department.Physics => 1
  | _ => 2

/-- Calculates the number of ways to select professors from a department -/
def department_selection_ways (d : Department) : Nat :=
  (professors_count d Gender.Male).choose (selection_count d) *
  (professors_count d Gender.Female).choose (selection_count d)

/-- Theorem: The number of possible committees is 729 -/
theorem committee_count : 
  (department_selection_ways Department.Mathematics) *
  (department_selection_ways Department.Statistics) *
  (department_selection_ways Department.ComputerScience) *
  (department_selection_ways Department.Physics) = 729 := by
  sorry

end committee_count_l3980_398045


namespace fundraiser_theorem_l3980_398002

def fundraiser (num_students : ℕ) (individual_cost : ℕ) (collective_cost : ℕ) 
                (day1_raised : ℕ) (day2_raised : ℕ) (day3_raised : ℕ) : ℕ :=
  let total_needed := num_students * individual_cost + collective_cost
  let first3days_raised := day1_raised + day2_raised + day3_raised
  let next4days_raised := first3days_raised / 2
  let total_raised := first3days_raised + next4days_raised
  let remaining := total_needed - total_raised
  remaining / num_students

theorem fundraiser_theorem : 
  fundraiser 6 450 3000 600 900 400 = 475 := by
  sorry

end fundraiser_theorem_l3980_398002


namespace max_value_sqrt_sum_l3980_398018

theorem max_value_sqrt_sum (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hsum : a + b + c = 7) :
  Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) ≤ Real.sqrt 69 := by
  sorry

end max_value_sqrt_sum_l3980_398018


namespace rays_grocery_bill_l3980_398079

/-- Calculates the total grocery bill for Ray's purchase with a store rewards discount --/
theorem rays_grocery_bill :
  let meat_price : ℚ := 5
  let crackers_price : ℚ := 3.5
  let vegetable_price : ℚ := 2
  let vegetable_quantity : ℕ := 4
  let cheese_price : ℚ := 3.5
  let discount_rate : ℚ := 0.1

  let total_before_discount : ℚ := 
    meat_price + crackers_price + (vegetable_price * vegetable_quantity) + cheese_price
  
  let discount_amount : ℚ := total_before_discount * discount_rate
  
  let final_bill : ℚ := total_before_discount - discount_amount

  final_bill = 18 := by sorry

end rays_grocery_bill_l3980_398079


namespace fraction_equality_sum_l3980_398093

theorem fraction_equality_sum (C D : ℚ) :
  (∀ x : ℚ, x ≠ 4 ∧ x ≠ 5 → (D * x - 13) / (x^2 - 9*x + 20) = C / (x - 4) + 5 / (x - 5)) →
  C + D = 1/5 := by
sorry

end fraction_equality_sum_l3980_398093


namespace ones_digit_largest_power_of_two_32_factorial_l3980_398084

/-- The largest power of 2 that divides n! -/
def largest_power_of_two (n : ℕ) : ℕ := sorry

/-- The ones digit of a natural number -/
def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_largest_power_of_two_32_factorial :
  ones_digit (2^(largest_power_of_two 32)) = 4 := by sorry

end ones_digit_largest_power_of_two_32_factorial_l3980_398084


namespace original_denominator_proof_l3980_398086

theorem original_denominator_proof (d : ℚ) : 
  (2 : ℚ) / d ≠ 0 →
  (2 + 7 : ℚ) / (d + 7) = (1 : ℚ) / 3 →
  d = 20 := by
sorry

end original_denominator_proof_l3980_398086


namespace range_of_m_l3980_398000

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁^2 + (2*m - 3)*x₁ + 1 = 0 ∧ x₂^2 + (2*m - 3)*x₂ + 1 = 0

def q (m : ℝ) : Prop := ∃ a b, a > 0 ∧ b > 0 ∧ a ≠ b ∧ ∀ x y, x^2/m + y^2/2 = 1 ↔ (x/a)^2 + (y/b)^2 = 1

-- Define the range of m
def m_range (m : ℝ) : Prop := m < 1/2 ∨ (2 < m ∧ m ≤ 5/2)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) → m_range m :=
sorry

end range_of_m_l3980_398000


namespace hair_sufficient_for_skin_l3980_398083

/-- Represents the state of having skin -/
def HasSkin : Prop := sorry

/-- Represents the state of having hair -/
def HasHair : Prop := sorry

/-- If there is no skin, there cannot be hair -/
axiom no_skin_no_hair : ¬HasSkin → ¬HasHair

/-- Prove that having hair is a sufficient condition for having skin -/
theorem hair_sufficient_for_skin : HasHair → HasSkin := by
  sorry

end hair_sufficient_for_skin_l3980_398083


namespace sin_450_degrees_l3980_398055

theorem sin_450_degrees : Real.sin (450 * π / 180) = 1 := by
  sorry

end sin_450_degrees_l3980_398055


namespace right_triangle_area_right_triangle_area_proof_l3980_398060

/-- The area of a right triangle with legs of length 36 and 48 is 864 -/
theorem right_triangle_area : ℝ → ℝ → ℝ → Prop :=
  fun leg1 leg2 area =>
    leg1 = 36 ∧ leg2 = 48 → area = (1 / 2) * leg1 * leg2 → area = 864

/-- Proof of the theorem -/
theorem right_triangle_area_proof : right_triangle_area 36 48 864 := by
  sorry

end right_triangle_area_right_triangle_area_proof_l3980_398060


namespace units_digit_of_17_power_28_l3980_398019

theorem units_digit_of_17_power_28 :
  ∃ n : ℕ, 17^28 ≡ 1 [ZMOD 10] ∧ 17 ≡ 7 [ZMOD 10] :=
by
  sorry

end units_digit_of_17_power_28_l3980_398019


namespace rebecca_eggs_l3980_398044

/-- The number of eggs Rebecca has -/
def num_eggs : ℕ := 3 * 6

/-- The number of groups Rebecca will create -/
def num_groups : ℕ := 3

/-- The number of eggs in each group -/
def eggs_per_group : ℕ := 6

/-- Theorem stating that Rebecca has 18 eggs -/
theorem rebecca_eggs : num_eggs = 18 := by
  sorry

end rebecca_eggs_l3980_398044


namespace cookie_sale_loss_l3980_398075

/-- Represents the cookie sale scenario --/
structure CookieSale where
  total_cookies : ℕ
  purchase_rate : ℚ  -- cookies per dollar
  selling_rate : ℚ   -- cookies per dollar

/-- Calculates the loss from a cookie sale --/
def calculate_loss (sale : CookieSale) : ℚ :=
  let cost := sale.total_cookies / sale.purchase_rate
  let revenue := sale.total_cookies / sale.selling_rate
  cost - revenue

/-- The main theorem stating the loss for the given scenario --/
theorem cookie_sale_loss : 
  let sale : CookieSale := {
    total_cookies := 800,
    purchase_rate := 4/3,  -- 4 cookies for $3
    selling_rate := 3/2    -- 3 cookies for $2
  }
  calculate_loss sale = 64 := by
  sorry


end cookie_sale_loss_l3980_398075


namespace apple_cost_per_kg_main_apple_cost_theorem_l3980_398034

/-- Represents the cost structure of apples -/
structure AppleCost where
  p : ℝ  -- Cost per kg for first 30 kgs
  q : ℝ  -- Cost per kg for additional kgs

/-- Theorem stating the cost per kg for first 30 kgs of apples -/
theorem apple_cost_per_kg (cost : AppleCost) : cost.p = 10 :=
  by
  have h1 : 30 * cost.p + 3 * cost.q = 360 := by sorry
  have h2 : 30 * cost.p + 6 * cost.q = 420 := by sorry
  have h3 : 25 * cost.p = 250 := by sorry
  sorry

/-- Main theorem proving the cost per kg for first 30 kgs of apples -/
theorem main_apple_cost_theorem : ∃ (cost : AppleCost), cost.p = 10 :=
  by
  sorry

end apple_cost_per_kg_main_apple_cost_theorem_l3980_398034


namespace smallest_digit_for_divisibility_l3980_398052

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem smallest_digit_for_divisibility :
  ∃ (d : ℕ), d < 10 ∧ 
    is_divisible_by_9 (529000 + d * 100 + 46) ∧
    ∀ (d' : ℕ), d' < d → ¬is_divisible_by_9 (529000 + d' * 100 + 46) :=
by
  use 1
  sorry

#check smallest_digit_for_divisibility

end smallest_digit_for_divisibility_l3980_398052


namespace mutual_greetings_l3980_398066

theorem mutual_greetings (n : ℕ) (min_sent : ℕ) (h1 : n = 30) (h2 : min_sent = 16) :
  let total_sent := n * min_sent
  let total_pairs := n * (n - 1) / 2
  let mutual_greetings := {x : ℕ // x ≤ total_pairs ∧ 2 * x + (total_sent - 2 * x) ≤ total_sent}
  ∃ (x : mutual_greetings), x.val ≥ 45 :=
by sorry

end mutual_greetings_l3980_398066


namespace painted_cubes_theorem_l3980_398006

/-- Represents the dimensions of a parallelepiped -/
structure Parallelepiped where
  m : ℕ
  n : ℕ
  k : ℕ
  h1 : 0 < k
  h2 : k ≤ n
  h3 : n ≤ m

/-- The set of possible numbers of painted cubes -/
def PaintedCubesCounts : Set ℕ := {60, 72, 84, 90, 120}

/-- 
  Given a parallelepiped where three faces sharing a common vertex are painted,
  if half of all cubes have at least one painted face, then the number of
  painted cubes is in the set PaintedCubesCounts
-/
theorem painted_cubes_theorem (p : Parallelepiped) :
  (p.m - 1) * (p.n - 1) * (p.k - 1) = p.m * p.n * p.k / 2 →
  (p.m * p.n * p.k - (p.m - 1) * (p.n - 1) * (p.k - 1)) ∈ PaintedCubesCounts := by
  sorry

end painted_cubes_theorem_l3980_398006


namespace hundred_days_from_friday_is_sunday_l3980_398031

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDay d m)

-- Theorem statement
theorem hundred_days_from_friday_is_sunday :
  advanceDay DayOfWeek.Friday 100 = DayOfWeek.Sunday := by
  sorry

end hundred_days_from_friday_is_sunday_l3980_398031


namespace brownie_ratio_l3980_398014

def total_brownies : ℕ := 15
def monday_brownies : ℕ := 5

def tuesday_brownies : ℕ := total_brownies - monday_brownies

theorem brownie_ratio :
  (tuesday_brownies : ℚ) / monday_brownies = 2 / 1 := by
  sorry

end brownie_ratio_l3980_398014


namespace houses_around_square_l3980_398081

/-- The number of houses around the square. -/
def n : ℕ := 32

/-- Maria's count for a given house. -/
def M (k : ℕ) : ℕ := k % n

/-- João's count for a given house. -/
def J (k : ℕ) : ℕ := k % n

/-- Theorem stating the number of houses around the square. -/
theorem houses_around_square :
  (M 5 = J 12) ∧ (J 5 = M 30) → n = 32 := by
  sorry

end houses_around_square_l3980_398081


namespace approximate_cost_of_bicycle_and_fan_l3980_398098

/-- The cost of a bicycle in yuan -/
def bicycle_cost : ℕ := 389

/-- The cost of an electric fan in yuan -/
def fan_cost : ℕ := 189

/-- The approximate total cost of buying a bicycle and an electric fan -/
def approximate_total_cost : ℕ := 600

/-- Theorem stating that the approximate total cost is 600 yuan -/
theorem approximate_cost_of_bicycle_and_fan :
  ∃ (error : ℕ), bicycle_cost + fan_cost = approximate_total_cost + error ∧ error < 100 := by
  sorry

end approximate_cost_of_bicycle_and_fan_l3980_398098


namespace arithmetic_sequence_sum_l3980_398043

/-- The sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem: The sum of the first 10 terms of the arithmetic sequence
    with first term -5 and common difference 6 is 220 -/
theorem arithmetic_sequence_sum :
  arithmetic_sum (-5) 6 10 = 220 := by
  sorry

end arithmetic_sequence_sum_l3980_398043


namespace unique_n_exists_l3980_398017

theorem unique_n_exists : ∃! n : ℤ,
  50 ≤ n ∧ n ≤ 200 ∧
  8 ∣ n ∧
  n % 6 = 4 ∧
  n % 7 = 3 ∧
  n = 136 := by
sorry

end unique_n_exists_l3980_398017


namespace binomial_square_proof_l3980_398056

theorem binomial_square_proof :
  ∃ (r s : ℚ), (r * x + s)^2 = (100 / 9 : ℚ) * x^2 + 20 * x + 9 := by
  sorry

end binomial_square_proof_l3980_398056


namespace gcd_20244_46656_l3980_398030

theorem gcd_20244_46656 : Nat.gcd 20244 46656 = 54 := by
  sorry

end gcd_20244_46656_l3980_398030


namespace f_has_one_zero_l3980_398054

/-- The function f(x) defined in terms of the parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * (m + 1) * x - 1

/-- The set of real numbers m for which f(x) has exactly one zero -/
def one_zero_set : Set ℝ := {m : ℝ | m = -3 ∨ m = 0}

/-- Theorem stating that f(x) has exactly one zero if and only if m is in the one_zero_set -/
theorem f_has_one_zero (m : ℝ) : 
  (∃! x : ℝ, f m x = 0) ↔ m ∈ one_zero_set :=
sorry

end f_has_one_zero_l3980_398054


namespace circle_passes_through_point_l3980_398096

theorem circle_passes_through_point :
  ∀ (a b r : ℝ),
  b^2 = 8*a →                          -- Center (a, b) is on the parabola y² = 8x
  (a + 2)^2 + b^2 = r^2 →              -- Circle is tangent to the line x + 2 = 0
  (2 - a)^2 + b^2 = r^2 :=             -- Circle passes through (2, 0)
by
  sorry

end circle_passes_through_point_l3980_398096


namespace sqrt_greater_than_sum_l3980_398038

theorem sqrt_greater_than_sum (a b x y : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) 
  (hab : a^2 + b^2 < 1) : 
  Real.sqrt (x^2 + y^2) > a*x + b*y := by
sorry

end sqrt_greater_than_sum_l3980_398038


namespace dot_product_is_2020_l3980_398078

/-- A trapezoid with perpendicular diagonals -/
structure PerpendicularDiagonalTrapezoid where
  -- Points of the trapezoid
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- AB is a base of length 101
  AB_length : dist A B = 101
  -- CD is a base of length 20
  CD_length : dist C D = 20
  -- ABCD is a trapezoid (parallel sides)
  is_trapezoid : (B.1 - A.1) * (D.2 - C.2) = (D.1 - C.1) * (B.2 - A.2)
  -- Diagonals are perpendicular
  diagonals_perpendicular : (C.1 - A.1) * (D.1 - B.1) + (C.2 - A.2) * (D.2 - B.2) = 0

/-- The dot product of vectors AD and BC in a trapezoid with perpendicular diagonals -/
def dot_product_AD_BC (t : PerpendicularDiagonalTrapezoid) : ℝ :=
  let AD := (t.D.1 - t.A.1, t.D.2 - t.A.2)
  let BC := (t.C.1 - t.B.1, t.C.2 - t.B.2)
  AD.1 * BC.1 + AD.2 * BC.2

/-- Theorem: The dot product of AD and BC is 2020 -/
theorem dot_product_is_2020 (t : PerpendicularDiagonalTrapezoid) :
  dot_product_AD_BC t = 2020 := by
  sorry

end dot_product_is_2020_l3980_398078


namespace prove_c_value_l3980_398077

-- Define the variables
variable (c k x y z : ℝ)

-- Define the conditions
axiom model : y = c * Real.exp (k * x)
axiom log_transform : z = Real.log y
axiom regression : z = 0.4 * x + 2

-- Theorem to prove
theorem prove_c_value : c = Real.exp 2 := by
  sorry

end prove_c_value_l3980_398077


namespace arithmetic_sequence_11th_term_l3980_398009

/-- Given an arithmetic sequence {a_n} with first term a and common difference d,
    prove that the 11th term is 3.5 under the given conditions. -/
theorem arithmetic_sequence_11th_term
  (a d : ℝ)
  (h1 : a + (a + 3 * d) + (a + 6 * d) = 31.5)
  (h2 : 9 * a + (9 * 8 / 2) * d = 85.5) :
  a + 10 * d = 3.5 := by
  sorry

end arithmetic_sequence_11th_term_l3980_398009


namespace ball_pit_problem_l3980_398016

theorem ball_pit_problem (total_balls : ℕ) (red_fraction : ℚ) (neither_red_nor_blue : ℕ) :
  total_balls = 360 →
  red_fraction = 1/4 →
  neither_red_nor_blue = 216 →
  (total_balls - red_fraction * total_balls - neither_red_nor_blue) / 
  (total_balls - red_fraction * total_balls) = 1/5 := by
  sorry

end ball_pit_problem_l3980_398016


namespace total_flowering_bulbs_l3980_398088

/-- Calculates the total number of small flowering bulbs that can be purchased given the costs and constraints. -/
theorem total_flowering_bulbs 
  (crocus_cost : ℚ)
  (daffodil_cost : ℚ)
  (total_budget : ℚ)
  (crocus_count : ℕ)
  (h1 : crocus_cost = 35/100)
  (h2 : daffodil_cost = 65/100)
  (h3 : total_budget = 2915/100)
  (h4 : crocus_count = 22) :
  ∃ (daffodil_count : ℕ), 
    (crocus_count : ℚ) * crocus_cost + (daffodil_count : ℚ) * daffodil_cost ≤ total_budget ∧
    crocus_count + daffodil_count = 55 :=
by sorry

end total_flowering_bulbs_l3980_398088


namespace h_has_two_roots_l3980_398049

/-- The function f(x) = 2x -/
def f (x : ℝ) : ℝ := 2 * x

/-- The function g(x) = 3 - x^2 -/
def g (x : ℝ) : ℝ := 3 - x^2

/-- The function h(x) = f(x) - g(x) -/
def h (x : ℝ) : ℝ := f x - g x

/-- The theorem stating that h(x) has exactly two distinct real roots -/
theorem h_has_two_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ h x₁ = 0 ∧ h x₂ = 0 ∧ ∀ x, h x = 0 → x = x₁ ∨ x = x₂ :=
sorry

end h_has_two_roots_l3980_398049


namespace probability_ratio_l3980_398021

def total_slips : ℕ := 50
def distinct_numbers : ℕ := 10
def slips_per_number : ℕ := 5
def drawn_slips : ℕ := 5

def p : ℚ := distinct_numbers / Nat.choose total_slips drawn_slips

def q : ℚ := (Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 3 * Nat.choose slips_per_number 2) / Nat.choose total_slips drawn_slips

theorem probability_ratio :
  q / p = 450 := by sorry

end probability_ratio_l3980_398021


namespace peggy_needs_825_stamps_l3980_398027

/-- The number of stamps Peggy needs to add to have as many as Bert -/
def stamps_to_add (peggy_stamps : ℕ) : ℕ :=
  4 * (3 * peggy_stamps) - peggy_stamps

/-- Theorem stating that Peggy needs to add 825 stamps to have as many as Bert -/
theorem peggy_needs_825_stamps : stamps_to_add 75 = 825 := by
  sorry

end peggy_needs_825_stamps_l3980_398027


namespace unknown_number_is_three_l3980_398040

theorem unknown_number_is_three (x n : ℝ) (h1 : (3/2) * x - n = 15) (h2 : x = 12) : n = 3 := by
  sorry

end unknown_number_is_three_l3980_398040


namespace classroom_students_classroom_students_proof_l3980_398033

theorem classroom_students : ℕ → Prop :=
  fun S : ℕ =>
    let boys := S / 3
    let girls := S - boys
    let girls_with_dogs := (40 * girls) / 100
    let girls_with_cats := (20 * girls) / 100
    let girls_without_pets := girls - girls_with_dogs - girls_with_cats
    girls_without_pets = 8 → S = 30

-- The proof goes here
theorem classroom_students_proof : classroom_students 30 := by
  sorry

end classroom_students_classroom_students_proof_l3980_398033


namespace juggler_path_radius_l3980_398073

/-- The equation of the path described by the juggler's balls -/
def path_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 5 = 2*x + 4*y

/-- The radius of the path described by the juggler's balls -/
def path_radius : ℝ := 0

/-- Theorem stating that the radius of the path is 0 -/
theorem juggler_path_radius :
  ∀ x y : ℝ, path_equation x y → (x - 1)^2 + (y - 2)^2 = path_radius^2 :=
by
  sorry


end juggler_path_radius_l3980_398073


namespace max_profit_price_l3980_398023

/-- Represents the sales volume as a function of unit price -/
def sales_volume (x : ℝ) : ℝ := -2 * x + 100

/-- Represents the profit as a function of unit price -/
def profit (x : ℝ) : ℝ := (x - 20) * (sales_volume x)

/-- Theorem: The unit price that maximizes profit is 35 yuan -/
theorem max_profit_price : 
  ∃ (x : ℝ), x = 35 ∧ ∀ (y : ℝ), profit y ≤ profit x :=
sorry

end max_profit_price_l3980_398023


namespace airplane_distance_theorem_l3980_398001

/-- Calculates the distance traveled by an airplane given its speed and time. -/
def airplane_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that an airplane flying for 38 hours at 30 miles per hour travels 1140 miles. -/
theorem airplane_distance_theorem :
  let speed : ℝ := 30
  let time : ℝ := 38
  airplane_distance speed time = 1140 := by
  sorry

end airplane_distance_theorem_l3980_398001


namespace quadratic_roots_conditions_l3980_398072

/-- The quadratic equation x^2 + 2x + 2m = 0 has two distinct real roots -/
def has_two_distinct_real_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 2*m = 0 ∧ x₂^2 + 2*x₂ + 2*m = 0

/-- The sum of squares of the roots of x^2 + 2x + 2m = 0 is 8 -/
def sum_of_squares_is_8 (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁^2 + 2*x₁ + 2*m = 0 ∧ x₂^2 + 2*x₂ + 2*m = 0 ∧ x₁^2 + x₂^2 = 8

theorem quadratic_roots_conditions (m : ℝ) :
  (has_two_distinct_real_roots m ↔ m < 1/2) ∧
  (sum_of_squares_is_8 m → m = -1) :=
by sorry

end quadratic_roots_conditions_l3980_398072


namespace max_quantity_a_theorem_l3980_398012

/-- Represents the prices and quantities of fertilizers A and B -/
structure Fertilizers where
  price_a : ℝ
  price_b : ℝ
  quantity_a : ℝ
  quantity_b : ℝ

/-- Conditions for the fertilizer problem -/
def fertilizer_conditions (f : Fertilizers) : Prop :=
  f.price_a = f.price_b + 100 ∧
  2 * f.price_a + f.price_b = 1700 ∧
  f.quantity_a + f.quantity_b = 10 ∧
  f.quantity_a * f.price_a + f.quantity_b * f.price_b ≤ 5600

/-- The maximum quantity of fertilizer A that can be purchased -/
def max_quantity_a (f : Fertilizers) : ℝ := 6

/-- Theorem stating the maximum quantity of fertilizer A that can be purchased -/
theorem max_quantity_a_theorem (f : Fertilizers) :
  fertilizer_conditions f → f.quantity_a ≤ max_quantity_a f := by
  sorry

end max_quantity_a_theorem_l3980_398012


namespace arithmetic_sequence_common_difference_l3980_398071

/-- An arithmetic sequence {a_n} with a_1 = 2 and a_3 + a_5 = 10 has a common difference of 1. -/
theorem arithmetic_sequence_common_difference : ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 2 →                            -- first term condition
  a 3 + a 5 = 10 →                     -- sum of 3rd and 5th terms condition
  a 2 - a 1 = 1 :=                     -- common difference is 1
by sorry

end arithmetic_sequence_common_difference_l3980_398071


namespace shortest_altitude_of_triangle_l3980_398062

/-- Given a triangle with sides 13, 14, and 15, the shortest altitude has length 168/15 -/
theorem shortest_altitude_of_triangle (a b c : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let h1 := 2 * area / a
  let h2 := 2 * area / b
  let h3 := 2 * area / c
  min h1 (min h2 h3) = 168 / 15 := by
sorry

end shortest_altitude_of_triangle_l3980_398062


namespace area_of_special_triangle_l3980_398026

/-- Given points A and B on the graph of y = 1/x in the first quadrant,
    if ∠OAB = 90° and OA = AB, then the area of triangle OAB is √5/2 -/
theorem area_of_special_triangle (A B : ℝ × ℝ) : 
  (A.2 = 1 / A.1) →  -- A is on y = 1/x
  (B.2 = 1 / B.1) →  -- B is on y = 1/x
  (A.1 > 0 ∧ A.2 > 0) →  -- A is in first quadrant
  (B.1 > 0 ∧ B.2 > 0) →  -- B is in first quadrant
  (A.1 * (B.1 - A.1) + A.2 * (B.2 - A.2) = 0) →  -- ∠OAB = 90°
  (A.1^2 + A.2^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2) →  -- OA = AB
  (1/2 * Real.sqrt (A.1^2 + A.2^2) = Real.sqrt 5 / 2) := by
sorry

end area_of_special_triangle_l3980_398026


namespace kylie_coins_left_l3980_398082

/-- Calculates the number of coins Kylie is left with after various transactions --/
def coins_left (piggy_bank : ℕ) (from_brother : ℕ) (from_father : ℕ) (given_to_friend : ℕ) : ℕ :=
  piggy_bank + from_brother + from_father - given_to_friend

/-- Theorem stating that Kylie is left with 15 coins --/
theorem kylie_coins_left : 
  coins_left 15 13 8 21 = 15 := by
  sorry

end kylie_coins_left_l3980_398082


namespace soap_brand_ratio_l3980_398074

/-- Given a survey of households and their soap brand preferences, 
    prove the ratio of households using only brand B to those using both brands. -/
theorem soap_brand_ratio 
  (total : ℕ) 
  (neither : ℕ) 
  (only_w : ℕ) 
  (both : ℕ) 
  (h1 : total = 200)
  (h2 : neither = 80)
  (h3 : only_w = 60)
  (h4 : both = 40)
  : (total - neither - only_w - both) / both = 1 / 2 := by
  sorry

end soap_brand_ratio_l3980_398074


namespace wire_length_between_poles_l3980_398005

theorem wire_length_between_poles (base_distance : ℝ) (short_pole_height : ℝ) (tall_pole_height : ℝ) 
  (h1 : base_distance = 20)
  (h2 : short_pole_height = 10)
  (h3 : tall_pole_height = 22) :
  Real.sqrt (base_distance ^ 2 + (tall_pole_height - short_pole_height) ^ 2) = Real.sqrt 544 := by
  sorry

end wire_length_between_poles_l3980_398005


namespace max_projection_sum_l3980_398097

-- Define the plane as ℝ²
def Plane := ℝ × ℝ

-- Define the dot product for vectors in the plane
def dot_product (v w : Plane) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define what it means for a vector to be a unit vector
def is_unit_vector (v : Plane) : Prop := dot_product v v = 1

-- State the theorem
theorem max_projection_sum 
  (a b c : Plane) 
  (ha : is_unit_vector a) 
  (hb : is_unit_vector b) 
  (hc : is_unit_vector c) 
  (hab : a ≠ b) 
  (hbc : b ≠ c) 
  (hac : a ≠ c)
  (hab_dot : dot_product a b = 1/2) 
  (hbc_dot : dot_product b c = 1/2) :
  ∃ (max : ℝ), max = 5 ∧ 
    ∀ (e : Plane), is_unit_vector e → 
      |dot_product a e| + |2 * dot_product b e| + 3 * |dot_product c e| ≤ max :=
sorry

end max_projection_sum_l3980_398097


namespace sin_negative_1665_degrees_l3980_398029

theorem sin_negative_1665_degrees :
  Real.sin ((-1665 : ℝ) * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_negative_1665_degrees_l3980_398029


namespace radio_selling_price_l3980_398087

/-- Calculates the selling price of a radio given its purchase price, overhead expenses, and profit percentage. -/
def calculate_selling_price (purchase_price : ℚ) (overhead_expenses : ℚ) (profit_percentage : ℚ) : ℚ :=
  let total_cost := purchase_price + overhead_expenses
  let profit_amount := (profit_percentage / 100) * total_cost
  total_cost + profit_amount

/-- Theorem stating that the selling price of a radio with given parameters is 350 Rs. -/
theorem radio_selling_price :
  let purchase_price : ℚ := 225
  let overhead_expenses : ℚ := 15
  let profit_percentage : ℚ := 45833333333333314 / 1000000000000000
  calculate_selling_price purchase_price overhead_expenses profit_percentage = 350 := by
  sorry


end radio_selling_price_l3980_398087


namespace books_per_day_calculation_l3980_398090

/-- Calculates the number of books read per day given the total books read and the number of reading days. -/
def books_per_day (total_books : ℕ) (reading_days : ℕ) : ℚ :=
  (total_books : ℚ) / (reading_days : ℚ)

/-- Represents the reading habits of a person over a period of weeks. -/
structure ReadingHabit where
  days_per_week : ℕ
  weeks : ℕ
  total_books : ℕ

theorem books_per_day_calculation (habit : ReadingHabit) 
    (h1 : habit.days_per_week = 2)
    (h2 : habit.weeks = 6)
    (h3 : habit.total_books = 48) :
  books_per_day habit.total_books (habit.days_per_week * habit.weeks) = 4 := by
  sorry

end books_per_day_calculation_l3980_398090


namespace charge_200_400_undetermined_l3980_398057

/-- Represents the monthly phone bill for a customer -/
structure PhoneBill where
  fixed_rental : ℝ
  free_calls : ℕ
  charge_200_400 : ℝ
  charge_400_plus : ℝ
  february_calls : ℕ
  march_calls : ℕ
  march_discount : ℝ

/-- The phone bill satisfies the given conditions -/
def satisfies_conditions (bill : PhoneBill) : Prop :=
  bill.fixed_rental = 350 ∧
  bill.free_calls = 200 ∧
  bill.charge_400_plus = 1.6 ∧
  bill.february_calls = 150 ∧
  bill.march_calls = 250 ∧
  bill.march_discount = 0.28

/-- Theorem stating that the charge per call when exceeding 200 calls cannot be determined -/
theorem charge_200_400_undetermined (bill : PhoneBill) 
  (h : satisfies_conditions bill) : 
  ¬ ∃ (x : ℝ), ∀ (b : PhoneBill), satisfies_conditions b → b.charge_200_400 = x :=
sorry

end charge_200_400_undetermined_l3980_398057


namespace basketball_cost_l3980_398025

theorem basketball_cost (initial_amount : ℕ) (jersey_cost : ℕ) (jersey_count : ℕ) (shorts_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 50 →
  jersey_cost = 2 →
  jersey_count = 5 →
  shorts_cost = 8 →
  remaining_amount = 14 →
  initial_amount - (jersey_cost * jersey_count + shorts_cost + remaining_amount) = 18 :=
by sorry

end basketball_cost_l3980_398025


namespace relay_race_total_time_l3980_398024

/-- The time taken by the relay team to finish the race -/
def relay_race_time (mary_time susan_time jen_time tiffany_time : ℕ) : ℕ :=
  mary_time + susan_time + jen_time + tiffany_time

/-- Theorem stating the total time for the relay race -/
theorem relay_race_total_time : ∃ (mary_time susan_time jen_time tiffany_time : ℕ),
  mary_time = 2 * susan_time ∧
  susan_time = jen_time + 10 ∧
  jen_time = 30 ∧
  tiffany_time = mary_time - 7 ∧
  relay_race_time mary_time susan_time jen_time tiffany_time = 223 := by
  sorry


end relay_race_total_time_l3980_398024


namespace polynomial_derivative_sum_l3980_398063

theorem polynomial_derivative_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (5*x - 4)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 25 := by
sorry

end polynomial_derivative_sum_l3980_398063


namespace monotone_increasing_condition_l3980_398085

theorem monotone_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Ioo (π/6) (π/3), Monotone (fun x => (a - Real.sin x) / Real.cos x)) →
  a ≥ 2 := by
  sorry

end monotone_increasing_condition_l3980_398085


namespace geometric_sum_four_l3980_398042

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sum_four (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  a 3 = 4 →
  a 2 + a 4 = -10 →
  |q| > 1 →
  a 1 + a 2 + a 3 + a 4 = -5 :=
by
  sorry

end geometric_sum_four_l3980_398042


namespace square_sum_ge_double_product_l3980_398051

theorem square_sum_ge_double_product (a b : ℝ) : a^2 + b^2 ≥ 2*a*b := by
  sorry

end square_sum_ge_double_product_l3980_398051


namespace buddy_program_fraction_l3980_398065

theorem buddy_program_fraction (s n : ℕ) (hs : s > 0) (hn : n > 0) : 
  (n / 4 : ℚ) = (s / 3 : ℚ) → 
  ((n / 4 + s / 3) / (n + s) : ℚ) = 2 / 7 := by
sorry

end buddy_program_fraction_l3980_398065


namespace probability_red_or_blue_specific_l3980_398059

/-- The probability of drawing either a red or blue marble from a bag -/
def probability_red_or_blue (red blue green yellow : ℕ) : ℚ :=
  (red + blue : ℚ) / (red + blue + green + yellow : ℚ)

/-- Theorem: The probability of drawing either a red or blue marble from a bag
    containing 5 red, 3 blue, 4 green, and 6 yellow marbles is 4/9 -/
theorem probability_red_or_blue_specific : probability_red_or_blue 5 3 4 6 = 4/9 := by
  sorry

end probability_red_or_blue_specific_l3980_398059


namespace parallel_vectors_imply_m_minus_n_equals_8_l3980_398099

def vector_a : Fin 3 → ℝ := ![1, 3, -2]
def vector_b (m n : ℝ) : Fin 3 → ℝ := ![2, m + 1, n - 1]

def parallel (u v : Fin 3 → ℝ) : Prop :=
  ∃ k : ℝ, ∀ i : Fin 3, v i = k * u i

theorem parallel_vectors_imply_m_minus_n_equals_8 (m n : ℝ) :
  parallel vector_a (vector_b m n) → m - n = 8 := by
  sorry

end parallel_vectors_imply_m_minus_n_equals_8_l3980_398099


namespace max_value_is_16_l3980_398080

/-- A function f(x) that is symmetric about x = -2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := (1 - x^2) * (x^2 + a*x + b)

/-- Symmetry condition: f(x) = f(-4-x) for all x -/
def is_symmetric (a b : ℝ) : Prop :=
  ∀ x, f a b x = f a b (-4-x)

/-- The maximum value of f(x) is 16 -/
theorem max_value_is_16 (a b : ℝ) (h : is_symmetric a b) :
  ∃ M, M = 16 ∧ ∀ x, f a b x ≤ M :=
sorry

end max_value_is_16_l3980_398080


namespace sum_of_squares_of_factors_72_l3980_398064

def sum_of_squares_of_factors (n : ℕ) : ℕ := sorry

theorem sum_of_squares_of_factors_72 : sum_of_squares_of_factors 72 = 7735 := by sorry

end sum_of_squares_of_factors_72_l3980_398064


namespace solution_y_original_amount_l3980_398035

/-- Represents the composition of a solution --/
structure Solution where
  total : ℝ
  liquid_x_percent : ℝ
  water_percent : ℝ

/-- The problem statement --/
theorem solution_y_original_amount
  (y : Solution)
  (h1 : y.liquid_x_percent = 0.3)
  (h2 : y.water_percent = 0.7)
  (h3 : y.liquid_x_percent + y.water_percent = 1)
  (evaporated_water : ℝ)
  (h4 : evaporated_water = 4)
  (added_solution : Solution)
  (h5 : added_solution.total = 4)
  (h6 : added_solution.liquid_x_percent = 0.3)
  (h7 : added_solution.water_percent = 0.7)
  (new_solution : Solution)
  (h8 : new_solution.total = y.total)
  (h9 : new_solution.liquid_x_percent = 0.45)
  (h10 : y.total * y.liquid_x_percent + added_solution.total * added_solution.liquid_x_percent
       = new_solution.total * new_solution.liquid_x_percent) :
  y.total = 8 := by
  sorry


end solution_y_original_amount_l3980_398035


namespace find_e_l3980_398070

theorem find_e : ∃ e : ℕ, (1/5 : ℝ)^e * (1/4 : ℝ)^18 = 1 / (2 * 10^35) ∧ e = 35 := by
  sorry

end find_e_l3980_398070


namespace remainder_2_1000_mod_17_l3980_398095

theorem remainder_2_1000_mod_17 (h : Prime 17) : 2^1000 % 17 = 0 := by
  sorry

end remainder_2_1000_mod_17_l3980_398095


namespace smallest_c_and_b_for_real_roots_l3980_398013

theorem smallest_c_and_b_for_real_roots (c b : ℝ) : 
  (∀ x : ℝ, x^4 - c*x^3 + b*x^2 - c*x + 1 = 0 → x > 0) →
  (c > 0) →
  (b > 0) →
  (∀ c' b' : ℝ, c' > 0 → b' > 0 → 
    (∀ x : ℝ, x^4 - c'*x^3 + b'*x^2 - c'*x + 1 = 0 → x > 0) →
    c ≤ c') →
  c = 4 ∧ b = 6 := by
sorry

end smallest_c_and_b_for_real_roots_l3980_398013


namespace inequality_proof_l3980_398003

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 1/3) :
  (1 - a) * (1 - b) ≤ 25/36 := by
  sorry

end inequality_proof_l3980_398003


namespace special_quadrilateral_angles_l3980_398089

/-- A quadrilateral with three equal sides and two specific angles -/
structure SpecialQuadrilateral where
  -- Three equal sides
  side : ℝ
  side_positive : side > 0
  -- Two angles formed by the equal sides
  angle1 : ℝ
  angle2 : ℝ
  -- Angle conditions
  angle1_is_90 : angle1 = 90
  angle2_is_150 : angle2 = 150

/-- The other two angles of the special quadrilateral -/
def other_angles (q : SpecialQuadrilateral) : ℝ × ℝ :=
  (45, 75)

/-- Theorem stating that the other two angles are 45° and 75° -/
theorem special_quadrilateral_angles (q : SpecialQuadrilateral) :
  other_angles q = (45, 75) := by
  sorry

end special_quadrilateral_angles_l3980_398089


namespace line_charts_reflect_changes_l3980_398010

/-- Represents a line chart --/
structure LineChart where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a change in data over time or across categories --/
structure DataChange where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Defines the property of a line chart being able to reflect changes clearly --/
def clearly_reflects_changes (chart : LineChart) : Prop :=
  ∀ (change : DataChange), ∃ (representation : LineChart → DataChange → Prop),
    representation chart change ∧ 
    (∀ (other_chart : LineChart), representation other_chart change → other_chart = chart)

/-- Theorem stating that line charts can clearly reflect changes in things --/
theorem line_charts_reflect_changes :
  ∀ (chart : LineChart), clearly_reflects_changes chart :=
sorry

end line_charts_reflect_changes_l3980_398010


namespace special_line_properties_l3980_398022

/-- A line passing through (2,3) with x-intercept twice the y-intercept -/
def special_line (x y : ℝ) : Prop := x + 2*y - 8 = 0

theorem special_line_properties :
  (special_line 2 3) ∧ 
  (∃ (a : ℝ), a ≠ 0 ∧ special_line (2*a) 0 ∧ special_line 0 a) :=
by sorry

end special_line_properties_l3980_398022


namespace symmetric_point_example_l3980_398053

/-- Given a point A and a point of symmetry, find the symmetric point -/
def symmetric_point (A : ℝ × ℝ × ℝ) (sym : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (2 * sym.1 - A.1, 2 * sym.2.1 - A.2.1, 2 * sym.2.2 - A.2.2)

/-- The theorem states that the symmetric point of A(3, -2, 4) with respect to (0, 1, -3) is (-3, 4, -10) -/
theorem symmetric_point_example : 
  symmetric_point (3, -2, 4) (0, 1, -3) = (-3, 4, -10) := by
  sorry

#eval symmetric_point (3, -2, 4) (0, 1, -3)

end symmetric_point_example_l3980_398053


namespace paint_on_third_day_l3980_398039

/-- The amount of paint available on the third day of a room refresh project -/
theorem paint_on_third_day (initial_paint : ℝ) (added_paint : ℝ) : 
  initial_paint = 80 → 
  added_paint = 20 → 
  (initial_paint / 2 + added_paint) / 2 = 30 := by
sorry

end paint_on_third_day_l3980_398039


namespace initial_meals_correct_l3980_398068

/-- The number of meals Colt and Curt initially prepared -/
def initial_meals : ℕ := 113

/-- The number of meals Sole Mart provided -/
def sole_mart_meals : ℕ := 50

/-- The number of meals given away -/
def meals_given_away : ℕ := 85

/-- The number of meals left to be distributed -/
def meals_left : ℕ := 78

/-- Theorem stating that the initial number of meals is correct -/
theorem initial_meals_correct : 
  initial_meals + sole_mart_meals = meals_given_away + meals_left := by
  sorry

end initial_meals_correct_l3980_398068


namespace roots_are_correct_all_roots_found_l3980_398069

/-- The roots of the equation 5x^4 - 28x^3 + 49x^2 - 28x + 5 = 0 -/
def roots : Set ℝ :=
  {2, 1/2, (5 + Real.sqrt 21)/5, (5 - Real.sqrt 21)/5}

/-- The polynomial function corresponding to the equation -/
def f (x : ℝ) : ℝ := 5*x^4 - 28*x^3 + 49*x^2 - 28*x + 5

theorem roots_are_correct : ∀ x ∈ roots, f x = 0 := by
  sorry

theorem all_roots_found : ∀ x, f x = 0 → x ∈ roots := by
  sorry

end roots_are_correct_all_roots_found_l3980_398069


namespace high_school_student_distribution_l3980_398032

theorem high_school_student_distribution :
  ∀ (total juniors sophomores freshmen seniors : ℕ),
    total = 800 →
    juniors = (27 * total) / 100 →
    sophomores = total - (75 * total) / 100 →
    seniors = 160 →
    freshmen = total - (juniors + sophomores + seniors) →
    freshmen - sophomores = 24 :=
by
  sorry

end high_school_student_distribution_l3980_398032


namespace vegetable_planting_methods_l3980_398036

theorem vegetable_planting_methods (n m : ℕ) (hn : n = 4) (hm : m = 3) :
  (n.choose m) * (m.factorial) = 24 := by
  sorry

end vegetable_planting_methods_l3980_398036


namespace quadratic_roots_relation_l3980_398037

theorem quadratic_roots_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∃ r₁ r₂ : ℝ, (r₁ + r₂ = -p ∧ r₁ * r₂ = m) ∧
               (3 * r₁ + 3 * r₂ = -m ∧ 9 * r₁ * r₂ = n)) →
  n / p = 27 := by
  sorry

end quadratic_roots_relation_l3980_398037


namespace inequality_solution_set_l3980_398008

theorem inequality_solution_set (x : ℝ) (h : x ≠ 0) :
  (1 / x^2 - 2 / x - 3 < 0) ↔ (x < -1 ∨ x > 1/3) :=
by sorry

end inequality_solution_set_l3980_398008


namespace probability_spade_face_diamond_l3980_398015

/-- Represents a standard 52-card deck --/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (h : cards.card = 52)

/-- Represents a suit in a deck of cards --/
inductive Suit
| Spade | Heart | Diamond | Club

/-- Represents a rank in a deck of cards --/
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- Checks if a rank is a face card --/
def is_face_card (r : Rank) : Bool :=
  match r with
  | Rank.Jack | Rank.Queen | Rank.King => true
  | _ => false

/-- Calculates the probability of drawing three specific cards --/
def probability_three_cards (d : Deck) (first : Suit) (second : Rank → Bool) (third : Suit) : ℚ :=
  sorry

/-- Theorem stating the probability of drawing a spade, then a face card, then a diamond --/
theorem probability_spade_face_diamond (d : Deck) :
  probability_three_cards d Suit.Spade is_face_card Suit.Diamond = 1911 / 132600 :=
sorry

end probability_spade_face_diamond_l3980_398015


namespace bacteria_growth_l3980_398061

/-- The number of times a bacteria culture doubles in 4 minutes -/
def doublings : ℕ := 240 / 30

/-- The final number of bacteria after 4 minutes -/
def final_count : ℕ := 524288

theorem bacteria_growth (n : ℕ) : n * 2^doublings = final_count ↔ n = 2048 := by
  sorry

end bacteria_growth_l3980_398061


namespace M_remainder_mod_32_l3980_398046

def M : ℕ := (List.filter (fun p => Nat.Prime p ∧ p % 2 = 1) (List.range 32)).prod

theorem M_remainder_mod_32 : M % 32 = 17 := by sorry

end M_remainder_mod_32_l3980_398046


namespace mans_speed_with_current_l3980_398094

/-- Given a man's speed against the current and the speed of the current,
    calculate the man's speed with the current. -/
def speed_with_current (speed_against_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- Theorem stating that given the specific conditions, 
    the man's speed with the current is 21 km/hr. -/
theorem mans_speed_with_current :
  let speed_against_current : ℝ := 16
  let current_speed : ℝ := 2.5
  speed_with_current speed_against_current current_speed = 21 := by
  sorry

#eval speed_with_current 16 2.5

end mans_speed_with_current_l3980_398094


namespace not_in_third_quadrant_l3980_398067

def linear_function (x : ℝ) : ℝ := -2 * x + 5

theorem not_in_third_quadrant :
  ∀ x y : ℝ, y = linear_function x → ¬(x < 0 ∧ y < 0) :=
by sorry

end not_in_third_quadrant_l3980_398067


namespace largest_quantity_l3980_398028

def A : ℚ := 3004 / 3003 + 3004 / 3005
def B : ℚ := 3006 / 3005 + 3006 / 3007
def C : ℚ := 3005 / 3004 + 3005 / 3006

theorem largest_quantity : A > B ∧ A > C := by
  sorry

end largest_quantity_l3980_398028


namespace age_problem_l3980_398020

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 22 →
  b = 8 :=
by sorry

end age_problem_l3980_398020


namespace floor_sum_count_l3980_398091

def count_integers (max : ℕ) : ℕ :=
  let count_for_form (k : ℕ) := (max - k) / 7 + 1
  (count_for_form 0) + (count_for_form 1) + (count_for_form 3) + (count_for_form 4)

theorem floor_sum_count :
  count_integers 1000 = 568 := by sorry

end floor_sum_count_l3980_398091


namespace problem_proof_l3980_398058

theorem problem_proof (a b : ℝ) (ha : a > 0) (h : Real.exp a + Real.log b = 1) :
  a * b < 1 ∧ a + b > 1 ∧ Real.exp a + b > 2 := by
  sorry

end problem_proof_l3980_398058


namespace candy_sales_theorem_l3980_398076

-- Define the candy sales for each week
structure CandySales :=
  (week1_initial : ℕ)
  (week1_monday : ℕ)
  (week1_tuesday : ℕ)
  (week1_wednesday_left : ℕ)
  (week2_initial : ℕ)
  (week2_monday : ℕ)
  (week2_tuesday : ℕ)
  (week2_wednesday : ℕ)
  (week2_thursday : ℕ)
  (week2_friday : ℕ)
  (week3_initial : ℕ)
  (week3_highest : ℕ)

-- Define the theorem
theorem candy_sales_theorem (sales : CandySales) 
  (h1 : sales.week1_initial = 80)
  (h2 : sales.week1_monday = 15)
  (h3 : sales.week1_tuesday = 2 * sales.week1_monday)
  (h4 : sales.week1_wednesday_left = 7)
  (h5 : sales.week2_initial = 100)
  (h6 : sales.week2_monday = 12)
  (h7 : sales.week2_tuesday = 18)
  (h8 : sales.week2_wednesday = 20)
  (h9 : sales.week2_thursday = 11)
  (h10 : sales.week2_friday = 25)
  (h11 : sales.week3_initial = 120)
  (h12 : sales.week3_highest = 40) :
  (sales.week1_initial - sales.week1_wednesday_left = 73) ∧
  (sales.week2_monday + sales.week2_tuesday + sales.week2_wednesday + sales.week2_thursday + sales.week2_friday = 86) ∧
  (sales.week3_highest = 40) := by
  sorry

end candy_sales_theorem_l3980_398076


namespace parabola_with_same_shape_and_vertex_l3980_398007

/-- A parabola with the same shape and opening direction as y = -3x^2 + 1 and vertex at (-1, 2) -/
theorem parabola_with_same_shape_and_vertex (x y : ℝ) : 
  y = -3 * (x + 1)^2 + 2 → 
  (∃ (a b c : ℝ), y = -3 * x^2 + b * x + c) ∧ 
  (y = -3 * (-1)^2 + 2 ∧ ∀ (h : ℝ), y ≤ -3 * (h + 1)^2 + 2) :=
sorry

end parabola_with_same_shape_and_vertex_l3980_398007


namespace find_b_l3980_398092

-- Define the real number √3
noncomputable def sqrt3 : ℝ := Real.sqrt 3

-- Define the equation (1 + √3)^5 = a + b√3
def equation (a b : ℚ) : Prop := (1 + sqrt3) ^ 5 = a + b * sqrt3

-- Theorem statement
theorem find_b : ∃ (a b : ℚ), equation a b ∧ b = 44 := by
  sorry

end find_b_l3980_398092
