import Mathlib

namespace acrobat_weight_l2820_282023

/-- Given weights of various objects, prove that an acrobat weighs twice as much as a lamb -/
theorem acrobat_weight (barrel dog acrobat lamb coil : ℝ) 
  (h1 : acrobat + dog = 2 * barrel)
  (h2 : dog = 2 * coil)
  (h3 : lamb + coil = barrel) :
  acrobat = 2 * lamb := by
  sorry

end acrobat_weight_l2820_282023


namespace inequality_solution_and_minimum_l2820_282095

-- Define the solution set
def solution_set (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4

-- Define the inequality
def inequality (x m n : ℝ) : Prop := |x - m| ≤ n

-- Define the constraint on a and b
def constraint (a b m n : ℝ) : Prop := a > 0 ∧ b > 0 ∧ a + b = m / a + n / b

theorem inequality_solution_and_minimum (m n : ℝ) :
  (∀ x, inequality x m n ↔ solution_set x) →
  (m = 2 ∧ n = 2) ∧
  (∀ a b, constraint a b m n → a + b ≥ 2 * Real.sqrt 2) ∧
  (∃ a b, constraint a b m n ∧ a + b = 2 * Real.sqrt 2) :=
sorry

end inequality_solution_and_minimum_l2820_282095


namespace circle_coloring_exists_l2820_282006

/-- A type representing the two colors we can use -/
inductive Color
  | Red
  | Blue

/-- A type representing a region in the plane -/
structure Region

/-- A type representing a circle in the plane -/
structure Circle

/-- A function that determines if two regions are adjacent (separated by an arc of a circle) -/
def adjacent (r1 r2 : Region) : Prop := sorry

/-- A coloring function that assigns a color to each region -/
def coloring (r : Region) : Color := sorry

/-- The main theorem stating that a valid coloring exists for any number of circles -/
theorem circle_coloring_exists (n : ℕ) (h : n ≥ 1) :
  ∃ (circles : Finset Circle) (regions : Finset Region),
    circles.card = n ∧
    (∀ r1 r2 : Region, r1 ∈ regions → r2 ∈ regions → adjacent r1 r2 → coloring r1 ≠ coloring r2) :=
  sorry

end circle_coloring_exists_l2820_282006


namespace consecutive_integers_base_sum_l2820_282098

/-- Given two consecutive positive integers X and Y, 
    if 241 in base X plus 52 in base Y equals 194 in base (X+Y), 
    then X + Y equals 15 -/
theorem consecutive_integers_base_sum (X Y : ℕ) : 
  X > 0 ∧ Y > 0 ∧ Y = X + 1 →
  (2 * X^2 + 4 * X + 1) + (5 * Y + 2) = ((X + Y)^2 + 9 * (X + Y) + 4) →
  X + Y = 15 := by
  sorry

end consecutive_integers_base_sum_l2820_282098


namespace solve_equation_l2820_282080

theorem solve_equation : ∃ x : ℝ, (3 * x) / 4 = 24 ∧ x = 32 := by
  sorry

end solve_equation_l2820_282080


namespace unique_x_value_l2820_282016

theorem unique_x_value : ∃! x : ℤ, 
  1 < x ∧ x < 9 ∧ 
  2 < x ∧ x < 15 ∧ 
  -1 < x ∧ x < 7 ∧ 
  0 < x ∧ x < 4 ∧ 
  x + 1 < 5 :=
by
  sorry

end unique_x_value_l2820_282016


namespace a_is_geometric_sequence_l2820_282050

/-- A linear function f(x) = bx + 1 where b is a constant not equal to 1 -/
def f (b : ℝ) (x : ℝ) : ℝ := b * x + 1

/-- A recursive function g(n) defined as:
    g(0) = 1
    g(n) = f(g(n-1)) for n ≥ 1 -/
def g (b : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => f b (g b n)

/-- The sequence a_n defined as a_n = g(n) - g(n-1) for n ∈ ℕ* -/
def a (b : ℝ) (n : ℕ) : ℝ := g b (n + 1) - g b n

/-- Theorem: The sequence {a_n} is a geometric sequence -/
theorem a_is_geometric_sequence (b : ℝ) (h : b ≠ 1) :
  ∃ r : ℝ, ∀ n : ℕ, a b (n + 1) = r * a b n :=
sorry

end a_is_geometric_sequence_l2820_282050


namespace words_exceeded_proof_l2820_282088

def word_limit : ℕ := 1000
def saturday_words : ℕ := 450
def sunday_words : ℕ := 650

theorem words_exceeded_proof :
  (saturday_words + sunday_words) - word_limit = 100 := by
  sorry

end words_exceeded_proof_l2820_282088


namespace right_triangle_area_l2820_282019

/-- The area of a right triangle with base 8 and hypotenuse 10 is 24 square units. -/
theorem right_triangle_area : 
  ∀ (base height hypotenuse : ℝ),
  base = 8 →
  hypotenuse = 10 →
  base ^ 2 + height ^ 2 = hypotenuse ^ 2 →
  (1 / 2) * base * height = 24 :=
by
  sorry

end right_triangle_area_l2820_282019


namespace money_division_l2820_282052

/-- Proof that the total sum of money is $320 given the specified conditions -/
theorem money_division (a b c d : ℝ) : 
  (∀ (x : ℝ), b = 0.75 * x → c = 0.5 * x → d = 0.25 * x → a = x) →
  c = 64 →
  a + b + c + d = 320 := by
  sorry

end money_division_l2820_282052


namespace balance_after_transactions_l2820_282093

def football_club_balance (initial_balance : ℝ) (players_sold : ℕ) (selling_price : ℝ) (players_bought : ℕ) (buying_price : ℝ) : ℝ :=
  initial_balance + players_sold * selling_price - players_bought * buying_price

theorem balance_after_transactions :
  football_club_balance 100 2 10 4 15 = 60 := by
  sorry

end balance_after_transactions_l2820_282093


namespace jean_sale_savings_l2820_282002

/-- Represents the total savings during a jean sale -/
def total_savings (fox_price pony_price : ℚ) (fox_discount pony_discount : ℚ) (fox_quantity pony_quantity : ℕ) : ℚ :=
  (fox_price * fox_quantity * fox_discount / 100) + (pony_price * pony_quantity * pony_discount / 100)

/-- Theorem stating the total savings during the jean sale -/
theorem jean_sale_savings :
  let fox_price : ℚ := 15
  let pony_price : ℚ := 18
  let fox_quantity : ℕ := 3
  let pony_quantity : ℕ := 2
  let pony_discount : ℚ := 13.999999999999993
  let fox_discount : ℚ := 22 - pony_discount
  total_savings fox_price pony_price fox_discount pony_discount fox_quantity pony_quantity = 864 / 100 :=
by
  sorry


end jean_sale_savings_l2820_282002


namespace algorithmC_is_best_l2820_282079

-- Define the durations of each task
def washAndBrush : ℕ := 5
def cleanKettle : ℕ := 2
def boilWater : ℕ := 8
def makeNoodles : ℕ := 3
def eat : ℕ := 10
def listenRadio : ℕ := 8

-- Define the algorithms
def algorithmA : ℕ := washAndBrush + cleanKettle + boilWater + makeNoodles + eat + listenRadio
def algorithmB : ℕ := cleanKettle + max boilWater washAndBrush + makeNoodles + eat + listenRadio
def algorithmC : ℕ := cleanKettle + max boilWater washAndBrush + makeNoodles + max eat listenRadio
def algorithmD : ℕ := max eat listenRadio + makeNoodles + max boilWater washAndBrush + cleanKettle

-- Theorem stating that algorithm C takes the least time
theorem algorithmC_is_best : 
  algorithmC ≤ algorithmA ∧ 
  algorithmC ≤ algorithmB ∧ 
  algorithmC ≤ algorithmD :=
sorry

end algorithmC_is_best_l2820_282079


namespace painted_faces_count_l2820_282029

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Represents a painted cube -/
structure PaintedCube (n : ℕ) extends Cube n where
  is_painted : Bool := true

/-- Counts the number of smaller cubes with at least two painted faces when a painted cube is cut into unit cubes -/
def count_painted_faces (c : PaintedCube 4) : ℕ :=
  sorry

/-- Theorem stating that a 4x4x4 painted cube cut into 1x1x1 cubes has 32 smaller cubes with at least two painted faces -/
theorem painted_faces_count (c : PaintedCube 4) : count_painted_faces c = 32 :=
  sorry

end painted_faces_count_l2820_282029


namespace product_of_three_numbers_l2820_282047

theorem product_of_three_numbers (a b c : ℝ) : 
  a + b + c = 45 ∧ 
  a = 2 * (b + c) ∧ 
  c = 4 * b → 
  a * b * c = 1080 := by
  sorry

end product_of_three_numbers_l2820_282047


namespace student_tickets_sold_l2820_282028

/-- Proves the number of student tickets sold given ticket prices and total sales information -/
theorem student_tickets_sold
  (adult_price : ℝ)
  (student_price : ℝ)
  (total_tickets : ℕ)
  (total_amount : ℝ)
  (h1 : adult_price = 4)
  (h2 : student_price = 2.5)
  (h3 : total_tickets = 59)
  (h4 : total_amount = 222.5) :
  ∃ (student_tickets : ℕ),
    student_tickets = 9 ∧
    (total_tickets - student_tickets) * adult_price + student_tickets * student_price = total_amount :=
by
  sorry

#check student_tickets_sold

end student_tickets_sold_l2820_282028


namespace complex_modulus_problem_l2820_282000

theorem complex_modulus_problem (z : ℂ) (h : z * (1 - Complex.I)^2 = 1 + Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_modulus_problem_l2820_282000


namespace sine_cosine_inequality_non_obtuse_triangle_l2820_282058

/-- For any non-obtuse triangle with angles α, β, and γ, the sum of the sines of these angles 
is greater than the sum of the cosines of these angles. -/
theorem sine_cosine_inequality_non_obtuse_triangle (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi)
  (h_non_obtuse : α ≤ Real.pi/2 ∧ β ≤ Real.pi/2 ∧ γ ≤ Real.pi/2) :
  Real.sin α + Real.sin β + Real.sin γ > Real.cos α + Real.cos β + Real.cos γ :=
sorry

end sine_cosine_inequality_non_obtuse_triangle_l2820_282058


namespace negation_of_not_all_zero_l2820_282042

theorem negation_of_not_all_zero (a b c : ℝ) :
  ¬(¬(a = 0 ∧ b = 0 ∧ c = 0)) ↔ (a = 0 ∧ b = 0 ∧ c = 0) := by
  sorry

end negation_of_not_all_zero_l2820_282042


namespace bob_distance_from_start_l2820_282090

-- Define the regular pentagon
def regularPentagon (sideLength : ℝ) : Set (ℝ × ℝ) :=
  sorry

-- Define Bob's position after walking a certain distance
def bobPosition (distance : ℝ) : ℝ × ℝ :=
  sorry

-- Theorem statement
theorem bob_distance_from_start :
  let pentagon := regularPentagon 3
  let finalPosition := bobPosition 7
  let distance := Real.sqrt ((finalPosition.1)^2 + (finalPosition.2)^2)
  distance = Real.sqrt 6.731 := by
  sorry

end bob_distance_from_start_l2820_282090


namespace function_triple_is_linear_l2820_282046

/-- A triple of injective functions from ℝ to ℝ satisfying specific conditions -/
structure FunctionTriple where
  f : ℝ → ℝ
  g : ℝ → ℝ
  h : ℝ → ℝ
  f_injective : Function.Injective f
  g_injective : Function.Injective g
  h_injective : Function.Injective h
  eq1 : ∀ x y, f (x + f y) = g x + h y
  eq2 : ∀ x y, g (x + g y) = h x + f y
  eq3 : ∀ x y, h (x + h y) = f x + g y

/-- The main theorem stating that any FunctionTriple consists of linear functions with the same constant term -/
theorem function_triple_is_linear (t : FunctionTriple) : 
  ∃ C : ℝ, ∀ x : ℝ, t.f x = x + C ∧ t.g x = x + C ∧ t.h x = x + C := by
  sorry


end function_triple_is_linear_l2820_282046


namespace object_length_increase_l2820_282061

/-- The daily increase factor for the object's length on day n -/
def daily_factor (n : ℕ) : ℚ := (n + 3) / (n + 2)

/-- The total multiplication factor after n days -/
def total_factor (n : ℕ) : ℚ := (n + 3) / 3

theorem object_length_increase (n : ℕ) : 
  n = 147 → total_factor n = 50 := by
  sorry

end object_length_increase_l2820_282061


namespace number_of_dimes_l2820_282024

/-- Proves the number of dimes given the number of pennies, nickels, and total value -/
theorem number_of_dimes 
  (num_pennies : ℕ) 
  (num_nickels : ℕ) 
  (total_value : ℚ) 
  (h_num_pennies : num_pennies = 9)
  (h_num_nickels : num_nickels = 4)
  (h_total_value : total_value = 59 / 100)
  (h_penny_value : ∀ n : ℕ, n * (1 / 100 : ℚ) = (n : ℚ) / 100)
  (h_nickel_value : ∀ n : ℕ, n * (5 / 100 : ℚ) = (5 * n : ℚ) / 100)
  (h_dime_value : ∀ n : ℕ, n * (10 / 100 : ℚ) = (10 * n : ℚ) / 100) :
  ∃ num_dimes : ℕ, 
    num_dimes = 3 ∧ 
    total_value = 
      num_pennies * (1 / 100 : ℚ) + 
      num_nickels * (5 / 100 : ℚ) + 
      num_dimes * (10 / 100 : ℚ) := by
  sorry


end number_of_dimes_l2820_282024


namespace fraction_comparison_l2820_282017

theorem fraction_comparison (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  ((1 + y) / x < 2) ∨ ((1 + x) / y < 2) :=
by sorry

end fraction_comparison_l2820_282017


namespace max_d_is_one_l2820_282081

def a (n : ℕ+) : ℕ := 100 + n^3

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_is_one : ∀ n : ℕ+, d n = 1 := by sorry

end max_d_is_one_l2820_282081


namespace multiples_of_3_or_5_not_6_up_to_200_l2820_282015

def count_multiples (n : ℕ) (m : ℕ) : ℕ :=
  (n / m : ℕ)

def count_multiples_of_3_or_5_not_6 (upper_bound : ℕ) : ℕ :=
  count_multiples upper_bound 3 +
  count_multiples upper_bound 5 -
  count_multiples upper_bound 15 -
  count_multiples upper_bound 6

theorem multiples_of_3_or_5_not_6_up_to_200 :
  count_multiples_of_3_or_5_not_6 200 = 60 := by
  sorry

end multiples_of_3_or_5_not_6_up_to_200_l2820_282015


namespace work_left_for_given_days_l2820_282035

/-- The fraction of work left after two workers collaborate for a given time --/
def work_left (a_days b_days collab_days : ℚ) : ℚ :=
  1 - collab_days * (1 / a_days + 1 / b_days)

/-- Theorem: If A can complete the work in 15 days and B in 20 days,
    then after working together for 3 days, the fraction of work left is 13/20 --/
theorem work_left_for_given_days :
  work_left 15 20 3 = 13 / 20 := by
  sorry

end work_left_for_given_days_l2820_282035


namespace equivalent_form_l2820_282049

theorem equivalent_form (x : ℝ) (h : x < 0) :
  Real.sqrt (x / (1 - (x + 1) / x)) = -x * Complex.I :=
by sorry

end equivalent_form_l2820_282049


namespace pizza_theorem_l2820_282030

def pizza_problem (total_slices pepperoni_slices mushroom_slices olive_slices : ℕ) : Prop :=
  ∃ (n a b c : ℕ),
    -- Total slices
    total_slices = 24 ∧
    -- Slices with each topping
    pepperoni_slices = 12 ∧
    mushroom_slices = 14 ∧
    olive_slices = 16 ∧
    -- Every slice has at least one topping
    (12 - n) + (14 - n) + (16 - n) + a + b + c + n = total_slices ∧
    -- Venn diagram constraint
    42 - 3*n - 2*(a + b + c) + a + b + c + n = total_slices ∧
    -- Number of slices with all three toppings
    n = 2

theorem pizza_theorem :
  ∀ (total_slices pepperoni_slices mushroom_slices olive_slices : ℕ),
    pizza_problem total_slices pepperoni_slices mushroom_slices olive_slices :=
by
  sorry

end pizza_theorem_l2820_282030


namespace product_selection_theorem_l2820_282041

/-- Represents the outcome of selecting two items from a batch of products -/
inductive Outcome
  | TwoGenuine
  | OneGenuineOneDefective
  | TwoDefective

/-- Represents a batch of products -/
structure Batch where
  genuine : ℕ
  defective : ℕ
  h_genuine : genuine > 2
  h_defective : defective > 2

/-- Probability of an outcome given a batch -/
def prob (b : Batch) (o : Outcome) : ℝ := sorry

/-- Event of having exactly one defective product -/
def exactly_one_defective (o : Outcome) : Prop :=
  o = Outcome.OneGenuineOneDefective

/-- Event of having exactly two defective products -/
def exactly_two_defective (o : Outcome) : Prop :=
  o = Outcome.TwoDefective

/-- Event of having at least one defective product -/
def at_least_one_defective (o : Outcome) : Prop :=
  o = Outcome.OneGenuineOneDefective ∨ o = Outcome.TwoDefective

/-- Event of having all genuine products -/
def all_genuine (o : Outcome) : Prop :=
  o = Outcome.TwoGenuine

theorem product_selection_theorem (b : Batch) :
  -- Statement ②: Exactly one defective and exactly two defective are mutually exclusive
  (∀ o : Outcome, ¬(exactly_one_defective o ∧ exactly_two_defective o)) ∧
  -- Statement ④: At least one defective and all genuine are mutually exclusive and complementary
  (∀ o : Outcome, ¬(at_least_one_defective o ∧ all_genuine o)) ∧
  (∀ o : Outcome, at_least_one_defective o ∨ all_genuine o) ∧
  -- Statements ① and ③ are incorrect (we don't need to prove them, just state that they're not included)
  True := by sorry

end product_selection_theorem_l2820_282041


namespace vector_collinearity_l2820_282013

/-- Given vectors a, b, and c in ℝ², prove that if b - a is collinear with c, then the n-coordinate of b equals -3. -/
theorem vector_collinearity (a b c : ℝ × ℝ) (n : ℝ) :
  a = (1, 2) →
  b = (n, 3) →
  c = (4, -1) →
  ∃ (k : ℝ), (b.1 - a.1, b.2 - a.2) = (k * c.1, k * c.2) →
  n = -3 := by
  sorry

end vector_collinearity_l2820_282013


namespace one_third_1206_percent_of_134_l2820_282055

theorem one_third_1206_percent_of_134 : 
  (1206 / 3) / 134 * 100 = 300 := by
  sorry

end one_third_1206_percent_of_134_l2820_282055


namespace milk_packet_price_problem_l2820_282059

/-- Given 5 packets of milk with an average price of 20 cents, if 2 packets are returned
    and the average price of the remaining 3 packets is 12 cents, then the average price
    of the 2 returned packets is 32 cents. -/
theorem milk_packet_price_problem (total_packets : Nat) (remaining_packets : Nat) 
    (initial_avg_price : ℚ) (remaining_avg_price : ℚ) :
  total_packets = 5 →
  remaining_packets = 3 →
  initial_avg_price = 20 →
  remaining_avg_price = 12 →
  let returned_packets := total_packets - remaining_packets
  let total_cost := total_packets * initial_avg_price
  let remaining_cost := remaining_packets * remaining_avg_price
  let returned_cost := total_cost - remaining_cost
  (returned_cost / returned_packets : ℚ) = 32 := by
sorry

end milk_packet_price_problem_l2820_282059


namespace bus_passengers_l2820_282092

theorem bus_passengers (initial_students : ℕ) (num_stops : ℕ) : 
  initial_students = 60 → num_stops = 4 → 
  ⌊(initial_students : ℚ) * (2/3)^num_stops⌋ = 11 := by
  sorry

end bus_passengers_l2820_282092


namespace flowers_per_pot_l2820_282014

/-- Given 141 pots and 10011 flowers in total, prove that each pot contains 71 flowers. -/
theorem flowers_per_pot (total_pots : ℕ) (total_flowers : ℕ) (h1 : total_pots = 141) (h2 : total_flowers = 10011) :
  total_flowers / total_pots = 71 := by
  sorry

end flowers_per_pot_l2820_282014


namespace perfect_squares_between_a_and_2a_l2820_282011

theorem perfect_squares_between_a_and_2a (a : ℕ) : 
  (a > 0) → 
  (∃ x : ℕ, x^2 > a ∧ (x+9)^2 < 2*a ∧ 
    ∀ y : ℕ, (y^2 > a ∧ y^2 < 2*a) → (x ≤ y ∧ y ≤ x+9)) →
  (481 ≤ a ∧ a ≤ 684) :=
by sorry

end perfect_squares_between_a_and_2a_l2820_282011


namespace daniel_initial_noodles_l2820_282026

/-- The number of noodles Daniel gave to William -/
def noodles_given : ℕ := 12

/-- The number of noodles Daniel has now -/
def noodles_remaining : ℕ := 54

/-- The initial number of noodles Daniel had -/
def initial_noodles : ℕ := noodles_given + noodles_remaining

theorem daniel_initial_noodles :
  initial_noodles = 66 :=
by sorry

end daniel_initial_noodles_l2820_282026


namespace parabola_focus_theorem_l2820_282066

-- Define the line on which the focus lies
def focus_line (x y : ℝ) : Prop := x + 2 * y + 3 = 0

-- Define the two possible standard equations for the parabola
def parabola_eq1 (x y : ℝ) : Prop := y^2 = -12 * x
def parabola_eq2 (x y : ℝ) : Prop := x^2 = -6 * y

-- Theorem statement
theorem parabola_focus_theorem :
  ∀ (x y : ℝ), focus_line x y →
  (parabola_eq1 x y ∨ parabola_eq2 x y) :=
by sorry

end parabola_focus_theorem_l2820_282066


namespace sum_of_roots_l2820_282091

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 := by
sorry

end sum_of_roots_l2820_282091


namespace log_expression_equals_two_l2820_282033

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_two :
  2 * log10 2 + log10 5 / log10 (Real.sqrt 10) = 2 := by
  sorry

end log_expression_equals_two_l2820_282033


namespace max_value_z_l2820_282053

theorem max_value_z (x y : ℝ) (h1 : x - y + 1 ≤ 0) (h2 : x - 2*y ≤ 0) (h3 : x + 2*y - 2 ≤ 0) :
  ∀ z, z = x + y → z ≤ 1 :=
by sorry

end max_value_z_l2820_282053


namespace original_light_wattage_l2820_282039

theorem original_light_wattage (W : ℝ) : 
  (W + 0.3 * W = 143) → W = 110 := by
  sorry

end original_light_wattage_l2820_282039


namespace binary_sum_equals_318_l2820_282021

/-- Convert a binary number represented as a string to its decimal equivalent -/
def binary_to_decimal (s : String) : ℕ :=
  s.foldl (fun acc c => 2 * acc + c.toString.toNat!) 0

/-- The sum of 11111111₂ and 111111₂ in base 10 -/
theorem binary_sum_equals_318 :
  binary_to_decimal "11111111" + binary_to_decimal "111111" = 318 := by
  sorry

end binary_sum_equals_318_l2820_282021


namespace tens_digit_of_3_to_100_l2820_282094

/-- The function that computes the last two digits of 3^n -/
def lastTwoDigits (n : ℕ) : ℕ := (3^n) % 100

/-- The cycle length of the last two digits of 3^n -/
def cycleLengthLastTwoDigits : ℕ := 20

/-- The tens digit of a number -/
def tensDigit (n : ℕ) : ℕ := (n / 10) % 10

theorem tens_digit_of_3_to_100 : tensDigit (lastTwoDigits 100) = 0 := by sorry

end tens_digit_of_3_to_100_l2820_282094


namespace min_throws_for_repeated_sum_l2820_282038

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice being thrown -/
def numDice : ℕ := 4

/-- The minimum possible sum when rolling the dice -/
def minSum : ℕ := numDice

/-- The maximum possible sum when rolling the dice -/
def maxSum : ℕ := numDice * sides

/-- The number of possible unique sums -/
def uniqueSums : ℕ := maxSum - minSum + 1

/-- The minimum number of throws needed to guarantee a repeated sum -/
def minThrows : ℕ := uniqueSums + 1

theorem min_throws_for_repeated_sum :
  minThrows = 22 := by sorry

end min_throws_for_repeated_sum_l2820_282038


namespace modulo_congruence_l2820_282044

theorem modulo_congruence : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -4792 - 242 [ZMOD 8] ∧ n = 6 := by
  sorry

end modulo_congruence_l2820_282044


namespace inequality_properties_l2820_282085

theorem inequality_properties (m n : ℝ) :
  (∀ a : ℝ, a ≠ 0 → m * a^2 < n * a^2 → m < n) ∧
  (m < n → n < 0 → n / m < 1) := by
  sorry

end inequality_properties_l2820_282085


namespace fifteenth_odd_multiple_of_five_l2820_282037

def nth_odd_multiple_of_five (n : ℕ) : ℕ := 10 * n - 5

theorem fifteenth_odd_multiple_of_five : 
  nth_odd_multiple_of_five 15 = 145 := by sorry

end fifteenth_odd_multiple_of_five_l2820_282037


namespace complete_factorization_l2820_282069

theorem complete_factorization (x : ℝ) :
  x^12 - 729 = (x^2 + 3) * (x^4 - 3*x^2 + 9) * (x^3 - 3) * (x^3 + 3) := by
  sorry

end complete_factorization_l2820_282069


namespace rectangle_perimeter_16_l2820_282034

def rectangle_perimeter (length width : ℚ) : ℚ := 2 * (length + width)

theorem rectangle_perimeter_16 :
  let length : ℚ := 5
  let width : ℚ := 30 / 10
  rectangle_perimeter length width = 16 := by sorry

end rectangle_perimeter_16_l2820_282034


namespace mathematics_players_count_l2820_282054

-- Define the set of all players
def TotalPlayers : ℕ := 30

-- Define the set of players taking physics
def PhysicsPlayers : ℕ := 15

-- Define the set of players taking both physics and mathematics
def BothSubjectsPlayers : ℕ := 7

-- Define the set of players taking mathematics
def MathematicsPlayers : ℕ := TotalPlayers - (PhysicsPlayers - BothSubjectsPlayers)

-- Theorem statement
theorem mathematics_players_count : MathematicsPlayers = 22 := by
  sorry

end mathematics_players_count_l2820_282054


namespace f_g_inequality_l2820_282027

open Set
open Function
open Topology

-- Define the interval [a, b]
variable (a b : ℝ) (hab : a < b)

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
variable (hf : DifferentiableOn ℝ f (Icc a b))
variable (hg : DifferentiableOn ℝ g (Icc a b))
variable (h_deriv : ∀ x ∈ Icc a b, deriv f x > deriv g x)

-- State the theorem
theorem f_g_inequality (x : ℝ) (hx : x ∈ Ioo a b) :
  f x + g a > g x + f a := by sorry

end f_g_inequality_l2820_282027


namespace min_distance_to_line_l2820_282045

theorem min_distance_to_line (x y : ℝ) :
  (3 * x + y = 10) → (x^2 + y^2 ≥ 10) := by sorry

end min_distance_to_line_l2820_282045


namespace total_beignets_l2820_282020

/-- The number of beignets eaten per day -/
def beignets_per_day : ℕ := 3

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of weeks we're considering -/
def weeks : ℕ := 16

/-- Theorem: The total number of beignets eaten in 16 weeks -/
theorem total_beignets : beignets_per_day * days_in_week * weeks = 336 := by
  sorry

end total_beignets_l2820_282020


namespace packets_in_box_l2820_282036

/-- The number of packets in a box of sugar substitute -/
def packets_per_box : ℕ := sorry

/-- The daily usage of sugar substitute packets -/
def daily_usage : ℕ := 2

/-- The number of days for which sugar substitute is needed -/
def duration : ℕ := 90

/-- The total cost of sugar substitute for the given duration -/
def total_cost : ℚ := 24

/-- The cost of one box of sugar substitute -/
def cost_per_box : ℚ := 4

/-- Theorem stating that the number of packets in a box is 30 -/
theorem packets_in_box :
  packets_per_box = 30 :=
by sorry

end packets_in_box_l2820_282036


namespace least_product_of_three_primes_greater_than_10_l2820_282007

theorem least_product_of_three_primes_greater_than_10 :
  ∃ (p q r : ℕ),
    Prime p ∧ Prime q ∧ Prime r ∧
    p > 10 ∧ q > 10 ∧ r > 10 ∧
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    p * q * r = 2431 ∧
    (∀ (a b c : ℕ),
      Prime a ∧ Prime b ∧ Prime c ∧
      a > 10 ∧ b > 10 ∧ c > 10 ∧
      a ≠ b ∧ a ≠ c ∧ b ≠ c →
      a * b * c ≥ 2431) :=
by sorry


end least_product_of_three_primes_greater_than_10_l2820_282007


namespace binomial_15_choose_3_l2820_282048

theorem binomial_15_choose_3 : Nat.choose 15 3 = 455 := by sorry

end binomial_15_choose_3_l2820_282048


namespace f_lower_bound_g_min_max_l2820_282073

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := x^2 + Real.log x

def g (x : ℝ) : ℝ := x^2 - 2 * Real.log x

-- State the theorems
theorem f_lower_bound (x : ℝ) (hx : x > 0) : f x ≥ (x^3 + x - 1) / x := by sorry

theorem g_min_max :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, g x ≥ 1) ∧
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, g x = 1) ∧
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, g x ≤ 4 - 2 * Real.log 2) ∧
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, g x = 4 - 2 * Real.log 2) := by sorry

end f_lower_bound_g_min_max_l2820_282073


namespace first_week_sales_l2820_282060

/-- Represents the sales of chips in a convenience store over a month -/
structure ChipSales where
  total : ℕ
  first_week : ℕ
  second_week : ℕ
  third_week : ℕ
  fourth_week : ℕ

/-- Theorem stating the conditions and the result to be proved -/
theorem first_week_sales (s : ChipSales) :
  s.total = 100 ∧
  s.second_week = 3 * s.first_week ∧
  s.third_week = 20 ∧
  s.fourth_week = 20 ∧
  s.total = s.first_week + s.second_week + s.third_week + s.fourth_week →
  s.first_week = 15 := by
  sorry

end first_week_sales_l2820_282060


namespace caitlin_age_l2820_282063

theorem caitlin_age (anna_age : ℕ) (brianna_age : ℕ) (caitlin_age : ℕ) : 
  anna_age = 48 →
  brianna_age = anna_age / 2 →
  caitlin_age = brianna_age - 7 →
  caitlin_age = 17 := by
sorry

end caitlin_age_l2820_282063


namespace original_price_is_10000_l2820_282025

/-- Calculates the original price of a machine given repair cost, transportation cost, profit percentage, and selling price. -/
def calculate_original_price (repair_cost : ℕ) (transport_cost : ℕ) (profit_percent : ℕ) (selling_price : ℕ) : ℕ :=
  let total_additional_cost := repair_cost + transport_cost
  let total_cost_multiplier := 100 + profit_percent
  ((selling_price * 100) / total_cost_multiplier) - total_additional_cost

/-- Theorem stating that given the specific conditions, the original price of the machine was 10000. -/
theorem original_price_is_10000 :
  calculate_original_price 5000 1000 50 24000 = 10000 := by
  sorry

end original_price_is_10000_l2820_282025


namespace sum_of_solutions_quadratic_l2820_282018

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (27 - 9*x - x^2 = 0) → 
  (∃ r s : ℝ, (27 - 9*r - r^2 = 0) ∧ (27 - 9*s - s^2 = 0) ∧ (r + s = 9)) :=
by sorry

end sum_of_solutions_quadratic_l2820_282018


namespace inequality_proof_l2820_282032

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0)
  (abs_sum_le_one : |x| + |y| + |z| ≤ 1) :
  x + y / 2 + z / 3 ≤ 1 / 3 := by
  sorry

end inequality_proof_l2820_282032


namespace exists_zero_of_f_n_l2820_282087

/-- The function f(x) = x^2 + 2017x + 1 -/
def f (x : ℝ) : ℝ := x^2 + 2017*x + 1

/-- n-fold composition of f -/
def f_n (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n+1 => f (f_n n x)

/-- For any positive integer n, there exists a real x such that f_n(x) = 0 -/
theorem exists_zero_of_f_n (n : ℕ) (hn : n ≥ 1) : ∃ x : ℝ, f_n n x = 0 := by
  sorry


end exists_zero_of_f_n_l2820_282087


namespace no_consecutive_perfect_squares_l2820_282022

theorem no_consecutive_perfect_squares (a b : ℤ) : a^2 - b^2 = 1 → (a = 1 ∧ b = 0) ∨ (a = -1 ∧ b = 0) := by
  sorry

end no_consecutive_perfect_squares_l2820_282022


namespace triangle_is_equilateral_l2820_282067

theorem triangle_is_equilateral (a b c : ℝ) (A B C : ℝ) 
  (h1 : b^2 + c^2 - a^2 = b*c)
  (h2 : 2 * Real.cos B * Real.sin C = Real.sin A)
  (h3 : A + B + C = π)
  (h4 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h5 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h6 : A < π ∧ B < π ∧ C < π) :
  a = b ∧ b = c := by
  sorry

end triangle_is_equilateral_l2820_282067


namespace prob_at_least_one_three_l2820_282043

/-- The number of sides on each die -/
def sides : ℕ := 8

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := sides * sides

/-- The number of outcomes where neither die shows a 3 -/
def neither_three : ℕ := (sides - 1) * (sides - 1)

/-- The number of outcomes where at least one die shows a 3 -/
def at_least_one_three : ℕ := total_outcomes - neither_three

/-- The probability of getting at least one 3 when rolling two 8-sided dice -/
theorem prob_at_least_one_three : 
  (at_least_one_three : ℚ) / total_outcomes = 15 / 64 := by
  sorry

end prob_at_least_one_three_l2820_282043


namespace students_not_both_count_l2820_282072

/-- Given information about students taking chemistry and physics classes -/
structure ClassData where
  both : ℕ         -- Number of students taking both chemistry and physics
  chemistry : ℕ    -- Total number of students taking chemistry
  only_physics : ℕ -- Number of students taking only physics

/-- Calculate the number of students taking chemistry or physics but not both -/
def students_not_both (data : ClassData) : ℕ :=
  (data.chemistry - data.both) + data.only_physics

/-- Theorem stating the number of students taking chemistry or physics but not both -/
theorem students_not_both_count (data : ClassData) 
  (h1 : data.both = 12)
  (h2 : data.chemistry = 30)
  (h3 : data.only_physics = 18) :
  students_not_both data = 36 := by
  sorry

#eval students_not_both ⟨12, 30, 18⟩

end students_not_both_count_l2820_282072


namespace pages_difference_l2820_282003

theorem pages_difference (total_pages book_length first_day fourth_day : ℕ) : 
  book_length = 354 → 
  first_day = 63 → 
  fourth_day = 29 → 
  total_pages = 4 → 
  (book_length - (first_day + 2 * first_day + fourth_day)) - 2 * first_day = 10 := by
  sorry

end pages_difference_l2820_282003


namespace equal_diagonal_distances_l2820_282004

/-- Represents a cuboid with edge lengths a, b, and c. -/
structure Cuboid where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a pair of diagonals on adjacent faces of a cuboid. -/
inductive DiagonalPair
  | AB_AC
  | AB_BC
  | AC_BC

/-- Calculates the distance between a pair of diagonals on adjacent faces of a cuboid. -/
def diagonalDistance (cuboid : Cuboid) (pair : DiagonalPair) : ℝ :=
  sorry

/-- Theorem stating that the distances between diagonals of each pair of adjacent faces are equal
    for a cuboid with edge lengths 7, 14, and 21. -/
theorem equal_diagonal_distances (cuboid : Cuboid)
    (h1 : cuboid.a = 7)
    (h2 : cuboid.b = 14)
    (h3 : cuboid.c = 21) :
    ∀ p q : DiagonalPair, diagonalDistance cuboid p = diagonalDistance cuboid q :=
  sorry

end equal_diagonal_distances_l2820_282004


namespace four_digit_number_count_special_four_digit_number_count_l2820_282051

/-- The set of available digits --/
def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

/-- A four-digit number with no repeating digits --/
structure FourDigitNumber where
  d₁ : Nat
  d₂ : Nat
  d₃ : Nat
  d₄ : Nat
  h₁ : d₁ ∈ digits
  h₂ : d₂ ∈ digits
  h₃ : d₃ ∈ digits
  h₄ : d₄ ∈ digits
  h₅ : d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₃ ≠ d₄
  h₆ : d₁ ≠ 0  -- Ensures it's a four-digit number

/-- The set of all valid four-digit numbers --/
def allFourDigitNumbers : Finset FourDigitNumber := sorry

/-- Four-digit numbers with tens digit larger than both units and hundreds digits --/
def specialFourDigitNumbers : Finset FourDigitNumber :=
  allFourDigitNumbers.filter (fun n => n.d₃ > n.d₂ ∧ n.d₃ > n.d₄)

theorem four_digit_number_count :
  Finset.card allFourDigitNumbers = 300 := by sorry

theorem special_four_digit_number_count :
  Finset.card specialFourDigitNumbers = 100 := by sorry

end four_digit_number_count_special_four_digit_number_count_l2820_282051


namespace fifth_term_of_sequence_l2820_282083

/-- A geometric sequence with the given first four terms -/
def geometric_sequence (y : ℝ) : ℕ → ℝ
  | 0 => 3
  | 1 => 9 * y
  | 2 => 27 * y^2
  | 3 => 81 * y^3
  | n + 4 => geometric_sequence y 3 * (3 * y)^(n + 1)

/-- The fifth term of the geometric sequence is 243y^4 -/
theorem fifth_term_of_sequence (y : ℝ) :
  geometric_sequence y 4 = 243 * y^4 := by
  sorry

end fifth_term_of_sequence_l2820_282083


namespace parabola_focus_distance_range_l2820_282065

theorem parabola_focus_distance_range :
  ∀ (A : ℝ × ℝ) (θ : ℝ),
    let F : ℝ × ℝ := (1/4, 0)
    let y : ℝ → ℝ := λ x => Real.sqrt x
    let l : ℝ → ℝ := λ x => Real.tan θ * (x - F.1) + F.2
    A.2 = y A.1 ∧  -- A is on the parabola
    A.2 > 0 ∧  -- A is above x-axis
    l A.1 = A.2 ∧  -- A is on line l
    θ ≥ π/4 →
    ∃ (FA : ℝ), FA > 1/4 ∧ FA ≤ 1 + Real.sqrt 2 / 2 ∧
      FA = Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) :=
by sorry

end parabola_focus_distance_range_l2820_282065


namespace right_triangle_circle_chord_length_l2820_282078

/-- Represents a triangle ABC with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Theorem: In a right triangle ABC with hypotenuse AB = 10, AC = 8, and BC = 6,
    if a circle P passes through C and is tangent to AB at its midpoint,
    then the length of the chord QR (where Q and R are the intersections of
    the circle with AC and BC respectively) is equal to 9.6. -/
theorem right_triangle_circle_chord_length
  (abc : Triangle)
  (p : Circle)
  (h1 : abc.a = 10 ∧ abc.b = 8 ∧ abc.c = 6)
  (h2 : abc.a^2 = abc.b^2 + abc.c^2)
  (h3 : p.center = (5, p.radius))
  (h4 : p.radius = abc.b * abc.c / abc.a) :
  2 * p.radius = 9.6 := by sorry

end right_triangle_circle_chord_length_l2820_282078


namespace find_m_value_l2820_282068

/-- Given two functions f and g, prove that m = -7 when f(5) - g(5) = 55 -/
theorem find_m_value (m : ℝ) : 
  let f : ℝ → ℝ := λ x => 5 * x^2 + 3 * x + 7
  let g : ℝ → ℝ := λ x => 2 * x^2 - m * x + 1
  (f 5 - g 5 = 55) → m = -7 := by
sorry

end find_m_value_l2820_282068


namespace zero_natural_number_ambiguity_l2820_282099

-- Define a type for natural number conventions
inductive NatConvention where
  | withZero    : NatConvention
  | withoutZero : NatConvention

-- Define a function that checks if 0 is a natural number based on the convention
def isZeroNatural (conv : NatConvention) : Prop :=
  match conv with
  | NatConvention.withZero    => True
  | NatConvention.withoutZero => False

-- Theorem statement
theorem zero_natural_number_ambiguity :
  ∃ (conv : NatConvention), isZeroNatural conv :=
sorry


end zero_natural_number_ambiguity_l2820_282099


namespace complement_of_union_MN_l2820_282056

def I : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2, 5}
def N : Set ℕ := {2, 3, 5}

theorem complement_of_union_MN :
  (M ∪ N)ᶜ = {4} :=
by
  sorry

end complement_of_union_MN_l2820_282056


namespace geometric_mean_minimum_l2820_282040

theorem geometric_mean_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (∀ x y, x > 0 → y > 0 → 1/x + 4/y ≥ 1/a + 4/b) → 1/a + 4/b = 9 :=
sorry

end geometric_mean_minimum_l2820_282040


namespace sum_remainder_mod_seven_l2820_282057

theorem sum_remainder_mod_seven : (5283 + 5284 + 5285 + 5286 + 5287) % 7 = 1 := by
  sorry

end sum_remainder_mod_seven_l2820_282057


namespace line_equation_represents_line_l2820_282074

/-- A line in the 2D plane defined by the equation y = mx + b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The set of points (x, y) satisfying a linear equation -/
def LinePoints (l : Line) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = l.m * p.1 + l.b}

theorem line_equation_represents_line :
  ∃ (l : Line), l.m = 2 ∧ l.b = 1 ∧
  LinePoints l = {p : ℝ × ℝ | p.2 = 2 * p.1 + 1} :=
by sorry

end line_equation_represents_line_l2820_282074


namespace dolls_distribution_count_l2820_282005

def distribute_dolls (n_dolls : ℕ) (n_houses : ℕ) : ℕ :=
  let choose_two := n_dolls.choose 2
  let select_house := n_houses
  let arrange_rest := (n_dolls - 2).factorial
  choose_two * select_house * arrange_rest

theorem dolls_distribution_count :
  distribute_dolls 7 6 = 15120 :=
by sorry

end dolls_distribution_count_l2820_282005


namespace warehouse_analysis_l2820_282097

/-- Represents the daily changes in goods, where positive values indicate goods entering
    and negative values indicate goods leaving the warehouse -/
def daily_changes : List Int := [31, -31, -16, 34, -38, -20]

/-- The final amount of goods in the warehouse after 6 days -/
def final_amount : Int := 430

/-- The fee for loading or unloading one ton of goods -/
def fee_per_ton : Int := 5

theorem warehouse_analysis :
  let net_change := daily_changes.sum
  let initial_amount := final_amount - net_change
  let total_fees := (daily_changes.map abs).sum * fee_per_ton
  (net_change < 0) ∧
  (initial_amount = 470) ∧
  (total_fees = 850) := by sorry

end warehouse_analysis_l2820_282097


namespace min_value_x_plus_2y_l2820_282086

theorem min_value_x_plus_2y (x y : ℝ) 
  (h1 : x > -1) 
  (h2 : y > 0) 
  (h3 : 1 / (x + 1) + 2 / y = 1) : 
  ∀ z, x + 2 * y ≤ z → 8 ≤ z :=
sorry

end min_value_x_plus_2y_l2820_282086


namespace frames_per_page_l2820_282077

theorem frames_per_page (total_frames : ℕ) (num_pages : ℕ) (h1 : total_frames = 143) (h2 : num_pages = 13) :
  total_frames / num_pages = 11 := by
  sorry

end frames_per_page_l2820_282077


namespace log_equation_solution_l2820_282064

theorem log_equation_solution (a x : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (h : (Real.log x) / (Real.log (a^3)) + (Real.log a) / (Real.log (x^2)) = 2) :
  x = a^(3 + (5 * Real.sqrt 3) / 2) ∨ x = a^(3 - (5 * Real.sqrt 3) / 2) :=
by sorry

end log_equation_solution_l2820_282064


namespace third_chest_coin_difference_l2820_282001

theorem third_chest_coin_difference (total_gold total_silver : ℕ) 
  (x1 y1 x2 y2 x3 y3 : ℕ) : 
  total_gold = 40 →
  total_silver = 40 →
  x1 + x2 + x3 = total_gold →
  y1 + y2 + y3 = total_silver →
  x1 = y1 + 7 →
  y2 = x2 - 15 →
  y3 - x3 = 22 :=
by sorry

end third_chest_coin_difference_l2820_282001


namespace combined_mean_of_two_sets_l2820_282082

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℝ) 
  (set2_count : ℕ) (set2_mean : ℝ) : 
  set1_count = 5 → 
  set1_mean = 16 → 
  set2_count = 8 → 
  set2_mean = 21 → 
  let total_count := set1_count + set2_count
  let combined_mean := (set1_count * set1_mean + set2_count * set2_mean) / total_count
  combined_mean = 19.08 := by
sorry

end combined_mean_of_two_sets_l2820_282082


namespace no_solution_equation_one_unique_solution_equation_two_l2820_282076

-- Problem 1
theorem no_solution_equation_one (x : ℝ) : 
  (x ≠ 2) → (1 / (x - 2) ≠ (1 - x) / (2 - x) - 3) :=
by sorry

-- Problem 2
theorem unique_solution_equation_two :
  ∃! x : ℝ, (x ≠ 1) ∧ (x^2 ≠ 1) ∧ (x / (x - 1) - (2*x - 1) / (x^2 - 1) = 1) :=
by sorry

end no_solution_equation_one_unique_solution_equation_two_l2820_282076


namespace tangent_line_at_2_minus_6_tangent_lines_slope_4_l2820_282031

-- Define the function f(x) = x³ + x - 16
def f (x : ℝ) : ℝ := x^3 + x - 16

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem for the tangent line at (2, -6)
theorem tangent_line_at_2_minus_6 :
  ∃ (m b : ℝ), m * 2 - b = f 2 ∧ 
               m = f' 2 ∧
               ∀ x, m * x - b = 13 * x - 32 :=
sorry

-- Theorem for tangent lines with slope 4
theorem tangent_lines_slope_4 :
  ∃ (x₁ x₂ b₁ b₂ : ℝ), 
    x₁ ≠ x₂ ∧
    f' x₁ = 4 ∧ f' x₂ = 4 ∧
    4 * x₁ - b₁ = f x₁ ∧
    4 * x₂ - b₂ = f x₂ ∧
    (∀ x, 4 * x - b₁ = 4 * x - 18) ∧
    (∀ x, 4 * x - b₂ = 4 * x - 14) :=
sorry

end tangent_line_at_2_minus_6_tangent_lines_slope_4_l2820_282031


namespace range_of_a_l2820_282089

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - (a^2 + a)*x + a^3 > 0 ↔ x < a^2 ∨ x > a) →
  0 ≤ a ∧ a ≤ 1 :=
by sorry

end range_of_a_l2820_282089


namespace sqrt_equation_solution_l2820_282084

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 11) = 10 → x = 89 := by
  sorry

end sqrt_equation_solution_l2820_282084


namespace bisecting_cross_section_dihedral_angle_l2820_282008

/-- Regular tetrahedron with specific dimensions -/
structure RegularTetrahedron where
  -- Base side length
  base_side : ℝ
  -- Side edge length
  side_edge : ℝ
  -- Assumption that base_side = 1 and side_edge = 2
  base_side_eq_one : base_side = 1
  side_edge_eq_two : side_edge = 2

/-- Cross-section that bisects the tetrahedron's volume -/
structure BisectingCrossSection (t : RegularTetrahedron) where
  -- The cross-section passes through edge AB of the base
  passes_through_base_edge : Prop

/-- Dihedral angle between the cross-section and the base -/
def dihedralAngle (t : RegularTetrahedron) (cs : BisectingCrossSection t) : ℝ :=
  sorry -- Definition of dihedral angle

/-- Main theorem -/
theorem bisecting_cross_section_dihedral_angle 
  (t : RegularTetrahedron) (cs : BisectingCrossSection t) : 
  Real.cos (dihedralAngle t cs) = 2 * Real.sqrt 15 / 15 := by
  sorry

end bisecting_cross_section_dihedral_angle_l2820_282008


namespace isosceles_triangle_angles_l2820_282010

theorem isosceles_triangle_angles (a b c : ℝ) : 
  -- The triangle is isosceles
  (a = b ∨ b = c ∨ a = c) →
  -- One of the interior angles is 50°
  (a = 50 ∨ b = 50 ∨ c = 50) →
  -- The sum of interior angles in a triangle is 180°
  a + b + c = 180 →
  -- The other two angles are either (65°, 65°) or (80°, 50°)
  ((a = 65 ∧ b = 65 ∧ c = 50) ∨ 
   (a = 65 ∧ c = 65 ∧ b = 50) ∨ 
   (b = 65 ∧ c = 65 ∧ a = 50) ∨
   (a = 80 ∧ b = 50 ∧ c = 50) ∨ 
   (a = 50 ∧ b = 80 ∧ c = 50) ∨ 
   (a = 50 ∧ b = 50 ∧ c = 80)) :=
by sorry

end isosceles_triangle_angles_l2820_282010


namespace largest_digit_sum_l2820_282096

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem largest_digit_sum (a b c y : ℕ) (ha : is_digit a) (hb : is_digit b) (hc : is_digit c)
  (hy : 0 < y ∧ y ≤ 15) (h_frac : (a * 100 + b * 10 + c : ℚ) / 1000 = 1 / y) :
  a + b + c ≤ 8 :=
sorry

end largest_digit_sum_l2820_282096


namespace max_value_product_sum_l2820_282070

theorem max_value_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (∀ a m c : ℕ, a + m + c = 15 →
    A * M * C + A * M + M * C + C * A + A + M + C ≥
    a * m * c + a * m + m * c + c * a + a + m + c) →
  A * M * C + A * M + M * C + C * A + A + M + C = 215 :=
sorry

end max_value_product_sum_l2820_282070


namespace quadratic_extrema_l2820_282071

-- Define the quadratic function
def f (x : ℝ) : ℝ := (x - 3)^2 - 1

-- Define the interval
def I : Set ℝ := Set.Icc 1 4

-- Theorem statement
theorem quadratic_extrema :
  (∃ (x : ℝ), x ∈ I ∧ ∀ (y : ℝ), y ∈ I → f y ≥ f x) ∧
  (∃ (x : ℝ), x ∈ I ∧ ∀ (y : ℝ), y ∈ I → f y ≤ f x) ∧
  (∀ (x : ℝ), x ∈ I → f x ≥ -1) ∧
  (∀ (x : ℝ), x ∈ I → f x ≤ 3) ∧
  (∃ (x : ℝ), x ∈ I ∧ f x = -1) ∧
  (∃ (x : ℝ), x ∈ I ∧ f x = 3) :=
by
  sorry

end quadratic_extrema_l2820_282071


namespace gcd_of_256_180_600_l2820_282009

theorem gcd_of_256_180_600 : Nat.gcd 256 (Nat.gcd 180 600) = 12 := by sorry

end gcd_of_256_180_600_l2820_282009


namespace bike_fundraising_days_l2820_282062

/-- The number of days required to raise money for a bike by selling bracelets -/
def days_to_raise_money (bike_cost : ℕ) (bracelet_price : ℕ) (bracelets_per_day : ℕ) : ℕ :=
  bike_cost / (bracelet_price * bracelets_per_day)

/-- Theorem: Given the specific costs and sales plan, it takes 14 days to raise money for the bike -/
theorem bike_fundraising_days :
  days_to_raise_money 112 1 8 = 14 := by
  sorry

end bike_fundraising_days_l2820_282062


namespace initial_red_marbles_l2820_282012

/-- Given a bag of red and green marbles with the following properties:
    1. The initial ratio of red to green marbles is 5:3
    2. After adding 15 red marbles and removing 9 green marbles, the new ratio is 3:1
    This theorem proves that the initial number of red marbles is 52.5 -/
theorem initial_red_marbles (r g : ℚ) : 
  r / g = 5 / 3 →
  (r + 15) / (g - 9) = 3 / 1 →
  r = 52.5 := by
sorry

end initial_red_marbles_l2820_282012


namespace max_cone_radius_in_crate_l2820_282075

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Checks if a cone fits upright in a crate -/
def fitsInCrate (cone : Cone) (crate : CrateDimensions) : Prop :=
  cone.height ≤ max crate.length (max crate.width crate.height) ∧
  2 * cone.radius ≤ min crate.length (min crate.width crate.height)

/-- The theorem stating the maximum radius of a cone that fits in the given crate -/
theorem max_cone_radius_in_crate :
  ∃ (maxRadius : ℝ),
    maxRadius = 2.5 ∧
    ∀ (c : Cone),
      fitsInCrate c (CrateDimensions.mk 5 8 12) →
      c.radius ≤ maxRadius :=
sorry

end max_cone_radius_in_crate_l2820_282075
