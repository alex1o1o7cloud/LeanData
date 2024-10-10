import Mathlib

namespace mikes_training_hours_l2364_236438

/-- Represents a day of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a week number -/
inductive Week
  | First
  | Second

/-- Returns true if the given day is a weekday, false otherwise -/
def isWeekday (d : Day) : Bool :=
  match d with
  | Day.Saturday | Day.Sunday => false
  | _ => true

/-- Returns the maximum training hours for a given day and week -/
def maxTrainingHours (d : Day) (w : Week) : Nat :=
  match w with
  | Week.First => if isWeekday d then 2 else 1
  | Week.Second => if isWeekday d then 3 else 2

/-- Returns true if the given day is a rest day, false otherwise -/
def isRestDay (dayNumber : Nat) : Bool :=
  dayNumber % 5 == 0

/-- Calculates the total training hours for Mike over two weeks -/
def totalTrainingHours : Nat :=
  let firstWeekHours := 12  -- 5 weekdays * 2 hours + 2 weekend days * 1 hour
  let secondWeekHours := 16 -- 4 weekdays * 3 hours + 2 weekend days * 2 hours (1 rest day)
  firstWeekHours + secondWeekHours

/-- Theorem stating that Mike's total training hours over two weeks is 28 -/
theorem mikes_training_hours : totalTrainingHours = 28 := by
  sorry

#eval totalTrainingHours  -- This should output 28

end mikes_training_hours_l2364_236438


namespace factorization_3x_squared_minus_27_l2364_236463

theorem factorization_3x_squared_minus_27 (x : ℝ) : 3 * x^2 - 27 = 3 * (x + 3) * (x - 3) := by
  sorry

end factorization_3x_squared_minus_27_l2364_236463


namespace overlapping_area_is_half_unit_l2364_236409

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given its three vertices -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

/-- Calculates the area of a quadrilateral given its four vertices -/
def quadrilateralArea (p1 p2 p3 p4 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p4.y) + p2.x * (p3.y - p1.y) + p3.x * (p4.y - p2.y) + p4.x * (p1.y - p3.y))

/-- The main theorem stating that the overlapping area is 0.5 square units -/
theorem overlapping_area_is_half_unit : 
  let t1p1 : Point := ⟨0, 0⟩
  let t1p2 : Point := ⟨6, 2⟩
  let t1p3 : Point := ⟨2, 6⟩
  let t2p1 : Point := ⟨6, 6⟩
  let t2p2 : Point := ⟨0, 2⟩
  let t2p3 : Point := ⟨2, 0⟩
  let ip1 : Point := ⟨2, 2⟩
  let ip2 : Point := ⟨4, 2⟩
  let ip3 : Point := ⟨3, 3⟩
  let ip4 : Point := ⟨2, 3⟩
  quadrilateralArea ip1 ip2 ip3 ip4 = 0.5 := by
  sorry

end overlapping_area_is_half_unit_l2364_236409


namespace garden_area_increase_l2364_236422

theorem garden_area_increase :
  let rect_length : ℝ := 60
  let rect_width : ℝ := 20
  let rect_perimeter := 2 * (rect_length + rect_width)
  let square_side := rect_perimeter / 4
  let rect_area := rect_length * rect_width
  let square_area := square_side * square_side
  square_area - rect_area = 400 := by
sorry

end garden_area_increase_l2364_236422


namespace carters_reading_rate_l2364_236490

/-- Given reading rates for Oliver, Lucy, and Carter, prove Carter's reading rate -/
theorem carters_reading_rate 
  (oliver_rate : ℕ) 
  (lucy_rate : ℕ) 
  (carter_rate : ℕ) 
  (h1 : oliver_rate = 40)
  (h2 : lucy_rate = oliver_rate + 20)
  (h3 : carter_rate = lucy_rate / 2) : 
  carter_rate = 30 := by
sorry

end carters_reading_rate_l2364_236490


namespace lattice_triangle_area_bound_l2364_236430

/-- A 3D lattice point is represented as a triple of integers -/
def LatticePoint3D := ℤ × ℤ × ℤ

/-- A triangle in 3D space is represented by its three vertices -/
structure Triangle3D where
  v1 : LatticePoint3D
  v2 : LatticePoint3D
  v3 : LatticePoint3D

/-- The area of a triangle -/
noncomputable def area (t : Triangle3D) : ℝ := sorry

/-- Theorem: The area of a triangle with vertices at 3D lattice points is at least 1/2 -/
theorem lattice_triangle_area_bound (t : Triangle3D) : area t ≥ 1/2 := by sorry

end lattice_triangle_area_bound_l2364_236430


namespace range_of_m_l2364_236494

-- Define the curves C₁ and C₂
def C₁ (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 = 1
def C₂ (m : ℝ) (x y : ℝ) : Prop := y^2 = 2*(x + m)

-- Define the condition for a single common point above x-axis
def single_common_point (a m : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, C₁ a p.1 p.2 ∧ C₂ m p.1 p.2 ∧ p.2 > 0

-- State the theorem
theorem range_of_m (a : ℝ) (h : a > 0) :
  (∀ m : ℝ, single_common_point a m →
    ((0 < a ∧ a < 1 → m = (a^2 + 1)/2 ∨ (-a < m ∧ m ≤ a)) ∧
     (a ≥ 1 → -a < m ∧ m < a))) :=
by sorry

end range_of_m_l2364_236494


namespace symmetry_of_point_l2364_236449

/-- A point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry of a point with respect to the origin -/
def symmetrical_to_origin (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

theorem symmetry_of_point :
  let P : Point := ⟨3, 2⟩
  let P' : Point := symmetrical_to_origin P
  P'.x = -3 ∧ P'.y = -2 := by sorry

end symmetry_of_point_l2364_236449


namespace dichromate_molecular_weight_l2364_236495

/-- The molecular weight of 9 moles of dichromate (Cr2O7^2-) -/
theorem dichromate_molecular_weight (Cr_weight O_weight : ℝ) 
  (h1 : Cr_weight = 52.00)
  (h2 : O_weight = 16.00) :
  9 * (2 * Cr_weight + 7 * O_weight) = 1944.00 := by
  sorry

end dichromate_molecular_weight_l2364_236495


namespace find_number_l2364_236445

theorem find_number : ∃! x : ℚ, (x + 305) / 16 = 31 := by
  sorry

end find_number_l2364_236445


namespace largest_prime_factor_of_M_l2364_236407

-- Define the sum of divisors function
def sumOfDivisors (n : ℕ) : ℕ := sorry

-- Define M as the sum of divisors of 300
def M : ℕ := sumOfDivisors 300

-- Define a function to get the largest prime factor of a number
def largestPrimeFactor (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem largest_prime_factor_of_M :
  largestPrimeFactor M = 31 := by sorry

end largest_prime_factor_of_M_l2364_236407


namespace sum_difference_inequality_l2364_236499

theorem sum_difference_inequality 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (ha : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0) 
  (hb : b₁ > 0 ∧ b₂ > 0 ∧ b₃ > 0) 
  (ha_sum : a₁*a₁ + a₁*a₂ + a₁*a₃ + a₂*a₂ + a₂*a₃ + a₃*a₃ ≤ 1) 
  (hb_sum : b₁*b₁ + b₁*b₂ + b₁*b₃ + b₂*b₂ + b₂*b₃ + b₃*b₃ ≤ 1) : 
  (a₁-b₁)*(a₁-b₁) + (a₁-b₁)*(a₂-b₂) + (a₁-b₁)*(a₃-b₃) + 
  (a₂-b₂)*(a₂-b₂) + (a₂-b₂)*(a₃-b₃) + (a₃-b₃)*(a₃-b₃) ≤ 1 := by
  sorry

end sum_difference_inequality_l2364_236499


namespace uncle_payment_ratio_l2364_236469

/-- Represents the cost structure and payment for James' singing lessons -/
structure LessonPayment where
  total_lessons : ℕ
  free_lessons : ℕ
  full_price_lessons : ℕ
  half_price_lessons : ℕ
  lesson_cost : ℕ
  james_payment : ℕ

/-- Calculates the total cost of lessons -/
def total_cost (l : LessonPayment) : ℕ :=
  l.lesson_cost * (l.full_price_lessons + l.half_price_lessons)

/-- Calculates the amount paid by James' uncle -/
def uncle_payment (l : LessonPayment) : ℕ :=
  total_cost l - l.james_payment

/-- Theorem stating the ratio of uncle's payment to total cost is 1:2 -/
theorem uncle_payment_ratio (l : LessonPayment) 
  (h1 : l.total_lessons = 20)
  (h2 : l.free_lessons = 1)
  (h3 : l.full_price_lessons = 10)
  (h4 : l.half_price_lessons = 4)
  (h5 : l.lesson_cost = 5)
  (h6 : l.james_payment = 35) :
  2 * uncle_payment l = total_cost l := by
  sorry

#check uncle_payment_ratio

end uncle_payment_ratio_l2364_236469


namespace rhombus_diagonal_length_l2364_236491

theorem rhombus_diagonal_length (d1 : ℝ) (d2 : ℝ) (square_side : ℝ) 
  (h1 : d1 = 16)
  (h2 : square_side = 8)
  (h3 : d1 * d2 / 2 = square_side ^ 2) :
  d2 = 8 := by
sorry

end rhombus_diagonal_length_l2364_236491


namespace cube_vertex_to_plane_distance_l2364_236437

/-- The distance from the closest vertex of a cube to a plane, given specific conditions --/
theorem cube_vertex_to_plane_distance (s : ℝ) (h₁ h₂ h₃ : ℝ) : 
  s = 8 ∧ h₁ = 8 ∧ h₂ = 9 ∧ h₃ = 10 → 
  ∃ (a b c d : ℝ), 
    a^2 + b^2 + c^2 = 1 ∧
    s * a + d = h₁ ∧
    s * b + d = h₂ ∧
    s * c + d = h₃ ∧
    d = (27 - Real.sqrt 186) / 3 := by
  sorry

#check cube_vertex_to_plane_distance

end cube_vertex_to_plane_distance_l2364_236437


namespace closest_multiple_l2364_236414

def target : ℕ := 2500
def divisor : ℕ := 18

-- Define a function to calculate the distance between two natural numbers
def distance (a b : ℕ) : ℕ := max a b - min a b

-- Define a function to check if a number is a multiple of another
def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

-- Statement of the theorem
theorem closest_multiple :
  ∀ n : ℕ, is_multiple n divisor →
    distance n target ≥ distance 2502 target :=
sorry

end closest_multiple_l2364_236414


namespace mutually_exclusive_events_l2364_236432

/-- Represents the outcome of tossing three coins -/
inductive CoinToss
  | HHH
  | HHT
  | HTH
  | HTT
  | THH
  | THT
  | TTH
  | TTT

/-- The event of getting no more than one heads -/
def noMoreThanOneHeads (t : CoinToss) : Prop :=
  t = CoinToss.HTT ∨ t = CoinToss.THT ∨ t = CoinToss.TTH ∨ t = CoinToss.TTT

/-- The event of getting at least two heads -/
def atLeastTwoHeads (t : CoinToss) : Prop :=
  t = CoinToss.HHH ∨ t = CoinToss.HHT ∨ t = CoinToss.HTH ∨ t = CoinToss.THH

/-- Theorem stating that the two events are mutually exclusive -/
theorem mutually_exclusive_events :
  ∀ t : CoinToss, ¬(noMoreThanOneHeads t ∧ atLeastTwoHeads t) :=
by
  sorry


end mutually_exclusive_events_l2364_236432


namespace max_expression_l2364_236468

/-- A permutation of the digits 1 to 9 -/
def Digits := Fin 9 → Fin 9

/-- Check if a permutation is valid (bijective) -/
def is_valid_permutation (p : Digits) : Prop :=
  Function.Bijective p

/-- Convert three consecutive digits in a permutation to a number -/
def to_number (p : Digits) (start : Fin 9) : ℕ :=
  100 * (p start).val + 10 * (p (start + 1)).val + (p (start + 2)).val

/-- The expression to be maximized -/
def expression (p : Digits) : ℤ :=
  (to_number p 0 : ℤ) + (to_number p 3 : ℤ) - (to_number p 6 : ℤ)

/-- The main theorem -/
theorem max_expression :
  ∃ (p : Digits), is_valid_permutation p ∧ 
    (∀ (q : Digits), is_valid_permutation q → expression q ≤ expression p) ∧
    expression p = 1716 := by sorry

end max_expression_l2364_236468


namespace function_must_be_constant_l2364_236486

-- Define the function type
def FunctionType := ℤ × ℤ → ℝ

-- Define the property of the function
def SatisfiesEquation (f : FunctionType) : Prop :=
  ∀ x y : ℤ, f (x, y) = (f (x - 1, y) + f (x, y - 1)) / 2

-- Define the range constraint
def InRange (f : FunctionType) : Prop :=
  ∀ x y : ℤ, 0 ≤ f (x, y) ∧ f (x, y) ≤ 1

-- Main theorem statement
theorem function_must_be_constant (f : FunctionType) 
  (h_eq : SatisfiesEquation f) (h_range : InRange f) : 
  ∃ c : ℝ, ∀ x y : ℤ, f (x, y) = c ∧ 0 ≤ c ∧ c ≤ 1 :=
sorry

end function_must_be_constant_l2364_236486


namespace quadrilateral_area_l2364_236406

theorem quadrilateral_area (rectangle_area shaded_triangles_area : ℝ) 
  (h1 : rectangle_area = 24)
  (h2 : shaded_triangles_area = 7.5) :
  rectangle_area - shaded_triangles_area = 16.5 := by
  sorry

end quadrilateral_area_l2364_236406


namespace circle_y_axis_intersection_l2364_236404

/-- A circle with diameter endpoints at (0,0) and (10,0) -/
def circle_with_diameter (x y : ℝ) : Prop :=
  (x - 5)^2 + y^2 = 25

/-- The y-axis -/
def y_axis (x : ℝ) : Prop := x = 0

/-- The intersection point of the circle and y-axis -/
def intersection_point (y : ℝ) : Prop :=
  circle_with_diameter 0 y ∧ y_axis 0

theorem circle_y_axis_intersection :
  ∃ y : ℝ, intersection_point y ∧ y = 0 :=
sorry

end circle_y_axis_intersection_l2364_236404


namespace vodka_mixture_profit_l2364_236474

/-- Represents the profit percentage of a mixture of two vodkas -/
def mixtureProfitPercentage (profit1 profit2 : ℚ) (increase1 increase2 : ℚ) : ℚ :=
  (profit1 * increase1 + profit2 * increase2) / 2

theorem vodka_mixture_profit :
  let initialProfit1 : ℚ := 10 / 100
  let initialProfit2 : ℚ := 40 / 100
  let increase1 : ℚ := 4 / 3
  let increase2 : ℚ := 5 / 3
  mixtureProfitPercentage initialProfit1 initialProfit2 increase1 increase2 = 40 / 100 := by
  sorry

end vodka_mixture_profit_l2364_236474


namespace ordering_abc_l2364_236447

noncomputable def a : ℝ := 0.98 + Real.sin 0.01
noncomputable def b : ℝ := Real.exp (-0.01)
noncomputable def c : ℝ := (Real.log 2022) / (Real.log 2023) / (Real.log 2021) / (Real.log 2023)

theorem ordering_abc : c > b ∧ b > a := by sorry

end ordering_abc_l2364_236447


namespace larger_number_given_hcf_lcm_factors_l2364_236483

theorem larger_number_given_hcf_lcm_factors (a b : ℕ) : 
  a > 0 → b > 0 → 
  Nat.gcd a b = 10 → 
  Nat.lcm a b = 10 * 11 * 15 → 
  max a b = 150 := by
sorry

end larger_number_given_hcf_lcm_factors_l2364_236483


namespace angle_sum_is_pi_over_two_l2364_236480

open Real

theorem angle_sum_is_pi_over_two (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : sin α ^ 2 + sin β ^ 2 - (Real.sqrt 6 / 2) * sin α - (Real.sqrt 10 / 2) * sin β + 1 = 0) : 
  α + β = π/2 := by
sorry

end angle_sum_is_pi_over_two_l2364_236480


namespace min_max_f_on_interval_l2364_236458

-- Define the function f(x) = x³ - 12x
def f (x : ℝ) : ℝ := x^3 - 12*x

-- State the theorem
theorem min_max_f_on_interval :
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc (-3) 3, f x ≥ min ∧ f x ≤ max) ∧
    (∃ x₁ ∈ Set.Icc (-3) 3, f x₁ = min) ∧
    (∃ x₂ ∈ Set.Icc (-3) 3, f x₂ = max) ∧
    min = -16 ∧ max = 16 := by
  sorry


end min_max_f_on_interval_l2364_236458


namespace january_more_expensive_l2364_236479

/-- Represents the cost of purchasing screws and bolts in different months -/
structure CostComparison where
  january_screws_per_dollar : ℕ
  january_bolts_per_dollar : ℕ
  february_set_screws : ℕ
  february_set_bolts : ℕ
  february_set_price : ℕ
  tractor_screws : ℕ
  tractor_bolts : ℕ

/-- Calculates the cost of purchasing screws and bolts for a tractor in January -/
def january_cost (c : CostComparison) : ℚ :=
  (c.tractor_screws : ℚ) / c.january_screws_per_dollar + (c.tractor_bolts : ℚ) / c.january_bolts_per_dollar

/-- Calculates the cost of purchasing screws and bolts for a tractor in February -/
def february_cost (c : CostComparison) : ℚ :=
  (c.february_set_price : ℚ) * (max (c.tractor_screws / c.february_set_screws) (c.tractor_bolts / c.february_set_bolts))

/-- Theorem stating that the cost in January is higher than in February -/
theorem january_more_expensive (c : CostComparison) 
    (h1 : c.january_screws_per_dollar = 40)
    (h2 : c.january_bolts_per_dollar = 60)
    (h3 : c.february_set_screws = 25)
    (h4 : c.february_set_bolts = 25)
    (h5 : c.february_set_price = 1)
    (h6 : c.tractor_screws = 600)
    (h7 : c.tractor_bolts = 600) :
  january_cost c > february_cost c := by
  sorry

end january_more_expensive_l2364_236479


namespace cow_population_characteristics_l2364_236439

/-- Represents the number of cows in each category --/
structure CowPopulation where
  total : ℕ
  male : ℕ
  female : ℕ
  transgender : ℕ

/-- Represents the characteristics of cows in each category --/
structure CowCharacteristics where
  hornedMalePercentage : ℚ
  spottedFemalePercentage : ℚ
  uniquePatternTransgenderPercentage : ℚ

/-- Theorem stating the relation between spotted females and the sum of horned males and uniquely patterned transgender cows --/
theorem cow_population_characteristics 
  (pop : CowPopulation)
  (char : CowCharacteristics)
  (h1 : pop.total = 450)
  (h2 : pop.male = 3 * pop.female / 2)
  (h3 : pop.female = 2 * pop.transgender)
  (h4 : pop.total = pop.male + pop.female + pop.transgender)
  (h5 : char.hornedMalePercentage = 3/5)
  (h6 : char.spottedFemalePercentage = 1/2)
  (h7 : char.uniquePatternTransgenderPercentage = 7/10) :
  ↑(pop.female * 1) * char.spottedFemalePercentage = 
  ↑(pop.male * 1) * char.hornedMalePercentage + ↑(pop.transgender * 1) * char.uniquePatternTransgenderPercentage - 112 :=
sorry

end cow_population_characteristics_l2364_236439


namespace exponent_law_multiplication_l2364_236436

theorem exponent_law_multiplication (y : ℝ) (n : ℤ) (h : y ≠ 0) :
  y * y^n = y^(n + 1) := by sorry

end exponent_law_multiplication_l2364_236436


namespace boy_scout_interest_l2364_236423

/-- Represents the simple interest calculation for a Boy Scout Troop's account --/
theorem boy_scout_interest (final_balance : ℝ) (rate : ℝ) (time : ℝ) (interest : ℝ) : 
  final_balance = 310.45 →
  rate = 0.06 →
  time = 0.25 →
  interest = final_balance - (final_balance / (1 + rate * time)) →
  interest = 4.54 := by
sorry

end boy_scout_interest_l2364_236423


namespace max_items_purchasable_l2364_236498

theorem max_items_purchasable (available : ℚ) (cost_per_item : ℚ) (h1 : available = 9.2) (h2 : cost_per_item = 1.05) :
  ⌊available / cost_per_item⌋ = 8 := by
  sorry

end max_items_purchasable_l2364_236498


namespace antons_winning_numbers_infinite_l2364_236441

theorem antons_winning_numbers_infinite :
  ∃ (f : ℕ → ℕ), Function.Injective f ∧
  ∀ (x : ℕ), 
    let n := f x
    (¬ ∃ (m : ℕ), n = m ^ 2) ∧ 
    ∃ (k : ℕ), (n + (n+1) + (n+2) + (n+3) + (n+4)) = k ^ 2 :=
sorry

end antons_winning_numbers_infinite_l2364_236441


namespace inequality_solution_l2364_236408

theorem inequality_solution (x : ℝ) : 
  (x - 2) / (x - 1) > (4 * x - 1) / (3 * x + 8) ↔ 
  (x > -3 ∧ x < -2) ∨ (x > -8/3 ∧ x < 1) := by
sorry

end inequality_solution_l2364_236408


namespace purely_imaginary_complex_number_l2364_236440

theorem purely_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := (1 + 2*I) / (a - I)
  (∃ (b : ℝ), z = b*I) → a = 2 :=
by sorry

end purely_imaginary_complex_number_l2364_236440


namespace max_value_quadratic_function_l2364_236416

theorem max_value_quadratic_function (p : ℝ) (hp : p > 0) :
  (∃ x ∈ Set.Icc (0 : ℝ) (4 / p), -1 / (2 * p) * x^2 + x > 1) ↔ 2 < p ∧ p < 1 + Real.sqrt 5 := by
  sorry

end max_value_quadratic_function_l2364_236416


namespace additional_girls_needed_prove_additional_girls_l2364_236488

theorem additional_girls_needed (initial_girls initial_boys : ℕ) 
  (target_ratio : ℚ) (additional_girls : ℕ) : Prop :=
  initial_girls = 2 →
  initial_boys = 6 →
  target_ratio = 5/8 →
  (initial_girls + additional_girls : ℚ) / 
    (initial_girls + initial_boys + additional_girls) = target_ratio →
  additional_girls = 8

theorem prove_additional_girls : 
  ∃ (additional_girls : ℕ), 
    additional_girls_needed 2 6 (5/8) additional_girls :=
sorry

end additional_girls_needed_prove_additional_girls_l2364_236488


namespace keith_and_jason_books_l2364_236418

/-- The number of books Keith and Jason have together -/
def total_books (keith_books jason_books : ℕ) : ℕ :=
  keith_books + jason_books

/-- Theorem: Keith and Jason have 41 books together -/
theorem keith_and_jason_books :
  total_books 20 21 = 41 := by
  sorry

end keith_and_jason_books_l2364_236418


namespace sqrt_meaningful_value_l2364_236419

theorem sqrt_meaningful_value (x : ℝ) : 
  (x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 3) → 
  (x - 2 ≥ 0 ↔ x = 3) :=
by sorry

end sqrt_meaningful_value_l2364_236419


namespace dime_nickel_difference_l2364_236417

/-- Proves that given 70 cents total and 2 nickels, the number of dimes exceeds the number of nickels by 4 -/
theorem dime_nickel_difference :
  ∀ (total_cents : ℕ) (num_nickels : ℕ) (nickel_value : ℕ) (dime_value : ℕ),
    total_cents = 70 →
    num_nickels = 2 →
    nickel_value = 5 →
    dime_value = 10 →
    ∃ (num_dimes : ℕ),
      num_dimes * dime_value + num_nickels * nickel_value = total_cents ∧
      num_dimes = num_nickels + 4 := by
  sorry

end dime_nickel_difference_l2364_236417


namespace percentage_boys_soccer_l2364_236412

def total_students : ℕ := 420
def boys : ℕ := 312
def soccer_players : ℕ := 250
def girls_not_playing : ℕ := 53

theorem percentage_boys_soccer : 
  (boys - (total_students - soccer_players - girls_not_playing)) / soccer_players * 100 = 78 := by
  sorry

end percentage_boys_soccer_l2364_236412


namespace power_product_equals_sum_of_exponents_l2364_236401

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^3 * a^6 = a^9 := by
  sorry

end power_product_equals_sum_of_exponents_l2364_236401


namespace distance_inequality_l2364_236465

theorem distance_inequality (x b : ℝ) (h1 : b > 0) (h2 : |x - 3| + |x - 5| < b) : b > 2 := by
  sorry

end distance_inequality_l2364_236465


namespace rope_sections_l2364_236433

/-- Given a rope of 50 feet, prove that after using 1/5 for art and giving half of the remainder
    to a friend, the number of 2-foot sections that can be cut from the remaining rope is 10. -/
theorem rope_sections (total_rope : ℝ) (art_fraction : ℝ) (friend_fraction : ℝ) (section_length : ℝ) :
  total_rope = 50 ∧
  art_fraction = 1/5 ∧
  friend_fraction = 1/2 ∧
  section_length = 2 →
  (total_rope - art_fraction * total_rope) * (1 - friend_fraction) / section_length = 10 := by
  sorry

end rope_sections_l2364_236433


namespace det_AB_eq_one_l2364_236462

open Matrix

variable {n : ℕ}

theorem det_AB_eq_one
  (A B : Matrix (Fin n) (Fin n) ℝ)
  (hA : IsUnit A)
  (hB : IsUnit B)
  (h : (A + B⁻¹)⁻¹ = A⁻¹ + B) :
  det (A * B) = 1 := by
  sorry

end det_AB_eq_one_l2364_236462


namespace four_solutions_l2364_236456

-- Define the two equations
def equation1 (x y : ℝ) : Prop := (x - y + 3) * (4 * x + y - 5) = 0
def equation2 (x y : ℝ) : Prop := (x + y - 3) * (3 * x - 4 * y + 6) = 0

-- Define a solution as a pair of real numbers satisfying both equations
def is_solution (p : ℝ × ℝ) : Prop :=
  equation1 p.1 p.2 ∧ equation2 p.1 p.2

-- State the theorem
theorem four_solutions :
  ∃ (s : Finset (ℝ × ℝ)), s.card = 4 ∧ (∀ p ∈ s, is_solution p) ∧
  (∀ p : ℝ × ℝ, is_solution p → p ∈ s) :=
sorry

end four_solutions_l2364_236456


namespace banana_permutations_eq_60_l2364_236476

/-- The number of distinct permutations of the letters in "BANANA" -/
def banana_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of distinct permutations of "BANANA" is 60 -/
theorem banana_permutations_eq_60 : banana_permutations = 60 := by
  sorry

end banana_permutations_eq_60_l2364_236476


namespace perfect_square_trinomial_m_value_l2364_236460

/-- A trinomial is a perfect square if it can be expressed as (x - a)^2 for some real number a -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (x - k)^2

theorem perfect_square_trinomial_m_value :
  ∀ m : ℝ, IsPerfectSquareTrinomial 1 (-m) 16 → m = 8 ∨ m = -8 := by
  sorry

end perfect_square_trinomial_m_value_l2364_236460


namespace x_squared_plus_reciprocal_l2364_236492

theorem x_squared_plus_reciprocal (x : ℝ) (h : 20 = x^6 + 1/x^6) : x^2 + 1/x^2 = 23 := by
  sorry

end x_squared_plus_reciprocal_l2364_236492


namespace right_triangle_legs_l2364_236427

theorem right_triangle_legs (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive lengths
  c = 25 →                 -- Hypotenuse is 25 cm
  a^2 + b^2 = c^2 →        -- Pythagorean theorem
  b / a = 4 / 3 →          -- Ratio of legs is 4:3
  a = 15 ∧ b = 20 :=       -- Legs are 15 cm and 20 cm
by sorry

end right_triangle_legs_l2364_236427


namespace jelly_bean_count_jelly_bean_theorem_l2364_236497

theorem jelly_bean_count : ℕ → Prop :=
  fun total_jelly_beans =>
    let red_jelly_beans := (3 * total_jelly_beans) / 4
    let coconut_red_jelly_beans := red_jelly_beans / 4
    coconut_red_jelly_beans = 750 →
    total_jelly_beans = 4000

-- Proof
theorem jelly_bean_theorem : jelly_bean_count 4000 := by
  sorry

end jelly_bean_count_jelly_bean_theorem_l2364_236497


namespace paper_flipping_difference_l2364_236473

theorem paper_flipping_difference (Y G : ℕ) : 
  Y - 152 = G + 152 + 346 → Y - G = 650 := by sorry

end paper_flipping_difference_l2364_236473


namespace fish_count_l2364_236410

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := 10

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 11

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := lilly_fish + rosy_fish

theorem fish_count : total_fish = 21 := by sorry

end fish_count_l2364_236410


namespace constant_remainder_l2364_236466

-- Define the polynomials
def f (b : ℚ) (x : ℚ) : ℚ := 12 * x^3 - 9 * x^2 + b * x + 8
def g (x : ℚ) : ℚ := 3 * x^2 - 4 * x + 2

-- Define the remainder function
def remainder (b : ℚ) (x : ℚ) : ℚ := f b x - g x * ((4 * x + 0) : ℚ)

-- Theorem statement
theorem constant_remainder :
  ∃ (c : ℚ), ∀ (x : ℚ), remainder (-4/3) x = c :=
sorry

end constant_remainder_l2364_236466


namespace lune_area_l2364_236425

/-- The area of the region inside a semicircle of diameter 2, outside a semicircle of diameter 4,
    and outside an inscribed square with side length 2 is equal to -π + 2. -/
theorem lune_area (π : ℝ) (h : π > 0) : 
  let small_semicircle_area := (1/2) * π * (2/2)^2
  let large_semicircle_area := (1/2) * π * (4/2)^2
  let square_area := 2^2
  let sector_area := (1/4) * large_semicircle_area
  small_semicircle_area - sector_area - square_area = -π + 2 := by
  sorry

end lune_area_l2364_236425


namespace goldfish_equality_l2364_236481

theorem goldfish_equality (n : ℕ) : (∃ m : ℕ, m < n ∧ 3^(m + 1) = 81 * 3^m) ↔ n > 3 :=
by sorry

end goldfish_equality_l2364_236481


namespace binomial_10_choose_5_l2364_236424

theorem binomial_10_choose_5 : Nat.choose 10 5 = 252 := by
  sorry

end binomial_10_choose_5_l2364_236424


namespace triangle_count_properties_l2364_236405

/-- Function that counts the number of congruent integer-sided triangles with perimeter n -/
def f (n : ℕ) : ℕ := sorry

/-- Theorem stating the properties of function f for specific values -/
theorem triangle_count_properties (h : ∀ n : ℕ, n ≥ 3 → f n = f n) :
  (f 1999 > f 1996) ∧ (f 2000 = f 1997) := by sorry

end triangle_count_properties_l2364_236405


namespace figure_perimeter_is_33_l2364_236420

/-- The perimeter of a figure composed of a 4x4 square with a 2x1 rectangle protruding from one side -/
def figurePerimeter (unitSquareSideLength : ℝ) : ℝ :=
  let largeSquareSide := 4 * unitSquareSideLength
  let rectangleWidth := 2 * unitSquareSideLength
  let rectangleHeight := unitSquareSideLength
  2 * largeSquareSide + rectangleWidth + rectangleHeight

theorem figure_perimeter_is_33 :
  figurePerimeter 2 = 33 := by
  sorry


end figure_perimeter_is_33_l2364_236420


namespace triangle_angles_from_exterior_l2364_236461

theorem triangle_angles_from_exterior (A B C : ℝ) : 
  A + B + C = 180 →
  (180 - B) / (180 - C) = 12 / 7 →
  (180 - B) - (180 - C) = 50 →
  (A = 10 ∧ B = 60 ∧ C = 110) := by
  sorry

end triangle_angles_from_exterior_l2364_236461


namespace x_eq_3_is_linear_l2364_236453

/-- Definition of a linear equation with one variable -/
def is_linear_equation_one_var (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation x = 3 -/
def f : ℝ → ℝ := λ x ↦ x - 3

/-- Theorem: x = 3 is a linear equation with one variable -/
theorem x_eq_3_is_linear : is_linear_equation_one_var f := by
  sorry


end x_eq_3_is_linear_l2364_236453


namespace saly_needs_ten_eggs_l2364_236443

/-- The number of eggs needed by various individuals and produced by the farm --/
structure EggNeeds where
  ben_weekly : ℕ  -- Ben's weekly egg needs
  ked_weekly : ℕ  -- Ked's weekly egg needs
  monthly_total : ℕ  -- Total eggs produced by the farm in a month
  weeks_in_month : ℕ  -- Number of weeks in a month

/-- Calculates Saly's weekly egg needs based on the given conditions --/
def saly_weekly_needs (e : EggNeeds) : ℕ :=
  (e.monthly_total - (e.ben_weekly + e.ked_weekly) * e.weeks_in_month) / e.weeks_in_month

/-- Theorem stating that Saly needs 10 eggs per week given the conditions --/
theorem saly_needs_ten_eggs (e : EggNeeds) 
  (h1 : e.ben_weekly = 14)
  (h2 : e.ked_weekly = e.ben_weekly / 2)
  (h3 : e.monthly_total = 124)
  (h4 : e.weeks_in_month = 4) : 
  saly_weekly_needs e = 10 := by
  sorry

end saly_needs_ten_eggs_l2364_236443


namespace smallest_four_digit_mod_8_5_l2364_236426

theorem smallest_four_digit_mod_8_5 : ∃ (n : ℕ), 
  (n ≥ 1000) ∧ 
  (n < 10000) ∧ 
  (n % 8 = 5) ∧ 
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 8 = 5 → m ≥ n) ∧
  (n = 1005) := by
sorry

end smallest_four_digit_mod_8_5_l2364_236426


namespace sqrt_combinability_with_sqrt_3_l2364_236459

theorem sqrt_combinability_with_sqrt_3 :
  ∃! x : ℝ, (x = Real.sqrt 32 ∨ x = -Real.sqrt 27 ∨ x = Real.sqrt 12 ∨ x = Real.sqrt (1/3)) ∧
  (∃ y : ℝ, x = y ∧ y ≠ 0 ∧ ∀ a b : ℝ, (y = a * Real.sqrt 3 + b → a = 0)) :=
by
  sorry

end sqrt_combinability_with_sqrt_3_l2364_236459


namespace valid_colorings_6x6_l2364_236402

/-- Recursive function for the number of valid colorings of an nxn grid -/
def f : ℕ → ℕ
| 0 => 0
| 1 => 0
| 2 => 1
| n + 2 => n * (n + 1) * f (n + 1) + (n * (n + 1)^2 / 2) * f n

/-- The size of the grid -/
def grid_size : ℕ := 6

/-- The number of red squares required in each row and column -/
def red_squares_per_line : ℕ := 2

/-- Theorem stating the number of valid colorings for a 6x6 grid -/
theorem valid_colorings_6x6 : f grid_size = 67950 := by
  sorry

end valid_colorings_6x6_l2364_236402


namespace smallest_number_proof_l2364_236421

theorem smallest_number_proof (x y z : ℝ) : 
  y = 2 * x →
  z = 4 * y →
  (x + y + z) / 3 = 165 →
  x = 45 := by
sorry

end smallest_number_proof_l2364_236421


namespace parallelepiped_to_cube_l2364_236413

/-- A rectangular parallelepiped with edges 8, 8, and 27 has the same volume as a cube with side length 12 -/
theorem parallelepiped_to_cube : 
  let parallelepiped_volume := 8 * 8 * 27
  let cube_volume := 12 * 12 * 12
  parallelepiped_volume = cube_volume := by
  sorry

#eval 8 * 8 * 27
#eval 12 * 12 * 12

end parallelepiped_to_cube_l2364_236413


namespace cave_depth_remaining_l2364_236411

theorem cave_depth_remaining (total_depth : ℕ) (traveled_distance : ℕ) 
  (h1 : total_depth = 1218)
  (h2 : traveled_distance = 849) :
  total_depth - traveled_distance = 369 := by
sorry

end cave_depth_remaining_l2364_236411


namespace parabola_focus_l2364_236429

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  y = 9 * x^2 + 6 * x - 2

/-- The focus of a parabola -/
def is_focus (f : ℝ × ℝ) (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a h k : ℝ), 
    (∀ x y, eq x y ↔ y = a * (x - h)^2 + k) ∧
    f = (h, k + 1 / (4 * a))

/-- Theorem: The focus of the parabola y = 9x^2 + 6x - 2 is (-1/3, -107/36) -/
theorem parabola_focus :
  is_focus (-1/3, -107/36) parabola_equation :=
sorry

end parabola_focus_l2364_236429


namespace expression_evaluation_l2364_236470

theorem expression_evaluation :
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 := by
  sorry

end expression_evaluation_l2364_236470


namespace red_white_red_probability_l2364_236448

/-- The probability of drawing a red marble, then a white marble, and then a red marble
    from a bag containing 4 red marbles and 6 white marbles, without replacement. -/
theorem red_white_red_probability (total_marbles : Nat) (red_marbles : Nat) (white_marbles : Nat)
    (h1 : total_marbles = red_marbles + white_marbles)
    (h2 : red_marbles = 4)
    (h3 : white_marbles = 6) :
    (red_marbles : ℚ) / total_marbles *
    (white_marbles : ℚ) / (total_marbles - 1) *
    (red_marbles - 1 : ℚ) / (total_marbles - 2) = 1 / 10 := by
  sorry

end red_white_red_probability_l2364_236448


namespace largest_angle_in_hexagon_l2364_236477

/-- Theorem: In a hexagon ABCDEF with given angle conditions, the largest angle measures 304°. -/
theorem largest_angle_in_hexagon (A B C D E F : ℝ) : 
  A = 100 →
  B = 120 →
  C = D →
  F = 3 * C + 10 →
  A + B + C + D + E + F = 720 →
  max A (max B (max C (max D (max E F)))) = 304 := by
sorry

end largest_angle_in_hexagon_l2364_236477


namespace orange_distribution_l2364_236487

theorem orange_distribution (total_oranges : ℕ) (pieces_per_orange : ℕ) (pieces_per_friend : ℕ) :
  total_oranges = 80 →
  pieces_per_orange = 10 →
  pieces_per_friend = 4 →
  (total_oranges * pieces_per_orange) / pieces_per_friend = 200 :=
by
  sorry

end orange_distribution_l2364_236487


namespace garden_bug_problem_l2364_236428

theorem garden_bug_problem (initial_plants : ℕ) (day1_eaten : ℕ) (day3_eaten : ℕ) : 
  initial_plants = 30 →
  day1_eaten = 20 →
  day3_eaten = 1 →
  initial_plants - day1_eaten - (initial_plants - day1_eaten) / 2 - day3_eaten = 4 :=
by
  sorry

end garden_bug_problem_l2364_236428


namespace fraction_always_defined_l2364_236446

theorem fraction_always_defined (y : ℝ) : (y^2 + 1) ≠ 0 :=
sorry

end fraction_always_defined_l2364_236446


namespace tom_tickets_left_l2364_236455

/-- The number of tickets Tom has left after winning some and spending some -/
def tickets_left (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (spent_tickets : ℕ) : ℕ :=
  whack_a_mole_tickets + skee_ball_tickets - spent_tickets

/-- Theorem stating that Tom has 50 tickets left -/
theorem tom_tickets_left : tickets_left 32 25 7 = 50 := by
  sorry

end tom_tickets_left_l2364_236455


namespace amoeba_growth_5_days_l2364_236403

def amoeba_population (initial_population : ℕ) (split_rate : ℕ) (days : ℕ) : ℕ :=
  initial_population * split_rate ^ days

theorem amoeba_growth_5_days :
  amoeba_population 1 3 5 = 243 := by
  sorry

end amoeba_growth_5_days_l2364_236403


namespace hyperbola_parabola_focus_l2364_236482

theorem hyperbola_parabola_focus (a : ℝ) (h1 : a > 0) : 
  (∃ (x y : ℝ), x^2/a^2 - y^2/3 = 1) ∧ 
  (∃ (x : ℝ), (2, 0) = (x, 0) ∧ x^2/a^2 - 0^2/3 = 1) →
  a = 1 := by sorry

end hyperbola_parabola_focus_l2364_236482


namespace jane_crayons_l2364_236431

theorem jane_crayons (initial_crayons : ℕ) (eaten_crayons : ℕ) : 
  initial_crayons = 87 → eaten_crayons = 7 → initial_crayons - eaten_crayons = 80 := by
  sorry

end jane_crayons_l2364_236431


namespace banana_pear_weight_equivalence_l2364_236450

/-- Given that 9 bananas weigh the same as 6 pears, prove that 36 bananas weigh the same as 24 pears. -/
theorem banana_pear_weight_equivalence (banana_weight pear_weight : ℝ) 
  (h : 9 * banana_weight = 6 * pear_weight) :
  36 * banana_weight = 24 * pear_weight := by
  sorry

end banana_pear_weight_equivalence_l2364_236450


namespace ps_length_l2364_236444

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  right_angle : (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0
  pq_length : Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = 15
  qr_length : Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) = 20

-- Define points S and T
def S (P R : ℝ × ℝ) : ℝ × ℝ := sorry
def T (Q R : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the theorem
theorem ps_length (P Q R : ℝ × ℝ) (h : Triangle P Q R) :
  let S := S P R
  let T := T Q R
  (S.1 - P.1) * (T.1 - S.1) + (S.2 - P.2) * (T.2 - S.2) = 0 →
  Real.sqrt ((T.1 - S.1)^2 + (T.2 - S.2)^2) = 12 →
  Real.sqrt ((S.1 - P.1)^2 + (S.2 - P.2)^2) = 15 := by
  sorry

end ps_length_l2364_236444


namespace map_scale_l2364_236496

/-- Given a map where 15 cm represents 90 km, prove that 20 cm represents 120 km -/
theorem map_scale (scale : ℝ → ℝ) (h1 : scale 15 = 90) : scale 20 = 120 := by
  sorry

end map_scale_l2364_236496


namespace tan_alpha_minus_pi_over_four_l2364_236478

theorem tan_alpha_minus_pi_over_four (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan β = 1 / 3) :
  Real.tan (α - π / 4) = - 8 / 9 := by
  sorry

end tan_alpha_minus_pi_over_four_l2364_236478


namespace f_max_min_on_interval_l2364_236415

def f (x : ℝ) := x^3 - 3*x^2 - 9*x + 2

theorem f_max_min_on_interval :
  ∃ (max min : ℝ), max = 2 ∧ min = -25 ∧
  (∀ x ∈ Set.Icc 0 4, f x ≤ max ∧ f x ≥ min) ∧
  (∃ x₁ ∈ Set.Icc 0 4, f x₁ = max) ∧
  (∃ x₂ ∈ Set.Icc 0 4, f x₂ = min) :=
sorry

end f_max_min_on_interval_l2364_236415


namespace polar_equation_of_line_l2364_236472

/-- The polar equation of a line passing through (5,0) and perpendicular to α = π/4 -/
theorem polar_equation_of_line (ρ θ : ℝ) : 
  (∃ (x y : ℝ), x = 5 ∧ y = 0 ∧ ρ * (Real.cos θ) = x ∧ ρ * (Real.sin θ) = y) →
  (∀ (α : ℝ), α = π/4 → (Real.tan α) * (Real.tan (α + π/2)) = -1) →
  ρ * Real.sin (π/4 + θ) = 5 * Real.sqrt 2 / 2 :=
by sorry

end polar_equation_of_line_l2364_236472


namespace larry_stickers_l2364_236464

/-- The number of stickers Larry starts with -/
def initial_stickers : ℕ := 93

/-- The number of stickers Larry loses -/
def lost_stickers : ℕ := 6

/-- The number of stickers Larry ends with -/
def final_stickers : ℕ := initial_stickers - lost_stickers

theorem larry_stickers : final_stickers = 87 := by
  sorry

end larry_stickers_l2364_236464


namespace base_prime_rep_945_l2364_236471

def base_prime_representation (n : ℕ) : List ℕ :=
  sorry

theorem base_prime_rep_945 : base_prime_representation 945 = [3, 1, 1, 0] := by
  sorry

end base_prime_rep_945_l2364_236471


namespace emily_pastry_production_l2364_236452

/-- Emily's pastry production problem -/
theorem emily_pastry_production (p h : ℕ) : 
  p = 3 * h →
  h = 1 →
  (p - 3) * (h + 3) - p * h = 3 := by
  sorry

end emily_pastry_production_l2364_236452


namespace stratified_sampling_low_income_l2364_236435

/-- Represents the number of households sampled from a given group -/
def sampleSize (totalSize : ℕ) (groupSize : ℕ) (sampledHighIncome : ℕ) (totalHighIncome : ℕ) : ℕ :=
  (sampledHighIncome * groupSize) / totalHighIncome

theorem stratified_sampling_low_income 
  (totalHouseholds : ℕ) 
  (highIncomeHouseholds : ℕ) 
  (lowIncomeHouseholds : ℕ) 
  (sampledHighIncome : ℕ) :
  totalHouseholds = 500 →
  highIncomeHouseholds = 125 →
  lowIncomeHouseholds = 95 →
  sampledHighIncome = 25 →
  sampleSize totalHouseholds lowIncomeHouseholds sampledHighIncome highIncomeHouseholds = 19 := by
  sorry

#check stratified_sampling_low_income

end stratified_sampling_low_income_l2364_236435


namespace proposition_problem_l2364_236493

theorem proposition_problem (a : ℝ) :
  ((∃ x₀ : ℝ, a * x₀^2 + 2 * x₀ + a < 0) ∨
   (∃ x y : ℝ, x ≠ y ∧ x < 0 ∧ y < 0 ∧ x^2 + a*x + 1 = 0 ∧ y^2 + a*y + 1 = 0)) →
  (a > 2 ∨ a < 1) :=
sorry

end proposition_problem_l2364_236493


namespace initial_concentrated_kola_percentage_l2364_236475

/-- Proves that the initial percentage of concentrated kola in a 340-liter solution is 5% -/
theorem initial_concentrated_kola_percentage
  (initial_volume : ℝ)
  (initial_water_percentage : ℝ)
  (added_sugar : ℝ)
  (added_water : ℝ)
  (added_concentrated_kola : ℝ)
  (new_volume : ℝ)
  (new_sugar_percentage : ℝ)
  (h1 : initial_volume = 340)
  (h2 : initial_water_percentage = 88 / 100)
  (h3 : added_sugar = 3.2)
  (h4 : added_water = 10)
  (h5 : added_concentrated_kola = 6.8)
  (h6 : new_volume = initial_volume + added_sugar + added_water + added_concentrated_kola)
  (h7 : new_sugar_percentage = 7.5 / 100) :
  ∃ (initial_concentrated_kola_percentage : ℝ),
    initial_concentrated_kola_percentage = 5 / 100 := by
  sorry

end initial_concentrated_kola_percentage_l2364_236475


namespace rectangle_y_coordinate_sum_l2364_236442

/-- Given a rectangle with opposite vertices (5,22) and (12,-3),
    the sum of the y-coordinates of the other two vertices is 19. -/
theorem rectangle_y_coordinate_sum :
  let v1 : ℝ × ℝ := (5, 22)
  let v2 : ℝ × ℝ := (12, -3)
  let mid_y : ℝ := (v1.2 + v2.2) / 2
  19 = 2 * mid_y := by sorry

end rectangle_y_coordinate_sum_l2364_236442


namespace tangent_intersection_for_specific_circles_l2364_236434

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Finds the x-coordinate of the intersection point between the common tangent line
    of two circles and the x-axis -/
def tangentIntersectionX (c1 c2 : Circle) : ℝ :=
  sorry

theorem tangent_intersection_for_specific_circles :
  let c1 : Circle := { center := (0, 0), radius := 3 }
  let c2 : Circle := { center := (12, 0), radius := 5 }
  tangentIntersectionX c1 c2 = 9 / 2 := by
  sorry

end tangent_intersection_for_specific_circles_l2364_236434


namespace largest_common_term_l2364_236485

def isInFirstSequence (x : ℕ) : Prop := ∃ n : ℕ, x = 3 + 8 * n

def isInSecondSequence (x : ℕ) : Prop := ∃ m : ℕ, x = 5 + 9 * m

theorem largest_common_term : 
  (∃ x : ℕ, x ≤ 200 ∧ isInFirstSequence x ∧ isInSecondSequence x ∧ 
    ∀ y : ℕ, y ≤ 200 → isInFirstSequence y → isInSecondSequence y → y ≤ x) ∧
  (∃ x : ℕ, x = 131 ∧ x ≤ 200 ∧ isInFirstSequence x ∧ isInSecondSequence x) :=
by sorry

end largest_common_term_l2364_236485


namespace integer_solution_abc_l2364_236484

theorem integer_solution_abc : ∀ a b c : ℕ,
  1 < a ∧ a < b ∧ b < c ∧ (abc - 1) % ((a - 1) * (b - 1) * (c - 1)) = 0 →
  (a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8) := by
  sorry

end integer_solution_abc_l2364_236484


namespace roots_sum_ln_abs_l2364_236400

theorem roots_sum_ln_abs (m : ℝ) (x₁ x₂ : ℝ) :
  (Real.log (|x₁ - 2|) = m) ∧ (Real.log (|x₂ - 2|) = m) →
  x₁ + x₂ = 4 := by
  sorry

end roots_sum_ln_abs_l2364_236400


namespace maria_green_towels_l2364_236467

/-- The number of green towels Maria bought -/
def green_towels : ℕ := 35

/-- The number of white towels Maria bought -/
def white_towels : ℕ := 21

/-- The number of towels Maria gave to her mother -/
def towels_given : ℕ := 34

/-- The number of towels Maria ended up with -/
def towels_left : ℕ := 22

/-- Theorem stating that the number of green towels Maria bought is 35 -/
theorem maria_green_towels :
  green_towels = 35 ∧
  green_towels + white_towels - towels_given = towels_left :=
by sorry

end maria_green_towels_l2364_236467


namespace rectangular_solid_edge_sum_l2364_236454

/-- A rectangular solid with volume 512 cm³, surface area 448 cm², and dimensions in geometric progression has a total edge length of 112 cm. -/
theorem rectangular_solid_edge_sum : 
  ∀ (a b c : ℝ),
    a > 0 → b > 0 → c > 0 →
    a * b * c = 512 →
    2 * (a * b + b * c + a * c) = 448 →
    ∃ (r : ℝ), r > 0 ∧ (a = b / r ∧ c = b * r) →
    4 * (a + b + c) = 112 := by
  sorry

end rectangular_solid_edge_sum_l2364_236454


namespace task_completion_ways_l2364_236451

theorem task_completion_ways (method1_count method2_count : ℕ) :
  method1_count + method2_count = 
  (number_of_ways_to_choose_person : ℕ) :=
by sorry

#check task_completion_ways 5 4

end task_completion_ways_l2364_236451


namespace K_change_implies_equilibrium_shift_l2364_236489

-- Define the equilibrium constant as a function of temperature
def K (temperature : ℝ) : ℝ := sorry

-- Define a predicate for equilibrium shift
def equilibrium_shift (initial_state final_state : ℝ) : Prop :=
  initial_state ≠ final_state

-- Define a predicate for K change
def K_change (initial_K final_K : ℝ) : Prop :=
  initial_K ≠ final_K

-- Theorem statement
theorem K_change_implies_equilibrium_shift
  (initial_temp final_temp : ℝ)
  (h_K_change : K_change (K initial_temp) (K final_temp)) :
  equilibrium_shift initial_temp final_temp :=
sorry

end K_change_implies_equilibrium_shift_l2364_236489


namespace solution_set_f_gt_5_empty_solution_set_condition_l2364_236457

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |2*x + 2|

-- Theorem for the solution of f(x) > 5
theorem solution_set_f_gt_5 :
  {x : ℝ | f x > 5} = {x : ℝ | x < -2 ∨ x > 4/3} :=
sorry

-- Theorem for the range of a where f(x) < a has no solution
theorem empty_solution_set_condition (a : ℝ) :
  ({x : ℝ | f x < a} = ∅) ↔ (a ≤ 2) :=
sorry

end solution_set_f_gt_5_empty_solution_set_condition_l2364_236457
