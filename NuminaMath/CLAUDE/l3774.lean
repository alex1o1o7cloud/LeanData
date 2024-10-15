import Mathlib

namespace NUMINAMATH_CALUDE_vending_machine_drinks_l3774_377474

def arcade_problem (num_machines : ℕ) (sections_per_machine : ℕ) (drinks_left : ℕ) (drinks_dispensed : ℕ) : Prop :=
  let drinks_per_section : ℕ := drinks_left + drinks_dispensed
  let drinks_per_machine : ℕ := drinks_per_section * sections_per_machine
  let total_drinks : ℕ := drinks_per_machine * num_machines
  total_drinks = 840

theorem vending_machine_drinks :
  arcade_problem 28 6 3 2 := by
  sorry

end NUMINAMATH_CALUDE_vending_machine_drinks_l3774_377474


namespace NUMINAMATH_CALUDE_equal_tasks_after_transfer_l3774_377469

/-- Given that Robyn has 4 tasks and Sasha has 14 tasks, prove that if Robyn takes 5 tasks from Sasha, they will have an equal number of tasks. -/
theorem equal_tasks_after_transfer (robyn_initial : Nat) (sasha_initial : Nat) (tasks_transferred : Nat) : 
  robyn_initial = 4 → 
  sasha_initial = 14 → 
  tasks_transferred = 5 → 
  (robyn_initial + tasks_transferred = sasha_initial - tasks_transferred) := by
  sorry

#check equal_tasks_after_transfer

end NUMINAMATH_CALUDE_equal_tasks_after_transfer_l3774_377469


namespace NUMINAMATH_CALUDE_factorization_problems_l3774_377401

theorem factorization_problems :
  (∀ x y : ℝ, 6*x*y - 9*x^2*y = 3*x*y*(2-3*x)) ∧
  (∀ a : ℝ, (a^2+1)^2 - 4*a^2 = (a+1)^2*(a-1)^2) :=
by sorry

end NUMINAMATH_CALUDE_factorization_problems_l3774_377401


namespace NUMINAMATH_CALUDE_lines_concurrent_l3774_377432

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the incidence relation
variable (lies_on : Point → Line → Prop)

-- Define the intersection of two lines
variable (intersect : Line → Line → Point)

-- Define the line passing through two points
variable (line_through : Point → Point → Line)

variable (A B C D E F P X Y Z W Q : Point)

-- Define the quadrilateral ABCD
variable (is_quadrilateral : Prop)

-- Define the conditions for E, F, P, X, Y, Z, W
variable (E_def : E = intersect (line_through A B) (line_through C D))
variable (F_def : F = intersect (line_through B C) (line_through D A))
variable (not_on_EF : ¬ lies_on P (line_through E F))
variable (X_def : X = intersect (line_through P A) (line_through E F))
variable (Y_def : Y = intersect (line_through P B) (line_through E F))
variable (Z_def : Z = intersect (line_through P C) (line_through E F))
variable (W_def : W = intersect (line_through P D) (line_through E F))

-- The theorem to prove
theorem lines_concurrent :
  ∃ Q : Point,
    lies_on Q (line_through A Z) ∧
    lies_on Q (line_through B W) ∧
    lies_on Q (line_through C X) ∧
    lies_on Q (line_through D Y) :=
sorry

end NUMINAMATH_CALUDE_lines_concurrent_l3774_377432


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l3774_377458

/-- The distance from point P(x, -5) to the y-axis is 10 units, given that the distance
    from P to the x-axis is half the distance from P to the y-axis. -/
theorem distance_to_y_axis (x : ℝ) : 
  let P : ℝ × ℝ := (x, -5)
  let dist_to_x_axis := |P.2|
  let dist_to_y_axis := |P.1|
  dist_to_x_axis = (1/2) * dist_to_y_axis → dist_to_y_axis = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l3774_377458


namespace NUMINAMATH_CALUDE_function_is_identity_l3774_377435

def is_valid_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) - f x - f y ∈ ({0, 1} : Set ℝ)) ∧
  (∀ x : ℝ, ⌊f x⌋ = ⌊x⌋)

theorem function_is_identity (f : ℝ → ℝ) (h : is_valid_function f) :
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_is_identity_l3774_377435


namespace NUMINAMATH_CALUDE_equilateral_triangle_circumcircle_area_l3774_377420

/-- The area of the circumcircle of an equilateral triangle with side length 4√3 is 16π -/
theorem equilateral_triangle_circumcircle_area :
  let side_length : ℝ := 4 * Real.sqrt 3
  let triangle_area : ℝ := (side_length ^ 2 * Real.sqrt 3) / 4
  let circumradius : ℝ := side_length / Real.sqrt 3
  circumradius ^ 2 * Real.pi = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_circumcircle_area_l3774_377420


namespace NUMINAMATH_CALUDE_caleb_dandelion_friends_l3774_377453

/-- The number of friends Caleb shared dandelion puffs with -/
def num_friends (total : ℕ) (mom sister grandma dog friend : ℕ) : ℕ :=
  (total - (mom + sister + grandma + dog)) / friend

/-- Theorem stating the number of friends Caleb shared dandelion puffs with -/
theorem caleb_dandelion_friends :
  num_friends 40 3 3 5 2 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_caleb_dandelion_friends_l3774_377453


namespace NUMINAMATH_CALUDE_integer_triple_divisibility_l3774_377486

theorem integer_triple_divisibility :
  ∀ a b c : ℤ,
    1 < a ∧ a < b ∧ b < c →
    (a - 1) * (b - 1) * (c - 1) ∣ a * b * c - 1 →
    ((a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8)) :=
by sorry

end NUMINAMATH_CALUDE_integer_triple_divisibility_l3774_377486


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l3774_377442

def R : Set ℝ := Set.univ

def M : Set ℝ := {-1, 0, 1, 5}

def N : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

theorem intersection_M_complement_N : M ∩ (R \ N) = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l3774_377442


namespace NUMINAMATH_CALUDE_sara_gave_four_limes_l3774_377473

def limes_from_sara (initial_limes final_limes : ℕ) : ℕ :=
  final_limes - initial_limes

theorem sara_gave_four_limes (initial_limes final_limes : ℕ) 
  (h1 : initial_limes = 9)
  (h2 : final_limes = 13) :
  limes_from_sara initial_limes final_limes = 4 := by
  sorry

end NUMINAMATH_CALUDE_sara_gave_four_limes_l3774_377473


namespace NUMINAMATH_CALUDE_congruence_problem_l3774_377412

theorem congruence_problem (x : ℤ) 
  (h1 : (2 + x) % 3 = 2^2 % 3)
  (h2 : (4 + x) % 5 = 3^2 % 5)
  (h3 : (6 + x) % 7 = 5^2 % 7) :
  x % 105 = 5 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3774_377412


namespace NUMINAMATH_CALUDE_max_value_product_max_value_achieved_l3774_377461

theorem max_value_product (x y z w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_w : w > 0)
  (sum_one : x + y + z + w = 1) :
  x^2 * y^2 * z^2 * w ≤ 64 / 823543 := by
sorry

theorem max_value_achieved (x y z w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_w : w > 0)
  (sum_one : x + y + z + w = 1) :
  ∃ x y z w, x^2 * y^2 * z^2 * w = 64 / 823543 ∧ 
             x + y + z + w = 1 ∧ 
             x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 := by
sorry

end NUMINAMATH_CALUDE_max_value_product_max_value_achieved_l3774_377461


namespace NUMINAMATH_CALUDE_line_intersection_condition_l3774_377407

/-- Given a directed line segment PQ and a line l, prove that l intersects
    the extended line segment PQ if and only if m is within a specific range. -/
theorem line_intersection_condition (m : ℝ) : 
  let P : ℝ × ℝ := (-1, 1)
  let Q : ℝ × ℝ := (2, 2)
  let l := {(x, y) : ℝ × ℝ | x + m * y + m = 0}
  (∃ (t : ℝ), (1 - t) • P + t • Q ∈ l) ↔ -3 < m ∧ m < -2/3 :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_condition_l3774_377407


namespace NUMINAMATH_CALUDE_five_ruble_coins_l3774_377426

/-- Represents the number of coins of each denomination -/
structure CoinCount where
  one : ℕ
  two : ℕ
  five : ℕ
  ten : ℕ

/-- The total number of coins -/
def total_coins : ℕ := 25

/-- The number of coins that are not of each denomination -/
def not_two_coins : ℕ := 19
def not_ten_coins : ℕ := 20
def not_one_coins : ℕ := 16

/-- Theorem stating the number of five-ruble coins -/
theorem five_ruble_coins (c : CoinCount) : c.five = 5 :=
  by
    have h1 : c.one + c.two + c.five + c.ten = total_coins := sorry
    have h2 : c.two = total_coins - not_two_coins := sorry
    have h3 : c.ten = total_coins - not_ten_coins := sorry
    have h4 : c.one = total_coins - not_one_coins := sorry
    sorry

end NUMINAMATH_CALUDE_five_ruble_coins_l3774_377426


namespace NUMINAMATH_CALUDE_odd_function_properties_l3774_377496

noncomputable def f (x : ℝ) : ℝ :=
  if -1 < x ∧ x < 0 then -2^(-x)
  else if x = 0 then 0
  else if 0 < x ∧ x < 1 then 2^x
  else 0

theorem odd_function_properties (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, f (-x) = -f x) →
  (∀ x ∈ Set.Ioo (0 : ℝ) 1, f x = 2^x) →
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, f x ≤ 2*a) →
  (∀ x ∈ Set.Ioo (-1 : ℝ) 0, f x = -2^(-x)) ∧
  (f 0 = 0) ∧
  (∀ x ∈ Set.Ioo (0 : ℝ) 1, f x = 2^x) ∧
  (a ≥ 1) := by sorry

end NUMINAMATH_CALUDE_odd_function_properties_l3774_377496


namespace NUMINAMATH_CALUDE_least_three_digit_with_digit_product_12_l3774_377416

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem least_three_digit_with_digit_product_12 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 12 → n ≥ 143 :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_with_digit_product_12_l3774_377416


namespace NUMINAMATH_CALUDE_age_difference_l3774_377495

theorem age_difference (A B : ℕ) : B = 38 → A + 10 = 2 * (B - 10) → A - B = 8 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3774_377495


namespace NUMINAMATH_CALUDE_quadratic_solutions_l3774_377444

-- Define the quadratic function
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 5

-- State the theorem
theorem quadratic_solutions (b : ℝ) :
  (∀ x, f b x = x^2 + b*x - 5) →  -- Definition of f
  (-b/(2:ℝ) = 2) →               -- Axis of symmetry condition
  (∀ x, f b x = 2*x - 13 ↔ x = 2 ∨ x = 4) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_solutions_l3774_377444


namespace NUMINAMATH_CALUDE_integer_roots_conditions_l3774_377427

theorem integer_roots_conditions (p q : ℤ) : 
  (∃ (a b c d : ℤ), (∀ x : ℤ, x^4 + 2*p*x^2 + q*x + p^2 - 36 = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) ∧
  (a + b + c + d = 0) ∧
  (a*b + a*c + a*d + b*c + b*d + c*d = 2*p) ∧
  (a*b*c*d = p^2 - 36)) →
  ∃ (x y z : ℕ), 18 = 2*x^2 + y^2 + z^2 ∧ 
  ((x = 0 ∧ y = 3 ∧ z = 3) ∨
   (x = 1 ∧ y = 4 ∧ z = 0) ∨
   (x = 1 ∧ y = 0 ∧ z = 4) ∨
   (x = 2 ∧ y = 3 ∧ z = 1) ∨
   (x = 2 ∧ y = 1 ∧ z = 3) ∨
   (x = 3 ∧ y = 0 ∧ z = 0)) :=
by sorry

end NUMINAMATH_CALUDE_integer_roots_conditions_l3774_377427


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_9_divisible_by_26_l3774_377487

def ends_in_9 (n : ℕ) : Prop := n % 10 = 9

theorem smallest_positive_integer_ending_in_9_divisible_by_26 :
  ∃ (n : ℕ), n > 0 ∧ ends_in_9 n ∧ n % 26 = 0 ∧
  ∀ (m : ℕ), m > 0 → ends_in_9 m → m % 26 = 0 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_9_divisible_by_26_l3774_377487


namespace NUMINAMATH_CALUDE_rectangular_field_width_l3774_377440

/-- Proves that a rectangular field with length 7/5 times its width and perimeter 432 meters has a width of 90 meters -/
theorem rectangular_field_width (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = (7 / 5) * width →
  perimeter = 432 →
  perimeter = 2 * length + 2 * width →
  width = 90 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l3774_377440


namespace NUMINAMATH_CALUDE_chromosome_stability_processes_l3774_377499

-- Define the type for physiological processes
inductive PhysiologicalProcess
  | Mitosis
  | Amitosis
  | Meiosis
  | Fertilization

-- Define the set of all physiological processes
def allProcesses : Set PhysiologicalProcess :=
  {PhysiologicalProcess.Mitosis, PhysiologicalProcess.Amitosis, 
   PhysiologicalProcess.Meiosis, PhysiologicalProcess.Fertilization}

-- Define the property of maintaining chromosome stability and continuity
def maintainsChromosomeStability (p : PhysiologicalProcess) : Prop :=
  match p with
  | PhysiologicalProcess.Meiosis => true
  | PhysiologicalProcess.Fertilization => true
  | _ => false

-- Theorem: The set of processes that maintain chromosome stability
--          is equal to {Meiosis, Fertilization}
theorem chromosome_stability_processes :
  {p ∈ allProcesses | maintainsChromosomeStability p} = 
  {PhysiologicalProcess.Meiosis, PhysiologicalProcess.Fertilization} :=
by
  sorry


end NUMINAMATH_CALUDE_chromosome_stability_processes_l3774_377499


namespace NUMINAMATH_CALUDE_restaurant_bill_split_l3774_377482

-- Define the meal costs and discounts
def sarah_meal : ℝ := 20
def mary_meal : ℝ := 22
def tuan_meal : ℝ := 18
def michael_meal : ℝ := 24
def linda_meal : ℝ := 16
def sarah_coupon : ℝ := 4
def student_discount : ℝ := 0.1
def sales_tax : ℝ := 0.08
def tip_percentage : ℝ := 0.15
def num_people : ℕ := 5

-- Define the theorem
theorem restaurant_bill_split (
  sarah_meal mary_meal tuan_meal michael_meal linda_meal : ℝ)
  (sarah_coupon student_discount sales_tax tip_percentage : ℝ)
  (num_people : ℕ) :
  let total_before_discount := sarah_meal + mary_meal + tuan_meal + michael_meal + linda_meal
  let sarah_discounted := sarah_meal - sarah_coupon
  let tuan_discounted := tuan_meal * (1 - student_discount)
  let linda_discounted := linda_meal * (1 - student_discount)
  let total_after_discount := sarah_discounted + mary_meal + tuan_discounted + michael_meal + linda_discounted
  let tax_amount := total_after_discount * sales_tax
  let tip_amount := total_before_discount * tip_percentage
  let final_bill := total_after_discount + tax_amount + tip_amount
  let individual_contribution := final_bill / num_people
  individual_contribution = 23 :=
by
  sorry


end NUMINAMATH_CALUDE_restaurant_bill_split_l3774_377482


namespace NUMINAMATH_CALUDE_max_circles_in_square_l3774_377490

/-- The maximum number of non-overlapping circles with radius 2 cm
    that can fit inside a square with side length 8 cm -/
def max_circles : ℕ := 4

/-- The side length of the square in cm -/
def square_side : ℝ := 8

/-- The radius of each circle in cm -/
def circle_radius : ℝ := 2

theorem max_circles_in_square :
  ∀ n : ℕ,
  (n : ℝ) * (2 * circle_radius) ≤ square_side →
  (n : ℝ) * (2 * circle_radius) > square_side - 2 * circle_radius →
  n * n = max_circles :=
by sorry

end NUMINAMATH_CALUDE_max_circles_in_square_l3774_377490


namespace NUMINAMATH_CALUDE_x_minus_y_power_2007_l3774_377410

theorem x_minus_y_power_2007 (x y : ℝ) :
  5 * x^2 - 4 * x * y + y^2 - 2 * x + 1 = 0 →
  (x - y)^2007 = -1 := by sorry

end NUMINAMATH_CALUDE_x_minus_y_power_2007_l3774_377410


namespace NUMINAMATH_CALUDE_distance_traveled_by_slower_person_l3774_377403

/-- The distance traveled by the slower person when two people walk towards each other -/
theorem distance_traveled_by_slower_person
  (total_distance : ℝ)
  (speed_1 : ℝ)
  (speed_2 : ℝ)
  (h1 : total_distance = 50)
  (h2 : speed_1 = 4)
  (h3 : speed_2 = 6)
  (h4 : speed_1 < speed_2) :
  speed_1 * (total_distance / (speed_1 + speed_2)) = 20 :=
by sorry

end NUMINAMATH_CALUDE_distance_traveled_by_slower_person_l3774_377403


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3774_377480

theorem sqrt_equation_solution (x : ℝ) (hx : x > 0) :
  3 * Real.sqrt (4 + x) + 3 * Real.sqrt (4 - x) = 5 * Real.sqrt 6 →
  x = Real.sqrt 43 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3774_377480


namespace NUMINAMATH_CALUDE_min_bailing_rate_calculation_l3774_377451

-- Define the problem parameters
def distance_to_shore : Real := 2 -- miles
def water_intake_rate : Real := 15 -- gallons per minute
def max_water_capacity : Real := 60 -- gallons
def rowing_speed : Real := 3 -- miles per hour

-- Define the theorem
theorem min_bailing_rate_calculation :
  let time_to_shore := distance_to_shore / rowing_speed * 60 -- Convert to minutes
  let total_water_intake := water_intake_rate * time_to_shore
  let water_to_bail := total_water_intake - max_water_capacity
  let min_bailing_rate := water_to_bail / time_to_shore
  min_bailing_rate = 13.5 := by
  sorry


end NUMINAMATH_CALUDE_min_bailing_rate_calculation_l3774_377451


namespace NUMINAMATH_CALUDE_function_composition_ratio_l3774_377472

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := 3 * x - 2

-- State the theorem
theorem function_composition_ratio :
  (f (g (f 3))) / (g (f (g 3))) = 53 / 49 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_ratio_l3774_377472


namespace NUMINAMATH_CALUDE_complex_magnitude_l3774_377455

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = -1 + Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3774_377455


namespace NUMINAMATH_CALUDE_inequality_proof_l3774_377448

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a^3 + b^3 + c^3 = 3) : 
  (1 / (a^4 + 3)) + (1 / (b^4 + 3)) + (1 / (c^4 + 3)) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3774_377448


namespace NUMINAMATH_CALUDE_fraction_divisibility_l3774_377498

theorem fraction_divisibility (a b n : ℕ) (hodd : Odd n) 
  (hnum : n ∣ (a^n + b^n)) (hden : n ∣ (a + b)) : 
  n ∣ ((a^n + b^n) / (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_divisibility_l3774_377498


namespace NUMINAMATH_CALUDE_integral_curves_satisfy_differential_equation_l3774_377443

/-- The differential equation in terms of x, y, dx, and dy -/
def differential_equation (x y : ℝ) (dx dy : ℝ) : Prop :=
  x * dx + y * dy + (x * dy - y * dx) / (x^2 + y^2) = 0

/-- The integral curve equation -/
def integral_curve (x y : ℝ) (C : ℝ) : Prop :=
  (x^2 + y^2) / 2 - y * Real.arctan (x / y) = C

/-- Theorem stating that the integral_curve satisfies the differential_equation -/
theorem integral_curves_satisfy_differential_equation :
  ∀ (x y : ℝ) (C : ℝ),
    x^2 + y^2 > 0 →
    integral_curve x y C →
    ∃ (dx dy : ℝ), differential_equation x y dx dy :=
sorry

end NUMINAMATH_CALUDE_integral_curves_satisfy_differential_equation_l3774_377443


namespace NUMINAMATH_CALUDE_stadium_length_in_feet_l3774_377445

/-- Proves that the length of a 61-yard stadium is 183 feet. -/
theorem stadium_length_in_feet :
  let stadium_length_yards : ℕ := 61
  let yards_to_feet_conversion : ℕ := 3
  stadium_length_yards * yards_to_feet_conversion = 183 :=
by sorry

end NUMINAMATH_CALUDE_stadium_length_in_feet_l3774_377445


namespace NUMINAMATH_CALUDE_smallest_x_value_l3774_377425

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (3 : ℚ) / 4 = y / (178 + x)) : 
  ∃ (x_min : ℕ+), x_min ≤ x ∧ ∃ (y_min : ℕ+), (3 : ℚ) / 4 = y_min / (178 + x_min) ∧ x_min = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3774_377425


namespace NUMINAMATH_CALUDE_cost_per_patch_l3774_377417

/-- Proves that the cost per patch is $1.25 given the order quantity, selling price, and net profit. -/
theorem cost_per_patch (order_quantity : ℕ) (selling_price : ℚ) (net_profit : ℚ) :
  order_quantity = 100 →
  selling_price = 12 →
  net_profit = 1075 →
  (order_quantity : ℚ) * selling_price - (order_quantity : ℚ) * (selling_price - net_profit / (order_quantity : ℚ)) = net_profit :=
by sorry

end NUMINAMATH_CALUDE_cost_per_patch_l3774_377417


namespace NUMINAMATH_CALUDE_chocolate_milk_probability_l3774_377418

theorem chocolate_milk_probability : 
  let n : ℕ := 7  -- number of days
  let k : ℕ := 4  -- number of successes (chocolate milk days)
  let p : ℚ := 2/3  -- probability of success on each day
  Nat.choose n k * p^k * (1-p)^(n-k) = 560/2187 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_milk_probability_l3774_377418


namespace NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l3774_377415

theorem triangle_perimeter_impossibility (a b x : ℝ) (h1 : a = 13) (h2 : b = 24) :
  (a + b + x = 78) → ¬(a + b > x ∧ a + x > b ∧ b + x > a) :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l3774_377415


namespace NUMINAMATH_CALUDE_wednesday_most_frequent_l3774_377492

/-- Represents days of the week -/
inductive DayOfWeek
  | sunday
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday

/-- Represents a date in the year 2014 -/
structure Date2014 where
  month : Nat
  day : Nat

def march_9_2014 : Date2014 := ⟨3, 9⟩

/-- The number of days in 2014 -/
def days_in_2014 : Nat := 365

/-- Function to determine the day of the week for a given date in 2014 -/
def dayOfWeek (d : Date2014) : DayOfWeek := sorry

/-- Function to count occurrences of each day of the week in 2014 -/
def countDayOccurrences (day : DayOfWeek) : Nat := sorry

/-- Theorem stating that Wednesday occurs most frequently in 2014 -/
theorem wednesday_most_frequent :
  (dayOfWeek march_9_2014 = DayOfWeek.sunday) →
  (∀ d : DayOfWeek, countDayOccurrences DayOfWeek.wednesday ≥ countDayOccurrences d) :=
by sorry

end NUMINAMATH_CALUDE_wednesday_most_frequent_l3774_377492


namespace NUMINAMATH_CALUDE_probability_at_least_one_six_l3774_377402

theorem probability_at_least_one_six (n : ℕ) (p : ℚ) : 
  n = 3 → p = 1/6 → (1 - (1 - p)^n) = 91/216 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_six_l3774_377402


namespace NUMINAMATH_CALUDE_factorization_of_4x_squared_minus_16_l3774_377449

theorem factorization_of_4x_squared_minus_16 (x : ℝ) : 4 * x^2 - 16 = 4 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4x_squared_minus_16_l3774_377449


namespace NUMINAMATH_CALUDE_count_not_divisible_by_5_and_7_l3774_377493

def count_not_divisible (n : ℕ) (a b : ℕ) : ℕ :=
  n - (n / a + n / b - n / (a * b))

theorem count_not_divisible_by_5_and_7 :
  count_not_divisible 499 5 7 = 343 := by sorry

end NUMINAMATH_CALUDE_count_not_divisible_by_5_and_7_l3774_377493


namespace NUMINAMATH_CALUDE_trapezoid_semicircle_area_l3774_377494

/-- Represents a trapezoid with semicircles on each side -/
structure TrapezoidWithSemicircles where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- Calculates the area of the region bounded by the semicircles -/
noncomputable def boundedArea (t : TrapezoidWithSemicircles) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem trapezoid_semicircle_area 
  (t : TrapezoidWithSemicircles) 
  (h1 : t.side1 = 10) 
  (h2 : t.side2 = 10) 
  (h3 : t.side3 = 10) 
  (h4 : t.side4 = 22) : 
  boundedArea t = 128 + 60.5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_semicircle_area_l3774_377494


namespace NUMINAMATH_CALUDE_ball_count_l3774_377488

theorem ball_count (white blue red : ℕ) : 
  blue = white + 12 →
  red = 2 * blue →
  white = 16 →
  white + blue + red = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_count_l3774_377488


namespace NUMINAMATH_CALUDE_increase_then_decrease_l3774_377428

theorem increase_then_decrease (x p q : ℝ) (hx : x = 80) (hp : p = 150) (hq : q = 30) :
  x * (1 + p / 100) * (1 - q / 100) = 140 := by
  sorry

end NUMINAMATH_CALUDE_increase_then_decrease_l3774_377428


namespace NUMINAMATH_CALUDE_binary_1001_equals_9_l3774_377483

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1001₂ -/
def binary_1001 : List Bool := [true, false, false, true]

theorem binary_1001_equals_9 : binary_to_decimal binary_1001 = 9 := by
  sorry

end NUMINAMATH_CALUDE_binary_1001_equals_9_l3774_377483


namespace NUMINAMATH_CALUDE_no_matching_units_digits_l3774_377478

theorem no_matching_units_digits :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 100 → (x % 10 ≠ (101 - x) % 10) :=
by sorry

end NUMINAMATH_CALUDE_no_matching_units_digits_l3774_377478


namespace NUMINAMATH_CALUDE_sum_fifth_powers_divisible_by_30_l3774_377463

theorem sum_fifth_powers_divisible_by_30 
  (n : ℕ) 
  (a : Fin n → ℕ) 
  (h : 30 ∣ (Finset.univ.sum (λ i => a i))) : 
  30 ∣ (Finset.univ.sum (λ i => (a i)^5)) :=
sorry

end NUMINAMATH_CALUDE_sum_fifth_powers_divisible_by_30_l3774_377463


namespace NUMINAMATH_CALUDE_smallest_sum_proof_l3774_377441

theorem smallest_sum_proof : 
  let sums : List ℚ := [1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/6, 1/3 + 1/7, 1/3 + 1/8]
  (∀ x ∈ sums, 1/3 + 1/8 ≤ x) ∧ (1/3 + 1/8 = 11/24) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_proof_l3774_377441


namespace NUMINAMATH_CALUDE_unique_number_with_two_perfect_square_increments_l3774_377484

theorem unique_number_with_two_perfect_square_increments : 
  ∃! n : ℕ, n > 1000 ∧ 
    ∃ a b : ℕ, (n + 79 = a^2) ∧ (n + 204 = b^2) ∧ 
    n = 3765 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_two_perfect_square_increments_l3774_377484


namespace NUMINAMATH_CALUDE_different_author_book_pairs_l3774_377422

/-- Given two groups of books, this theorem proves that the number of different pairs
    that can be formed by selecting one book from each group is equal to the product
    of the number of books in each group. -/
theorem different_author_book_pairs (group1 group2 : ℕ) (h1 : group1 = 6) (h2 : group2 = 9) :
  group1 * group2 = 54 := by
  sorry

end NUMINAMATH_CALUDE_different_author_book_pairs_l3774_377422


namespace NUMINAMATH_CALUDE_det_A_eq_48_l3774_377489

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 1, -2; 8, 5, -4; 3, 3, 6]

theorem det_A_eq_48 : Matrix.det A = 48 := by sorry

end NUMINAMATH_CALUDE_det_A_eq_48_l3774_377489


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3774_377430

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  (2 * X^4 + 10 * X^3 - 45 * X^2 - 52 * X + 63) = 
  (X^2 + 6 * X - 7) * q + (48 * X - 70) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3774_377430


namespace NUMINAMATH_CALUDE_hexagonal_tile_difference_l3774_377460

theorem hexagonal_tile_difference (initial_blue : ℕ) (initial_green : ℕ) (border_tiles : ℕ) : 
  initial_blue = 20 → initial_green = 15 → border_tiles = 18 →
  (initial_green + 2 * border_tiles) - initial_blue = 31 := by
sorry

end NUMINAMATH_CALUDE_hexagonal_tile_difference_l3774_377460


namespace NUMINAMATH_CALUDE_percent_relation_l3774_377414

theorem percent_relation (x y z : ℝ) 
  (h1 : 0.45 * z = 0.72 * y) 
  (h2 : y = 0.75 * x) : 
  z = 1.2 * x := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l3774_377414


namespace NUMINAMATH_CALUDE_acid_dilution_l3774_377413

theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (final_concentration : ℝ) (water_added : ℝ) : 
  initial_volume = 50 →
  initial_concentration = 0.4 →
  final_concentration = 0.25 →
  water_added = 30 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by
  sorry

#check acid_dilution

end NUMINAMATH_CALUDE_acid_dilution_l3774_377413


namespace NUMINAMATH_CALUDE_diminished_value_proof_diminished_value_l3774_377450

theorem diminished_value_proof (n : Nat) (divisors : List Nat) : Prop :=
  let smallest := 1013
  let value := 5
  let lcm := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 12 16) 18) 21) 28
  (∀ d ∈ divisors, (smallest - value) % d = 0) ∧
  (smallest = lcm + value) ∧
  (∀ m < smallest, ∃ d ∈ divisors, (m - value) % d ≠ 0)

/-- The value that needs to be diminished from 1013 to make it divisible by 12, 16, 18, 21, and 28 is 5. -/
theorem diminished_value :
  diminished_value_proof 1013 [12, 16, 18, 21, 28] :=
by sorry

end NUMINAMATH_CALUDE_diminished_value_proof_diminished_value_l3774_377450


namespace NUMINAMATH_CALUDE_chord_intersection_angle_l3774_377497

theorem chord_intersection_angle (θ : Real) : 
  θ ∈ Set.Icc 0 (Real.pi / 2) →
  (∃ (x y : Real), 
    x * Real.sin θ + y * Real.cos θ - 1 = 0 ∧
    (x - 1)^2 + (y - Real.cos θ)^2 = 1/4 ∧
    ∃ (x' y' : Real), 
      x' * Real.sin θ + y' * Real.cos θ - 1 = 0 ∧
      (x' - 1)^2 + (y' - Real.cos θ)^2 = 1/4 ∧
      (x - x')^2 + (y - y')^2 = 3/4) →
  θ = Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_chord_intersection_angle_l3774_377497


namespace NUMINAMATH_CALUDE_star_polygon_angles_l3774_377481

/-- Given a star polygon where the sum of five angles is 500°, 
    prove that the sum of the other five angles is 140°. -/
theorem star_polygon_angles (p q r s t A B C D E : ℝ) 
  (h1 : p + q + r + s + t = 500) 
  (h2 : A + B + C + D + E = x) : x = 140 := by
  sorry

end NUMINAMATH_CALUDE_star_polygon_angles_l3774_377481


namespace NUMINAMATH_CALUDE_max_prime_angle_in_isosceles_triangle_l3774_377405

def IsIsosceles (a b c : ℕ) : Prop := a + b + c = 180 ∧ a = b

def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem max_prime_angle_in_isosceles_triangle :
  ∀ x : ℕ,
    IsIsosceles x x (180 - 2*x) →
    IsPrime x →
    x ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_prime_angle_in_isosceles_triangle_l3774_377405


namespace NUMINAMATH_CALUDE_hockey_league_games_l3774_377404

/-- The number of games played in a hockey league season --/
def hockey_games (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) / 2 * m

/-- Theorem: In a hockey league with 25 teams, where each team faces all other teams 15 times, 
    the total number of games played in the season is 4500. --/
theorem hockey_league_games : hockey_games 25 15 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l3774_377404


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3774_377477

theorem triangle_angle_calculation (A B C : ℝ) : 
  A = 60 → B = 2 * C → A + B + C = 180 → B = 80 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l3774_377477


namespace NUMINAMATH_CALUDE_average_first_20_even_numbers_l3774_377433

theorem average_first_20_even_numbers : 
  let first_20_even : List ℕ := List.range 20 |>.map (fun i => 2 * (i + 1))
  (first_20_even.sum / first_20_even.length : ℚ) = 21 := by
  sorry

end NUMINAMATH_CALUDE_average_first_20_even_numbers_l3774_377433


namespace NUMINAMATH_CALUDE_extra_fruits_calculation_l3774_377431

theorem extra_fruits_calculation (red_ordered green_ordered oranges_ordered : ℕ)
                                 (red_chosen green_chosen oranges_chosen : ℕ)
                                 (h1 : red_ordered = 43)
                                 (h2 : green_ordered = 32)
                                 (h3 : oranges_ordered = 25)
                                 (h4 : red_chosen = 7)
                                 (h5 : green_chosen = 5)
                                 (h6 : oranges_chosen = 4) :
  (red_ordered - red_chosen) + (green_ordered - green_chosen) + (oranges_ordered - oranges_chosen) = 84 :=
by sorry

end NUMINAMATH_CALUDE_extra_fruits_calculation_l3774_377431


namespace NUMINAMATH_CALUDE_items_after_price_drop_l3774_377475

/-- Calculates the number of items that can be purchased after a price drop -/
theorem items_after_price_drop (original_price : ℚ) (original_quantity : ℕ) (new_price : ℚ) :
  original_price > 0 →
  new_price > 0 →
  new_price < original_price →
  (original_price * original_quantity) / new_price = 20 :=
by
  sorry

#check items_after_price_drop (4 : ℚ) 15 (3 : ℚ)

end NUMINAMATH_CALUDE_items_after_price_drop_l3774_377475


namespace NUMINAMATH_CALUDE_sum_of_squared_complements_geq_two_l3774_377419

theorem sum_of_squared_complements_geq_two 
  (a b c : ℝ) 
  (h1 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
  (h2 : a + b + c = 1) : 
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_complements_geq_two_l3774_377419


namespace NUMINAMATH_CALUDE_symmetry_x_axis_correct_l3774_377457

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the x-axis -/
def symmetricXAxis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

theorem symmetry_x_axis_correct :
  let M : Point3D := { x := -1, y := 2, z := 1 }
  let M' : Point3D := { x := -1, y := -2, z := -1 }
  symmetricXAxis M = M' := by sorry

end NUMINAMATH_CALUDE_symmetry_x_axis_correct_l3774_377457


namespace NUMINAMATH_CALUDE_triangle_with_long_altitudes_l3774_377454

theorem triangle_with_long_altitudes (a b c : ℝ) (ma mb : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_altitudes : ma ≥ a ∧ mb ≥ b)
  (h_area : a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) / 2 = a * ma / 2)
  (h_area_alt : a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) / 2 = b * mb / 2) :
  a = b ∧ c^2 = 2 * a^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_long_altitudes_l3774_377454


namespace NUMINAMATH_CALUDE_fixed_point_on_graph_l3774_377408

theorem fixed_point_on_graph :
  ∀ (k : ℝ), 112 = 7 * (4 : ℝ)^2 + k * 4 - 4 * k := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_graph_l3774_377408


namespace NUMINAMATH_CALUDE_service_charge_percentage_l3774_377471

theorem service_charge_percentage (salmon_cost black_burger_cost chicken_katsu_cost : ℝ)
  (tip_percentage : ℝ) (total_paid change : ℝ) :
  salmon_cost = 40 →
  black_burger_cost = 15 →
  chicken_katsu_cost = 25 →
  tip_percentage = 0.05 →
  total_paid = 100 →
  change = 8 →
  let food_cost := salmon_cost + black_burger_cost + chicken_katsu_cost
  let tip := food_cost * tip_percentage
  let service_charge := total_paid - change - food_cost - tip
  service_charge / food_cost = 0.1 := by
sorry

end NUMINAMATH_CALUDE_service_charge_percentage_l3774_377471


namespace NUMINAMATH_CALUDE_weight_loss_challenge_l3774_377468

/-- 
Given an initial weight W, a weight loss percentage, and a clothing weight percentage,
calculates the final measured weight loss percentage.
-/
def measured_weight_loss_percentage (initial_weight_loss : Real) (clothing_weight_percent : Real) : Real :=
  let remaining_weight_percent := 1 - initial_weight_loss
  let final_weight_percent := remaining_weight_percent * (1 + clothing_weight_percent)
  (1 - final_weight_percent) * 100

/-- 
Proves that given an initial weight loss of 15% and clothes that add 2% to the final weight,
the measured weight loss percentage at the final weigh-in is 13.3%.
-/
theorem weight_loss_challenge (ε : Real) :
  ∃ δ > 0, ∀ x, |x - 0.133| < δ → |measured_weight_loss_percentage 0.15 0.02 - x| < ε :=
sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l3774_377468


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l3774_377400

theorem correct_mean_calculation (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 20 ∧ original_mean = 150 ∧ incorrect_value = 135 ∧ correct_value = 160 →
  (n : ℚ) * original_mean - incorrect_value + correct_value = n * (151.25 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l3774_377400


namespace NUMINAMATH_CALUDE_factor_3x_squared_minus_75_l3774_377411

theorem factor_3x_squared_minus_75 (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_3x_squared_minus_75_l3774_377411


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l3774_377464

theorem parallelogram_side_length (s : ℝ) : 
  s > 0 → -- side length is positive
  let angle : ℝ := 30 * π / 180 -- 30 degrees in radians
  let area : ℝ := 12 * Real.sqrt 3 -- area of the parallelogram
  s * (s * Real.sin angle) = area → -- area formula for parallelogram
  s = 2 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l3774_377464


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l3774_377470

def f (x : ℝ) := x^3 - 12*x

theorem max_min_f_on_interval :
  let a := -3
  let b := 5
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc a b, f x ≤ max) ∧
    (∃ x ∈ Set.Icc a b, f x = max) ∧
    (∀ x ∈ Set.Icc a b, min ≤ f x) ∧
    (∃ x ∈ Set.Icc a b, f x = min) ∧
    max = 65 ∧ min = -16 := by
  sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l3774_377470


namespace NUMINAMATH_CALUDE_jason_retirement_age_l3774_377479

/-- Jason's career in the military -/
def military_career (joining_age : ℕ) (years_to_chief : ℕ) (additional_years : ℕ) : Prop :=
  let years_to_master_chief : ℕ := years_to_chief + (years_to_chief / 4)
  let total_years : ℕ := years_to_chief + years_to_master_chief + additional_years
  let retirement_age : ℕ := joining_age + total_years
  retirement_age = 46

theorem jason_retirement_age :
  military_career 18 8 10 :=
by
  sorry

end NUMINAMATH_CALUDE_jason_retirement_age_l3774_377479


namespace NUMINAMATH_CALUDE_range_of_a_l3774_377424

/-- The range of a given the conditions in the problem -/
theorem range_of_a (a : ℝ) : 
  (a < 0) → 
  (∀ x : ℝ, (x^2 - 4*a*x + 3*a^2 < 0) → (x^2 + 2*x - 8 > 0)) →
  (∃ x : ℝ, (x^2 - 4*a*x + 3*a^2 < 0) ∧ (x^2 + 2*x - 8 ≤ 0)) →
  a ≤ -4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3774_377424


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3774_377462

theorem quadratic_factorization (x : ℝ) : 4 - 4*x + x^2 = (2 - x)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3774_377462


namespace NUMINAMATH_CALUDE_at_least_two_primes_of_form_l3774_377446

theorem at_least_two_primes_of_form (n : ℕ) : ∃ (a b : ℕ), 2 ≤ a ∧ 2 ≤ b ∧ a ≠ b ∧ 
  Nat.Prime (a^3 + a + 1) ∧ Nat.Prime (b^3 + b + 1) :=
sorry

end NUMINAMATH_CALUDE_at_least_two_primes_of_form_l3774_377446


namespace NUMINAMATH_CALUDE_digit_sum_problem_l3774_377465

theorem digit_sum_problem (x y z w : ℕ) : 
  (x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w) →
  (x < 10 ∧ y < 10 ∧ z < 10 ∧ w < 10) →
  (100 * x + 10 * y + z) + (100 * w + 10 * z + x) = 1000 →
  z + x ≥ 10 →
  y + z < 10 →
  x + y + z + w = 19 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l3774_377465


namespace NUMINAMATH_CALUDE_swimming_speed_in_still_water_l3774_377406

/-- Proves that a person's swimming speed in still water is 4 km/h given the conditions -/
theorem swimming_speed_in_still_water 
  (water_speed : ℝ) 
  (swimming_time : ℝ) 
  (swimming_distance : ℝ) 
  (h1 : water_speed = 2)
  (h2 : swimming_time = 5)
  (h3 : swimming_distance = 10)
  : ∃ (still_water_speed : ℝ), 
    swimming_distance = (still_water_speed - water_speed) * swimming_time ∧ 
    still_water_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_swimming_speed_in_still_water_l3774_377406


namespace NUMINAMATH_CALUDE_probability_odd_divisor_15_factorial_l3774_377459

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_odd (n : ℕ) : Prop := n % 2 = 1

def count_divisors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (fun acc (p, e) => acc * (e + 1)) 1

def count_odd_divisors (factors : List (ℕ × ℕ)) : ℕ :=
  (factors.filter (fun (p, _) => p ≠ 2)).foldl (fun acc (_, e) => acc * (e + 1)) 1

theorem probability_odd_divisor_15_factorial :
  let f15 := factorial 15
  let factors := prime_factorization f15
  let total_divisors := count_divisors factors
  let odd_divisors := count_odd_divisors factors
  (odd_divisors : ℚ) / total_divisors = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_probability_odd_divisor_15_factorial_l3774_377459


namespace NUMINAMATH_CALUDE_full_bucket_weight_formula_l3774_377437

/-- Represents the weight of a bucket with water -/
structure BucketWeight where
  twoThirdsFull : ℝ  -- Weight when 2/3 full
  halfFull : ℝ       -- Weight when 1/2 full

/-- Calculates the weight of a bucket when it's full of water -/
def fullBucketWeight (bw : BucketWeight) : ℝ :=
  3 * bw.twoThirdsFull - 2 * bw.halfFull

/-- Theorem stating that the weight of a full bucket is 3a - 2b given the weights at 2/3 and 1/2 full -/
theorem full_bucket_weight_formula (bw : BucketWeight) :
  fullBucketWeight bw = 3 * bw.twoThirdsFull - 2 * bw.halfFull := by
  sorry

end NUMINAMATH_CALUDE_full_bucket_weight_formula_l3774_377437


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_of_60_l3774_377476

def is_factor (n m : ℕ) : Prop := m % n = 0

theorem smallest_non_factor_product_of_60 (a b : ℕ) :
  a ≠ b →
  a > 0 →
  b > 0 →
  is_factor a 60 →
  is_factor b 60 →
  ¬ is_factor (a * b) 60 →
  ∀ c d : ℕ, c ≠ d → c > 0 → d > 0 → is_factor c 60 → is_factor d 60 → ¬ is_factor (c * d) 60 → a * b ≤ c * d :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_of_60_l3774_377476


namespace NUMINAMATH_CALUDE_expected_value_special_coin_l3774_377491

/-- The expected value of winnings for a special coin flip -/
theorem expected_value_special_coin : 
  let p_heads : ℚ := 2 / 5
  let p_tails : ℚ := 3 / 5
  let win_heads : ℚ := 4
  let lose_tails : ℚ := 3
  p_heads * win_heads - p_tails * lose_tails = -1 / 5 := by
sorry

end NUMINAMATH_CALUDE_expected_value_special_coin_l3774_377491


namespace NUMINAMATH_CALUDE_debt_average_payment_l3774_377466

/-- Prove that the average payment for a debt with specific payment structure is $442.50 -/
theorem debt_average_payment (n : ℕ) (first_payment second_payment : ℚ) : 
  n = 40 →
  first_payment = 410 →
  second_payment = first_payment + 65 →
  (n / 2 * first_payment + n / 2 * second_payment) / n = 442.5 := by
  sorry

end NUMINAMATH_CALUDE_debt_average_payment_l3774_377466


namespace NUMINAMATH_CALUDE_shoe_comparison_l3774_377438

theorem shoe_comparison (bobby_shoes : ℕ) (bonny_shoes : ℕ) : 
  bobby_shoes = 27 →
  bonny_shoes = 13 →
  ∃ (becky_shoes : ℕ), 
    bobby_shoes = 3 * becky_shoes ∧
    2 * becky_shoes - bonny_shoes = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_shoe_comparison_l3774_377438


namespace NUMINAMATH_CALUDE_lemonade_stand_profit_l3774_377439

/-- Calculate the profit from a lemonade stand -/
theorem lemonade_stand_profit 
  (price_per_cup : ℕ) 
  (cups_sold : ℕ) 
  (lemon_cost sugar_cost cup_cost : ℕ) : 
  price_per_cup * cups_sold - (lemon_cost + sugar_cost + cup_cost) = 66 :=
by
  sorry

#check lemonade_stand_profit 4 21 10 5 3

end NUMINAMATH_CALUDE_lemonade_stand_profit_l3774_377439


namespace NUMINAMATH_CALUDE_intersection_P_complement_Q_l3774_377447

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 < 1}
def Q : Set ℝ := {x | x ≥ 0}

-- State the theorem
theorem intersection_P_complement_Q : 
  P ∩ (Set.univ \ Q) = {x : ℝ | -1 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_intersection_P_complement_Q_l3774_377447


namespace NUMINAMATH_CALUDE_fred_remaining_cards_l3774_377436

def initial_cards : ℕ := 40
def purchase_percentage : ℚ := 375 / 1000

theorem fred_remaining_cards :
  initial_cards - (purchase_percentage * initial_cards).floor = 25 := by
  sorry

end NUMINAMATH_CALUDE_fred_remaining_cards_l3774_377436


namespace NUMINAMATH_CALUDE_profit_percentage_is_10_percent_l3774_377421

def cost_price : ℚ := 340
def selling_price : ℚ := 374

theorem profit_percentage_is_10_percent :
  (selling_price - cost_price) / cost_price * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_10_percent_l3774_377421


namespace NUMINAMATH_CALUDE_problem_statement_l3774_377452

theorem problem_statement (a b c : ℝ) : 
  a^2 + b^2 + c^2 + 4 ≤ a*b + 3*b + 2*c → 200*a + 9*b + c = 219 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3774_377452


namespace NUMINAMATH_CALUDE_parabola_directrix_through_ellipse_focus_l3774_377409

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/9 + y^2/5 = 1

-- Define the focus of the ellipse
def ellipse_focus : ℝ × ℝ := (2, 0)

-- Define the directrix of the parabola
def parabola_directrix (p : ℝ) : ℝ → Prop := λ x ↦ x = -p/2

-- Theorem statement
theorem parabola_directrix_through_ellipse_focus :
  ∀ p : ℝ, (∃ x y : ℝ, parabola p x y ∧ ellipse x y ∧ 
    parabola_directrix p (ellipse_focus.1)) →
  parabola_directrix p = λ x ↦ x = -2 := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_through_ellipse_focus_l3774_377409


namespace NUMINAMATH_CALUDE_bowling_ball_weight_is_16_l3774_377434

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 16

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := 24

/-- Theorem stating the weight of one bowling ball is 16 pounds -/
theorem bowling_ball_weight_is_16 : 
  (9 * bowling_ball_weight = 6 * canoe_weight) → 
  (5 * canoe_weight = 120) → 
  bowling_ball_weight = 16 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_is_16_l3774_377434


namespace NUMINAMATH_CALUDE_peters_exam_score_l3774_377485

theorem peters_exam_score :
  ∀ (e m h : ℕ),
  e + m + h = 25 →
  2 * e + 3 * m + 5 * h = 84 →
  m % 2 = 0 →
  h % 3 = 0 →
  2 * e + 3 * (m / 2) + 5 * (h / 3) = 40 :=
by sorry

end NUMINAMATH_CALUDE_peters_exam_score_l3774_377485


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3774_377423

/-- An arithmetic sequence with given parameters has 13 terms -/
theorem arithmetic_sequence_terms (a d l : ℤ) (h1 : a = -5) (h2 : d = 5) (h3 : l = 55) :
  ∃ n : ℕ, n = 13 ∧ l = a + (n - 1) * d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3774_377423


namespace NUMINAMATH_CALUDE_cube_surface_area_l3774_377467

/-- The surface area of a cube with edge length 4a is 96a² -/
theorem cube_surface_area (a : ℝ) : 
  let edge_length : ℝ := 4 * a
  let surface_area : ℝ := 6 * (edge_length ^ 2)
  surface_area = 96 * (a ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3774_377467


namespace NUMINAMATH_CALUDE_petya_strategy_works_l3774_377456

/-- Represents a non-zero digit (1-9) -/
def NonZeroDigit := {n : Nat // 1 ≤ n ∧ n ≤ 9}

/-- Represents a 3-digit number -/
def ThreeDigitNumber := {n : Nat // 100 ≤ n ∧ n ≤ 999}

/-- The main theorem stating that any 12 non-zero digits can be arranged into four 3-digit numbers whose product is divisible by 9 -/
theorem petya_strategy_works (digits : Fin 12 → NonZeroDigit) : 
  ∃ (a b c d : ThreeDigitNumber), (a.val * b.val * c.val * d.val) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_petya_strategy_works_l3774_377456


namespace NUMINAMATH_CALUDE_expression_value_l3774_377429

theorem expression_value (a b : ℝ) 
  (h1 : |a| ≠ |b|) 
  (h2 : (a + b) / (a - b) + (a - b) / (a + b) = 6) : 
  (a^3 + b^3) / (a^3 - b^3) + (a^3 - b^3) / (a^3 + b^3) = 18/7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3774_377429
