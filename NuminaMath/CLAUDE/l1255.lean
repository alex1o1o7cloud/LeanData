import Mathlib

namespace NUMINAMATH_CALUDE_solution_approximation_l1255_125586

-- Define the equation
def equation (x y : ℝ) : Prop :=
  ((3 * x^2 - 7)^2 / 9) + 5 * y = x^3 - 2 * x

-- State the theorem
theorem solution_approximation :
  ∃ y : ℝ, equation 4 y ∧ abs (y + 26.155) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_solution_approximation_l1255_125586


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l1255_125524

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q → q = 2 → a 3 = 3 → a 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l1255_125524


namespace NUMINAMATH_CALUDE_problem_solution_l1255_125574

theorem problem_solution (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 * y = 2)
  (h2 : y^2 * z = 4)
  (h3 : z^2 / x = 5) :
  x = 5^(1/7) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1255_125574


namespace NUMINAMATH_CALUDE_sum_and_square_difference_implies_difference_l1255_125599

theorem sum_and_square_difference_implies_difference
  (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 190) :
  x - y = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_square_difference_implies_difference_l1255_125599


namespace NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_planes_perpendicular_to_same_line_are_parallel_l1255_125505

-- Define the concept of a plane
variable (Plane : Type)

-- Define the concept of a line
variable (Line : Type)

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the property of being non-coincident
variable (non_coincident : Plane → Plane → Prop)

-- Theorem 1: Two non-coincident planes parallel to the same plane are parallel
theorem planes_parallel_to_same_plane_are_parallel 
  (α β γ : Plane) 
  (h1 : parallel α γ) 
  (h2 : parallel β γ) 
  (h3 : non_coincident α β) : 
  parallel α β :=
sorry

-- Theorem 2: Two non-coincident planes perpendicular to the same line are parallel
theorem planes_perpendicular_to_same_line_are_parallel 
  (α β : Plane) 
  (a : Line) 
  (h1 : perpendicular a α) 
  (h2 : perpendicular a β) 
  (h3 : non_coincident α β) : 
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_to_same_plane_are_parallel_planes_perpendicular_to_same_line_are_parallel_l1255_125505


namespace NUMINAMATH_CALUDE_range_of_a_for_solution_a_value_for_minimum_l1255_125564

-- Define the function f
def f (a x : ℝ) : ℝ := |2*x - a| + |x - 1|

-- Theorem for part I
theorem range_of_a_for_solution (a : ℝ) :
  (∃ x, f a x ≤ 2 - |x - 1|) ↔ (0 ≤ a ∧ a ≤ 4) :=
sorry

-- Theorem for part II
theorem a_value_for_minimum (a : ℝ) :
  a < 2 → (∀ x, f a x ≥ 3) → (∃ x, f a x = 3) → a = -4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_solution_a_value_for_minimum_l1255_125564


namespace NUMINAMATH_CALUDE_cone_cut_ratio_sum_l1255_125550

/-- Represents a right circular cone --/
structure Cone where
  height : ℝ
  baseRadius : ℝ

/-- Represents the result of cutting a cone parallel to its base --/
structure CutCone where
  originalCone : Cone
  cutRadius : ℝ

def surfaceAreaRatio (cc : CutCone) : ℝ := sorry

def volumeRatio (cc : CutCone) : ℝ := sorry

def isCoprime (m n : ℕ) : Prop := sorry

theorem cone_cut_ratio_sum (m n : ℕ) :
  let originalCone : Cone := { height := 6, baseRadius := 5 }
  let cc : CutCone := { originalCone := originalCone, cutRadius := 25/8 }
  surfaceAreaRatio cc = m / n →
  volumeRatio cc = m / n →
  isCoprime m n →
  m + n = 20 := by sorry

end NUMINAMATH_CALUDE_cone_cut_ratio_sum_l1255_125550


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1255_125516

theorem inequality_system_solution_set
  (x : ℝ) :
  (2 * x ≤ -2 ∧ x + 3 < 4) ↔ x ≤ -1 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1255_125516


namespace NUMINAMATH_CALUDE_function_properties_l1255_125503

noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + (a^2 - 1) * x + b

theorem function_properties (a b : ℝ) :
  (∃ y, f a b 1 = y ∧ x + y - 3 = 0) →
  (∀ x ∈ Set.Icc (-2 : ℝ) 4, f a b x ≤ 8) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 4, f a b x ≥ -4) ∧
  (∃ x ∈ Set.Ioo (-1 : ℝ) 1, ∃ y ∈ Set.Ioo (-1 : ℝ) 1, x < y ∧ f a b x > f a b y) →
  a ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioo 0 2 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1255_125503


namespace NUMINAMATH_CALUDE_inverse_proportion_wrench_force_l1255_125583

/-- Proof that for inversely proportional quantities, if F₁ * L₁ = k and F₂ * L₂ = k,
    where F₁ = 300, L₁ = 12, and L₂ = 18, then F₂ = 200. -/
theorem inverse_proportion_wrench_force (k : ℝ) (F₁ F₂ L₁ L₂ : ℝ) 
    (h1 : F₁ * L₁ = k)
    (h2 : F₂ * L₂ = k)
    (h3 : F₁ = 300)
    (h4 : L₁ = 12)
    (h5 : L₂ = 18) :
    F₂ = 200 := by
  sorry

#check inverse_proportion_wrench_force

end NUMINAMATH_CALUDE_inverse_proportion_wrench_force_l1255_125583


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1255_125591

/-- Given a circle and a line with specific properties, prove the center and radius of the circle -/
theorem circle_center_and_radius 
  (x y : ℝ) 
  (m : ℝ) 
  (circle_eq : x^2 + y^2 + x - 6*y + m = 0) 
  (line_eq : x + 2*y - 3 = 0) 
  (P Q : ℝ × ℝ) 
  (intersect : (P.1^2 + P.2^2 + P.1 - 6*P.2 + m = 0 ∧ P.1 + 2*P.2 - 3 = 0) ∧ 
               (Q.1^2 + Q.2^2 + Q.1 - 6*Q.2 + m = 0 ∧ Q.1 + 2*Q.2 - 3 = 0)) 
  (perpendicular : P.1 * Q.1 + P.2 * Q.2 = 0) : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    center = (-1/2, 3) ∧ radius = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1255_125591


namespace NUMINAMATH_CALUDE_regular_star_polygon_points_l1255_125578

/-- A regular star polygon with n points, where each point has two associated angles. -/
structure RegularStarPolygon where
  n : ℕ
  A : Fin n → ℝ
  B : Fin n → ℝ
  all_A_congruent : ∀ i j, A i = A j
  all_B_congruent : ∀ i j, B i = B j
  A_less_than_B : ∀ i, A i = B i - 20

/-- The number of points in a regular star polygon satisfying the given conditions is 18. -/
theorem regular_star_polygon_points (p : RegularStarPolygon) : p.n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_star_polygon_points_l1255_125578


namespace NUMINAMATH_CALUDE_function_properties_l1255_125551

/-- A function is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function is monotonically increasing on an interval [a, b] if
    for all x, y in [a, b], x ≤ y implies f(x) ≤ f(y) -/
def MonoIncOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

/-- Main theorem -/
theorem function_properties (f : ℝ → ℝ) 
    (heven : IsEven f)
    (hmono : MonoIncOn f (-1) 0)
    (hcond : ∀ x, f (1 - x) + f (1 + x) = 0) :
    (f (-3) = 0) ∧
    (MonoIncOn f 1 2) ∧
    (∀ x, f x = f (2 - x)) := by
  sorry


end NUMINAMATH_CALUDE_function_properties_l1255_125551


namespace NUMINAMATH_CALUDE_sugar_content_per_bar_l1255_125588

/-- The sugar content of each chocolate bar -/
def sugar_per_bar (total_sugar total_bars lollipop_sugar : ℕ) : ℚ :=
  (total_sugar - lollipop_sugar) / total_bars

/-- Proof that the sugar content of each chocolate bar is 10 grams -/
theorem sugar_content_per_bar :
  sugar_per_bar 177 14 37 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sugar_content_per_bar_l1255_125588


namespace NUMINAMATH_CALUDE_perpendicular_lines_condition_l1255_125517

-- Define the slopes of the two lines
def slope1 (a : ℝ) := a
def slope2 (a : ℝ) := -4 * a

-- Define perpendicularity condition
def isPerpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines_condition (a : ℝ) :
  (isPerpendicular (slope1 a) (slope2 a) → (a = 1/2 ∨ a = -1/2)) ∧
  ¬(a = 1/2 → isPerpendicular (slope1 a) (slope2 a)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_condition_l1255_125517


namespace NUMINAMATH_CALUDE_students_at_start_l1255_125598

theorem students_at_start (students_left : ℕ) (new_students : ℕ) (final_students : ℕ) : 
  students_left = 4 → new_students = 42 → final_students = 48 → 
  final_students - (new_students - students_left) = 10 :=
by sorry

end NUMINAMATH_CALUDE_students_at_start_l1255_125598


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1255_125540

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the problem statement
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 3 + a 8) / 2 = 10 →
  a 1 + a 10 = 20 := by
    sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1255_125540


namespace NUMINAMATH_CALUDE_better_deal_gives_three_contacts_per_dollar_l1255_125547

/-- Represents a box of contacts with a given number of contacts and price --/
structure ContactBox where
  contacts : ℕ
  price : ℚ

/-- Calculates the number of contacts per dollar for a given box --/
def contactsPerDollar (box : ContactBox) : ℚ :=
  box.contacts / box.price

theorem better_deal_gives_three_contacts_per_dollar
  (box1 box2 : ContactBox)
  (h1 : box1 = ⟨50, 25⟩)
  (h2 : box2 = ⟨99, 33⟩)
  (h3 : contactsPerDollar box2 > contactsPerDollar box1) :
  contactsPerDollar box2 = 3 := by
  sorry

#check better_deal_gives_three_contacts_per_dollar

end NUMINAMATH_CALUDE_better_deal_gives_three_contacts_per_dollar_l1255_125547


namespace NUMINAMATH_CALUDE_pet_store_cages_l1255_125553

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 2

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 7

/-- The total number of birds in the pet store -/
def total_birds : ℕ := 72

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := total_birds / (parrots_per_cage + parakeets_per_cage)

theorem pet_store_cages : num_cages = 8 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l1255_125553


namespace NUMINAMATH_CALUDE_circle_radius_l1255_125559

/-- Given a circle with area P and circumference Q, if P/Q = 25, then the radius is 50 -/
theorem circle_radius (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) (h3 : P / Q = 25) :
  ∃ (r : ℝ), P = π * r^2 ∧ Q = 2 * π * r ∧ r = 50 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_l1255_125559


namespace NUMINAMATH_CALUDE_jack_house_height_correct_l1255_125538

/-- The height of Jack's house -/
def jackHouseHeight : ℝ := 49

/-- The length of the shadow cast by Jack's house -/
def jackHouseShadow : ℝ := 56

/-- The height of the tree -/
def treeHeight : ℝ := 21

/-- The length of the shadow cast by the tree -/
def treeShadow : ℝ := 24

/-- Theorem stating that the calculated height of Jack's house is correct -/
theorem jack_house_height_correct :
  jackHouseHeight = (jackHouseShadow * treeHeight) / treeShadow :=
by sorry

end NUMINAMATH_CALUDE_jack_house_height_correct_l1255_125538


namespace NUMINAMATH_CALUDE_sqrt_2a_plus_b_is_6_l1255_125504

/-- Given that the square root of (a + 9) is -5 and the cube root of (2b - a) is -2,
    prove that the arithmetic square root of (2a + b) is 6 -/
theorem sqrt_2a_plus_b_is_6 (a b : ℝ) 
  (h1 : Real.sqrt (a + 9) = -5)
  (h2 : (2 * b - a) ^ (1/3 : ℝ) = -2) :
  Real.sqrt (2 * a + b) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2a_plus_b_is_6_l1255_125504


namespace NUMINAMATH_CALUDE_largest_lcm_with_15_l1255_125511

theorem largest_lcm_with_15 : 
  let lcm_list := [Nat.lcm 15 3, Nat.lcm 15 5, Nat.lcm 15 6, Nat.lcm 15 9, Nat.lcm 15 10, Nat.lcm 15 15]
  List.maximum lcm_list = some 45 := by
sorry

end NUMINAMATH_CALUDE_largest_lcm_with_15_l1255_125511


namespace NUMINAMATH_CALUDE_notebook_duration_example_l1255_125584

/-- The number of days notebooks last given the number of notebooks, pages per notebook, and pages used per day. -/
def notebook_duration (num_notebooks : ℕ) (pages_per_notebook : ℕ) (pages_per_day : ℕ) : ℕ :=
  (num_notebooks * pages_per_notebook) / pages_per_day

/-- Theorem stating that 5 notebooks with 40 pages each, using 4 pages per day, last for 50 days. -/
theorem notebook_duration_example : notebook_duration 5 40 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_notebook_duration_example_l1255_125584


namespace NUMINAMATH_CALUDE_intersection_A_B_union_B_complement_A_l1255_125556

-- Define the universal set U
def U : Set ℝ := {x | -5 ≤ x ∧ x ≤ 5}

-- Define set A
def A : Set ℝ := {x ∈ U | 0 < x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {x ∈ U | -2 ≤ x ∧ x ≤ 1}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x ∈ U | 0 < x ∧ x ≤ 1} := by sorry

-- Theorem for B ∪ (ᶜA)
theorem union_B_complement_A : B ∪ (U \ A) = {x ∈ U | x ≤ 1 ∨ 3 < x} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_B_complement_A_l1255_125556


namespace NUMINAMATH_CALUDE_find_b_l1255_125596

theorem find_b (x₁ x₂ c : ℝ) (h₁ : x₁ ≠ x₂) 
  (h₂ : ∃ y, y^2 + 2*x₁*y + 2*x₂ = 0 ∧ y^2 + 2*x₂*y + 2*x₁ = 0)
  (h₃ : x₁^2 + 5*(1/10)*x₁ + c = 0)
  (h₄ : x₂^2 + 5*(1/10)*x₂ + c = 0) :
  ∃ b : ℝ, b = 1/10 ∧ x₁^2 + 5*b*x₁ + c = 0 ∧ x₂^2 + 5*b*x₂ + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_find_b_l1255_125596


namespace NUMINAMATH_CALUDE_fraction_unchanged_l1255_125525

theorem fraction_unchanged (x y : ℝ) : 
  (2*x) * (2*y) / ((2*x)^2 - (2*y)^2) = x * y / (x^2 - y^2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l1255_125525


namespace NUMINAMATH_CALUDE_part_one_part_two_l1255_125581

-- Define the sets A and B
def A : Set ℝ := {x | (x + 2) * (x - 3) < 0}
def B (a : ℝ) : Set ℝ := {x | x - a > 0}

-- Part 1
theorem part_one : (Aᶜ ∪ B 1) = {x : ℝ | x ≤ -2 ∨ x > 1} := by sorry

-- Part 2
theorem part_two : A ⊆ B a → a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1255_125581


namespace NUMINAMATH_CALUDE_newspapers_sold_l1255_125577

theorem newspapers_sold (magazines : ℕ) (total : ℕ) (newspapers : ℕ) : 
  magazines = 425 → total = 700 → newspapers = total - magazines → newspapers = 275 := by
  sorry

end NUMINAMATH_CALUDE_newspapers_sold_l1255_125577


namespace NUMINAMATH_CALUDE_car_wash_solution_l1255_125509

/-- Represents the car wash problem --/
structure CarWash where
  car_price : ℕ
  truck_price : ℕ
  suv_price : ℕ
  total_raised : ℕ
  num_suvs : ℕ
  num_cars : ℕ

/-- The solution to the car wash problem --/
def solve_car_wash (cw : CarWash) : ℕ :=
  (cw.total_raised - cw.car_price * cw.num_cars - cw.suv_price * cw.num_suvs) / cw.truck_price

/-- Theorem stating the solution to the specific problem --/
theorem car_wash_solution :
  let cw : CarWash := {
    car_price := 5,
    truck_price := 6,
    suv_price := 7,
    total_raised := 100,
    num_suvs := 5,
    num_cars := 7
  }
  solve_car_wash cw = 5 := by
  sorry

end NUMINAMATH_CALUDE_car_wash_solution_l1255_125509


namespace NUMINAMATH_CALUDE_interest_rate_is_five_paise_l1255_125523

/-- Calculates the interest rate in paise per rupee per month given the principal, time, and simple interest -/
def interest_rate_paise (principal : ℚ) (time_months : ℚ) (simple_interest : ℚ) : ℚ :=
  (simple_interest / (principal * time_months)) * 100

/-- Theorem stating that for the given conditions, the interest rate is 5 paise per rupee per month -/
theorem interest_rate_is_five_paise 
  (principal : ℚ) 
  (time_months : ℚ) 
  (simple_interest : ℚ) 
  (h1 : principal = 20)
  (h2 : time_months = 6)
  (h3 : simple_interest = 6) :
  interest_rate_paise principal time_months simple_interest = 5 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_is_five_paise_l1255_125523


namespace NUMINAMATH_CALUDE_total_investment_l1255_125589

theorem total_investment (T : ℝ) : T = 2000 :=
  let invested_at_8_percent : ℝ := 600
  let invested_at_10_percent : ℝ := T - 600
  let income_difference : ℝ := 92
  have h1 : 0.10 * invested_at_10_percent - 0.08 * invested_at_8_percent = income_difference := by sorry
  sorry

end NUMINAMATH_CALUDE_total_investment_l1255_125589


namespace NUMINAMATH_CALUDE_zyx_syndrome_ratio_is_one_to_three_l1255_125532

/-- Represents the ratio of patients with ZYX syndrome to those without it -/
structure ZYXRatio where
  with_syndrome : ℕ
  without_syndrome : ℕ

/-- The clinic's patient information -/
structure ClinicInfo where
  total_patients : ℕ
  diagnosed_patients : ℕ

/-- Calculates the ZYX syndrome ratio given clinic information -/
def calculate_zyx_ratio (info : ClinicInfo) : ZYXRatio :=
  { with_syndrome := info.diagnosed_patients,
    without_syndrome := info.total_patients - info.diagnosed_patients }

/-- Simplifies a ZYX ratio by dividing both numbers by their GCD -/
def simplify_ratio (ratio : ZYXRatio) : ZYXRatio :=
  let gcd := Nat.gcd ratio.with_syndrome ratio.without_syndrome
  { with_syndrome := ratio.with_syndrome / gcd,
    without_syndrome := ratio.without_syndrome / gcd }

theorem zyx_syndrome_ratio_is_one_to_three :
  let clinic_info : ClinicInfo := { total_patients := 52, diagnosed_patients := 13 }
  let ratio := simplify_ratio (calculate_zyx_ratio clinic_info)
  ratio.with_syndrome = 1 ∧ ratio.without_syndrome = 3 := by sorry

end NUMINAMATH_CALUDE_zyx_syndrome_ratio_is_one_to_three_l1255_125532


namespace NUMINAMATH_CALUDE_freezer_temp_calculation_l1255_125542

-- Define the temperature of the refrigerator compartment
def refrigerator_temp : ℝ := 4

-- Define the temperature difference between compartments
def temp_difference : ℝ := 22

-- Theorem to prove
theorem freezer_temp_calculation :
  refrigerator_temp - temp_difference = -18 := by
  sorry

end NUMINAMATH_CALUDE_freezer_temp_calculation_l1255_125542


namespace NUMINAMATH_CALUDE_min_value_fraction_l1255_125575

theorem min_value_fraction (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (m : ℝ), m = 1 - Real.sqrt 2 ∧ ∀ z, z = (2 * x * y) / (x + y - 1) → m ≤ z := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1255_125575


namespace NUMINAMATH_CALUDE_zoe_family_members_l1255_125590

/-- Proves that Zoe is buying for 5 family members given the problem conditions -/
theorem zoe_family_members :
  let cost_per_person : ℚ := 3/2  -- $1.50
  let total_cost : ℚ := 9
  ∀ x : ℚ, (x + 1) * cost_per_person = total_cost → x = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_zoe_family_members_l1255_125590


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1255_125530

theorem polynomial_division_theorem (x : ℝ) :
  (x - 3) * (x^3 - 19*x^2 - 45*x - 148) + (-435) = x^4 - 22*x^3 + 12*x^2 - 13*x + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1255_125530


namespace NUMINAMATH_CALUDE_friends_team_assignment_l1255_125545

theorem friends_team_assignment (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 4) :
  k^n = 65536 := by
  sorry

end NUMINAMATH_CALUDE_friends_team_assignment_l1255_125545


namespace NUMINAMATH_CALUDE_calculate_expression_l1255_125554

theorem calculate_expression : (-2)^49 + 2^(4^4 + 3^2 - 5^2) = -2^49 + 2^240 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1255_125554


namespace NUMINAMATH_CALUDE_smaller_number_problem_l1255_125592

theorem smaller_number_problem (x y : ℤ) (h1 : x + y = 62) (h2 : y = x + 12) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l1255_125592


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1255_125567

theorem sqrt_equation_solution (x : ℝ) (h : 4 * x - 3 > 0) :
  (Real.sqrt (4 * x - 3) + 14 / Real.sqrt (4 * x - 3) = 8) ↔
  (x = (21 + 8 * Real.sqrt 2) / 4 ∨ x = (21 - 8 * Real.sqrt 2) / 4) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1255_125567


namespace NUMINAMATH_CALUDE_add_9999_seconds_to_5_45_00_l1255_125587

/-- Represents a time of day in hours, minutes, and seconds -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds seconds to a given time and returns the new time -/
def addSeconds (time : TimeOfDay) (seconds : Nat) : TimeOfDay :=
  sorry

/-- Converts a TimeOfDay to a string in the format "HH:MM:SS" -/
def TimeOfDay.toString (time : TimeOfDay) : String :=
  sorry

theorem add_9999_seconds_to_5_45_00 :
  let initialTime : TimeOfDay := ⟨17, 45, 0⟩
  let secondsToAdd : Nat := 9999
  let finalTime := addSeconds initialTime secondsToAdd
  finalTime.toString = "20:31:39" :=
sorry

end NUMINAMATH_CALUDE_add_9999_seconds_to_5_45_00_l1255_125587


namespace NUMINAMATH_CALUDE_vehicles_meeting_time_l1255_125543

/-- The time taken for two vehicles to meet when traveling towards each other -/
theorem vehicles_meeting_time (distance : ℝ) (speed1 speed2 : ℝ) (h1 : distance = 480) 
  (h2 : speed1 = 65) (h3 : speed2 = 55) : 
  (distance / (speed1 + speed2)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_vehicles_meeting_time_l1255_125543


namespace NUMINAMATH_CALUDE_coin_game_probabilities_l1255_125565

-- Define the coin probabilities
def p_heads : ℚ := 3/4
def p_tails : ℚ := 1/4

-- Define the games
def game_A : ℕ := 3  -- number of tosses in Game A
def game_C : ℕ := 4  -- number of tosses in Game C

-- Define the winning probability functions
def win_prob (n : ℕ) : ℚ := p_heads^n + p_tails^n

-- Theorem statement
theorem coin_game_probabilities :
  (win_prob game_A = 7/16) ∧ (win_prob game_C = 41/128) :=
sorry

end NUMINAMATH_CALUDE_coin_game_probabilities_l1255_125565


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l1255_125529

theorem quadratic_equation_m_value : 
  ∀ m : ℤ, 
  (∀ x : ℝ, ∃ a b c : ℝ, (m - 1) * x^(m^2 + 1) + 2*x - 3 = a*x^2 + b*x + c) →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l1255_125529


namespace NUMINAMATH_CALUDE_friend_reading_time_l1255_125555

/-- Given a person who reads at half the speed of their friend and takes 4 hours to read a book,
    prove that their friend will take 120 minutes to read the same book. -/
theorem friend_reading_time (my_speed friend_speed : ℝ) (my_time friend_time : ℝ) :
  my_speed = (1/2) * friend_speed →
  my_time = 4 →
  friend_time = 2 →
  friend_time * 60 = 120 := by
  sorry

end NUMINAMATH_CALUDE_friend_reading_time_l1255_125555


namespace NUMINAMATH_CALUDE_money_division_l1255_125566

theorem money_division (total : ℝ) (p q r : ℝ) : 
  p + q + r = total →
  p / q = 3 / 7 →
  q / r = 7 / 12 →
  q - p = 3200 →
  r - q = 4000 :=
by sorry

end NUMINAMATH_CALUDE_money_division_l1255_125566


namespace NUMINAMATH_CALUDE_complex_z_value_l1255_125546

theorem complex_z_value (z : ℂ) : z / Complex.I = 2 - 3 * Complex.I → z = 3 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_z_value_l1255_125546


namespace NUMINAMATH_CALUDE_set_equality_l1255_125522

def S : Set ℕ := {x | ∃ k : ℤ, 12 = k * (6 - x)}

theorem set_equality : S = {0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 18} := by sorry

end NUMINAMATH_CALUDE_set_equality_l1255_125522


namespace NUMINAMATH_CALUDE_trains_catch_up_l1255_125576

/-- The time (in hours after midnight) when the first train starts -/
def first_train_start : ℝ := 14

/-- The speed of the first train in km/h -/
def first_train_speed : ℝ := 70

/-- The time (in hours after midnight) when the second train starts -/
def second_train_start : ℝ := 15

/-- The speed of the second train in km/h -/
def second_train_speed : ℝ := 80

/-- The time (in hours after midnight) when the second train catches the first train -/
def catch_up_time : ℝ := 22

theorem trains_catch_up :
  let t := catch_up_time - second_train_start
  first_train_speed * (t + (second_train_start - first_train_start)) = second_train_speed * t :=
by sorry

end NUMINAMATH_CALUDE_trains_catch_up_l1255_125576


namespace NUMINAMATH_CALUDE_cyclist_speed_problem_l1255_125512

/-- The speed of cyclist C in miles per hour -/
def speed_C : ℝ := 10

/-- The speed of cyclist D in miles per hour -/
def speed_D : ℝ := speed_C + 5

/-- The distance to the town in miles -/
def distance_to_town : ℝ := 90

/-- The distance from the meeting point to the town in miles -/
def distance_meeting_to_town : ℝ := 18

theorem cyclist_speed_problem :
  /- Given conditions -/
  (speed_D = speed_C + 5) →
  (distance_to_town = 90) →
  (distance_meeting_to_town = 18) →
  /- The time taken by C to reach the meeting point equals
     the time taken by D to reach the town and return to the meeting point -/
  ((distance_to_town - distance_meeting_to_town) / speed_C =
   (distance_to_town + distance_meeting_to_town) / speed_D) →
  /- Conclusion: The speed of cyclist C is 10 mph -/
  speed_C = 10 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_problem_l1255_125512


namespace NUMINAMATH_CALUDE_books_per_shelf_l1255_125507

theorem books_per_shelf (total_books : ℕ) (num_shelves : ℕ) (h1 : total_books = 315) (h2 : num_shelves = 7) :
  total_books / num_shelves = 45 := by
sorry

end NUMINAMATH_CALUDE_books_per_shelf_l1255_125507


namespace NUMINAMATH_CALUDE_school_sections_l1255_125571

theorem school_sections (boys girls : ℕ) (h1 : boys = 408) (h2 : girls = 192) :
  let gcd := Nat.gcd boys girls
  let boys_sections := boys / gcd
  let girls_sections := girls / gcd
  boys_sections + girls_sections = 25 := by
sorry

end NUMINAMATH_CALUDE_school_sections_l1255_125571


namespace NUMINAMATH_CALUDE_fixed_point_of_f_l1255_125549

/-- The logarithmic function with base a > 0 and a ≠ 1 -/
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

/-- The function f(x) = 1 + log_a(x-1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + log a (x - 1)

/-- Theorem: For any base a > 0 and a ≠ 1, f(x) passes through the point (2,1) -/
theorem fixed_point_of_f (a : ℝ) (ha : a > 0 ∧ a ≠ 1) : f a 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_f_l1255_125549


namespace NUMINAMATH_CALUDE_larger_number_proof_l1255_125502

theorem larger_number_proof (S L : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 20) : L = 1634 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1255_125502


namespace NUMINAMATH_CALUDE_trigonometric_propositions_l1255_125557

open Real

theorem trigonometric_propositions :
  (¬ ∃ x : ℝ, sin x + cos x = 2) ∧
  (∃ x : ℝ, sin (2 * x) = sin x) ∧
  (∀ x ∈ Set.Icc (-π/2) (π/2), Real.sqrt ((1 + cos (2 * x)) / 2) = cos x) ∧
  (¬ ∀ x ∈ Set.Ioo 0 π, sin x > cos x) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_propositions_l1255_125557


namespace NUMINAMATH_CALUDE_percentage_problem_l1255_125515

theorem percentage_problem (P : ℝ) (x : ℝ) (h1 : x = 264) (h2 : (P / 100) * x = (1 / 3) * x + 110) : P = 75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1255_125515


namespace NUMINAMATH_CALUDE_solution_set_implies_a_b_values_f_1_negative_implies_a_conditions_l1255_125533

-- Define the function f
def f (a b x : ℝ) : ℝ := -3 * x^2 + a * (5 - a) * x + b

-- Part 1
theorem solution_set_implies_a_b_values (a b : ℝ) :
  (∀ x : ℝ, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  ((a = 2 ∧ b = 9) ∨ (a = 3 ∧ b = 9)) :=
sorry

-- Part 2
theorem f_1_negative_implies_a_conditions (b : ℝ) :
  (∀ a : ℝ, f a b 1 < 0 ↔
    (b < -13/4 ∧ a ∈ Set.univ) ∨
    (b = -13/4 ∧ a ≠ 5/2) ∨
    (b > -13/4 ∧ (a > (5 + Real.sqrt (4*b + 13))/2 ∨ a < (5 - Real.sqrt (4*b + 13))/2))) :=
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_b_values_f_1_negative_implies_a_conditions_l1255_125533


namespace NUMINAMATH_CALUDE_odd_number_2009_group_l1255_125544

/-- The cumulative sum of odd numbers up to the n-th group -/
def cumulative_sum (n : ℕ) : ℕ := n^2

/-- The size of the n-th group -/
def group_size (n : ℕ) : ℕ := 2*n - 1

/-- The theorem stating that 2009 belongs to the 32nd group -/
theorem odd_number_2009_group : 
  (cumulative_sum 31 < 2009) ∧ (2009 ≤ cumulative_sum 32) := by sorry

end NUMINAMATH_CALUDE_odd_number_2009_group_l1255_125544


namespace NUMINAMATH_CALUDE_square_table_correctness_l1255_125582

/-- Converts a base 60 number to base 10 -/
def base60ToBase10 (x : List Nat) : Nat :=
  x.enum.foldl (fun acc (i, digit) => acc + digit * (60 ^ i)) 0

/-- Converts a base 10 number to base 60 -/
def base10ToBase60 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 60) ((m % 60) :: acc)
    aux n []

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

/-- Represents the table of squares in base 60 -/
def squareTable : Nat → List Nat := sorry

theorem square_table_correctness :
  ∀ n : Nat, 1 ≤ n ∧ n ≤ 60 →
    base60ToBase10 (squareTable n) = n * n ∧
    isPerfectSquare (base60ToBase10 (squareTable n)) := by
  sorry

end NUMINAMATH_CALUDE_square_table_correctness_l1255_125582


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_l1255_125585

theorem smallest_k_no_real_roots : 
  ∃ k : ℕ, k = 4 ∧ 
  (∀ x : ℝ, 3 * x * (k * x - 5) - 2 * x^2 + 7 ≠ 0) ∧
  (∀ m : ℕ, m < k → ∃ x : ℝ, 3 * x * (m * x - 5) - 2 * x^2 + 7 = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_l1255_125585


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1255_125560

theorem quadratic_equations_solutions :
  ∃ (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ),
    (x₁^2 - 6*x₁ + 5 = 0 ∧ x₂^2 - 6*x₂ + 5 = 0 ∧ x₁ = 5 ∧ x₂ = 1) ∧
    (3*x₃*(2*x₃ - 1) = 4*x₃ - 2 ∧ 3*x₄*(2*x₄ - 1) = 4*x₄ - 2 ∧ x₃ = 1/2 ∧ x₄ = 2/3) ∧
    (x₅^2 - 2*Real.sqrt 2*x₅ - 2 = 0 ∧ x₆^2 - 2*Real.sqrt 2*x₆ - 2 = 0 ∧ 
     x₅ = Real.sqrt 2 + 2 ∧ x₆ = Real.sqrt 2 - 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1255_125560


namespace NUMINAMATH_CALUDE_students_in_both_clubs_l1255_125568

theorem students_in_both_clubs 
  (total_students : ℕ)
  (drama_club : ℕ)
  (science_club : ℕ)
  (in_either_club : ℕ)
  (h1 : total_students = 300)
  (h2 : drama_club = 80)
  (h3 : science_club = 130)
  (h4 : in_either_club = 190) :
  drama_club + science_club - in_either_club = 20 := by
  sorry

#check students_in_both_clubs

end NUMINAMATH_CALUDE_students_in_both_clubs_l1255_125568


namespace NUMINAMATH_CALUDE_range_when_a_is_one_a_values_for_all_x_geq_one_l1255_125513

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := |x - a| + |x + 4|

-- Theorem for part I
theorem range_when_a_is_one :
  Set.range (fun x => f x 1) = Set.Ici 5 := by sorry

-- Theorem for part II
theorem a_values_for_all_x_geq_one :
  {a : ℝ | ∀ x, f x a ≥ 1} = Set.Iic (-5) ∪ Set.Ici (-3) := by sorry

end NUMINAMATH_CALUDE_range_when_a_is_one_a_values_for_all_x_geq_one_l1255_125513


namespace NUMINAMATH_CALUDE_fraction_of_b_equal_to_third_of_a_prove_fraction_of_b_equal_to_third_of_a_l1255_125537

theorem fraction_of_b_equal_to_third_of_a : ℝ → ℝ → ℝ → Prop :=
  fun a b x =>
    a + b = 1210 →
    b = 484 →
    (1/3) * a = x * b →
    x = 1/2

-- Proof
theorem prove_fraction_of_b_equal_to_third_of_a :
  ∃ (a b x : ℝ), fraction_of_b_equal_to_third_of_a a b x :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_of_b_equal_to_third_of_a_prove_fraction_of_b_equal_to_third_of_a_l1255_125537


namespace NUMINAMATH_CALUDE_inequality_proof_l1255_125508

theorem inequality_proof (x y z : ℝ) : 
  x^2 / (x^2 + 2*y*z) + y^2 / (y^2 + 2*z*x) + z^2 / (z^2 + 2*x*y) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1255_125508


namespace NUMINAMATH_CALUDE_cassidy_poster_collection_l1255_125597

/-- The number of posters Cassidy has now -/
def current_posters : ℕ := 22

/-- The number of posters Cassidy will add -/
def added_posters : ℕ := 6

/-- The number of posters Cassidy had two years ago -/
def posters_two_years_ago : ℕ := 14

theorem cassidy_poster_collection :
  2 * posters_two_years_ago = current_posters + added_posters :=
by sorry

end NUMINAMATH_CALUDE_cassidy_poster_collection_l1255_125597


namespace NUMINAMATH_CALUDE_absolute_value_of_z_squared_minus_two_z_l1255_125506

theorem absolute_value_of_z_squared_minus_two_z (z : ℂ) : 
  z = 1 + I → Complex.abs (z^2 - 2*z) = 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_of_z_squared_minus_two_z_l1255_125506


namespace NUMINAMATH_CALUDE_intersection_M_N_l1255_125535

def M : Set ℝ := {x | 0 < x ∧ x < 4}
def N : Set ℝ := {x | 1/3 ≤ x ∧ x ≤ 5}

theorem intersection_M_N : M ∩ N = {x | 1/3 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1255_125535


namespace NUMINAMATH_CALUDE_square_of_binomial_l1255_125562

theorem square_of_binomial (m : ℝ) : 
  (∃ c : ℝ, ∀ x : ℝ, x^2 - 12*x + m = (x + c)^2) → m = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l1255_125562


namespace NUMINAMATH_CALUDE_toy_car_production_l1255_125563

theorem toy_car_production (yesterday : ℕ) (today : ℕ) : 
  yesterday = 60 → today = 2 * yesterday → yesterday + today = 180 := by
  sorry

end NUMINAMATH_CALUDE_toy_car_production_l1255_125563


namespace NUMINAMATH_CALUDE_student_ticket_price_l1255_125539

theorem student_ticket_price 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (student_tickets : ℕ) 
  (non_student_price : ℕ) :
  total_tickets = 2000 →
  total_revenue = 20960 →
  student_tickets = 520 →
  non_student_price = 11 →
  ∃ (student_price : ℕ),
    student_price * student_tickets + 
    non_student_price * (total_tickets - student_tickets) = 
    total_revenue ∧ student_price = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_student_ticket_price_l1255_125539


namespace NUMINAMATH_CALUDE_rainfall_rate_calculation_l1255_125514

/-- Proves that the rainfall rate is 5 cm/hour given the specified conditions -/
theorem rainfall_rate_calculation (depth : ℝ) (area : ℝ) (time : ℝ) 
  (h_depth : depth = 15)
  (h_area : area = 300)
  (h_time : time = 3) :
  (depth * area) / (time * area) = 5 := by
sorry

end NUMINAMATH_CALUDE_rainfall_rate_calculation_l1255_125514


namespace NUMINAMATH_CALUDE_probability_of_card_sequence_l1255_125580

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards of each suit in a standard deck -/
def CardsPerSuit : ℕ := 13

/-- Calculates the probability of the specified card sequence -/
def probabilityOfSequence : ℚ :=
  (CardsPerSuit : ℚ) / StandardDeck *
  (CardsPerSuit - 1) / (StandardDeck - 1) *
  CardsPerSuit / (StandardDeck - 2) *
  CardsPerSuit / (StandardDeck - 3)

/-- Theorem stating that the probability of drawing two hearts, 
    followed by one diamond, and then one club from a standard 
    52-card deck is equal to 39/63875 -/
theorem probability_of_card_sequence :
  probabilityOfSequence = 39 / 63875 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_card_sequence_l1255_125580


namespace NUMINAMATH_CALUDE_jungkook_has_most_apples_l1255_125500

def jungkook_initial : ℕ := 6
def jungkook_additional : ℕ := 3
def yoongi_apples : ℕ := 4
def yuna_apples : ℕ := 5

def jungkook_total : ℕ := jungkook_initial + jungkook_additional

theorem jungkook_has_most_apples :
  jungkook_total > yoongi_apples ∧ jungkook_total > yuna_apples :=
by sorry

end NUMINAMATH_CALUDE_jungkook_has_most_apples_l1255_125500


namespace NUMINAMATH_CALUDE_cubic_factorization_l1255_125573

theorem cubic_factorization (x : ℝ) : x^3 - 8*x^2 + 16*x = x*(x-4)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1255_125573


namespace NUMINAMATH_CALUDE_room_length_to_perimeter_ratio_l1255_125594

/-- The ratio of a rectangular room's length to its perimeter -/
theorem room_length_to_perimeter_ratio :
  let length : ℚ := 23
  let width : ℚ := 13
  let perimeter : ℚ := 2 * (length + width)
  (length : ℚ) / perimeter = 23 / 72 := by sorry

end NUMINAMATH_CALUDE_room_length_to_perimeter_ratio_l1255_125594


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l1255_125593

/-- A rectangle with perimeter 46 and area 108 has a shorter side of 9 -/
theorem rectangle_shorter_side : ∃ (a b : ℝ), 
  a > 0 ∧ b > 0 ∧  -- positive sides
  a ≥ b ∧          -- a is the longer side
  2 * (a + b) = 46 ∧  -- perimeter condition
  a * b = 108 ∧    -- area condition
  b = 9 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l1255_125593


namespace NUMINAMATH_CALUDE_max_value_quadratic_l1255_125552

theorem max_value_quadratic (a : ℝ) : 
  8 * a^2 + 6 * a + 2 = 0 → (∃ (x : ℝ), 3 * a + 2 ≤ x ∧ x = 5/4) :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l1255_125552


namespace NUMINAMATH_CALUDE_line_equation_equivalence_l1255_125548

/-- Given a line described by the dot product equation (-1, 2) · ((x, y) - (3, -4)) = 0,
    prove that it is equivalent to the line y = (1/2)x - 11/2 -/
theorem line_equation_equivalence (x y : ℝ) :
  (-1 : ℝ) * (x - 3) + 2 * (y - (-4)) = 0 ↔ y = (1/2 : ℝ) * x - 11/2 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_equivalence_l1255_125548


namespace NUMINAMATH_CALUDE_dolphin_count_l1255_125519

theorem dolphin_count (initial : ℕ) (joining_factor : ℕ) (h1 : initial = 65) (h2 : joining_factor = 3) :
  initial + joining_factor * initial = 260 := by
  sorry

end NUMINAMATH_CALUDE_dolphin_count_l1255_125519


namespace NUMINAMATH_CALUDE_range_of_a_l1255_125531

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |4*x - 3| ≤ 1 → x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0) ∧ 
  (∃ x : ℝ, |4*x - 3| ≤ 1 ∧ x^2 - (2*a + 1)*x + a*(a + 1) > 0) → 
  0 ≤ a ∧ a ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1255_125531


namespace NUMINAMATH_CALUDE_vector_operation_proof_l1255_125510

def vector_operation : ℝ × ℝ := sorry

theorem vector_operation_proof :
  vector_operation = (5, 2) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l1255_125510


namespace NUMINAMATH_CALUDE_g_zero_value_l1255_125501

-- Define polynomials f, g, and h
variable (f g h : ℝ[X])

-- Define the relationship between h, f, and g
axiom h_eq_f_mul_g : h = f * g

-- Define the constant term of f
axiom f_const_term : f.coeff 0 = 2

-- Define the constant term of h
axiom h_const_term : h.coeff 0 = -6

-- Theorem to prove
theorem g_zero_value : g.eval 0 = -3 := by sorry

end NUMINAMATH_CALUDE_g_zero_value_l1255_125501


namespace NUMINAMATH_CALUDE_point_transformation_l1255_125518

/-- Rotation of a point (x, y) by 90° counterclockwise around (h, k) -/
def rotate90 (x y h k : ℝ) : ℝ × ℝ :=
  (k - (y - h) + h, h + (x - h) + k)

/-- Reflection of a point (x, y) about the line y = x -/
def reflectYEqualsX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (a b : ℝ) :
  let (x₁, y₁) := rotate90 a b 2 3
  let (x₂, y₂) := reflectYEqualsX x₁ y₁
  (x₂ = -3 ∧ y₂ = 1) → b - a = -6 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l1255_125518


namespace NUMINAMATH_CALUDE_probability_of_one_in_first_20_rows_l1255_125558

/-- Calculates the number of elements in the first n rows of Pascal's Triangle. -/
def elementsInRows (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Calculates the number of ones in the first n rows of Pascal's Triangle. -/
def onesInRows (n : ℕ) : ℕ := if n = 0 then 1 else 2 * n - 1

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle. -/
def probabilityOfOne (n : ℕ) : ℚ :=
  (onesInRows n) / (elementsInRows n)

theorem probability_of_one_in_first_20_rows :
  probabilityOfOne 20 = 13 / 70 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_one_in_first_20_rows_l1255_125558


namespace NUMINAMATH_CALUDE_dog_weight_difference_l1255_125572

theorem dog_weight_difference (labrador_initial : ℝ) (dachshund_initial : ℝ) 
  (growth_rate : ℝ) (h1 : labrador_initial = 40) (h2 : dachshund_initial = 12) 
  (h3 : growth_rate = 0.25) : 
  labrador_initial * (1 + growth_rate) - dachshund_initial * (1 + growth_rate) = 35 := by
  sorry

end NUMINAMATH_CALUDE_dog_weight_difference_l1255_125572


namespace NUMINAMATH_CALUDE_eggs_used_for_omelet_l1255_125528

theorem eggs_used_for_omelet (initial_eggs : ℕ) (chickens : ℕ) (eggs_per_chicken : ℕ) (final_eggs : ℕ) : 
  initial_eggs = 10 →
  chickens = 2 →
  eggs_per_chicken = 3 →
  final_eggs = 11 →
  initial_eggs + chickens * eggs_per_chicken - final_eggs = 7 :=
by
  sorry

#check eggs_used_for_omelet

end NUMINAMATH_CALUDE_eggs_used_for_omelet_l1255_125528


namespace NUMINAMATH_CALUDE_square_area_error_l1255_125579

theorem square_area_error (s : ℝ) (s' : ℝ) (h : s' = s * (1 + 0.02)) :
  (s'^2 - s^2) / s^2 * 100 = 4.04 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l1255_125579


namespace NUMINAMATH_CALUDE_tangent_parallel_to_line_l1255_125595

theorem tangent_parallel_to_line (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.sin x + Real.cos x
  let tangent_point : ℝ × ℝ := (π / 2, 1)
  let tangent_slope : ℝ := (deriv f) (π / 2)
  let line_slope : ℝ := 1 / a
  (tangent_slope = line_slope) → a = -1 := by
sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_line_l1255_125595


namespace NUMINAMATH_CALUDE_buses_needed_l1255_125536

theorem buses_needed (classrooms : ℕ) (students_per_classroom : ℕ) (seats_per_bus : ℕ) : 
  classrooms = 67 → students_per_classroom = 66 → seats_per_bus = 6 →
  (classrooms * students_per_classroom + seats_per_bus - 1) / seats_per_bus = 738 := by
  sorry

end NUMINAMATH_CALUDE_buses_needed_l1255_125536


namespace NUMINAMATH_CALUDE_choose_3_from_10_l1255_125561

theorem choose_3_from_10 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_3_from_10_l1255_125561


namespace NUMINAMATH_CALUDE_square_with_ascending_digits_l1255_125520

theorem square_with_ascending_digits : ∃ n : ℕ, 
  (n^2).repr.takeRight 5 = "23456" ∧ 
  n^2 = 54563456 := by
  sorry

end NUMINAMATH_CALUDE_square_with_ascending_digits_l1255_125520


namespace NUMINAMATH_CALUDE_total_photos_after_bali_trip_l1255_125570

/-- Calculates the total number of photos after a trip to Bali -/
theorem total_photos_after_bali_trip 
  (initial_photos : ℕ) 
  (first_week_photos : ℕ) 
  (third_fourth_week_photos : ℕ) 
  (h1 : initial_photos = 100)
  (h2 : first_week_photos = 50)
  (h3 : third_fourth_week_photos = 80) :
  initial_photos + first_week_photos + 2 * first_week_photos + third_fourth_week_photos = 330 :=
by sorry

#check total_photos_after_bali_trip

end NUMINAMATH_CALUDE_total_photos_after_bali_trip_l1255_125570


namespace NUMINAMATH_CALUDE_ladder_problem_l1255_125521

theorem ladder_problem (ladder_length height_on_wall : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height_on_wall = 12) :
  ∃ (base_distance : ℝ), 
    base_distance^2 + height_on_wall^2 = ladder_length^2 ∧ 
    base_distance = 5 :=
by sorry

end NUMINAMATH_CALUDE_ladder_problem_l1255_125521


namespace NUMINAMATH_CALUDE_average_of_pqrs_l1255_125534

theorem average_of_pqrs (p q r s : ℝ) (h : (8 / 5) * (p + q + r + s) = 20) :
  (p + q + r + s) / 4 = 3.125 := by
sorry

end NUMINAMATH_CALUDE_average_of_pqrs_l1255_125534


namespace NUMINAMATH_CALUDE_line_equation_problem_l1255_125569

/-- Two distinct lines in the xy-plane -/
structure TwoLines where
  ℓ : Set (ℝ × ℝ)
  m : Set (ℝ × ℝ)
  distinct : ℓ ≠ m

/-- Point in ℝ² -/
def Point := ℝ × ℝ

/-- Reflection of a point about a line -/
def reflect (p : Point) (line : Set Point) : Point := sorry

/-- The problem statement -/
theorem line_equation_problem (lines : TwoLines) 
  (h1 : (0, 0) ∈ lines.ℓ ∩ lines.m)
  (h2 : ∀ x y, (x, y) ∈ lines.ℓ ↔ 3 * x + 4 * y = 0)
  (h3 : reflect (-2, 3) lines.ℓ = reflect (3, -2) lines.m) :
  ∀ x y, (x, y) ∈ lines.m ↔ 7 * x - 25 * y = 0 := by sorry

end NUMINAMATH_CALUDE_line_equation_problem_l1255_125569


namespace NUMINAMATH_CALUDE_derivative_sin_3x_at_pi_9_l1255_125541

theorem derivative_sin_3x_at_pi_9 :
  let f : ℝ → ℝ := λ x ↦ Real.sin (3 * x)
  let x₀ : ℝ := π / 9
  (deriv f) x₀ = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_3x_at_pi_9_l1255_125541


namespace NUMINAMATH_CALUDE_essay_writing_speed_l1255_125527

/-- Represents the essay writing scenario -/
structure EssayWriting where
  total_words : ℕ
  initial_speed : ℕ
  initial_hours : ℕ
  total_hours : ℕ

/-- Calculates the words written per hour after the initial period -/
def words_per_hour_after (e : EssayWriting) : ℕ :=
  (e.total_words - e.initial_speed * e.initial_hours) / (e.total_hours - e.initial_hours)

/-- Theorem stating that under the given conditions, the writing speed after
    the first two hours is 200 words per hour -/
theorem essay_writing_speed (e : EssayWriting) 
    (h1 : e.total_words = 1200)
    (h2 : e.initial_speed = 400)
    (h3 : e.initial_hours = 2)
    (h4 : e.total_hours = 4) : 
  words_per_hour_after e = 200 := by
  sorry

#eval words_per_hour_after { total_words := 1200, initial_speed := 400, initial_hours := 2, total_hours := 4 }

end NUMINAMATH_CALUDE_essay_writing_speed_l1255_125527


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1255_125526

theorem trigonometric_identity (α β : Real) 
  (h : (Real.sin α)^2 / (Real.cos β)^2 + (Real.cos α)^2 / (Real.sin β)^2 = 4) :
  (Real.cos β)^2 / (Real.sin α)^2 + (Real.sin β)^2 / (Real.cos α)^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1255_125526
