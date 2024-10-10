import Mathlib

namespace pencil_count_l1563_156348

theorem pencil_count (num_students : ℕ) (pencils_per_student : ℕ) 
  (h1 : num_students = 2) 
  (h2 : pencils_per_student = 9) : 
  num_students * pencils_per_student = 18 := by
  sorry

end pencil_count_l1563_156348


namespace triangle_area_comparison_l1563_156302

/-- The area of a triangle given its side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (s * (s - a) * (s - b) * (s - c)).sqrt

theorem triangle_area_comparison : 
  triangleArea 30 30 45 > triangleArea 30 30 55 := by sorry

end triangle_area_comparison_l1563_156302


namespace cos_15_cos_30_minus_sin_15_sin_150_l1563_156392

theorem cos_15_cos_30_minus_sin_15_sin_150 : 
  Real.cos (15 * π / 180) * Real.cos (30 * π / 180) - 
  Real.sin (15 * π / 180) * Real.sin (150 * π / 180) = 
  Real.sqrt 2 / 2 :=
by
  -- Assuming sin 150° = sin 30°
  have h1 : Real.sin (150 * π / 180) = Real.sin (30 * π / 180) := by sorry
  sorry

end cos_15_cos_30_minus_sin_15_sin_150_l1563_156392


namespace derivative_at_e_l1563_156362

open Real

theorem derivative_at_e (f : ℝ → ℝ) (h : Differentiable ℝ f) :
  (∀ x, f x = 2 * x * (deriv f e) - log x) →
  deriv f e = 1 / e :=
by sorry

end derivative_at_e_l1563_156362


namespace general_laborer_pay_general_laborer_pay_is_90_l1563_156310

/-- The daily pay for general laborers given the following conditions:
  - There are 35 people hired in total
  - The total payroll is 3950 dollars
  - 19 of the hired people are general laborers
  - Heavy equipment operators are paid 140 dollars per day
-/
theorem general_laborer_pay (total_hired : ℕ) (total_payroll : ℕ) 
  (num_laborers : ℕ) (operator_pay : ℕ) : ℕ :=
  let num_operators := total_hired - num_laborers
  let operator_total_pay := num_operators * operator_pay
  let laborer_total_pay := total_payroll - operator_total_pay
  laborer_total_pay / num_laborers

/-- Proof that the daily pay for general laborers is 90 dollars -/
theorem general_laborer_pay_is_90 : 
  general_laborer_pay 35 3950 19 140 = 90 := by
  sorry

end general_laborer_pay_general_laborer_pay_is_90_l1563_156310


namespace line_through_point_with_angle_l1563_156361

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parametric line -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Checks if a point lies on a parametric line -/
def pointOnLine (p : Point) (l : ParametricLine) : Prop :=
  ∃ t : ℝ, l.x t = p.x ∧ l.y t = p.y

/-- Calculates the angle between a parametric line and the positive x-axis -/
noncomputable def lineAngle (l : ParametricLine) : ℝ :=
  Real.arctan ((l.y 1 - l.y 0) / (l.x 1 - l.x 0))

theorem line_through_point_with_angle (M : Point) (θ : ℝ) :
  let l : ParametricLine := {
    x := λ t => 1 + (1/2) * t,
    y := λ t => 5 + (Real.sqrt 3 / 2) * t
  }
  pointOnLine M l ∧ lineAngle l = θ ∧ M.x = 1 ∧ M.y = 5 ∧ θ = π/3 := by
  sorry

end line_through_point_with_angle_l1563_156361


namespace special_matrix_product_l1563_156396

/-- A 5x5 matrix with special properties -/
structure SpecialMatrix where
  a : Fin 5 → Fin 5 → ℝ
  first_row_arithmetic : ∀ i j k : Fin 5, a 0 j - a 0 i = a 0 k - a 0 j
    → j - i = k - j
  columns_geometric : ∃ q : ℝ, ∀ i j : Fin 5, a (i+1) j = q * a i j
  a24_eq_4 : a 1 3 = 4
  a41_eq_neg2 : a 3 0 = -2
  a43_eq_10 : a 3 2 = 10

/-- The product of a₁₁ and a₅₅ is -11 -/
theorem special_matrix_product (m : SpecialMatrix) : m.a 0 0 * m.a 4 4 = -11 := by
  sorry

end special_matrix_product_l1563_156396


namespace roots_reciprocal_sum_l1563_156316

theorem roots_reciprocal_sum (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 2 * x₁ - 1 = 0) → 
  (2 * x₂^2 - 2 * x₂ - 1 = 0) → 
  (x₁ ≠ x₂) →
  (1 / x₁ + 1 / x₂ = -2) := by
  sorry

end roots_reciprocal_sum_l1563_156316


namespace quadratic_solution_l1563_156336

theorem quadratic_solution (x : ℝ) (h1 : 2 * x^2 - 6 * x = 0) (h2 : x ≠ 0) : x = 3 := by
  sorry

end quadratic_solution_l1563_156336


namespace fraction_product_l1563_156354

theorem fraction_product : (2 : ℚ) / 9 * 5 / 8 = 5 / 36 := by
  sorry

end fraction_product_l1563_156354


namespace midpoint_locus_of_square_l1563_156343

/-- The locus of the midpoint of a square with side length 2a, where two consecutive vertices
    are always on the x- and y-axes respectively in the first quadrant, is a circle with
    radius a centered at the origin. -/
theorem midpoint_locus_of_square (a : ℝ) (h : a > 0) :
  ∃ (C : ℝ × ℝ), (∀ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ x^2 + y^2 = (2*a)^2 →
    C = (x/2, y/2) ∧ C.1^2 + C.2^2 = a^2) :=
sorry

end midpoint_locus_of_square_l1563_156343


namespace shortest_wire_for_given_poles_l1563_156325

/-- Represents a cylindrical pole with a given diameter -/
structure Pole where
  diameter : ℝ

/-- Calculates the shortest wire length to wrap around three poles -/
def shortestWireLength (pole1 pole2 pole3 : Pole) : ℝ :=
  sorry

/-- The theorem stating the shortest wire length for the given poles -/
theorem shortest_wire_for_given_poles :
  let pole1 : Pole := ⟨6⟩
  let pole2 : Pole := ⟨18⟩
  let pole3 : Pole := ⟨12⟩
  shortestWireLength pole1 pole2 pole3 = 6 * Real.sqrt 3 + 6 * Real.sqrt 6 + 18 * Real.pi :=
by sorry

end shortest_wire_for_given_poles_l1563_156325


namespace egypt_promotion_free_tourists_l1563_156358

/-- Represents the number of tourists who went to Egypt for free -/
def free_tourists : ℕ := 29

/-- Represents the number of tourists who came on their own -/
def self_tourists : ℕ := 13

/-- Represents the number of tourists who brought no one -/
def no_referral_tourists : ℕ := 100

theorem egypt_promotion_free_tourists :
  ∃ (total_tourists : ℕ),
    total_tourists = self_tourists + 4 * free_tourists ∧
    total_tourists = free_tourists + no_referral_tourists ∧
    free_tourists * 4 + self_tourists = free_tourists + no_referral_tourists :=
by sorry

end egypt_promotion_free_tourists_l1563_156358


namespace arrangement_count_proof_l1563_156338

/-- The number of ways to arrange 2 female students and 4 male students in a row,
    such that female student A is to the left of female student B. -/
def arrangement_count : ℕ := 360

/-- The total number of students -/
def total_students : ℕ := 6

/-- The number of female students -/
def female_students : ℕ := 2

/-- The number of male students -/
def male_students : ℕ := 4

theorem arrangement_count_proof :
  arrangement_count = (Nat.factorial total_students) / 2 :=
sorry

end arrangement_count_proof_l1563_156338


namespace line_equation_through_bisecting_point_l1563_156342

/-- Given a parabola and a line with specific properties, prove the equation of the line -/
theorem line_equation_through_bisecting_point (x y : ℝ) :
  (∀ x y, y^2 = 16*x) → -- parabola equation
  (∃ x1 y1 x2 y2 : ℝ, 
    y1^2 = 16*x1 ∧ y2^2 = 16*x2 ∧ -- intersection points on parabola
    (x1 + x2)/2 = 2 ∧ (y1 + y2)/2 = 1) → -- midpoint is (2, 1)
  (8*x - y - 15 = 0) :=
by
  sorry


end line_equation_through_bisecting_point_l1563_156342


namespace photographer_profit_percentage_l1563_156374

/-- Calculates the profit percentage for a photographer's business --/
theorem photographer_profit_percentage
  (selling_price : ℝ)
  (production_cost : ℝ)
  (sale_probability : ℝ)
  (h1 : selling_price = 600)
  (h2 : production_cost = 100)
  (h3 : sale_probability = 1/4)
  : (((sale_probability * selling_price - production_cost) / production_cost) * 100 = 50) :=
by sorry

end photographer_profit_percentage_l1563_156374


namespace max_discussions_left_l1563_156318

/-- Represents a group of politicians at a summit --/
structure PoliticianGroup where
  size : Nat
  has_talked : Fin size → Fin size → Bool
  all_pairs_plan_to_talk : ∀ i j, i ≠ j → has_talked i j = false → True
  four_politician_condition : ∀ a b c d, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d →
    (has_talked a b ∧ has_talked a c ∧ has_talked a d) ∨
    (has_talked b a ∧ has_talked b c ∧ has_talked b d) ∨
    (has_talked c a ∧ has_talked c b ∧ has_talked c d) ∨
    (has_talked d a ∧ has_talked d b ∧ has_talked d c)

/-- The theorem stating the maximum number of discussions yet to be held --/
theorem max_discussions_left (g : PoliticianGroup) (h : g.size = 2018) :
  (∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    ¬g.has_talked a b ∧ ¬g.has_talked b c ∧ ¬g.has_talked a c) ∧
  (∀ a b c d, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d →
    g.has_talked a b ∨ g.has_talked b c ∨ g.has_talked a c ∨
    g.has_talked a d ∨ g.has_talked b d ∨ g.has_talked c d) :=
by sorry

end max_discussions_left_l1563_156318


namespace simple_interest_problem_l1563_156340

-- Define the variables
variable (P : ℝ) -- Principal amount
variable (R : ℝ) -- Original interest rate in percentage

-- Define the theorem
theorem simple_interest_problem :
  (P * (R + 3) * 2) / 100 - (P * R * 2) / 100 = 300 →
  P = 5000 := by
sorry

end simple_interest_problem_l1563_156340


namespace sum_xy_given_condition_l1563_156324

theorem sum_xy_given_condition (x y : ℝ) : 
  |x + 3| + (y - 2)^2 = 0 → x + y = -1 := by sorry

end sum_xy_given_condition_l1563_156324


namespace quartic_equation_roots_l1563_156393

theorem quartic_equation_roots (p : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ 
    x₁^4 + p*x₁^3 + 3*x₁^2 + p*x₁ + 4 = 0 ∧
    x₂^4 + p*x₂^3 + 3*x₂^2 + p*x₂ + 4 = 0) ↔ 
  (p ≤ -2 ∨ p ≥ 2) :=
sorry

end quartic_equation_roots_l1563_156393


namespace F_3_f_4_equals_7_l1563_156345

-- Define the functions f and F
def f (a : ℝ) : ℝ := a - 2
def F (a b : ℝ) : ℝ := b^2 + a

-- State the theorem
theorem F_3_f_4_equals_7 : F 3 (f 4) = 7 := by
  sorry

end F_3_f_4_equals_7_l1563_156345


namespace partitions_count_l1563_156372

/-- The number of partitions of a set with n+1 elements into n subsets -/
def num_partitions (n : ℕ) : ℕ := (2^n - 1) * n + 1

/-- Theorem stating the number of partitions of a set with n+1 elements into n subsets -/
theorem partitions_count (n : ℕ) (h : n > 0) :
  num_partitions n = (2^n - 1) * n + 1 :=
by sorry

end partitions_count_l1563_156372


namespace compound_interest_problem_l1563_156330

theorem compound_interest_problem (P r : ℝ) : 
  P > 0 → r > 0 →
  P * (1 + r)^2 = 7000 →
  P * (1 + r)^3 = 9261 →
  P = 4000 := by
sorry

end compound_interest_problem_l1563_156330


namespace shaded_area_is_thirty_l1563_156397

/-- An isosceles right triangle with legs of length 10 -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  is_ten : leg_length = 10

/-- The large triangle partitioned into 25 congruent smaller triangles -/
def num_partitions : ℕ := 25

/-- The number of shaded smaller triangles -/
def num_shaded : ℕ := 15

/-- Theorem stating the area of the shaded region -/
theorem shaded_area_is_thirty (t : IsoscelesRightTriangle) : 
  (t.leg_length^2 / 2) * (num_shaded / num_partitions) = 30 := by
  sorry

end shaded_area_is_thirty_l1563_156397


namespace couples_matching_l1563_156363

structure Couple where
  wife : String
  husband : String
  wife_bottles : ℕ
  husband_bottles : ℕ

def total_bottles : ℕ := 44

def couples : List Couple := [
  ⟨"Anna", "", 2, 0⟩,
  ⟨"Betty", "", 3, 0⟩,
  ⟨"Carol", "", 4, 0⟩,
  ⟨"Dorothy", "", 5, 0⟩
]

def husbands : List String := ["Brown", "Green", "White", "Smith"]

theorem couples_matching :
  ∃ (matched_couples : List Couple),
    matched_couples.length = 4 ∧
    (matched_couples.map (λ c => c.wife_bottles + c.husband_bottles)).sum = total_bottles ∧
    (∃ c ∈ matched_couples, c.wife = "Anna" ∧ c.husband = "Smith" ∧ c.husband_bottles = 4 * c.wife_bottles) ∧
    (∃ c ∈ matched_couples, c.wife = "Betty" ∧ c.husband = "White" ∧ c.husband_bottles = 3 * c.wife_bottles) ∧
    (∃ c ∈ matched_couples, c.wife = "Carol" ∧ c.husband = "Green" ∧ c.husband_bottles = 2 * c.wife_bottles) ∧
    (∃ c ∈ matched_couples, c.wife = "Dorothy" ∧ c.husband = "Brown" ∧ c.husband_bottles = c.wife_bottles) ∧
    (matched_couples.map (λ c => c.husband)).toFinset = husbands.toFinset :=
by sorry

end couples_matching_l1563_156363


namespace security_deposit_is_1110_l1563_156335

/-- Calculates the security deposit for a cabin rental --/
def calculate_security_deposit (weeks : ℕ) (daily_rate : ℚ) (pet_fee : ℚ) (service_fee_rate : ℚ) (deposit_rate : ℚ) : ℚ :=
  let days := weeks * 7
  let rental_fee := daily_rate * days
  let total_rental := rental_fee + pet_fee
  let service_fee := service_fee_rate * total_rental
  let total_cost := total_rental + service_fee
  deposit_rate * total_cost

/-- Theorem: The security deposit for the given conditions is $1,110.00 --/
theorem security_deposit_is_1110 :
  calculate_security_deposit 2 125 100 (1/5) (1/2) = 1110 := by
  sorry

end security_deposit_is_1110_l1563_156335


namespace binary_subtraction_l1563_156341

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def a : List Bool := [true, true, true, true, true, true, true, true, true]
def b : List Bool := [true, true, true, true]

theorem binary_subtraction :
  binary_to_decimal a - binary_to_decimal b = 496 := by
  sorry

end binary_subtraction_l1563_156341


namespace craig_travel_difference_l1563_156394

theorem craig_travel_difference :
  let bus_distance : ℝ := 3.83
  let walk_distance : ℝ := 0.17
  bus_distance - walk_distance = 3.66 := by sorry

end craig_travel_difference_l1563_156394


namespace f_composition_negative_two_l1563_156384

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 2^x

theorem f_composition_negative_two : f (f (-2)) = 1/2 := by
  sorry

end f_composition_negative_two_l1563_156384


namespace job_completion_time_l1563_156300

/-- Calculates the remaining days to complete a job given initial and additional workers -/
def remaining_days (initial_workers : ℕ) (initial_days : ℕ) (days_worked : ℕ) (additional_workers : ℕ) : ℚ :=
  let total_work := initial_workers * initial_days
  let work_done := initial_workers * days_worked
  let remaining_work := total_work - work_done
  let total_workers := initial_workers + additional_workers
  remaining_work / total_workers

theorem job_completion_time : remaining_days 6 8 3 4 = 3 := by
  sorry

end job_completion_time_l1563_156300


namespace years_until_double_age_l1563_156307

/-- Represents the age difference problem between a father and son -/
structure AgeDifference where
  son_age : ℕ
  father_age : ℕ
  years_until_double : ℕ

/-- The age difference scenario satisfies the given conditions -/
def valid_age_difference (ad : AgeDifference) : Prop :=
  ad.son_age = 10 ∧
  ad.father_age = 40 ∧
  ad.father_age = 4 * ad.son_age ∧
  ad.father_age + ad.years_until_double = 2 * (ad.son_age + ad.years_until_double)

/-- Theorem stating that the number of years until the father is twice as old as the son is 20 -/
theorem years_until_double_age : ∀ ad : AgeDifference, valid_age_difference ad → ad.years_until_double = 20 := by
  sorry

end years_until_double_age_l1563_156307


namespace largest_prime_factor_of_4872_l1563_156328

theorem largest_prime_factor_of_4872 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 4872 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 4872 → q ≤ p :=
by sorry

end largest_prime_factor_of_4872_l1563_156328


namespace triangle_area_l1563_156347

-- Define the triangle DEF
structure Triangle :=
  (D E F : ℝ × ℝ)

-- Define the point L on EF
def L (t : Triangle) : ℝ × ℝ := sorry

-- State that DL is an altitude of triangle DEF
def is_altitude (t : Triangle) : Prop :=
  let (dx, dy) := t.D
  let (lx, ly) := L t
  (lx - dx) * (t.F.1 - t.E.1) + (ly - dy) * (t.F.2 - t.E.2) = 0

-- Define the lengths
def DE (t : Triangle) : ℝ := sorry
def EL (t : Triangle) : ℝ := sorry
def EF (t : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_area (t : Triangle) 
  (h1 : is_altitude t)
  (h2 : DE t = 14)
  (h3 : EL t = 9)
  (h4 : EF t = 17) :
  let area := (EF t * Real.sqrt ((DE t)^2 - (EL t)^2)) / 2
  area = (17 * Real.sqrt 115) / 2 :=
sorry

end triangle_area_l1563_156347


namespace shaded_area_in_circle_l1563_156339

theorem shaded_area_in_circle (r : ℝ) (h : r = 6) : 
  let angle : ℝ := π / 3  -- 60° in radians
  let triangle_area : ℝ := (1/2) * r * r * Real.sin angle
  let sector_area : ℝ := (angle / (2 * π)) * π * r^2
  2 * triangle_area + 2 * sector_area = 36 * Real.sqrt 3 + 12 * π := by
sorry

end shaded_area_in_circle_l1563_156339


namespace derivative_x_plus_one_squared_times_x_minus_one_l1563_156320

theorem derivative_x_plus_one_squared_times_x_minus_one (x : ℝ) :
  deriv (λ x => (x + 1)^2 * (x - 1)) x = 3*x^2 + 2*x - 1 := by
  sorry

end derivative_x_plus_one_squared_times_x_minus_one_l1563_156320


namespace octal_to_decimal_fraction_l1563_156332

theorem octal_to_decimal_fraction (c d : ℕ) : 
  (c < 10 ∧ d < 10) →  -- c and d are base-10 digits
  (435 : Nat) = 4 * 8^2 + 3 * 8 + 5 →  -- 435 in octal
  285 = 200 + 10 * c + d →  -- 2cd in decimal
  (c + d) / 12 = 5 / 6 := by
  sorry

end octal_to_decimal_fraction_l1563_156332


namespace ratio_fraction_equality_l1563_156368

theorem ratio_fraction_equality (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 6) :
  (2 * A + 3 * B) / (5 * C - 2 * A) = 1 / 2 := by
  sorry

end ratio_fraction_equality_l1563_156368


namespace election_winner_votes_l1563_156319

theorem election_winner_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (65 : ℚ) / 100 * total_votes - (35 : ℚ) / 100 * total_votes = 300) : 
  (65 : ℚ) / 100 * total_votes = 650 := by
  sorry

end election_winner_votes_l1563_156319


namespace meatballs_on_plate_l1563_156315

theorem meatballs_on_plate (num_sons : ℕ) (fraction_eaten : ℚ) (meatballs_left : ℕ) : 
  num_sons = 3 → 
  fraction_eaten = 2/3 → 
  meatballs_left = 3 → 
  ∃ (initial_meatballs : ℕ), 
    initial_meatballs = 3 ∧ 
    (num_sons : ℚ) * ((1 : ℚ) - fraction_eaten) * initial_meatballs = meatballs_left :=
by sorry

end meatballs_on_plate_l1563_156315


namespace average_speed_two_segment_trip_l1563_156379

theorem average_speed_two_segment_trip (d1 d2 v1 v2 : ℝ) 
  (h1 : d1 = 45) (h2 : d2 = 15) (h3 : v1 = 15) (h4 : v2 = 45) :
  (d1 + d2) / ((d1 / v1) + (d2 / v2)) = 18 := by
  sorry

end average_speed_two_segment_trip_l1563_156379


namespace expression_value_l1563_156301

theorem expression_value (m : ℝ) (h : 1 / (m - 2) = 1) : 2 / (m - 2) - m + 2 = 1 := by
  sorry

end expression_value_l1563_156301


namespace comparison_sqrt_l1563_156333

theorem comparison_sqrt : 3 * Real.sqrt 2 > Real.sqrt 15 := by
  sorry

end comparison_sqrt_l1563_156333


namespace system_solution_l1563_156360

theorem system_solution (x y : ℝ) : 
  (4 * x + y = 6 ∧ 3 * x - y = 1) ↔ (x = 1 ∧ y = 2) := by
  sorry

end system_solution_l1563_156360


namespace quadratic_roots_opposite_signs_l1563_156391

theorem quadratic_roots_opposite_signs (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - (a + 3) * x + 2 = 0 ∧ 
               a * y^2 - (a + 3) * y + 2 = 0 ∧ 
               x * y < 0) ↔ 
  a < 0 := by
sorry

end quadratic_roots_opposite_signs_l1563_156391


namespace josh_remaining_money_l1563_156304

def initial_amount : ℚ := 9
def first_expense : ℚ := 1.75
def second_expense : ℚ := 1.25

theorem josh_remaining_money :
  initial_amount - first_expense - second_expense = 6 := by sorry

end josh_remaining_money_l1563_156304


namespace multiple_problem_l1563_156364

theorem multiple_problem (x y : ℕ) (k m : ℕ) : 
  x = 11 → 
  x + y = 55 → 
  y = k * x + m → 
  k = 4 ∧ m = 0 := by
sorry

end multiple_problem_l1563_156364


namespace cube_sum_inequality_l1563_156395

theorem cube_sum_inequality (x y z : ℝ) : 
  x^3 + y^3 + z^3 + 3*x*y*z ≥ x^2*(y+z) + y^2*(z+x) + z^2*(x+y) := by
  sorry

end cube_sum_inequality_l1563_156395


namespace parallelogram_construction_l1563_156380

-- Define a structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a structure for a circle
structure Circle where
  center : Point2D
  radius : ℝ

-- Define the problem statement
theorem parallelogram_construction (A B C : Point2D) (r : ℝ) 
  (h1 : ∃ (circle : Circle), circle.center = A ∧ circle.radius = r ∧ 
    (B.x - A.x)^2 + (B.y - A.y)^2 ≤ r^2 ∧ 
    (C.x - A.x)^2 + (C.y - A.y)^2 ≤ r^2) :
  ∃ (D : Point2D), 
    (A.x + C.x = B.x + D.x) ∧ 
    (A.y + C.y = B.y + D.y) ∧
    (A.x - B.x = D.x - C.x) ∧ 
    (A.y - B.y = D.y - C.y) :=
sorry

end parallelogram_construction_l1563_156380


namespace translation_problem_l1563_156337

/-- A translation of the complex plane -/
def ComplexTranslation (w : ℂ) : ℂ → ℂ := fun z ↦ z + w

theorem translation_problem (t : ℂ → ℂ) (h : t (1 + 3*I) = 4 + 2*I) :
  ∃ w : ℂ, t = ComplexTranslation w ∧ t (3 - 2*I) = 6 - 3*I := by
  sorry

end translation_problem_l1563_156337


namespace distance_on_number_line_l1563_156322

theorem distance_on_number_line (A B C : ℝ) : 
  (|B - A| = 5) → (|C - B| = 3) → (|C - A| = 2 ∨ |C - A| = 8) :=
by sorry

end distance_on_number_line_l1563_156322


namespace prism_square_intersection_angle_l1563_156305

theorem prism_square_intersection_angle (d : ℝ) (h : d > 0) : 
  let rhombus_acute_angle : ℝ := 60 * π / 180
  let rhombus_diagonal : ℝ := d * Real.sqrt 3
  let intersection_angle : ℝ := Real.arccos (Real.sqrt 3 / 3)
  intersection_angle = Real.arccos (d / rhombus_diagonal) :=
by sorry

end prism_square_intersection_angle_l1563_156305


namespace simplify_complex_fraction_l1563_156367

theorem simplify_complex_fraction :
  1 / (1 / (Real.sqrt 2 + 1) + 2 / (Real.sqrt 3 - 1) + 3 / (Real.sqrt 5 + 2)) =
  1 / (Real.sqrt 2 + 2 * Real.sqrt 3 + 3 * Real.sqrt 5 - 5) := by
  sorry

end simplify_complex_fraction_l1563_156367


namespace jerry_age_l1563_156371

/-- Given that Mickey's age is 8 years less than 200% of Jerry's age,
    and Mickey is 16 years old, prove that Jerry is 12 years old. -/
theorem jerry_age (mickey_age jerry_age : ℕ) 
  (h1 : mickey_age = 16)
  (h2 : mickey_age = 2 * jerry_age - 8) : 
  jerry_age = 12 := by
sorry

end jerry_age_l1563_156371


namespace expression_range_l1563_156381

theorem expression_range (x y : ℝ) (h : (x - 1)^2 + (y - 4)^2 = 1) :
  0 ≤ (x*y - x) / (x^2 + (y - 1)^2) ∧ (x*y - x) / (x^2 + (y - 1)^2) ≤ 12/25 := by
  sorry

end expression_range_l1563_156381


namespace negation_of_universal_positive_quadratic_l1563_156388

theorem negation_of_universal_positive_quadratic :
  (¬ ∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) := by
  sorry

end negation_of_universal_positive_quadratic_l1563_156388


namespace line_only_count_l1563_156327

/-- Represents an alphabet with letters containing dots and straight lines -/
structure Alphabet :=
  (total_letters : ℕ)
  (dot_and_line : ℕ)
  (dot_only : ℕ)
  (h_total : total_letters = 40)
  (h_dot_and_line : dot_and_line = 9)
  (h_dot_only : dot_only = 7)
  (h_all_contain : total_letters = dot_and_line + dot_only + (total_letters - (dot_and_line + dot_only)))

/-- The number of letters containing a straight line but not a dot -/
def line_only (α : Alphabet) : ℕ := α.total_letters - (α.dot_and_line + α.dot_only)

theorem line_only_count (α : Alphabet) : line_only α = 24 := by
  sorry

end line_only_count_l1563_156327


namespace hexacontagon_triangles_l1563_156314

/-- The number of sides in a regular hexacontagon -/
def n : ℕ := 60

/-- The number of triangles that can be formed using the vertices of a regular hexacontagon,
    without using any three consecutive vertices -/
def num_triangles : ℕ := Nat.choose n 3 - n

theorem hexacontagon_triangles : num_triangles = 34160 := by
  sorry

end hexacontagon_triangles_l1563_156314


namespace gift_cost_equation_l1563_156306

/-- Represents the cost equation for Xiaofen's gift purchase -/
theorem gift_cost_equation (x : ℝ) : 
  (15 : ℝ) * (x + 2 * 20) = 900 ↔ 
  (∃ (total_cost num_gifts num_lollipops_per_gift lollipop_cost : ℝ),
    total_cost = 900 ∧
    num_gifts = 15 ∧
    num_lollipops_per_gift = 2 ∧
    lollipop_cost = 20 ∧
    total_cost = num_gifts * (x + num_lollipops_per_gift * lollipop_cost)) :=
by sorry

end gift_cost_equation_l1563_156306


namespace rose_bush_price_is_75_l1563_156356

-- Define the given conditions
def total_rose_bushes : ℕ := 6
def friend_rose_bushes : ℕ := 2
def aloe_count : ℕ := 2
def aloe_price : ℕ := 100
def total_spent_self : ℕ := 500

-- Define the function to calculate the price of each rose bush
def rose_bush_price : ℕ :=
  let self_rose_bushes := total_rose_bushes - friend_rose_bushes
  let aloe_total := aloe_count * aloe_price
  let rose_bushes_total := total_spent_self - aloe_total
  rose_bushes_total / self_rose_bushes

-- Theorem statement
theorem rose_bush_price_is_75 : rose_bush_price = 75 := by
  sorry

end rose_bush_price_is_75_l1563_156356


namespace normal_distribution_probability_l1563_156350

-- Define a random variable following normal distribution
def normal_distribution (μ σ : ℝ) : Type := ℝ

-- Define probability function
noncomputable def probability {α : Type} (event : Set α) : ℝ := sorry

-- Theorem statement
theorem normal_distribution_probability 
  (ξ : normal_distribution 1 σ) 
  (h1 : probability {x | x < 1} = 1/2) 
  (h2 : probability {x | x > 2} = p) :
  probability {x | 0 < x ∧ x < 1} = 1/2 - p :=
sorry

end normal_distribution_probability_l1563_156350


namespace pasta_cost_is_one_dollar_l1563_156317

/-- The cost of pasta per box for Sam's spaghetti and meatballs dinner -/
def pasta_cost (total_cost sauce_cost meatballs_cost : ℚ) : ℚ :=
  total_cost - (sauce_cost + meatballs_cost)

/-- Theorem: The cost of pasta per box is $1.00 -/
theorem pasta_cost_is_one_dollar :
  pasta_cost 8 2 5 = 1 := by
  sorry

end pasta_cost_is_one_dollar_l1563_156317


namespace cubic_inequality_l1563_156366

theorem cubic_inequality (x : ℝ) : x^3 - 4*x^2 - 7*x + 10 > 0 ↔ x < -2 ∨ x > 5 := by
  sorry

end cubic_inequality_l1563_156366


namespace crayons_remaining_l1563_156329

theorem crayons_remaining (initial : ℕ) (taken : ℕ) (remaining : ℕ) : 
  initial = 7 → taken = 3 → remaining = initial - taken → remaining = 4 := by
sorry

end crayons_remaining_l1563_156329


namespace dividend_mod_31_l1563_156331

theorem dividend_mod_31 (divisor quotient remainder dividend : ℕ) : 
  divisor = 37 → 
  quotient = 214 → 
  remainder = 12 → 
  dividend = divisor * quotient + remainder →
  dividend % 31 = 25 := by
  sorry

end dividend_mod_31_l1563_156331


namespace no_reciprocal_roots_l1563_156377

theorem no_reciprocal_roots (a b c : ℤ) (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ¬∃ (n : ℕ), a * (1 / n : ℚ)^2 + b * (1 / n : ℚ) + c = 0 := by
  sorry

end no_reciprocal_roots_l1563_156377


namespace crate_weight_l1563_156309

/-- Given an empty truck weighing 9600 kg and a total weight of 38000 kg when loaded with 40 identical crates, 
    prove that each crate weighs 710 kg. -/
theorem crate_weight (empty_truck_weight : ℕ) (loaded_truck_weight : ℕ) (num_crates : ℕ) :
  empty_truck_weight = 9600 →
  loaded_truck_weight = 38000 →
  num_crates = 40 →
  (loaded_truck_weight - empty_truck_weight) / num_crates = 710 :=
by sorry

end crate_weight_l1563_156309


namespace max_distance_MP_l1563_156359

-- Define the equilateral triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = 8 ∧ dist B C = 8 ∧ dist C A = 8

-- Define the point O satisfying the given condition
def PointO (A B C O : ℝ × ℝ) : Prop :=
  (O.1 - A.1, O.2 - A.2) = 2 • (O.1 - B.1, O.2 - B.2) + 3 • (O.1 - C.1, O.2 - C.2)

-- Define a point M on the sides of triangle ABC
def PointM (A B C M : ℝ × ℝ) : Prop :=
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)) ∨
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2)) ∨
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (t * C.1 + (1 - t) * A.1, t * C.2 + (1 - t) * A.2))

-- Define a point P such that |OP| = √19
def PointP (O P : ℝ × ℝ) : Prop :=
  dist O P = Real.sqrt 19

theorem max_distance_MP (A B C O M P : ℝ × ℝ) :
  Triangle A B C →
  PointO A B C O →
  PointM A B C M →
  PointP O P →
  (∀ M' P', PointM A B C M' → PointP O P' → dist M P ≤ dist M' P') →
  dist M P = 3 * Real.sqrt 19 :=
sorry

end max_distance_MP_l1563_156359


namespace room_width_calculation_l1563_156313

/-- Given a rectangular room with known length, flooring cost per square meter, 
    and total flooring cost, calculate the width of the room. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) :
  length = 5.5 →
  cost_per_sqm = 800 →
  total_cost = 16500 →
  (total_cost / cost_per_sqm) / length = 3.75 := by
  sorry

#check room_width_calculation

end room_width_calculation_l1563_156313


namespace point_on_terminal_side_l1563_156323

theorem point_on_terminal_side (m : ℝ) (α : ℝ) :
  (2 : ℝ) / Real.sqrt (m^2 + 4) = (1 : ℝ) / 3 →
  m = 4 * Real.sqrt 2 ∨ m = -4 * Real.sqrt 2 := by
  sorry

end point_on_terminal_side_l1563_156323


namespace range_of_a_solution_set_l1563_156373

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- Theorem for part I
theorem range_of_a (a : ℝ) :
  (∃ x, f x < 2 * a - 1) ↔ a > 2 :=
sorry

-- Theorem for part II
theorem solution_set :
  {x : ℝ | f x ≥ x^2 - 2*x} = {x : ℝ | -1 ≤ x ∧ x ≤ 2 + Real.sqrt 3} :=
sorry

end range_of_a_solution_set_l1563_156373


namespace quadratic_roots_property_l1563_156378

/-- Given a quadratic function y = x^2 - 1840x + 2009 with roots m and n,
    prove that (m^2 - 1841m + 2009)(n^2 - 1841n + 2009) = 2009 -/
theorem quadratic_roots_property (m n : ℝ) : 
  m^2 - 1840*m + 2009 = 0 →
  n^2 - 1840*n + 2009 = 0 →
  (m^2 - 1841*m + 2009) * (n^2 - 1841*n + 2009) = 2009 := by
  sorry

end quadratic_roots_property_l1563_156378


namespace complex_equation_sum_l1563_156399

theorem complex_equation_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 → (a - 2 * i) * i = b - i → a + b = 1 := by
  sorry

end complex_equation_sum_l1563_156399


namespace f_is_quadratic_l1563_156387

/-- A quadratic equation in x is of the form ax² + bx + c = 0, where a, b, and c are constants, and a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² + x - 4 = 0 -/
def f (x : ℝ) : ℝ := x^2 + x - 4

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by sorry

end f_is_quadratic_l1563_156387


namespace intersection_of_A_and_B_l1563_156386

def A : Set ℝ := {x | -3 ≤ x ∧ x < 4}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 4} := by sorry

end intersection_of_A_and_B_l1563_156386


namespace safari_animal_ratio_l1563_156352

theorem safari_animal_ratio :
  let antelopes : ℕ := 80
  let rabbits : ℕ := antelopes + 34
  let hyenas : ℕ := antelopes + rabbits - 42
  let wild_dogs : ℕ := hyenas + 50
  let total_animals : ℕ := 605
  let leopards : ℕ := total_animals - (antelopes + rabbits + hyenas + wild_dogs)
  leopards * 2 = rabbits :=
by sorry

end safari_animal_ratio_l1563_156352


namespace symmetric_point_correct_l1563_156382

/-- Given a point (x, y) and a line y = mx + b, 
    returns the symmetric point with respect to the line -/
def symmetricPoint (x y m b : ℝ) : ℝ × ℝ := sorry

/-- The line of symmetry y = x - 1 -/
def lineOfSymmetry : ℝ → ℝ := fun x ↦ x - 1

theorem symmetric_point_correct : 
  symmetricPoint (-1) 2 1 (-1) = (3, -2) := by sorry

end symmetric_point_correct_l1563_156382


namespace quadratic_equation_solution_l1563_156326

theorem quadratic_equation_solution :
  ∀ x : ℝ, (x - 2)^2 - 4 = 0 ↔ x = 4 ∨ x = 0 := by
sorry

end quadratic_equation_solution_l1563_156326


namespace mildred_blocks_l1563_156355

theorem mildred_blocks (initial_blocks found_blocks : ℕ) : 
  initial_blocks = 2 → found_blocks = 84 → initial_blocks + found_blocks = 86 := by
  sorry

end mildred_blocks_l1563_156355


namespace sin_negative_1920_degrees_l1563_156369

theorem sin_negative_1920_degrees : 
  Real.sin ((-1920 : ℝ) * π / 180) = -Real.sqrt 3 / 2 := by sorry

end sin_negative_1920_degrees_l1563_156369


namespace equation_solution_l1563_156344

theorem equation_solution :
  let f : ℝ → ℝ := λ x => x + 30 / (x - 4)
  ∃ (x₁ x₂ : ℝ), (f x₁ = -8 ∧ f x₂ = -8) ∧ 
    x₁ = -2 + Real.sqrt 6 ∧ x₂ = -2 - Real.sqrt 6 ∧
    ∀ x : ℝ, f x = -8 → (x = x₁ ∨ x = x₂) :=
by sorry

end equation_solution_l1563_156344


namespace cubic_factorization_sum_of_squares_l1563_156351

theorem cubic_factorization_sum_of_squares :
  ∀ (a b c d e f : ℤ),
  (∀ x : ℝ, 1001 * x^3 - 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 3458 :=
by sorry

end cubic_factorization_sum_of_squares_l1563_156351


namespace opposite_of_negative_two_l1563_156385

theorem opposite_of_negative_two :
  ∃ x : ℝ, x + (-2) = 0 ∧ x = 2 := by sorry

end opposite_of_negative_two_l1563_156385


namespace function_inequality_l1563_156389

-- Define the condition (1-x)/f'(x) ≥ 0
def condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (1 - x) / (deriv f x) ≥ 0

-- State the theorem
theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) (h : condition f) :
  f 0 + f 2 < 2 * f 1 := by
  sorry

end function_inequality_l1563_156389


namespace trig_calculation_l1563_156398

theorem trig_calculation : 
  (6 * (Real.tan (45 * π / 180))) - (2 * (Real.cos (60 * π / 180))) = 5 := by
  sorry

end trig_calculation_l1563_156398


namespace tv_sale_effect_l1563_156312

-- Define the price reduction percentage
def price_reduction : ℝ := 0.18

-- Define the sales increase percentage
def sales_increase : ℝ := 0.88

-- Define the net effect on sale value
def net_effect : ℝ := 0.5416

-- Theorem statement
theorem tv_sale_effect :
  let new_price_factor := 1 - price_reduction
  let new_sales_factor := 1 + sales_increase
  (new_price_factor * new_sales_factor - 1) = net_effect := by sorry

end tv_sale_effect_l1563_156312


namespace max_intersecting_chords_2017_l1563_156383

/-- The maximum number of intersecting chords for a circle with n points -/
def max_intersecting_chords (n : ℕ) : ℕ :=
  let k := (n - 1) / 2
  k * (n - 1 - k) + (n - 1)

/-- Theorem stating the maximum number of intersecting chords for 2017 points -/
theorem max_intersecting_chords_2017 :
  max_intersecting_chords 2017 = 1018080 := by
  sorry

#eval max_intersecting_chords 2017

end max_intersecting_chords_2017_l1563_156383


namespace paper_folding_l1563_156311

theorem paper_folding (s : ℝ) (x : ℝ) :
  s^2 = 18 →
  (1/2) * x^2 = 18 - x^2 →
  ∃ d : ℝ, d = 2 * Real.sqrt 6 ∧ d^2 = 2 * x^2 :=
by sorry

end paper_folding_l1563_156311


namespace smallest_number_of_eggs_l1563_156346

theorem smallest_number_of_eggs (n : ℕ) (c : ℕ) : 
  n > 150 →
  n = 15 * c - 6 →
  c ≥ 11 →
  (∀ m : ℕ, m > 150 ∧ (∃ k : ℕ, m = 15 * k - 6) → m ≥ n) →
  n = 159 :=
by sorry

end smallest_number_of_eggs_l1563_156346


namespace rowing_time_ratio_l1563_156365

theorem rowing_time_ratio (man_speed stream_speed : ℝ) 
  (h1 : man_speed = 36)
  (h2 : stream_speed = 18) :
  (man_speed - stream_speed) / (man_speed + stream_speed) = 1 / 3 := by
  sorry

end rowing_time_ratio_l1563_156365


namespace balls_removed_l1563_156353

def initial_balls : ℕ := 8
def current_balls : ℕ := 6

theorem balls_removed : initial_balls - current_balls = 2 := by
  sorry

end balls_removed_l1563_156353


namespace melissa_commission_l1563_156334

/-- Calculates the commission earned by Melissa based on vehicle sales --/
def calculate_commission (coupe_price suv_price luxury_sedan_price motorcycle_price truck_price : ℕ)
  (coupe_sold suv_sold luxury_sedan_sold motorcycle_sold truck_sold : ℕ) : ℕ :=
  let total_sales := coupe_price * coupe_sold + (2 * coupe_price) * suv_sold +
                     luxury_sedan_price * luxury_sedan_sold + motorcycle_price * motorcycle_sold +
                     truck_price * truck_sold
  let total_vehicles := coupe_sold + suv_sold + luxury_sedan_sold + motorcycle_sold + truck_sold
  let commission_rate := if total_vehicles ≤ 2 then 2
                         else if total_vehicles ≤ 4 then 25
                         else 3
  (total_sales * commission_rate) / 100

theorem melissa_commission :
  calculate_commission 30000 60000 80000 15000 40000 3 2 1 4 2 = 12900 :=
by sorry

end melissa_commission_l1563_156334


namespace cost_per_person_l1563_156308

def total_cost : ℚ := 12100
def num_people : ℕ := 11

theorem cost_per_person :
  total_cost / num_people = 1100 :=
sorry

end cost_per_person_l1563_156308


namespace third_group_men_count_l1563_156370

/-- Represents the work rate of a person -/
structure WorkRate where
  rate : ℝ
  rate_pos : rate > 0

/-- Represents a group of workers -/
structure WorkGroup where
  men : ℕ
  women : ℕ

/-- Calculates the total work rate of a group -/
def totalWorkRate (m w : WorkRate) (g : WorkGroup) : ℝ :=
  g.men * m.rate + g.women * w.rate

theorem third_group_men_count 
  (m w : WorkRate) 
  (g1 g2 : WorkGroup) 
  (h1 : totalWorkRate m w g1 = totalWorkRate m w g2)
  (h2 : g1.men = 3 ∧ g1.women = 8)
  (h3 : g2.men = 6 ∧ g2.women = 2)
  (g3 : WorkGroup)
  (h4 : g3.women = 3)
  (h5 : totalWorkRate m w g3 = 0.5 * totalWorkRate m w g1) :
  g3.men = 2 := by
sorry

end third_group_men_count_l1563_156370


namespace no_real_solution_cubic_equation_l1563_156376

theorem no_real_solution_cubic_equation :
  ∀ x : ℝ, x > 0 → 4 * x^(1/3) - 3 * (x / x^(2/3)) ≠ 10 + 2 * x^(1/3) + x^(2/3) :=
by
  sorry

end no_real_solution_cubic_equation_l1563_156376


namespace max_snacks_l1563_156349

theorem max_snacks (S : ℕ) : 
  (∀ n : ℕ, n ≤ S → n > 6 * 18 ∧ n < 7 * 18) → 
  S = 125 := by
  sorry

end max_snacks_l1563_156349


namespace train_crossing_time_l1563_156303

/-- Proves that a train 600 meters long, traveling at 144 km/hr, takes 15 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 600 →
  train_speed_kmh = 144 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 15 := by
  sorry


end train_crossing_time_l1563_156303


namespace sqrt_88200_simplification_l1563_156375

theorem sqrt_88200_simplification : Real.sqrt 88200 = 882 * Real.sqrt 10 := by
  sorry

end sqrt_88200_simplification_l1563_156375


namespace ellipsoid_sum_center_axes_l1563_156390

/-- The equation of a tilted three-dimensional ellipsoid -/
def ellipsoid_equation (x y z x₀ y₀ z₀ A B C : ℝ) : Prop :=
  (x - x₀)^2 / A^2 + (y - y₀)^2 / B^2 + (z - z₀)^2 / C^2 = 1

/-- Theorem: Sum of center coordinates and semi-major axes lengths -/
theorem ellipsoid_sum_center_axes :
  ∀ (x₀ y₀ z₀ A B C : ℝ),
  ellipsoid_equation x y z x₀ y₀ z₀ A B C →
  x₀ = -2 →
  y₀ = 3 →
  z₀ = 1 →
  A = 6 →
  B = 4 →
  C = 2 →
  x₀ + y₀ + z₀ + A + B + C = 14 :=
by sorry

end ellipsoid_sum_center_axes_l1563_156390


namespace system_solution_l1563_156321

theorem system_solution :
  ∀ (x y z : ℝ),
    (x + 1) * y * z = 12 ∧
    (y + 1) * z * x = 4 ∧
    (z + 1) * x * y = 4 →
    ((x = 2 ∧ y = -2 ∧ z = -2) ∨ (x = 1/3 ∧ y = 3 ∧ z = 3)) :=
by sorry

end system_solution_l1563_156321


namespace f_extremum_f_two_zeros_harmonic_sum_bound_l1563_156357

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - a / x + a * Real.log (1 / x)

theorem f_extremum :
  let f₁ := f 1
  (∃ x₀ > 0, ∀ x > 0, f₁ x ≤ f₁ x₀) ∧
  f₁ 1 = 0 ∧
  (¬∃ x₀ > 0, ∀ x > 0, f₁ x ≥ f₁ x₀) := by sorry

theorem f_two_zeros (a : ℝ) :
  (∃ x y, 1 / Real.exp 1 < x ∧ x < y ∧ y < Real.exp 1 ∧ f a x = 0 ∧ f a y = 0) ↔
  (Real.exp 1 / (Real.exp 1 + 1) < a ∧ a < 1) := by sorry

theorem harmonic_sum_bound (n : ℕ) (hn : n ≥ 3) :
  Real.log ((n + 1) / 3) < (Finset.range (n - 2)).sum (λ i => 1 / (i + 3 : ℝ)) := by sorry

end f_extremum_f_two_zeros_harmonic_sum_bound_l1563_156357
