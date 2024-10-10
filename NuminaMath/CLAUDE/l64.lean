import Mathlib

namespace simple_interest_time_period_l64_6475

/-- Theorem: Simple Interest Time Period Calculation -/
theorem simple_interest_time_period 
  (P : ℝ) -- Principal amount
  (r : ℝ) -- Rate of interest per annum
  (t : ℝ) -- Time period in years
  (h1 : r = 12) -- Given rate is 12% per annum
  (h2 : (P * r * t) / 100 = (6/5) * P) -- Simple interest equation
  : t = 10 := by
sorry

end simple_interest_time_period_l64_6475


namespace x_equation_implies_polynomial_value_l64_6403

theorem x_equation_implies_polynomial_value :
  ∀ x : ℝ, x + 1/x = 2 → x^9 - 5*x^5 + x = -3 := by
  sorry

end x_equation_implies_polynomial_value_l64_6403


namespace smallest_possible_b_l64_6490

theorem smallest_possible_b (a b : ℝ) : 
  (1 < a ∧ a < b) →
  (1 + a ≤ b) →
  (1/a + 1/b ≤ 1) →
  b ≥ (3 + Real.sqrt 5) / 2 :=
by sorry

end smallest_possible_b_l64_6490


namespace product_equals_expansion_l64_6495

-- Define the binomials
def binomial1 (x : ℝ) : ℝ := 4 * x + 3
def binomial2 (x : ℝ) : ℝ := 2 * x - 7

-- Define the product using the distributive property
def product (x : ℝ) : ℝ := binomial1 x * binomial2 x

-- Theorem stating that the product equals the expanded form
theorem product_equals_expansion (x : ℝ) : 
  product x = 8 * x^2 - 22 * x - 21 := by sorry

end product_equals_expansion_l64_6495


namespace smallest_number_with_conditions_l64_6450

theorem smallest_number_with_conditions : ∃ A : ℕ,
  (A % 10 = 6) ∧
  (4 * A = 6 * (A / 10)) ∧
  (∀ B : ℕ, B < A → ¬(B % 10 = 6 ∧ 4 * B = 6 * (B / 10))) ∧
  A = 153846 := by
sorry

end smallest_number_with_conditions_l64_6450


namespace club_members_proof_l64_6485

theorem club_members_proof (total : ℕ) (left_handed : ℕ) (rock_fans : ℕ) (right_handed_non_rock : ℕ) 
  (h1 : total = 25)
  (h2 : left_handed = 10)
  (h3 : rock_fans = 18)
  (h4 : right_handed_non_rock = 4)
  (h5 : left_handed + (total - left_handed) = total) :
  ∃ x : ℕ, x = 7 ∧ 
    x ≤ left_handed ∧ 
    x ≤ rock_fans ∧
    x + (left_handed - x) + (rock_fans - x) + right_handed_non_rock = total :=
by
  sorry

#check club_members_proof

end club_members_proof_l64_6485


namespace income_ratio_proof_l64_6496

/-- Given two persons P1 and P2 with the following conditions:
    1. The ratio of their expenditures is 3:2
    2. Each saves 2200 at the end of the year
    3. The income of P1 is 5500
    Prove that the ratio of their incomes is 5:4 -/
theorem income_ratio_proof (income_P1 income_P2 expenditure_P1 expenditure_P2 : ℕ) : 
  income_P1 = 5500 →
  expenditure_P1 = income_P1 - 2200 →
  expenditure_P2 = income_P2 - 2200 →
  3 * expenditure_P2 = 2 * expenditure_P1 →
  5 * income_P2 = 4 * income_P1 := by
  sorry

#check income_ratio_proof

end income_ratio_proof_l64_6496


namespace michaels_regular_hours_l64_6469

/-- Proves that given the conditions of Michael's work schedule and earnings,
    the number of regular hours worked before overtime is 40. -/
theorem michaels_regular_hours
  (regular_rate : ℝ)
  (total_earnings : ℝ)
  (total_hours : ℝ)
  (h1 : regular_rate = 7)
  (h2 : total_earnings = 320)
  (h3 : total_hours = 42.857142857142854)
  : ∃ (regular_hours : ℝ),
    regular_hours = 40 ∧
    regular_hours * regular_rate +
    (total_hours - regular_hours) * (2 * regular_rate) = total_earnings :=
by sorry

end michaels_regular_hours_l64_6469


namespace crayons_division_l64_6454

/-- Given 24 crayons equally divided among 3 people, prove that each person gets 8 crayons. -/
theorem crayons_division (total_crayons : ℕ) (num_people : ℕ) (crayons_per_person : ℕ) :
  total_crayons = 24 →
  num_people = 3 →
  crayons_per_person = total_crayons / num_people →
  crayons_per_person = 8 := by
  sorry

end crayons_division_l64_6454


namespace difference_of_squares_division_l64_6407

theorem difference_of_squares_division : (204^2 - 196^2) / 16 = 200 := by
  sorry

end difference_of_squares_division_l64_6407


namespace simplify_power_expression_l64_6416

theorem simplify_power_expression (y : ℝ) : (3 * y^4)^2 = 9 * y^8 := by
  sorry

end simplify_power_expression_l64_6416


namespace expand_expression_l64_6492

theorem expand_expression (x : ℝ) : 5 * (-3 * x^3 + 4 * x^2 - 2 * x + 7) = -15 * x^3 + 20 * x^2 - 10 * x + 35 := by
  sorry

end expand_expression_l64_6492


namespace mrs_hilt_shopping_l64_6449

/-- Mrs. Hilt's shopping problem -/
theorem mrs_hilt_shopping (pencil_cost candy_cost remaining_money : ℕ) 
  (h1 : pencil_cost = 20)
  (h2 : candy_cost = 5)
  (h3 : remaining_money = 18) :
  pencil_cost + candy_cost + remaining_money = 43 :=
by sorry

end mrs_hilt_shopping_l64_6449


namespace halfway_between_one_eighth_and_one_third_l64_6467

theorem halfway_between_one_eighth_and_one_third :
  (1 / 8 + 1 / 3) / 2 = 11 / 48 := by
  sorry

end halfway_between_one_eighth_and_one_third_l64_6467


namespace population_growth_proof_l64_6438

/-- Represents the annual growth rate of the population -/
def growth_rate : ℝ := 0.20

/-- Represents the population after one year of growth -/
def final_population : ℝ := 12000

/-- Represents the initial population before growth -/
def initial_population : ℝ := 10000

/-- Theorem stating that if a population grows by 20% in one year to reach 12,000,
    then the initial population was 10,000 -/
theorem population_growth_proof :
  final_population = initial_population * (1 + growth_rate) :=
by sorry

end population_growth_proof_l64_6438


namespace train_travel_time_equation_l64_6442

/-- Proves that the equation for the difference in travel times between two trains is correct -/
theorem train_travel_time_equation (x : ℝ) (h : x > 0) : 
  700 / x - 700 / (2.8 * x) = 3.6 := by
  sorry

end train_travel_time_equation_l64_6442


namespace right_triangle_side_length_l64_6425

/-- In a right-angled triangle XYZ, given the following conditions:
  - ∠X = 90°
  - YZ = 20
  - tan Z = 3 sin Y
  Prove that XY = (40√2) / 3 -/
theorem right_triangle_side_length (X Y Z : ℝ) (h1 : X^2 + Y^2 = Z^2) 
  (h2 : Z = 20) (h3 : Real.tan X = 3 * Real.sin Y) : 
  Y = (40 * Real.sqrt 2) / 3 := by
  sorry

end right_triangle_side_length_l64_6425


namespace new_class_mean_l64_6497

theorem new_class_mean (total_students : ℕ) (initial_students : ℕ) (later_students : ℕ)
  (initial_mean : ℚ) (later_mean : ℚ) :
  total_students = initial_students + later_students →
  initial_students = 30 →
  later_students = 6 →
  initial_mean = 72 / 100 →
  later_mean = 78 / 100 →
  (initial_students * initial_mean + later_students * later_mean) / total_students = 73 / 100 :=
by sorry

end new_class_mean_l64_6497


namespace largest_digit_divisible_by_six_l64_6426

theorem largest_digit_divisible_by_six : 
  ∀ N : ℕ, N ≤ 9 → (5678 * 10 + N) % 6 = 0 → N ≤ 4 :=
by
  sorry

end largest_digit_divisible_by_six_l64_6426


namespace admission_criteria_correct_l64_6470

/-- Represents the admission score criteria for art students in a high school. -/
structure AdmissionCriteria where
  x : ℝ  -- Professional score
  y : ℝ  -- Total score of liberal arts
  z : ℝ  -- Physical education score

/-- Defines the correct admission criteria based on the given conditions. -/
def correct_criteria (c : AdmissionCriteria) : Prop :=
  c.x ≥ 95 ∧ c.y > 380 ∧ c.z > 45

/-- Theorem stating that the given inequalities correctly represent the admission criteria. -/
theorem admission_criteria_correct (c : AdmissionCriteria) :
  (c.x ≥ 95 ∧ c.y > 380 ∧ c.z > 45) ↔ correct_criteria c :=
by sorry

end admission_criteria_correct_l64_6470


namespace intersection_M_N_l64_6482

def M : Set ℝ := {x | -2 < x ∧ x < 3}
def N : Set ℝ := {-2, -1, 0, 1}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by sorry

end intersection_M_N_l64_6482


namespace line_passes_through_fixed_point_and_max_distance_and_segment_length_l64_6484

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := 2*x + (1+m)*y + 2*m = 0

-- Define point P
def P : ℝ × ℝ := (-1, 0)

-- Define point Q
def Q : ℝ × ℝ := (1, -2)

-- Define point N
def N : ℝ × ℝ := (2, 1)

theorem line_passes_through_fixed_point_and_max_distance_and_segment_length :
  (∀ m : ℝ, line_l m Q.1 Q.2) ∧
  (∃ d : ℝ, d = 2 * Real.sqrt 2 ∧ 
    ∀ m : ℝ, ∀ x y : ℝ, line_l m x y → Real.sqrt ((x - P.1)^2 + (y - P.2)^2) ≤ d) ∧
  (∀ M : ℝ × ℝ, (M.1 - 0)^2 + (M.2 + 1)^2 = 2 →
    Real.sqrt 2 ≤ Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) ≤ 3 * Real.sqrt 2) :=
by sorry

end line_passes_through_fixed_point_and_max_distance_and_segment_length_l64_6484


namespace wolf_sheep_eating_time_l64_6431

/-- If 7 wolves eat 7 sheep in 7 days, then 9 wolves will eat 9 sheep in 7 days. -/
theorem wolf_sheep_eating_time (initial_wolves initial_sheep initial_days : ℕ) 
  (new_wolves new_sheep : ℕ) : 
  initial_wolves = 7 → initial_sheep = 7 → initial_days = 7 →
  new_wolves = 9 → new_sheep = 9 →
  initial_wolves * initial_sheep * new_days = new_wolves * new_sheep * initial_days →
  new_days = 7 :=
by
  sorry

#check wolf_sheep_eating_time

end wolf_sheep_eating_time_l64_6431


namespace number_of_students_l64_6424

theorem number_of_students (group_size : ℕ) (num_groups : ℕ) (h1 : group_size = 5) (h2 : num_groups = 6) :
  group_size * num_groups = 30 := by
  sorry

end number_of_students_l64_6424


namespace sean_initial_blocks_l64_6491

/-- The number of blocks Sean had initially -/
def initial_blocks : ℕ := sorry

/-- The number of blocks eaten by the hippopotamus -/
def blocks_eaten : ℕ := 29

/-- The number of blocks remaining after the hippopotamus ate some -/
def blocks_remaining : ℕ := 26

/-- Theorem stating that Sean initially had 55 blocks -/
theorem sean_initial_blocks : initial_blocks = 55 := by sorry

end sean_initial_blocks_l64_6491


namespace urn_problem_l64_6476

theorem urn_problem (N : ℕ) : 
  let urn1_red : ℕ := 5
  let urn1_yellow : ℕ := 8
  let urn2_red : ℕ := 18
  let urn2_yellow : ℕ := N
  let total1 : ℕ := urn1_red + urn1_yellow
  let total2 : ℕ := urn2_red + urn2_yellow
  let prob_same_color : ℚ := (urn1_red / total1) * (urn2_red / total2) + 
                             (urn1_yellow / total1) * (urn2_yellow / total2)
  prob_same_color = 62/100 → N = 59 := by
sorry


end urn_problem_l64_6476


namespace circle_centers_distance_l64_6400

/-- Given a right triangle XYZ with side lengths XY = 7, XZ = 24, and YZ = 25,
    and two circles: one centered at O tangent to XZ at Z and passing through Y,
    and another centered at P tangent to XY at Y and passing through Z,
    prove that the length of OP is 25. -/
theorem circle_centers_distance (X Y Z O P : ℝ × ℝ) : 
  -- Right triangle XYZ with given side lengths
  (Y.1 - X.1)^2 + (Y.2 - X.2)^2 = 7^2 →
  (Z.1 - X.1)^2 + (Z.2 - X.2)^2 = 24^2 →
  (Z.1 - Y.1)^2 + (Z.2 - Y.2)^2 = 25^2 →
  -- Circle O is tangent to XZ at Z and passes through Y
  ((O.1 - Z.1)^2 + (O.2 - Z.2)^2 = (O.1 - Y.1)^2 + (O.2 - Y.2)^2) →
  ((O.1 - Z.1) * (Z.1 - X.1) + (O.2 - Z.2) * (Z.2 - X.2) = 0) →
  -- Circle P is tangent to XY at Y and passes through Z
  ((P.1 - Y.1)^2 + (P.2 - Y.2)^2 = (P.1 - Z.1)^2 + (P.2 - Z.2)^2) →
  ((P.1 - Y.1) * (Y.1 - X.1) + (P.2 - Y.2) * (Y.2 - X.2) = 0) →
  -- The distance between O and P is 25
  (O.1 - P.1)^2 + (O.2 - P.2)^2 = 25^2 := by
sorry


end circle_centers_distance_l64_6400


namespace c_share_is_63_l64_6430

/-- Represents a person renting the pasture -/
structure Renter where
  oxen : ℕ
  months : ℕ

/-- Calculates the share of rent for a given renter -/
def calculateShare (renter : Renter) (totalRent : ℕ) (totalOxMonths : ℕ) : ℚ :=
  (renter.oxen * renter.months : ℚ) / totalOxMonths * totalRent

theorem c_share_is_63 (a b c : Renter) (totalRent : ℕ) :
  a.oxen = 10 →
  a.months = 7 →
  b.oxen = 12 →
  b.months = 5 →
  c.oxen = 15 →
  c.months = 3 →
  totalRent = 245 →
  calculateShare c totalRent (a.oxen * a.months + b.oxen * b.months + c.oxen * c.months) = 63 := by
  sorry

#eval calculateShare (Renter.mk 15 3) 245 175

end c_share_is_63_l64_6430


namespace jury_seating_arrangements_l64_6480

/-- Represents the number of jury members -/
def n : ℕ := 12

/-- Represents the number of jury members excluding Nikolai Nikolaevich and the person whose seat he took -/
def m : ℕ := n - 2

/-- A function that calculates the number of distinct seating arrangements -/
def seating_arrangements (n : ℕ) : ℕ := 2^(n - 2)

/-- Theorem stating that the number of distinct seating arrangements for 12 jury members is 2^10 -/
theorem jury_seating_arrangements :
  seating_arrangements n = 2^m :=
by sorry

end jury_seating_arrangements_l64_6480


namespace geometric_sequence_first_term_l64_6494

theorem geometric_sequence_first_term 
  (a : ℝ) -- first term
  (r : ℝ) -- common ratio
  (h1 : a * r^5 = Nat.factorial 9)  -- 6th term is 9!
  (h2 : a * r^8 = Nat.factorial 10) -- 9th term is 10!
  : a = (Nat.factorial 9) / (10 ^ (5/3)) :=
by
  sorry

end geometric_sequence_first_term_l64_6494


namespace new_profit_percentage_l64_6448

/-- Given the initial and new manufacturing costs, and the initial profit percentage,
    calculate the new profit percentage of the selling price. -/
theorem new_profit_percentage
  (initial_cost : ℝ)
  (new_cost : ℝ)
  (initial_profit_percentage : ℝ)
  (h_initial_cost : initial_cost = 70)
  (h_new_cost : new_cost = 50)
  (h_initial_profit_percentage : initial_profit_percentage = 30)
  : (1 - new_cost / (initial_cost / (1 - initial_profit_percentage / 100))) * 100 = 50 := by
  sorry

#check new_profit_percentage

end new_profit_percentage_l64_6448


namespace hyperbola_properties_l64_6474

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the point that lies on the hyperbola
def point_on_hyperbola (a b : ℝ) : Prop :=
  hyperbola a b (Real.sqrt 6) (Real.sqrt 3)

-- Define the focus of the hyperbola
def focus (a b : ℝ) : Prop :=
  hyperbola a b (-Real.sqrt 6) 0

-- Define the intersection line
def intersection_line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 2

-- Theorem statement
theorem hyperbola_properties :
  ∀ a b : ℝ,
  point_on_hyperbola a b →
  focus a b →
  (∀ x y : ℝ, hyperbola a b x y ↔ x^2 - y^2 = 3) ∧
  (∀ k : ℝ, (∃ x₁ x₂ y₁ y₂ : ℝ, 
    x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    hyperbola a b x₁ y₁ ∧ hyperbola a b x₂ y₂ ∧
    intersection_line k x₁ y₁ ∧ intersection_line k x₂ y₂)
    ↔ -Real.sqrt (21 / 9) < k ∧ k < -1) :=
sorry

end hyperbola_properties_l64_6474


namespace condition_necessary_not_sufficient_l64_6443

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x = -2 → x^2 = 4) ∧
  ¬(∀ x : ℝ, x^2 = 4 → x = -2) := by
  sorry

end condition_necessary_not_sufficient_l64_6443


namespace perpendicular_line_x_intercept_l64_6423

/-- Given a line L1 defined by 3x - 2y = 6, prove that a line L2 perpendicular to L1
    with y-intercept 2 has x-intercept 3. -/
theorem perpendicular_line_x_intercept :
  ∀ (L1 L2 : Set (ℝ × ℝ)),
  (∀ (x y : ℝ), (x, y) ∈ L1 ↔ 3 * x - 2 * y = 6) →
  (∃ (m : ℝ), ∀ (x y : ℝ), (x, y) ∈ L2 ↔ y = m * x + 2) →
  (∀ (x1 y1 x2 y2 : ℝ), (x1, y1) ∈ L1 → (x2, y2) ∈ L2 → 
    (x2 - x1) * (3 * (y2 - y1) + 2 * (x2 - x1)) = 0) →
  (3, 0) ∈ L2 :=
by sorry

end perpendicular_line_x_intercept_l64_6423


namespace horner_method_v3_l64_6471

/-- The polynomial f(x) = 2 + 0.35x + 1.8x² - 3.66x³ + 6x⁴ - 5.2x⁵ + x⁶ -/
def f (x : ℝ) : ℝ := 2 + 0.35*x + 1.8*x^2 - 3.66*x^3 + 6*x^4 - 5.2*x^5 + x^6

/-- Horner's method for calculating v₃ -/
def horner_v3 (x : ℝ) : ℝ :=
  let v0 : ℝ := 1
  let v1 : ℝ := v0 * x - 5.2
  let v2 : ℝ := v1 * x + 6
  v2 * x - 3.66

theorem horner_method_v3 :
  horner_v3 (-1) = -15.86 := by sorry

end horner_method_v3_l64_6471


namespace ten_people_two_vip_seats_l64_6466

/-- The number of ways to arrange n people around a round table with k marked VIP seats,
    where arrangements are considered the same if rotations preserve who sits in the VIP seats -/
def roundTableArrangements (n : ℕ) (k : ℕ) : ℕ :=
  (n.choose k) * (n - k).factorial

/-- Theorem stating that for 10 people and 2 VIP seats, there are 1,814,400 arrangements -/
theorem ten_people_two_vip_seats :
  roundTableArrangements 10 2 = 1814400 := by
  sorry

#eval roundTableArrangements 10 2

end ten_people_two_vip_seats_l64_6466


namespace ticket_123123123_is_red_l64_6402

/-- Represents the color of a lottery ticket -/
inductive TicketColor
| Red
| Blue
| Green

/-- Represents a 9-digit lottery ticket number -/
def TicketNumber := Fin 9 → Fin 3

/-- The coloring function for tickets -/
def ticketColor : TicketNumber → TicketColor := sorry

/-- Check if two tickets differ in all places -/
def differInAllPlaces (t1 t2 : TicketNumber) : Prop :=
  ∀ i : Fin 9, t1 i ≠ t2 i

/-- The main theorem to prove -/
theorem ticket_123123123_is_red :
  (∀ t1 t2 : TicketNumber, differInAllPlaces t1 t2 → ticketColor t1 ≠ ticketColor t2) →
  ticketColor (λ i => if i.val % 3 = 0 then 0 else if i.val % 3 = 1 then 1 else 2) = TicketColor.Red →
  ticketColor (λ _ => 1) = TicketColor.Green →
  ticketColor (λ i => i.val % 3) = TicketColor.Red :=
sorry

end ticket_123123123_is_red_l64_6402


namespace min_area_is_zero_l64_6404

/-- Represents a rectangle with one integer dimension and one half-integer dimension -/
structure Rectangle where
  x : ℕ  -- Integer dimension
  y : ℚ  -- Half-integer dimension
  y_half_int : ∃ (n : ℕ), y = n + 1/2
  perimeter_150 : 2 * (x + y) = 150

/-- The area of a rectangle -/
def area (r : Rectangle) : ℚ :=
  r.x * r.y

/-- Theorem stating that the minimum area of a rectangle with the given conditions is 0 -/
theorem min_area_is_zero :
  ∃ (r : Rectangle), ∀ (s : Rectangle), area r ≤ area s :=
sorry

end min_area_is_zero_l64_6404


namespace integer_roots_of_polynomial_l64_6459

def polynomial (a₂ a₁ : ℤ) (x : ℤ) : ℤ := x^3 + a₂ * x^2 + a₁ * x - 18

def possible_roots : Set ℤ := {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18}

theorem integer_roots_of_polynomial (a₂ a₁ : ℤ) :
  ∀ x : ℤ, polynomial a₂ a₁ x = 0 → x ∈ possible_roots :=
sorry

end integer_roots_of_polynomial_l64_6459


namespace triangle_side_length_l64_6489

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if a = 3, C = 120°, and the area of the triangle is 15√3/4, then c = 7. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  a = 3 → 
  C = 2 * π / 3 → 
  (1/2) * a * b * Real.sin C = (15 * Real.sqrt 3) / 4 →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = 7 := by
  sorry


end triangle_side_length_l64_6489


namespace power_of_two_condition_l64_6478

theorem power_of_two_condition (n : ℕ+) : 
  (∃ k : ℕ, n.val^3 + n.val - 2 = 2^k) ↔ (n.val = 2 ∨ n.val = 5) := by
sorry

end power_of_two_condition_l64_6478


namespace episodes_per_day_l64_6405

/-- Given a TV series with 3 seasons of 20 episodes each, watched over 30 days,
    the number of episodes watched per day is 2. -/
theorem episodes_per_day (seasons : ℕ) (episodes_per_season : ℕ) (total_days : ℕ)
    (h1 : seasons = 3)
    (h2 : episodes_per_season = 20)
    (h3 : total_days = 30) :
    (seasons * episodes_per_season) / total_days = 2 := by
  sorry

end episodes_per_day_l64_6405


namespace rectangle_variability_l64_6460

-- Define the rectangle
structure Rectangle where
  length : ℝ
  width : ℝ

-- Define the perimeter, area, and one side length
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)
def area (r : Rectangle) : ℝ := r.length * r.width
def oneSideLength (r : Rectangle) : ℝ := r.length

-- State the theorem
theorem rectangle_variability (fixedPerimeter : ℝ) (r : Rectangle) 
  (h : perimeter r = fixedPerimeter) :
  ∃ (r' : Rectangle), 
    perimeter r' = fixedPerimeter ∧ 
    area r' ≠ area r ∧
    oneSideLength r' ≠ oneSideLength r :=
sorry

end rectangle_variability_l64_6460


namespace point_symmetry_l64_6477

/-- The line with respect to which we're finding symmetry -/
def symmetry_line (x y : ℝ) : Prop := x - y - 1 = 0

/-- Definition of symmetry with respect to a line -/
def symmetric_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  symmetry_line midpoint_x midpoint_y ∧
  (y₂ - y₁) / (x₂ - x₁) = -1

theorem point_symmetry :
  symmetric_points (-1) 1 2 (-2) := by sorry

end point_symmetry_l64_6477


namespace simple_interest_calculation_l64_6409

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : principal = 10000)
  (h2 : rate = 0.04)
  (h3 : time = 1) :
  principal * rate * time = 400 :=
by sorry

end simple_interest_calculation_l64_6409


namespace arithmetic_mean_of_fractions_l64_6408

theorem arithmetic_mean_of_fractions (x a : ℝ) (hx : x ≠ 0) :
  (1 / 2) * ((2 * x + a) / x + (2 * x - a) / x) = 2 := by
  sorry

end arithmetic_mean_of_fractions_l64_6408


namespace investigator_strategy_equivalence_l64_6473

/-- Represents the investigator's questioning strategy -/
structure InvestigatorStrategy where
  num_questions : ℕ
  max_lie : ℕ

/-- Defines the original strategy with all truthful answers -/
def original_strategy : InvestigatorStrategy :=
  { num_questions := 91
  , max_lie := 0 }

/-- Defines the new strategy allowing for one possible lie -/
def new_strategy : InvestigatorStrategy :=
  { num_questions := 105
  , max_lie := 1 }

/-- Represents the information obtained from questioning -/
def Information : Type := Unit

/-- Function to obtain information given a strategy -/
def obtain_information (strategy : InvestigatorStrategy) : Information := sorry

theorem investigator_strategy_equivalence :
  obtain_information original_strategy = obtain_information new_strategy :=
by sorry

end investigator_strategy_equivalence_l64_6473


namespace trigonometric_identity_l64_6421

theorem trigonometric_identity (α : Real) : 
  4.10 * (Real.cos (π/4 - α))^2 - (Real.cos (π/3 + α))^2 - 
  Real.cos (5*π/12) * Real.sin (5*π/12 - 2*α) = Real.sin (2*α) := by
  sorry

end trigonometric_identity_l64_6421


namespace original_item_is_mirror_l64_6462

-- Define the code language as a function
def code (x : String) : String :=
  match x with
  | "item" => "pencil"
  | "pencil" => "mirror"
  | "mirror" => "board"
  | _ => x

-- Define the useful item to write on paper
def useful_item : String := "pencil"

-- Define the coded useful item
def coded_useful_item : String := "2"

-- Theorem to prove
theorem original_item_is_mirror :
  (code useful_item = coded_useful_item) → 
  (∃ x, code x = useful_item ∧ code (code x) = coded_useful_item) →
  (∃ y, code y = useful_item ∧ y = "mirror") :=
by sorry

end original_item_is_mirror_l64_6462


namespace stone_distance_l64_6410

theorem stone_distance (n : ℕ) (total_distance : ℝ) : 
  n = 31 → 
  n % 2 = 1 → 
  total_distance = 4.8 → 
  (2 * (n / 2) * (n / 2 + 1) / 2) * (total_distance / (2 * (n / 2) * (n / 2 + 1) / 2)) = 0.02 := by
  sorry

end stone_distance_l64_6410


namespace division_remainder_proof_l64_6452

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (h1 : dividend = 2944) (h2 : divisor = 72) (h3 : quotient = 40) :
  dividend - divisor * quotient = 64 := by
sorry

end division_remainder_proof_l64_6452


namespace indefinite_integral_of_3x_squared_plus_1_l64_6444

theorem indefinite_integral_of_3x_squared_plus_1 (x : ℝ) (C : ℝ) :
  deriv (fun x => x^3 + x + C) x = 3 * x^2 + 1 := by
  sorry

end indefinite_integral_of_3x_squared_plus_1_l64_6444


namespace toy_store_revenue_ratio_l64_6499

theorem toy_store_revenue_ratio :
  ∀ (N D J : ℝ),
  N > 0 →
  N = (2/5) * D →
  D = 3.75 * ((N + J) / 2) →
  J / N = 1/3 :=
by
  sorry

end toy_store_revenue_ratio_l64_6499


namespace complex_cube_root_l64_6453

theorem complex_cube_root (x y : ℕ+) :
  (↑x + ↑y * I : ℂ)^3 = 2 + 11 * I →
  ↑x + ↑y * I = 2 + I :=
by sorry

end complex_cube_root_l64_6453


namespace equation_solution_l64_6406

theorem equation_solution :
  ∃ (x y z u : ℝ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ u > 0 ∧
    -1/x + 1/y + 1/z + 1/u = 2 ∧
    x = 1 ∧ y = 2 ∧ z = 3 ∧ u = 6 := by
  sorry

end equation_solution_l64_6406


namespace least_lcm_value_l64_6465

def lcm_problem (a b c : ℕ) : Prop :=
  (Nat.lcm a b = 40) ∧ 
  (Nat.lcm b c = 21) ∧
  (Nat.lcm a c ≥ 24) ∧
  ∀ x y, (Nat.lcm x y = 40) → (Nat.lcm y c = 21) → (Nat.lcm x c ≥ 24)

theorem least_lcm_value : ∃ a b c, lcm_problem a b c ∧ Nat.lcm a c = 24 :=
sorry

end least_lcm_value_l64_6465


namespace lcm_of_5_6_10_18_l64_6468

theorem lcm_of_5_6_10_18 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 10 18)) = 90 := by
  sorry

end lcm_of_5_6_10_18_l64_6468


namespace negation_of_universal_proposition_l64_6427

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ ∃ x : ℝ, x^2 - x + 2 < 0 :=
by sorry

end negation_of_universal_proposition_l64_6427


namespace find_b_l64_6481

-- Define the relationship between a and b
def inverse_relation (a b : ℝ) : Prop := ∃ k : ℝ, a^2 * Real.sqrt b = k

-- Define the theorem
theorem find_b (a b : ℝ) (h1 : inverse_relation a b) (h2 : a = 2 ∧ b = 81) (h3 : a * b = 48) :
  b = 16 := by
  sorry

end find_b_l64_6481


namespace parabola_equation_l64_6464

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space of the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola in general form ax^2 + bxy + cy^2 + dx + ey + f = 0 -/
structure Parabola where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- The greatest common divisor of the absolute values of all coefficients is 1 -/
def coefficientsAreCoprime (p : Parabola) : Prop :=
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs p.a) (Int.natAbs p.b)) (Int.natAbs p.c)) (Int.natAbs p.d)) (Int.natAbs p.e)) (Int.natAbs p.f) = 1

/-- The theorem stating the equation of the parabola -/
theorem parabola_equation (focus : Point) (directrix : Line) : 
  focus.x = 4 ∧ focus.y = 2 ∧ 
  directrix.a = 2 ∧ directrix.b = 5 ∧ directrix.c = 20 →
  ∃ (p : Parabola), 
    p.a = 25 ∧ p.b = -20 ∧ p.c = 4 ∧ p.d = -152 ∧ p.e = 84 ∧ p.f = -180 ∧
    p.a > 0 ∧
    coefficientsAreCoprime p :=
by sorry

end parabola_equation_l64_6464


namespace square_sum_equals_negative_45_l64_6441

theorem square_sum_equals_negative_45 (x y : ℝ) 
  (h1 : x - 3 * y = 3) 
  (h2 : x * y = -9) : 
  x^2 + 9 * y^2 = -45 := by
sorry

end square_sum_equals_negative_45_l64_6441


namespace minus_one_circle_plus_four_equals_zero_l64_6437

-- Define the new operation ⊕
def circle_plus (a b : ℝ) : ℝ := a * b + b

-- Theorem statement
theorem minus_one_circle_plus_four_equals_zero : 
  circle_plus (-1) 4 = 0 := by sorry

end minus_one_circle_plus_four_equals_zero_l64_6437


namespace mrs_hilt_animal_legs_l64_6458

/-- The number of legs for each animal type -/
def dog_legs : ℕ := 4
def chicken_legs : ℕ := 2
def spider_legs : ℕ := 8
def octopus_legs : ℕ := 8

/-- The number of each animal type Mrs. Hilt saw -/
def dogs_seen : ℕ := 3
def chickens_seen : ℕ := 4
def spiders_seen : ℕ := 2
def octopuses_seen : ℕ := 1

/-- The total number of animal legs Mrs. Hilt saw -/
def total_legs : ℕ := dogs_seen * dog_legs + chickens_seen * chicken_legs + 
                      spiders_seen * spider_legs + octopuses_seen * octopus_legs

theorem mrs_hilt_animal_legs : total_legs = 44 := by
  sorry

end mrs_hilt_animal_legs_l64_6458


namespace balls_after_2010_actions_l64_6401

/-- Represents the state of boxes with balls -/
def BoxState := List Nat

/-- Adds a ball to the first available box and empties boxes to its left -/
def addBall (state : BoxState) : BoxState :=
  match state with
  | [] => [1]
  | (h::t) => if h < 6 then (h+1)::t else 0::addBall t

/-- Performs the ball-adding process n times -/
def performActions (n : Nat) : BoxState :=
  match n with
  | 0 => []
  | n+1 => addBall (performActions n)

/-- Calculates the sum of balls in all boxes -/
def totalBalls (state : BoxState) : Nat :=
  state.sum

/-- The main theorem to prove -/
theorem balls_after_2010_actions :
  totalBalls (performActions 2010) = 16 := by
  sorry

end balls_after_2010_actions_l64_6401


namespace lowest_possible_score_l64_6445

-- Define the parameters of the problem
def mean : ℝ := 60
def std_dev : ℝ := 10
def z_score_95_percentile : ℝ := 1.645

-- Define the function to calculate the score from z-score
def score_from_z (z : ℝ) : ℝ := z * std_dev + mean

-- Define the conditions
def within_top_5_percent (score : ℝ) : Prop := score ≥ score_from_z z_score_95_percentile
def within_2_std_dev (score : ℝ) : Prop := score ≤ mean + 2 * std_dev

-- The theorem to prove
theorem lowest_possible_score :
  ∃ (score : ℕ), 
    (score : ℝ) = ⌈score_from_z z_score_95_percentile⌉ ∧
    within_top_5_percent score ∧
    within_2_std_dev score ∧
    ∀ (s : ℕ), s < score → ¬(within_top_5_percent s ∧ within_2_std_dev s) :=
by sorry

end lowest_possible_score_l64_6445


namespace sequence_properties_l64_6488

def sequence_a (n : ℕ) : ℤ := 2^n - n - 2

def sequence_c (n : ℕ) : ℤ := sequence_a n + n + 2

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → sequence_a (n + 1) = 2 * sequence_a n + n + 1) ∧
  (sequence_a 1 = -1) ∧
  (∀ n : ℕ, n > 0 → sequence_c (n + 1) = 2 * sequence_c n) :=
sorry

end sequence_properties_l64_6488


namespace equation_three_solutions_l64_6412

theorem equation_three_solutions (a : ℝ) :
  (∃! (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (∀ x : ℝ, x^2 * a - 2*x + 1 = 3 * |x| ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃))) ↔
  a = (1 : ℝ) / 4 :=
by sorry

end equation_three_solutions_l64_6412


namespace value_of_c_l64_6472

theorem value_of_c (x b c : ℝ) (h1 : x - 1/x = 2*b) (h2 : x^3 - 1/x^3 = c) : c = 8*b^3 + 6*b := by
  sorry

end value_of_c_l64_6472


namespace work_completion_time_l64_6455

/-- 
Given that:
- A and B can do the same work
- B can do the work in 16 days
- A and B together can do the work in 16/3 days
Prove that A can do the work alone in 8 days
-/
theorem work_completion_time (b_time a_and_b_time : ℚ) 
  (hb : b_time = 16)
  (hab : a_and_b_time = 16 / 3) : 
  ∃ (a_time : ℚ), a_time = 8 := by
  sorry

end work_completion_time_l64_6455


namespace trick_or_treat_duration_l64_6440

/-- The number of hours Tim and his children were out trick or treating -/
def trick_or_treat_hours (num_children : ℕ) (houses_per_hour : ℕ) (treats_per_child_per_house : ℕ) (total_treats : ℕ) : ℕ :=
  total_treats / (num_children * houses_per_hour * treats_per_child_per_house)

/-- Theorem stating that Tim and his children were out for 4 hours -/
theorem trick_or_treat_duration :
  trick_or_treat_hours 3 5 3 180 = 4 := by
  sorry

#eval trick_or_treat_hours 3 5 3 180

end trick_or_treat_duration_l64_6440


namespace point_distance_from_two_l64_6432

theorem point_distance_from_two : ∀ x : ℝ, |x - 2| = 3 → x = -1 ∨ x = 5 := by
  sorry

end point_distance_from_two_l64_6432


namespace ram_independent_time_l64_6419

/-- The number of days Gohul takes to complete the job independently -/
def gohul_days : ℝ := 15

/-- The number of days Ram and Gohul take to complete the job together -/
def combined_days : ℝ := 6

/-- The number of days Ram takes to complete the job independently -/
def ram_days : ℝ := 10

/-- Theorem stating that given Gohul's time and the combined time, Ram's independent time is 10 days -/
theorem ram_independent_time : 
  (1 / ram_days + 1 / gohul_days = 1 / combined_days) → ram_days = 10 := by
  sorry

end ram_independent_time_l64_6419


namespace bisecting_chord_equation_l64_6498

/-- The equation of a line bisecting a chord of a parabola -/
theorem bisecting_chord_equation (x y : ℝ → ℝ) :
  (∀ t, (y t)^2 = 16 * (x t)) →  -- Parabola equation
  (∃ t₁ t₂, t₁ ≠ t₂ ∧ 
    (x t₁ + x t₂) / 2 = 2 ∧ 
    (y t₁ + y t₂) / 2 = 1) →  -- Midpoint condition
  (∃ a b c : ℝ, ∀ t, a * (x t) + b * (y t) + c = 0 ∧ 
    a = 8 ∧ b = -1 ∧ c = -15) := by
sorry

end bisecting_chord_equation_l64_6498


namespace triangle_vertices_l64_6486

-- Define the lines
def d₁ (x y : ℝ) : Prop := 2 * x - y - 2 = 0
def d₂ (x y : ℝ) : Prop := x + y - 4 = 0
def d₃ (x y : ℝ) : Prop := y = 2
def d₄ (x y : ℝ) : Prop := x - 4 * y + 3 = 0

-- Define the points
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 4)
def C : ℝ × ℝ := (5, 2)

-- Define what it means for a line to be a median
def is_median (line : (ℝ → ℝ → Prop)) (triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  sorry

-- Define what it means for a line to be an altitude
def is_altitude (line : (ℝ → ℝ → Prop)) (triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  sorry

theorem triangle_vertices : 
  is_median d₁ (A, B, C) ∧ 
  is_median d₂ (A, B, C) ∧ 
  is_median d₃ (A, B, C) ∧ 
  is_altitude d₄ (A, B, C) → 
  (A = (1, 0) ∧ B = (0, 4) ∧ C = (5, 2)) :=
sorry

end triangle_vertices_l64_6486


namespace tan_sum_problem_l64_6414

theorem tan_sum_problem (α β : ℝ) 
  (h1 : Real.tan (α + 2 * β) = 2) 
  (h2 : Real.tan β = -3) : 
  Real.tan (α + β) = -1 := by
  sorry

end tan_sum_problem_l64_6414


namespace min_value_quadratic_l64_6461

theorem min_value_quadratic (x y : ℝ) : x^2 + 4*x*y + 5*y^2 - 8*x - 6*y ≥ 7 := by
  sorry

end min_value_quadratic_l64_6461


namespace starters_count_theorem_l64_6413

def number_of_players : ℕ := 15
def number_of_starters : ℕ := 5

-- Define a function to calculate the number of ways to choose starters
def choose_starters (total : ℕ) (choose : ℕ) : ℕ := Nat.choose total choose

-- Define a function to calculate the number of ways to choose starters excluding both twins
def choose_starters_excluding_twins (total : ℕ) (choose : ℕ) : ℕ :=
  choose_starters total choose - choose_starters (total - 2) (choose - 2)

theorem starters_count_theorem : 
  choose_starters_excluding_twins number_of_players number_of_starters = 2717 := by
  sorry

end starters_count_theorem_l64_6413


namespace sin_alpha_value_l64_6417

theorem sin_alpha_value (α : Real) : 
  (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3)) ∈ {(x, y) | ∃ r > 0, x = r * Real.cos α ∧ y = r * Real.sin α} → 
  Real.sin α = -1/2 := by
sorry

end sin_alpha_value_l64_6417


namespace lucia_hip_hop_classes_l64_6422

/-- Represents the number of hip-hop classes Lucia takes in a week -/
def hip_hop_classes : ℕ := sorry

/-- Represents the cost of one hip-hop class -/
def hip_hop_cost : ℕ := 10

/-- Represents the number of ballet classes Lucia takes in a week -/
def ballet_classes : ℕ := 2

/-- Represents the cost of one ballet class -/
def ballet_cost : ℕ := 12

/-- Represents the number of jazz classes Lucia takes in a week -/
def jazz_classes : ℕ := 1

/-- Represents the cost of one jazz class -/
def jazz_cost : ℕ := 8

/-- Represents the total cost of Lucia's dance classes in one week -/
def total_cost : ℕ := 52

/-- Theorem stating that Lucia takes 2 hip-hop classes in a week -/
theorem lucia_hip_hop_classes : 
  hip_hop_classes = 2 :=
by sorry

end lucia_hip_hop_classes_l64_6422


namespace first_quadrant_sufficient_not_necessary_l64_6436

-- Define what it means for an angle to be in the first quadrant
def is_first_quadrant (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2

-- Define the condition we're interested in
def condition (α : Real) : Prop := Real.sin α * Real.cos α > 0

-- Theorem statement
theorem first_quadrant_sufficient_not_necessary :
  (∀ α : Real, is_first_quadrant α → condition α) ∧
  (∃ α : Real, condition α ∧ ¬is_first_quadrant α) := by sorry

end first_quadrant_sufficient_not_necessary_l64_6436


namespace sum_of_greatest_b_values_l64_6428

theorem sum_of_greatest_b_values (b : ℝ) : 
  4 * b^4 - 41 * b^2 + 100 = 0 →
  ∃ (b1 b2 : ℝ), b1 ≥ b2 ∧ b2 ≥ 0 ∧ 
    (4 * b1^4 - 41 * b1^2 + 100 = 0) ∧
    (4 * b2^4 - 41 * b2^2 + 100 = 0) ∧
    b1 + b2 = 4.5 ∧
    ∀ (x : ℝ), (4 * x^4 - 41 * x^2 + 100 = 0) → x ≤ b1 :=
by sorry

end sum_of_greatest_b_values_l64_6428


namespace harriett_us_dollars_l64_6463

def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10
def nickel_value : ℚ := 0.05
def penny_value : ℚ := 0.01
def half_dollar_value : ℚ := 0.50
def dollar_coin_value : ℚ := 1.00

def num_quarters : ℕ := 23
def num_dimes : ℕ := 15
def num_nickels : ℕ := 17
def num_pennies : ℕ := 29
def num_half_dollars : ℕ := 6
def num_dollar_coins : ℕ := 10

def total_us_dollars : ℚ := 
  num_quarters * quarter_value + 
  num_dimes * dime_value + 
  num_nickels * nickel_value + 
  num_pennies * penny_value + 
  num_half_dollars * half_dollar_value + 
  num_dollar_coins * dollar_coin_value

theorem harriett_us_dollars : total_us_dollars = 21.39 := by
  sorry

end harriett_us_dollars_l64_6463


namespace power_division_l64_6451

theorem power_division (a : ℝ) : a^6 / a^3 = a^3 := by
  sorry

end power_division_l64_6451


namespace sample_capacity_l64_6415

theorem sample_capacity (frequency : ℕ) (frequency_rate : ℚ) (n : ℕ) 
  (h1 : frequency = 30)
  (h2 : frequency_rate = 1/4)
  (h3 : n = frequency / frequency_rate) :
  n = 120 := by
sorry

end sample_capacity_l64_6415


namespace triangle_angle_C_l64_6434

theorem triangle_angle_C (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → 
  5 * Real.sin A + 2 * Real.cos B = 3 →
  2 * Real.sin B + 5 * Real.tan A = 7 →
  Real.sin C = Real.sin (A + B) := by sorry

end triangle_angle_C_l64_6434


namespace star_calculation_l64_6420

-- Define the * operation
def star (a b : ℤ) : ℤ := a * (a - b)

-- State the theorem
theorem star_calculation : star 2 3 + star (6 - 2) 4 = -2 := by
  sorry

end star_calculation_l64_6420


namespace inverse_99_mod_101_l64_6429

theorem inverse_99_mod_101 : ∃ x : ℕ, x ∈ Finset.range 101 ∧ (99 * x) % 101 = 1 := by
  use 51
  sorry

end inverse_99_mod_101_l64_6429


namespace function_inequality_implies_m_value_l64_6411

/-- Given functions f and g, prove that if f(x) ≥ g(x) holds exactly for x ∈ [-1, 2], then m = 2 -/
theorem function_inequality_implies_m_value (m : ℝ) :
  (∀ x : ℝ, (x^2 - 3*x + m ≥ 2*x^2 - 4*x) ↔ (-1 ≤ x ∧ x ≤ 2)) →
  m = 2 := by
  sorry

end function_inequality_implies_m_value_l64_6411


namespace symmetric_complex_product_l64_6446

theorem symmetric_complex_product (z₁ z₂ : ℂ) :
  z₁ = 2 + I →
  (z₁.re = -z₂.re ∧ z₁.im = z₂.im) →
  z₁ * z₂ = -5 := by
  sorry

end symmetric_complex_product_l64_6446


namespace odd_integer_sum_theorem_l64_6493

/-- The sum of 60 non-consecutive, odd integers starting from -29 in increasing order -/
def oddIntegerSum : ℤ := 5340

/-- The first term of the sequence -/
def firstTerm : ℤ := -29

/-- The number of terms in the sequence -/
def numTerms : ℕ := 60

/-- The common difference between consecutive terms -/
def commonDiff : ℤ := 4

/-- The last term of the sequence -/
def lastTerm : ℤ := firstTerm + (numTerms - 1) * commonDiff

theorem odd_integer_sum_theorem :
  oddIntegerSum = (numTerms : ℤ) * (firstTerm + lastTerm) / 2 :=
sorry

end odd_integer_sum_theorem_l64_6493


namespace manuscript_revision_cost_l64_6439

theorem manuscript_revision_cost (total_pages : ℕ) (revised_once : ℕ) (revised_twice : ℕ)
  (initial_cost_per_page : ℚ) (total_cost : ℚ)
  (h1 : total_pages = 100)
  (h2 : revised_once = 30)
  (h3 : revised_twice = 20)
  (h4 : initial_cost_per_page = 5)
  (h5 : total_cost = 710)
  (h6 : total_pages = revised_once + revised_twice + (total_pages - revised_once - revised_twice)) :
  let revision_cost : ℚ := (total_cost - (initial_cost_per_page * total_pages)) / (revised_once + 2 * revised_twice)
  revision_cost = 3 := by sorry

end manuscript_revision_cost_l64_6439


namespace m_range_l64_6447

-- Define the point M
def M : ℝ × ℝ := (1, 2)

-- Define the proposition p
def p (m : ℝ) : Prop := M.1 - M.2 + m < 0

-- Define the proposition q
def q (m : ℝ) : Prop := m ≠ -2

-- Define the theorem
theorem m_range (m : ℝ) : p m ∧ q m ↔ m ∈ Set.Ioo (-Real.pi) (-2) ∪ Set.Ioo (-2) 1 :=
sorry

end m_range_l64_6447


namespace least_people_second_caterer_cheaper_l64_6487

def first_caterer_cost (people : ℕ) : ℕ := 100 + 15 * people
def second_caterer_cost (people : ℕ) : ℕ := 200 + 12 * people

theorem least_people_second_caterer_cheaper :
  (∀ n : ℕ, n < 34 → first_caterer_cost n ≤ second_caterer_cost n) ∧
  (second_caterer_cost 34 < first_caterer_cost 34) := by
  sorry

end least_people_second_caterer_cheaper_l64_6487


namespace power_tower_mod_500_l64_6418

theorem power_tower_mod_500 : 7^(7^(7^7)) % 500 = 343 := by
  sorry

end power_tower_mod_500_l64_6418


namespace gardening_project_cost_correct_l64_6457

def gardening_project_cost (rose_bushes : ℕ) (rose_bush_cost : ℕ) (fertilizer_cost : ℕ) 
  (gardener_hours : List ℕ) (gardener_rate : ℕ) (soil_volume : ℕ) (soil_cost : ℕ) : ℕ :=
  let bush_total := rose_bushes * rose_bush_cost
  let fertilizer_total := rose_bushes * fertilizer_cost
  let labor_total := (List.sum gardener_hours) * gardener_rate
  let soil_total := soil_volume * soil_cost
  bush_total + fertilizer_total + labor_total + soil_total

theorem gardening_project_cost_correct : 
  gardening_project_cost 20 150 25 [6, 5, 4, 7] 30 100 5 = 4660 := by
  sorry

end gardening_project_cost_correct_l64_6457


namespace correct_multiplication_l64_6435

theorem correct_multiplication (x : ℝ) : x * 51 = 244.8 → x * 15 = 72 := by
  sorry

end correct_multiplication_l64_6435


namespace travel_time_calculation_l64_6483

/-- Given a distance of 200 miles and a speed of 25 miles per hour, the time taken is 8 hours. -/
theorem travel_time_calculation (distance : ℝ) (speed : ℝ) (time : ℝ) :
  distance = 200 ∧ speed = 25 → time = distance / speed → time = 8 := by
  sorry

end travel_time_calculation_l64_6483


namespace pauls_initial_amount_l64_6479

/-- The amount of money Paul initially had for shopping --/
def initial_amount : ℕ := 15

/-- The cost of bread --/
def bread_cost : ℕ := 2

/-- The cost of butter --/
def butter_cost : ℕ := 3

/-- The cost of juice (twice the price of bread) --/
def juice_cost : ℕ := 2 * bread_cost

/-- The amount Paul had left after shopping --/
def amount_left : ℕ := 6

/-- Theorem stating that Paul's initial amount equals the sum of his purchases and remaining money --/
theorem pauls_initial_amount :
  initial_amount = bread_cost + butter_cost + juice_cost + amount_left := by
  sorry

end pauls_initial_amount_l64_6479


namespace intersection_points_range_l64_6456

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem intersection_points_range (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f x₁ = a ∧ f x₂ = a ∧ f x₃ = a) ↔ 
  -2 < a ∧ a < 2 :=
sorry

end intersection_points_range_l64_6456


namespace success_probability_given_expectation_l64_6433

/-- A two-point distribution random variable -/
structure TwoPointDistribution where
  X : ℝ → ℝ  -- The random variable
  p : ℝ      -- Success probability
  h1 : 0 ≤ p ∧ p ≤ 1  -- Probability is between 0 and 1

/-- Expected value of a two-point distribution -/
def expectedValue (T : TwoPointDistribution) : ℝ :=
  T.p * 1 + (1 - T.p) * 0

theorem success_probability_given_expectation 
  (T : TwoPointDistribution) 
  (h : expectedValue T = 0.7) : 
  T.p = 0.7 := by
  sorry


end success_probability_given_expectation_l64_6433
