import Mathlib

namespace inverse_f_486_l3367_336708

def f (x : ℝ) : ℝ := sorry

theorem inverse_f_486 (h1 : f 5 = 2) (h2 : ∀ x, f (3 * x) = 3 * f x) :
  f 1215 = 486 := by sorry

end inverse_f_486_l3367_336708


namespace inscribed_cube_volume_l3367_336748

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid :=
  (base_side : ℝ)
  (base_is_square : base_side > 0)
  (lateral_faces_equilateral : True)

/-- A cube inscribed in a pyramid -/
structure InscribedCube (p : Pyramid) :=
  (side_length : ℝ)
  (base_on_pyramid_base : True)
  (top_face_on_lateral_faces : True)

/-- The volume of the inscribed cube -/
def cube_volume (p : Pyramid) (c : InscribedCube p) : ℝ :=
  c.side_length ^ 3

/-- The theorem stating the volume of the inscribed cube -/
theorem inscribed_cube_volume (p : Pyramid) (c : InscribedCube p)
  (h : p.base_side = 1) :
  cube_volume p c = 5 * Real.sqrt 2 - 7 :=
sorry

end inscribed_cube_volume_l3367_336748


namespace smallest_pascal_family_pascal_family_with_five_children_l3367_336795

/-- Represents a family with children -/
structure Family :=
  (boys : ℕ)
  (girls : ℕ)

/-- Defines the conditions for the Pascal family -/
def isPascalFamily (f : Family) : Prop :=
  f.boys ≥ 3 ∧ f.girls ≥ 2

/-- The total number of children in a family -/
def totalChildren (f : Family) : ℕ := f.boys + f.girls

/-- Theorem: The smallest possible number of children in a Pascal family is 5 -/
theorem smallest_pascal_family :
  ∀ f : Family, isPascalFamily f → totalChildren f ≥ 5 :=
by
  sorry

/-- Theorem: There exists a Pascal family with exactly 5 children -/
theorem pascal_family_with_five_children :
  ∃ f : Family, isPascalFamily f ∧ totalChildren f = 5 :=
by
  sorry

end smallest_pascal_family_pascal_family_with_five_children_l3367_336795


namespace min_sum_squares_l3367_336704

/-- A random variable with normal distribution N(1, σ²) -/
def X (σ : ℝ) : Type := Unit

/-- The probability that X is less than or equal to a -/
def P_le (σ : ℝ) (X : X σ) (a : ℝ) : ℝ := sorry

/-- The probability that X is greater than or equal to b -/
def P_ge (σ : ℝ) (X : X σ) (b : ℝ) : ℝ := sorry

/-- The theorem stating that the minimum value of a² + b² is 2 -/
theorem min_sum_squares (σ : ℝ) (X : X σ) (a b : ℝ) 
  (h : P_le σ X a = P_ge σ X b) : 
  ∃ (min : ℝ), min = 2 ∧ ∀ (x y : ℝ), P_le σ X x = P_ge σ X y → x^2 + y^2 ≥ min :=
sorry

end min_sum_squares_l3367_336704


namespace exponential_properties_l3367_336750

theorem exponential_properties (a : ℝ) (x y : ℝ) 
  (hx : a^x = 2) (hy : a^y = 3) : 
  a^(x + y) = 6 ∧ a^(2*x - 3*y) = 4/27 := by
  sorry

end exponential_properties_l3367_336750


namespace simplify_fraction_sum_l3367_336796

theorem simplify_fraction_sum (n d : Nat) : 
  n = 75 → d = 100 → ∃ (a b : Nat), (a.gcd b = 1) ∧ (n * b = d * a) ∧ (a + b = 7) := by
  sorry

end simplify_fraction_sum_l3367_336796


namespace largest_quotient_is_15_l3367_336780

def S : Set Int := {-30, -4, -2, 2, 4, 10}

def is_even (n : Int) : Prop := ∃ k : Int, n = 2 * k

theorem largest_quotient_is_15 :
  ∀ a b : Int,
    a ∈ S → b ∈ S →
    is_even a → is_even b →
    a < 0 → b > 0 →
    (-a : ℚ) / b ≤ 15 :=
by
  sorry

end largest_quotient_is_15_l3367_336780


namespace dave_candy_bars_l3367_336768

/-- Proves that Dave paid for 6 candy bars given the problem conditions -/
theorem dave_candy_bars (total_bars : ℕ) (cost_per_bar : ℚ) (john_paid : ℚ) : 
  total_bars = 20 →
  cost_per_bar = 3/2 →
  john_paid = 21 →
  (total_bars : ℚ) * cost_per_bar - john_paid = 6 * cost_per_bar :=
by sorry

end dave_candy_bars_l3367_336768


namespace sequence_formula_l3367_336716

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem sequence_formula :
  let a₁ : ℝ := 20
  let d : ℝ := -9
  ∀ n : ℕ, arithmetic_sequence a₁ d n = -9 * n + 29 := by
sorry

end sequence_formula_l3367_336716


namespace samantha_bus_time_l3367_336773

/-- Represents Samantha's daily schedule --/
structure Schedule where
  wakeUpTime : Nat
  busTime : Nat
  classCount : Nat
  classDuration : Nat
  lunchDuration : Nat
  chessClubDuration : Nat
  arrivalTime : Nat

/-- Calculates the total time spent on the bus given a schedule --/
def busTimeDuration (s : Schedule) : Nat :=
  let totalAwayTime := s.arrivalTime - s.busTime
  let totalSchoolTime := s.classCount * s.classDuration + s.lunchDuration + s.chessClubDuration
  totalAwayTime - totalSchoolTime

/-- Samantha's actual schedule --/
def samanthaSchedule : Schedule :=
  { wakeUpTime := 7 * 60
    busTime := 8 * 60
    classCount := 7
    classDuration := 45
    lunchDuration := 45
    chessClubDuration := 90
    arrivalTime := 17 * 60 + 30 }

/-- Theorem stating that Samantha spends 120 minutes on the bus --/
theorem samantha_bus_time :
  busTimeDuration samanthaSchedule = 120 := by
  sorry

end samantha_bus_time_l3367_336773


namespace fraction_problem_l3367_336774

theorem fraction_problem (N : ℝ) (F : ℝ) : 
  (3/10 : ℝ) * N = 64.8 →
  F * ((1/4 : ℝ) * N) = 18 →
  F = 1/3 := by
sorry

end fraction_problem_l3367_336774


namespace greatest_four_digit_multiple_of_17_l3367_336789

theorem greatest_four_digit_multiple_of_17 : ∃ n : ℕ, 
  n ≤ 9999 ∧ 
  n > 999 ∧
  n % 17 = 0 ∧
  ∀ m : ℕ, m ≤ 9999 ∧ m > 999 ∧ m % 17 = 0 → m ≤ n :=
by
  -- The proof would go here
  sorry

end greatest_four_digit_multiple_of_17_l3367_336789


namespace exp_inequality_l3367_336784

/-- The function f(x) = (x-3)³ + 2x - 6 -/
def f (x : ℝ) : ℝ := (x - 3)^3 + 2*x - 6

/-- Theorem stating that if f(2a-b) + f(6-b) > 0, then e^a > e^b -/
theorem exp_inequality (a b : ℝ) (h : f (2*a - b) + f (6 - b) > 0) : Real.exp a > Real.exp b := by
  sorry

end exp_inequality_l3367_336784


namespace cos2α_plus_sin2α_for_point_l3367_336725

theorem cos2α_plus_sin2α_for_point (α : Real) :
  (∃ r : Real, r > 0 ∧ r * Real.cos α = -3 ∧ r * Real.sin α = 4) →
  Real.cos (2 * α) + Real.sin (2 * α) = -31/25 := by
  sorry

end cos2α_plus_sin2α_for_point_l3367_336725


namespace company_growth_rate_equation_l3367_336702

/-- Represents the average annual growth rate of a company's payment. -/
def average_annual_growth_rate (initial_payment final_payment : ℝ) (years : ℕ) : ℝ → Prop :=
  λ x => initial_payment * (1 + x) ^ years = final_payment

/-- Theorem stating that the equation 40(1 + x)^2 = 48.4 correctly represents
    the average annual growth rate of the company's payment. -/
theorem company_growth_rate_equation :
  average_annual_growth_rate 40 48.4 2 = λ x => 40 * (1 + x)^2 = 48.4 := by
  sorry

end company_growth_rate_equation_l3367_336702


namespace existence_of_small_difference_l3367_336783

theorem existence_of_small_difference (a : Fin 101 → ℝ)
  (h_increasing : ∀ i j, i < j → a i < a j)
  (h_bound : a 100 - a 0 ≤ 1000) :
  ∃ i j, i < j ∧ 0 < a j - a i ∧ a j - a i ≤ 10 :=
by sorry

end existence_of_small_difference_l3367_336783


namespace unique_solution_is_six_l3367_336798

theorem unique_solution_is_six :
  ∃! n : ℤ, ⌊(n^2 : ℚ) / 3⌋ - ⌊(n : ℚ) / 2⌋^2 = 3 := by
  sorry

end unique_solution_is_six_l3367_336798


namespace second_replaced_man_age_is_23_l3367_336700

/-- The age of the second replaced man in a group where:
  * There are 8 men initially
  * Two men are replaced
  * The average age increases by 2 years after replacement
  * One of the replaced men is 21 years old
  * The average age of the two new men is 30 years
-/
def second_replaced_man_age : ℕ := by
  -- Define the initial number of men
  let initial_count : ℕ := 8
  -- Define the age increase after replacement
  let age_increase : ℕ := 2
  -- Define the age of the first replaced man
  let first_replaced_age : ℕ := 21
  -- Define the average age of the new men
  let new_men_avg_age : ℕ := 30

  -- The actual proof would go here
  sorry

theorem second_replaced_man_age_is_23 : second_replaced_man_age = 23 := by
  -- The actual proof would go here
  sorry

end second_replaced_man_age_is_23_l3367_336700


namespace p_necessary_not_sufficient_for_q_l3367_336764

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, (5 * x - 6 > x^2) → (|x + 1| > 2)) ∧
  (∃ x : ℝ, (|x + 1| > 2) ∧ ¬(5 * x - 6 > x^2)) := by
  sorry

end p_necessary_not_sufficient_for_q_l3367_336764


namespace quadratic_prime_roots_l3367_336794

theorem quadratic_prime_roots (k : ℕ) : 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p + q = 63 ∧ p * q = k ∧ ∀ x : ℝ, x^2 - 63*x + k = 0 ↔ (x = p ∨ x = q)) → 
  k = 122 :=
sorry

end quadratic_prime_roots_l3367_336794


namespace remainder_problem_l3367_336735

theorem remainder_problem (n : ℕ) (h : n % 12 = 8) : n % 6 = 2 := by
  sorry

end remainder_problem_l3367_336735


namespace french_not_english_speakers_l3367_336760

/-- The number of students who speak French but not English in a survey -/
theorem french_not_english_speakers (total : ℕ) (french_speakers : ℕ) (both_speakers : ℕ) 
  (h1 : total = 200)
  (h2 : french_speakers = total / 4)
  (h3 : both_speakers = 10) :
  french_speakers - both_speakers = 40 := by
  sorry

end french_not_english_speakers_l3367_336760


namespace max_angle_ratio_theorem_l3367_336701

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

-- Define the line
def line (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 8 + 2 * Real.sqrt 3 = 0

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  F₁ = (-2 * Real.sqrt 3, 0) ∧ F₂ = (2 * Real.sqrt 3, 0)

-- Define the point P on the line
def point_on_line (P : ℝ × ℝ) : Prop :=
  line P.1 P.2

-- Define the angle F₁PF₂
def angle_F₁PF₂ (F₁ F₂ P : ℝ × ℝ) : ℝ := sorry

-- Define the distance between two points
def distance (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem max_angle_ratio_theorem 
  (F₁ F₂ P : ℝ × ℝ) 
  (h_ellipse : ellipse F₁.1 F₁.2 ∧ ellipse F₂.1 F₂.2)
  (h_foci : foci F₁ F₂)
  (h_point : point_on_line P)
  (h_max_angle : ∀ Q, point_on_line Q → angle_F₁PF₂ F₁ F₂ P ≥ angle_F₁PF₂ F₁ F₂ Q) :
  distance P F₁ / distance P F₂ = Real.sqrt 3 - 1 := by
  sorry

end max_angle_ratio_theorem_l3367_336701


namespace complex_equation_implies_sum_l3367_336747

theorem complex_equation_implies_sum (a t : ℝ) :
  (a + Complex.I) / (1 + 2 * Complex.I) = t * Complex.I →
  t + a = -1 := by sorry

end complex_equation_implies_sum_l3367_336747


namespace no_solution_exists_l3367_336790

theorem no_solution_exists : ¬∃ (n k : ℕ), 
  n ≠ 0 ∧ k ≠ 0 ∧ (n ∣ k^n - 1) ∧ Nat.gcd n (k - 1) = 1 :=
by sorry

end no_solution_exists_l3367_336790


namespace problem_statement_l3367_336740

theorem problem_statement (x y : ℝ) (hx : x = Real.sqrt 2 + 1) (hy : y = Real.sqrt 2 - 1) :
  (x + y) * (x - y) = 4 * Real.sqrt 2 := by
  sorry

end problem_statement_l3367_336740


namespace second_month_sale_l3367_336759

theorem second_month_sale 
  (first_month : ℕ) 
  (third_month : ℕ) 
  (fourth_month : ℕ) 
  (fifth_month : ℕ) 
  (sixth_month : ℕ) 
  (average_sale : ℕ) 
  (h1 : first_month = 5435)
  (h2 : third_month = 5855)
  (h3 : fourth_month = 6230)
  (h4 : fifth_month = 5562)
  (h5 : sixth_month = 3991)
  (h6 : average_sale = 5500) :
  ∃ (second_month : ℕ), 
    (first_month + second_month + third_month + fourth_month + fifth_month + sixth_month) / 6 = average_sale ∧ 
    second_month = 5927 :=
by sorry

end second_month_sale_l3367_336759


namespace total_recovery_time_l3367_336710

/-- Calculates the total recovery time for James after a hand burn, considering initial healing,
    post-surgery recovery, physical therapy sessions, and medication effects. -/
theorem total_recovery_time (initial_healing : ℝ) (A : ℝ) : 
  initial_healing = 4 →
  let post_surgery := initial_healing * 1.5
  let total_before_reduction := post_surgery
  let therapy_reduction := total_before_reduction * (0.1 * A)
  let medication_reduction := total_before_reduction * 0.2
  total_before_reduction - therapy_reduction - medication_reduction = 4.8 - 0.6 * A := by
  sorry

end total_recovery_time_l3367_336710


namespace correct_mark_l3367_336749

theorem correct_mark (wrong_mark : ℝ) (class_size : ℕ) (average_increase : ℝ) : 
  wrong_mark = 85 → 
  class_size = 80 → 
  average_increase = 0.5 →
  (wrong_mark - (wrong_mark - class_size * average_increase)) = 45 := by
  sorry

end correct_mark_l3367_336749


namespace system_solution_l3367_336797

theorem system_solution (k j : ℝ) (h1 : 64 / k = 8) (h2 : k * j = 128) : k = 8 ∧ j = 16 := by
  sorry

end system_solution_l3367_336797


namespace jesse_carpet_amount_l3367_336721

/-- The amount of carpet Jesse already has -/
def carpet_already_has (room_length room_width additional_carpet_needed : ℝ) : ℝ :=
  room_length * room_width - additional_carpet_needed

/-- Theorem: Jesse already has 16 square feet of carpet -/
theorem jesse_carpet_amount :
  carpet_already_has 11 15 149 = 16 := by
  sorry

end jesse_carpet_amount_l3367_336721


namespace peculiar_quadratic_minimum_l3367_336739

/-- A quadratic polynomial q(x) = x^2 + bx + c is peculiar if q(q(x)) = 0 has exactly four real roots, including a triple root. -/
def IsPeculiar (q : ℝ → ℝ) : Prop :=
  ∃ b c : ℝ, (∀ x, q x = x^2 + b*x + c) ∧
  (∃ r₁ r₂ r₃ r₄ : ℝ, (∀ x, q (q x) = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) ∧
  (r₁ = r₂ ∧ r₂ = r₃ ∧ r₃ ≠ r₄))

theorem peculiar_quadratic_minimum :
  ∃! q : ℝ → ℝ, IsPeculiar q ∧
  (∀ p : ℝ → ℝ, IsPeculiar p → q 0 ≤ p 0) ∧
  (∀ x, q x = x^2 - 1/2) ∧
  q 0 = -1/2 := by sorry

end peculiar_quadratic_minimum_l3367_336739


namespace max_distance_complex_l3367_336765

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs ((1 + 2*Complex.I)*z^3 - z^6) ≤ Real.sqrt 5 + 1 ∧
  ∃ z₀ : ℂ, Complex.abs z₀ = 1 ∧ Complex.abs ((1 + 2*Complex.I)*z₀^3 - z₀^6) = Real.sqrt 5 + 1 :=
sorry

end max_distance_complex_l3367_336765


namespace match_box_dozens_l3367_336761

theorem match_box_dozens (total_matches : ℕ) (matches_per_box : ℕ) (boxes_per_dozen : ℕ) : 
  total_matches = 1200 →
  matches_per_box = 20 →
  boxes_per_dozen = 12 →
  (total_matches / matches_per_box) / boxes_per_dozen = 5 :=
by sorry

end match_box_dozens_l3367_336761


namespace even_function_implies_a_value_l3367_336703

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (2 * x + 3 * a)

-- State the theorem
theorem even_function_implies_a_value :
  (∀ x : ℝ, f a x = f a (-x)) → a = -2/3 := by
  sorry

end even_function_implies_a_value_l3367_336703


namespace parabola_hyperbola_focus_coincide_l3367_336788

/-- The value of p for which the focus of the parabola y^2 = 2px coincides with 
    the right focus of the hyperbola x^2/3 - y^2/1 = 1 -/
theorem parabola_hyperbola_focus_coincide : ∃ p : ℝ, 
  (∀ x y : ℝ, y^2 = 2*p*x → x^2/3 - y^2 = 1 → 
   ∃ f : ℝ × ℝ, f = (p, 0) ∧ f = (2, 0)) → 
  p = 2 := by sorry

end parabola_hyperbola_focus_coincide_l3367_336788


namespace system_solution_l3367_336730

theorem system_solution (x y : ℝ) : 
  (x + 2*y = 6 ∧ 5*x - 4*y = 2) ↔ (x = 2 ∧ y = 2) :=
by sorry

end system_solution_l3367_336730


namespace three_distinct_zeros_l3367_336720

-- Define the piecewise function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.exp (-x) - 1/2
  else x^3 - 3*m*x - 2

-- Theorem statement
theorem three_distinct_zeros (m : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f m x = 0 ∧ f m y = 0 ∧ f m z = 0) ↔ m > 1 :=
sorry

end three_distinct_zeros_l3367_336720


namespace simple_annual_interest_rate_l3367_336776

/-- Calculate the simple annual interest rate given monthly interest payment and principal amount -/
theorem simple_annual_interest_rate 
  (monthly_interest : ℝ) 
  (principal : ℝ) 
  (h1 : monthly_interest = 234)
  (h2 : principal = 31200) :
  (monthly_interest * 12 / principal) * 100 = 8.99 := by
sorry

end simple_annual_interest_rate_l3367_336776


namespace escalator_travel_time_l3367_336756

/-- Proves that a person walking on a moving escalator takes 10 seconds to cover its length -/
theorem escalator_travel_time (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) 
  (h1 : escalator_speed = 20)
  (h2 : person_speed = 5)
  (h3 : escalator_length = 250) :
  escalator_length / (escalator_speed + person_speed) = 10 := by
  sorry

end escalator_travel_time_l3367_336756


namespace bc_length_l3367_336753

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the properties of the triangle
def isObtuseTriangle (t : Triangle) : Prop := sorry

def triangleArea (t : Triangle) : ℝ := sorry

def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem bc_length (ABC : Triangle) 
  (h1 : isObtuseTriangle ABC)
  (h2 : triangleArea ABC = 10 * Real.sqrt 3)
  (h3 : distance ABC.A ABC.B = 5)
  (h4 : distance ABC.A ABC.C = 8) :
  distance ABC.B ABC.C = Real.sqrt 129 := by sorry

end bc_length_l3367_336753


namespace cubic_fifth_power_roots_l3367_336772

/-- The roots of a cubic polynomial x^3 + ax^2 + bx + c = 0 are the fifth powers of the roots of x^3 - 3x + 1 = 0 if and only if a = 15, b = -198, and c = 1 -/
theorem cubic_fifth_power_roots (a b c : ℝ) : 
  (∀ x : ℂ, x^3 - 3*x + 1 = 0 → ∃ y : ℂ, y^3 + a*y^2 + b*y + c = 0 ∧ y = x^5) ↔ 
  (a = 15 ∧ b = -198 ∧ c = 1) :=
by sorry

end cubic_fifth_power_roots_l3367_336772


namespace quadratic_properties_l3367_336746

/-- A quadratic function with a < 0, f(-1) = 0, and axis of symmetry x = 1 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_neg : a < 0
  root_neg_one : a * (-1)^2 + b * (-1) + c = 0
  axis_sym : -b / (2 * a) = 1

/-- Properties of the quadratic function -/
theorem quadratic_properties (f : QuadraticFunction) :
  (f.a - f.b + f.c = 0) ∧
  (∀ m : ℝ, f.a * m^2 + f.b * m + f.c ≤ -4 * f.a) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → 
    f.a * x₁^2 + f.b * x₁ + f.c + 1 = 0 → 
    f.a * x₂^2 + f.b * x₂ + f.c + 1 = 0 → 
    x₁ < -1 ∧ x₂ > 3) := by
  sorry

end quadratic_properties_l3367_336746


namespace greatest_circle_center_distance_l3367_336706

theorem greatest_circle_center_distance
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (circle_diameter : ℝ)
  (h_width : rectangle_width = 18)
  (h_height : rectangle_height = 20)
  (h_diameter : circle_diameter = 8)
  (h_fit : circle_diameter ≤ min rectangle_width rectangle_height) :
  ∃ (d : ℝ), d = 2 * Real.sqrt 61 ∧
    ∀ (d' : ℝ), d' ≤ d ∧
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        0 ≤ x₁ ∧ x₁ ≤ rectangle_width ∧
        0 ≤ y₁ ∧ y₁ ≤ rectangle_height ∧
        0 ≤ x₂ ∧ x₂ ≤ rectangle_width ∧
        0 ≤ y₂ ∧ y₂ ≤ rectangle_height ∧
        (x₁ - circle_diameter / 2 ≥ 0) ∧
        (y₁ - circle_diameter / 2 ≥ 0) ∧
        (x₁ + circle_diameter / 2 ≤ rectangle_width) ∧
        (y₁ + circle_diameter / 2 ≤ rectangle_height) ∧
        (x₂ - circle_diameter / 2 ≥ 0) ∧
        (y₂ - circle_diameter / 2 ≥ 0) ∧
        (x₂ + circle_diameter / 2 ≤ rectangle_width) ∧
        (y₂ + circle_diameter / 2 ≤ rectangle_height) ∧
        d' = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) :=
by sorry

end greatest_circle_center_distance_l3367_336706


namespace min_value_theorem_l3367_336745

theorem min_value_theorem (x : ℝ) (h : x > 1) : x + 4 / (x - 1) ≥ 5 ∧ ∃ y > 1, y + 4 / (y - 1) = 5 := by
  sorry

end min_value_theorem_l3367_336745


namespace sum_is_square_l3367_336718

theorem sum_is_square (x y z : ℕ+) 
  (h1 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / z)
  (h2 : Nat.gcd (Nat.gcd x.val y.val) z.val = 1) :
  ∃ n : ℕ, x.val + y.val = n ^ 2 := by
  sorry

end sum_is_square_l3367_336718


namespace integer_solutions_cubic_equation_l3367_336751

theorem integer_solutions_cubic_equation :
  ∀ x y : ℤ, y^2 = x^3 + (x + 1)^2 ↔ (x = 0 ∧ y = 1) ∨ (x = 0 ∧ y = -1) :=
by sorry

end integer_solutions_cubic_equation_l3367_336751


namespace mod_inverse_89_mod_90_l3367_336793

theorem mod_inverse_89_mod_90 : ∃ x : ℕ, 0 ≤ x ∧ x < 90 ∧ (89 * x) % 90 = 1 :=
by
  use 89
  sorry

end mod_inverse_89_mod_90_l3367_336793


namespace axis_of_symmetry_shifted_sine_l3367_336732

theorem axis_of_symmetry_shifted_sine (k : ℤ) :
  let f : ℝ → ℝ := λ x => 2 * Real.sin (2 * (x + π / 12))
  let axis : ℝ := k * π / 2 + π / 6
  ∀ x : ℝ, f (axis - x) = f (axis + x) :=
by
  sorry

end axis_of_symmetry_shifted_sine_l3367_336732


namespace color_drawing_cost_theorem_l3367_336755

/-- The cost of a color drawing given the cost of a black and white drawing and the additional percentage for color. -/
def color_drawing_cost (bw_cost : ℝ) (color_percentage : ℝ) : ℝ :=
  bw_cost * (1 + color_percentage)

/-- Theorem: The cost of a color drawing is $240 when a black and white drawing costs $160 and color is 50% more expensive. -/
theorem color_drawing_cost_theorem :
  color_drawing_cost 160 0.5 = 240 := by
  sorry

end color_drawing_cost_theorem_l3367_336755


namespace amusement_park_tickets_l3367_336792

theorem amusement_park_tickets (total_cost : ℕ) (adult_price child_price : ℕ) (adult_child_diff : ℕ) : 
  total_cost = 720 →
  adult_price = 15 →
  child_price = 8 →
  adult_child_diff = 25 →
  ∃ (num_children : ℕ), 
    num_children * child_price + (num_children + adult_child_diff) * adult_price = total_cost ∧ 
    num_children = 15 :=
by sorry

end amusement_park_tickets_l3367_336792


namespace amount_distribution_l3367_336791

/-- The total amount distributed among boys -/
def total_amount : ℕ := 5040

/-- The number of boys in the first distribution -/
def boys_first : ℕ := 14

/-- The number of boys in the second distribution -/
def boys_second : ℕ := 18

/-- The difference in amount received by each boy between the two distributions -/
def difference : ℕ := 80

theorem amount_distribution :
  total_amount / boys_first = total_amount / boys_second + difference :=
sorry

end amount_distribution_l3367_336791


namespace bill_with_late_charges_l3367_336785

/-- Calculates the final amount owed after applying three consecutive 2% increases to an original bill. -/
def final_amount (original_bill : ℝ) : ℝ :=
  original_bill * (1 + 0.02)^3

/-- Theorem stating that given an original bill of $500 and three consecutive 2% increases, 
    the final amount owed is $530.604 (rounded to 3 decimal places) -/
theorem bill_with_late_charges :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0005 ∧ |final_amount 500 - 530.604| < ε :=
sorry

end bill_with_late_charges_l3367_336785


namespace largest_n_for_integer_differences_l3367_336741

theorem largest_n_for_integer_differences : ∃ (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℤ),
  (∀ k : ℕ, k ≤ 9 → 
    (∃ (i j : Fin 4), i < j ∧ (k : ℤ) = |x₁ - x₂| ∨ k = |x₁ - x₃| ∨ k = |x₁ - x₄| ∨ 
                               k = |x₂ - x₃| ∨ k = |x₂ - x₄| ∨ k = |x₃ - x₄| ∨
                               k = |y₁ - y₂| ∨ k = |y₁ - y₃| ∨ k = |y₁ - y₄| ∨ 
                               k = |y₂ - y₃| ∨ k = |y₂ - y₄| ∨ k = |y₃ - y₄|)) ∧
  (∀ n : ℕ, n > 9 → 
    ¬∃ (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℤ),
      ∀ k : ℕ, k ≤ n → 
        (∃ (i j : Fin 4), i < j ∧ (k : ℤ) = |a₁ - a₂| ∨ k = |a₁ - a₃| ∨ k = |a₁ - a₄| ∨ 
                                   k = |a₂ - a₃| ∨ k = |a₂ - a₄| ∨ k = |a₃ - a₄| ∨
                                   k = |b₁ - b₂| ∨ k = |b₁ - b₃| ∨ k = |b₁ - b₄| ∨ 
                                   k = |b₂ - b₃| ∨ k = |b₂ - b₄| ∨ k = |b₃ - b₄|)) :=
by sorry

end largest_n_for_integer_differences_l3367_336741


namespace black_pens_per_student_l3367_336738

/-- Represents the problem of calculating the number of black pens each student received. -/
theorem black_pens_per_student (num_students : ℕ) (red_pens_per_student : ℕ) 
  (pens_taken_first_month : ℕ) (pens_taken_second_month : ℕ) (remaining_pens_per_student : ℕ) :
  num_students = 3 →
  red_pens_per_student = 62 →
  pens_taken_first_month = 37 →
  pens_taken_second_month = 41 →
  remaining_pens_per_student = 79 →
  (num_students * (red_pens_per_student + 43) - pens_taken_first_month - pens_taken_second_month) / num_students = remaining_pens_per_student :=
by sorry

#check black_pens_per_student

end black_pens_per_student_l3367_336738


namespace matrix_power_sum_l3367_336714

/-- Given a 3x3 matrix C and a natural number m, 
    if C^m equals a specific matrix and C has a specific form,
    then b + m = 310 where b is an element of C. -/
theorem matrix_power_sum (b m : ℕ) (C : Matrix (Fin 3) (Fin 3) ℕ) : 
  C^m = !![1, 33, 3080; 1, 1, 65; 1, 0, 1] ∧ 
  C = !![1, 3, b; 0, 1, 5; 1, 0, 1] → 
  b + m = 310 := by sorry

end matrix_power_sum_l3367_336714


namespace binomial_coefficient_divisibility_l3367_336744

theorem binomial_coefficient_divisibility (p n : ℕ) (hp : p.Prime) (hn : n ≥ p) :
  ∃ k : ℤ, (Nat.choose n p : ℤ) - (n / p : ℤ) = k * p := by
  sorry

end binomial_coefficient_divisibility_l3367_336744


namespace bee_count_l3367_336771

theorem bee_count (initial_bees : ℕ) (h1 : initial_bees = 144) : 
  ⌊(3 * initial_bees : ℚ) * (1 - 0.2)⌋ = 346 := by
  sorry

end bee_count_l3367_336771


namespace c_equals_zero_l3367_336728

theorem c_equals_zero (a b c : ℝ) (h1 : a + b = 5) (h2 : c^2 = a*b + b - 9) : c = 0 := by
  sorry

end c_equals_zero_l3367_336728


namespace median_mean_difference_l3367_336758

theorem median_mean_difference (x : ℤ) (a : ℤ) : 
  x > 0 → x + a > 0 → x + 4 > 0 → x + 7 > 0 → x + 37 > 0 →
  (x + (x + a) + (x + 4) + (x + 7) + (x + 37)) / 5 = (x + 4) + 6 →
  a = 2 := by sorry

end median_mean_difference_l3367_336758


namespace discount_calculation_l3367_336782

/-- Calculates the discounted price of an item given the original price and discount rate. -/
def discounted_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

/-- Proves that a 20% discount on a $120 item results in a price of $96. -/
theorem discount_calculation :
  let original_price : ℝ := 120
  let discount_rate : ℝ := 0.2
  discounted_price original_price discount_rate = 96 := by
  sorry

end discount_calculation_l3367_336782


namespace mud_weight_after_evaporation_l3367_336778

/-- 
Given a train car with mud, prove that the final weight after water evaporation
is 4000 pounds, given the initial conditions and final water percentage.
-/
theorem mud_weight_after_evaporation 
  (initial_weight : ℝ) 
  (initial_water_percent : ℝ)
  (final_water_percent : ℝ)
  (hw : initial_weight = 6000)
  (hiw : initial_water_percent = 88)
  (hfw : final_water_percent = 82) :
  (initial_weight * (100 - initial_water_percent) / 100) / ((100 - final_water_percent) / 100) = 4000 :=
by sorry

end mud_weight_after_evaporation_l3367_336778


namespace sequence_property_l3367_336723

theorem sequence_property (a : ℕ → ℕ) 
  (h1 : ∀ (p q : ℕ), a (p + q) = a p + a q) 
  (h2 : a 2 = 4) : 
  a 9 = 18 := by
sorry

end sequence_property_l3367_336723


namespace points_on_line_implies_a_equals_two_l3367_336763

-- Define the points
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (5, 1)
def C (a : ℝ) : ℝ × ℝ := (-4, 2*a)

-- Define the condition for points being on the same line
def collinear (p q r : ℝ × ℝ) : Prop :=
  (q.2 - p.2) * (r.1 - p.1) = (r.2 - p.2) * (q.1 - p.1)

-- Theorem statement
theorem points_on_line_implies_a_equals_two :
  collinear A B (C a) → a = 2 := by
  sorry

end points_on_line_implies_a_equals_two_l3367_336763


namespace quadratic_completing_square_l3367_336799

theorem quadratic_completing_square :
  ∀ x : ℝ, (x^2 - 4*x - 6 = 0) ↔ ((x - 2)^2 = 10) :=
by
  sorry

end quadratic_completing_square_l3367_336799


namespace medication_price_reduction_l3367_336770

theorem medication_price_reduction (a : ℝ) :
  let new_price := a
  let reduction_rate := 0.4
  let original_price := (5 / 3) * a
  (1 - reduction_rate) * original_price = new_price :=
by sorry

end medication_price_reduction_l3367_336770


namespace probability_seven_chairs_probability_n_chairs_l3367_336727

/-- The probability of three knights being seated at a round table with empty chairs on either side of each knight. -/
def knight_seating_probability (n : ℕ) : ℚ :=
  if n = 7 then 1 / 35
  else if n ≥ 6 then (n - 4) * (n - 5) / ((n - 1) * (n - 2))
  else 0

/-- Theorem stating the probability for 7 chairs -/
theorem probability_seven_chairs :
  knight_seating_probability 7 = 1 / 35 := by sorry

/-- Theorem stating the probability for n chairs (n ≥ 6) -/
theorem probability_n_chairs (n : ℕ) (h : n ≥ 6) :
  knight_seating_probability n = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) := by sorry

end probability_seven_chairs_probability_n_chairs_l3367_336727


namespace partnership_capital_fraction_l3367_336717

theorem partnership_capital_fraction :
  ∀ (T : ℚ) (x : ℚ),
    x > 0 →
    T > 0 →
    x * T + (1/4) * T + (1/5) * T + ((11/20 - x) * T) = T →
    805 / 2415 = x →
    x = 161 / 483 := by
  sorry

end partnership_capital_fraction_l3367_336717


namespace archer_arrow_cost_l3367_336734

/-- Represents the archer's arrow usage and costs -/
structure ArcherData where
  shots_per_week : ℕ
  recovery_rate : ℚ
  personal_expense_rate : ℚ
  personal_expense : ℚ

/-- Calculates the cost per arrow given the archer's data -/
def cost_per_arrow (data : ArcherData) : ℚ :=
  let total_cost := data.personal_expense / data.personal_expense_rate
  let arrows_lost := data.shots_per_week * (1 - data.recovery_rate)
  total_cost / arrows_lost

/-- Theorem stating that the cost per arrow is $5.50 given the specific conditions -/
theorem archer_arrow_cost :
  let data : ArcherData := {
    shots_per_week := 800,
    recovery_rate := 1/5,
    personal_expense_rate := 3/10,
    personal_expense := 1056
  }
  cost_per_arrow data = 11/2 := by
  sorry

end archer_arrow_cost_l3367_336734


namespace swimming_pool_volume_l3367_336736

/-- The volume of a cylindrical swimming pool -/
theorem swimming_pool_volume (diameter : ℝ) (depth : ℝ) (volume : ℝ) :
  diameter = 16 →
  depth = 4 →
  volume = π * (diameter / 2)^2 * depth →
  volume = 256 * π := by
  sorry

end swimming_pool_volume_l3367_336736


namespace prism_volume_l3367_336719

/-- The volume of a right rectangular prism with face areas 18, 12, and 8 square inches -/
theorem prism_volume (x y z : ℝ) 
  (h1 : x * y = 18) 
  (h2 : y * z = 12) 
  (h3 : x * z = 8) : 
  x * y * z = 24 * Real.sqrt 3 := by
  sorry

end prism_volume_l3367_336719


namespace x_fourth_plus_inverse_fourth_l3367_336775

theorem x_fourth_plus_inverse_fourth (x : ℝ) (h : x + 1/x = 3) :
  x^4 + 1/x^4 = 47 := by sorry

end x_fourth_plus_inverse_fourth_l3367_336775


namespace first_five_average_l3367_336767

theorem first_five_average (total_average : ℝ) (last_seven_average : ℝ) (fifth_result : ℝ) :
  total_average = 42 →
  last_seven_average = 52 →
  fifth_result = 147 →
  (5 * ((11 * total_average - (7 * last_seven_average - fifth_result)) / 5) = 245) ∧
  ((11 * total_average - (7 * last_seven_average - fifth_result)) / 5 = 49) :=
by
  sorry

end first_five_average_l3367_336767


namespace train_speed_crossing_bridge_l3367_336733

/-- Calculate the speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 500) 
  (h2 : bridge_length = 350) 
  (h3 : crossing_time = 60) : 
  ∃ (speed : ℝ), abs (speed - 14.1667) < 0.0001 :=
by
  sorry

end train_speed_crossing_bridge_l3367_336733


namespace no_intersection_implies_k_equals_three_l3367_336724

theorem no_intersection_implies_k_equals_three (k : ℕ+) :
  (∀ x y : ℝ, x^2 + y^2 = k^2 → x * y ≠ k) → k = 3 := by
  sorry

end no_intersection_implies_k_equals_three_l3367_336724


namespace point_transformation_l3367_336713

def rotate_y_180 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, -z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, z)

def transform_point (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  p |> rotate_y_180 |> reflect_yz |> reflect_xz |> rotate_y_180 |> reflect_xz

theorem point_transformation :
  transform_point (2, 3, 4) = (-2, 3, 4) := by
  sorry


end point_transformation_l3367_336713


namespace cistern_wet_surface_area_l3367_336712

/-- Calculates the total wet surface area of a rectangular cistern -/
def wetSurfaceArea (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth) + 2 * (width * depth)

/-- Theorem: The wet surface area of a cistern with given dimensions is 134 square meters -/
theorem cistern_wet_surface_area :
  wetSurfaceArea 10 8 1.5 = 134 := by
  sorry

end cistern_wet_surface_area_l3367_336712


namespace lemniscate_polar_to_rect_l3367_336766

/-- The lemniscate equation in polar coordinates -/
def lemniscate_polar (r φ a : ℝ) : Prop :=
  r^2 = 2 * a^2 * Real.cos (2 * φ)

/-- The lemniscate equation in rectangular coordinates -/
def lemniscate_rect (x y a : ℝ) : Prop :=
  (x^2 + y^2)^2 = 2 * a^2 * (x^2 - y^2)

/-- Theorem stating that the rectangular equation represents the lemniscate -/
theorem lemniscate_polar_to_rect (a : ℝ) :
  ∀ (x y r φ : ℝ), 
    x = r * Real.cos φ →
    y = r * Real.sin φ →
    lemniscate_polar r φ a →
    lemniscate_rect x y a :=
by
  sorry

end lemniscate_polar_to_rect_l3367_336766


namespace sin_4theta_from_exp_itheta_l3367_336737

theorem sin_4theta_from_exp_itheta (θ : ℝ) :
  Complex.exp (θ * Complex.I) = (2 + Complex.I * Real.sqrt 5) / 3 →
  Real.sin (4 * θ) = -8 * Real.sqrt 5 / 81 := by
  sorry

end sin_4theta_from_exp_itheta_l3367_336737


namespace pauls_erasers_l3367_336726

/-- Represents the number of crayons and erasers Paul has --/
structure PaulsSupplies where
  initialCrayons : ℕ
  finalCrayons : ℕ
  erasers : ℕ

/-- Defines the conditions of Paul's supplies --/
def validSupplies (s : PaulsSupplies) : Prop :=
  s.initialCrayons = 601 ∧
  s.finalCrayons = 336 ∧
  s.erasers = s.finalCrayons + 70

/-- Theorem stating the number of erasers Paul got for his birthday --/
theorem pauls_erasers (s : PaulsSupplies) (h : validSupplies s) : s.erasers = 406 := by
  sorry

end pauls_erasers_l3367_336726


namespace lucas_february_bill_l3367_336787

/-- Calculates the total cost of a cell phone plan based on given parameters. -/
def calculate_phone_bill (base_cost : ℚ) (text_cost : ℚ) (extra_cost_30_31 : ℚ) 
  (extra_cost_beyond_31 : ℚ) (num_texts : ℕ) (talk_time : ℚ) : ℚ :=
  let text_total := num_texts * text_cost
  let extra_time := max (talk_time - 30) 0
  let extra_cost := 
    if extra_time ≤ 1 then
      extra_time * 60 * extra_cost_30_31
    else
      60 * extra_cost_30_31 + (extra_time - 1) * 60 * extra_cost_beyond_31
  base_cost + text_total + extra_cost

/-- Theorem stating that Lucas's phone bill for February is $55.00 -/
theorem lucas_february_bill : 
  calculate_phone_bill 25 0.1 0.15 0.2 150 31.5 = 55 := by
  sorry


end lucas_february_bill_l3367_336787


namespace original_savings_proof_l3367_336742

def lindas_savings : ℝ := 880
def tv_cost : ℝ := 220

theorem original_savings_proof :
  (1 / 4 : ℝ) * lindas_savings = tv_cost →
  lindas_savings = 880 := by
sorry

end original_savings_proof_l3367_336742


namespace first_group_size_l3367_336777

/-- The amount of work done by one person in one day -/
def work_unit : ℝ := 1

/-- The number of days -/
def days : ℕ := 3

/-- The number of people in the second group -/
def people_second_group : ℕ := 8

/-- The amount of work done by the first group -/
def work_first_group : ℝ := 3

/-- The amount of work done by the second group -/
def work_second_group : ℝ := 8

/-- The number of people in the first group -/
def people_first_group : ℕ := 3

theorem first_group_size :
  (people_first_group : ℝ) * days * work_unit = work_first_group ∧
  (people_second_group : ℝ) * days * work_unit = work_second_group →
  people_first_group = 3 := by
  sorry

end first_group_size_l3367_336777


namespace original_number_proof_l3367_336711

theorem original_number_proof (x : ℝ) : x * 1.2 = 288 → x = 240 := by sorry

end original_number_proof_l3367_336711


namespace missing_digit_is_4_l3367_336729

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem missing_digit_is_4 (n : ℕ) (h1 : n ≥ 35204 ∧ n < 35304) 
  (h2 : is_divisible_by_9 n) : 
  ∃ (d : ℕ), d < 10 ∧ n = 35204 + d * 10 ∧ d = 4 := by
  sorry

#check missing_digit_is_4

end missing_digit_is_4_l3367_336729


namespace concentric_circle_through_point_l3367_336715

/-- Given a circle with equation x^2 + y^2 - 4x + 6y + 3 = 0,
    prove that (x - 2)^2 + (y + 3)^2 = 25 represents a circle
    that is concentric with the given circle and passes through (-1, 1) -/
theorem concentric_circle_through_point
  (h : ∀ x y : ℝ, x^2 + y^2 - 4*x + 6*y + 3 = 0 → (x - 2)^2 + (y + 3)^2 = 10) :
  (∀ x y : ℝ, (x - 2)^2 + (y + 3)^2 = 25 →
    ∃ k : ℝ, k > 0 ∧ (x - 2)^2 + (y + 3)^2 = k * ((x - 2)^2 + (y + 3)^2 - 10)) ∧
  ((-1 - 2)^2 + (1 + 3)^2 = 25) :=
by sorry

end concentric_circle_through_point_l3367_336715


namespace rounding_bounds_l3367_336705

def rounded_value : ℕ := 1300000

theorem rounding_bounds :
  ∀ n : ℕ,
  (n + 50000) / 100000 * 100000 = rounded_value →
  n ≤ 1304999 ∧ n ≥ 1295000 :=
by sorry

end rounding_bounds_l3367_336705


namespace order_of_rationals_l3367_336786

theorem order_of_rationals (a b : ℚ) (h : b < 0) : a + b < a ∧ a < a - b := by
  sorry

end order_of_rationals_l3367_336786


namespace similarity_condition_l3367_336757

theorem similarity_condition (a b : ℝ) :
  (∃ h : ℝ → ℝ, 
    (∀ y : ℝ, ∃ x : ℝ, h x = y) ∧ 
    (∀ x₁ x₂ : ℝ, h x₁ = h x₂ → x₁ = x₂) ∧
    (∀ x : ℝ, h (x^2 + a*x + b) = (h x)^2)) →
  b = a*(a + 2)/4 := by
sorry

end similarity_condition_l3367_336757


namespace rotation_theorem_l3367_336762

/-- The original line -/
def original_line (x y : ℝ) : Prop := 2 * x - y - 2 = 0

/-- The point where the original line intersects the y-axis -/
def intersection_point : ℝ × ℝ := (0, -2)

/-- The rotated line -/
def rotated_line (x y : ℝ) : Prop := x + 2 * y + 4 = 0

/-- Theorem stating that rotating the original line 90° counterclockwise around the intersection point results in the rotated line -/
theorem rotation_theorem :
  ∀ (x y : ℝ),
  original_line x y →
  ∃ (x' y' : ℝ),
  (x' - intersection_point.1) ^ 2 + (y' - intersection_point.2) ^ 2 = (x - intersection_point.1) ^ 2 + (y - intersection_point.2) ^ 2 ∧
  (x' - intersection_point.1) * (x - intersection_point.1) + (y' - intersection_point.2) * (y - intersection_point.2) = 0 ∧
  rotated_line x' y' :=
sorry

end rotation_theorem_l3367_336762


namespace floor_negative_seven_fourths_l3367_336781

theorem floor_negative_seven_fourths : ⌊(-7 : ℤ) / 4⌋ = -2 := by
  sorry

end floor_negative_seven_fourths_l3367_336781


namespace probability_not_adjacent_correct_l3367_336731

/-- The number of chairs in a row -/
def total_chairs : ℕ := 10

/-- The number of available chairs (excluding the last one) -/
def available_chairs : ℕ := total_chairs - 1

/-- The probability that two people don't sit next to each other 
    when randomly selecting from the first 9 chairs of 10 -/
def probability_not_adjacent : ℚ := 7 / 9

/-- Theorem stating the probability of two people not sitting adjacent 
    when randomly selecting from 9 out of 10 chairs -/
theorem probability_not_adjacent_correct : 
  probability_not_adjacent = 1 - (2 * available_chairs - 2) / (available_chairs * (available_chairs - 1)) :=
by sorry

end probability_not_adjacent_correct_l3367_336731


namespace equal_tuesdays_thursdays_30_days_l3367_336769

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- A function that determines if a given day is a valid starting day for a 30-day month with equal Tuesdays and Thursdays -/
def isValidStartDay (d : DayOfWeek) : Prop := sorry

/-- The number of valid starting days for a 30-day month with equal Tuesdays and Thursdays -/
def numValidStartDays : ℕ := sorry

theorem equal_tuesdays_thursdays_30_days :
  numValidStartDays = 5 := by sorry

end equal_tuesdays_thursdays_30_days_l3367_336769


namespace imo_1996_p5_l3367_336743

theorem imo_1996_p5 (n p q : ℕ+) (x : ℕ → ℤ)
  (h_npq : n > p + q)
  (h_x0 : x 0 = 0)
  (h_xn : x n = 0)
  (h_diff : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → (x i - x (i-1) = p ∨ x i - x (i-1) = -q)) :
  ∃ i j : ℕ, 0 ≤ i ∧ i < j ∧ j ≤ n ∧ (i, j) ≠ (0, n) ∧ x i = x j :=
sorry

end imo_1996_p5_l3367_336743


namespace regular_star_points_l3367_336722

/-- Represents an n-pointed regular star with alternating angles --/
structure RegularStar where
  n : ℕ
  A : ℕ → ℝ
  B : ℕ → ℝ
  A_congruent : ∀ i j, A i = A j
  B_congruent : ∀ i j, B i = B j
  angle_difference : ∀ i, B i - A i = 20

/-- Theorem stating that the only possible number of points for the given conditions is 18 --/
theorem regular_star_points (star : RegularStar) : star.n = 18 := by
  sorry

end regular_star_points_l3367_336722


namespace first_prize_winners_l3367_336754

theorem first_prize_winners (n : ℕ) : 
  (30 ≤ n ∧ n ≤ 55) ∧ 
  (n % 3 = 2) ∧ 
  (n % 5 = 4) ∧ 
  (n % 7 = 2) → 
  n = 44 := by
sorry

end first_prize_winners_l3367_336754


namespace product_of_conjugates_l3367_336709

theorem product_of_conjugates (P Q R S : ℝ) : 
  P = Real.sqrt 2023 + Real.sqrt 2024 →
  Q = -Real.sqrt 2023 - Real.sqrt 2024 →
  R = Real.sqrt 2023 - Real.sqrt 2024 →
  S = Real.sqrt 2024 - Real.sqrt 2023 →
  P * Q * R * S = 1 := by
  sorry

end product_of_conjugates_l3367_336709


namespace coins_can_be_all_heads_l3367_336779

/-- Represents the state of a coin (heads or tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents a sequence of 100 coins -/
def CoinSequence := Fin 100 → CoinState

/-- Represents an operation that flips 7 coins at equal intervals -/
structure FlipOperation where
  start : Fin 100  -- Starting position of the flip
  interval : Nat   -- Interval between flipped coins
  valid : start.val + 6 * interval < 100  -- Ensure operation is within bounds

/-- Applies a flip operation to a coin sequence -/
def applyFlip (seq : CoinSequence) (op : FlipOperation) : CoinSequence :=
  λ i => if ∃ k : Fin 7, i.val = op.start.val + k.val * op.interval
         then match seq i with
              | CoinState.Heads => CoinState.Tails
              | CoinState.Tails => CoinState.Heads
         else seq i

/-- Checks if all coins in the sequence are heads -/
def allHeads (seq : CoinSequence) : Prop :=
  ∀ i : Fin 100, seq i = CoinState.Heads

/-- The main theorem: it's possible to make all coins heads -/
theorem coins_can_be_all_heads :
  ∀ (initial : CoinSequence),
  ∃ (ops : List FlipOperation),
  allHeads (ops.foldl applyFlip initial) :=
sorry

end coins_can_be_all_heads_l3367_336779


namespace billion_to_scientific_notation_l3367_336707

/-- Represents scientific notation as a pair of a coefficient and an exponent -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_to_scientific_notation :
  let billion : ℝ := 1000000000
  let amount : ℝ := 10.58 * billion
  toScientificNotation amount = ScientificNotation.mk 1.058 10 (by norm_num) :=
by sorry

end billion_to_scientific_notation_l3367_336707


namespace range_of_a_l3367_336752

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 2) * Real.exp x + a * (Real.log x - x + 1)

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f a x ≥ -Real.exp 1) →
  (∃ x > 0, f a x = -Real.exp 1) →
  a ≤ 0 :=
sorry

end range_of_a_l3367_336752
