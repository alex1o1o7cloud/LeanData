import Mathlib

namespace first_shift_members_l3385_338568

-- Define the number of shifts
def num_shifts : ℕ := 3

-- Define the total number of workers in the company
def total_workers (shift1 : ℕ) : ℕ := shift1 + 50 + 40

-- Define the participation rate for each shift
def participation_rate1 : ℚ := 1/5
def participation_rate2 : ℚ := 2/5
def participation_rate3 : ℚ := 1/10

-- Define the total number of participants in the pension program
def total_participants (shift1 : ℕ) : ℚ :=
  participation_rate1 * shift1 + participation_rate2 * 50 + participation_rate3 * 40

-- State the theorem
theorem first_shift_members :
  ∃ (shift1 : ℕ), 
    shift1 > 0 ∧
    (total_participants shift1) / (total_workers shift1) = 6/25 ∧
    shift1 = 60 :=
by sorry

end first_shift_members_l3385_338568


namespace matthews_water_bottle_size_l3385_338538

/-- Calculates the size of Matthew's water bottle based on his drinking habits -/
theorem matthews_water_bottle_size 
  (glasses_per_day : ℕ) 
  (ounces_per_glass : ℕ) 
  (fills_per_week : ℕ) 
  (h1 : glasses_per_day = 4)
  (h2 : ounces_per_glass = 5)
  (h3 : fills_per_week = 4) :
  (glasses_per_day * ounces_per_glass * 7) / fills_per_week = 35 := by
  sorry

end matthews_water_bottle_size_l3385_338538


namespace cookie_ratio_l3385_338504

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The total number of cookies Meena baked -/
def total_cookies : ℕ := 5 * dozen

/-- The number of cookies Mr. Stone bought -/
def stone_cookies : ℕ := 2 * dozen

/-- The number of cookies Brock bought -/
def brock_cookies : ℕ := 7

/-- The number of cookies left -/
def cookies_left : ℕ := 15

/-- The number of cookies Katy bought -/
def katy_cookies : ℕ := total_cookies - stone_cookies - brock_cookies - cookies_left

theorem cookie_ratio : 
  (katy_cookies : ℚ) / brock_cookies = 2 := by sorry

end cookie_ratio_l3385_338504


namespace initial_snack_eaters_l3385_338534

/-- The number of snack eaters after a series of events -/
def final_snack_eaters (S : ℕ) : ℕ :=
  ((S + 20) / 2 + 10 - 30) / 2

/-- Theorem stating that the initial number of snack eaters was 100 -/
theorem initial_snack_eaters :
  ∃ S : ℕ, final_snack_eaters S = 20 ∧ S = 100 := by
  sorry

end initial_snack_eaters_l3385_338534


namespace kamals_biology_marks_l3385_338588

-- Define the known marks and average
def english_marks : ℕ := 76
def math_marks : ℕ := 60
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def average_marks : ℕ := 74
def num_subjects : ℕ := 5

-- Define the theorem
theorem kamals_biology_marks :
  let total_marks := average_marks * num_subjects
  let known_marks_sum := english_marks + math_marks + physics_marks + chemistry_marks
  let biology_marks := total_marks - known_marks_sum
  biology_marks = 85 := by sorry

end kamals_biology_marks_l3385_338588


namespace circle1_correct_circle2_correct_l3385_338550

-- Define the points
def M : ℝ × ℝ := (-5, 3)
def A1 : ℝ × ℝ := (-8, -1)
def A2 : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (-1, 3)
def C : ℝ × ℝ := (2, 6)

-- Define the circle equations
def circle1_eq (x y : ℝ) : Prop := (x + 5)^2 + (y - 3)^2 = 25
def circle2_eq (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 5

-- Theorem for the first circle
theorem circle1_correct : 
  (∀ x y : ℝ, circle1_eq x y ↔ 
    ((x, y) = M ∨ (∃ t : ℝ, (x, y) = M + t • (A1 - M) ∧ 0 < t ∧ t < 1))) := by sorry

-- Theorem for the second circle
theorem circle2_correct : 
  (∀ x y : ℝ, circle2_eq x y ↔ 
    ((x, y) = A2 ∨ (x, y) = B ∨ (x, y) = C ∨ 
    (∃ t : ℝ, ((x, y) = A2 + t • (B - A2) ∨ 
               (x, y) = B + t • (C - B) ∨ 
               (x, y) = C + t • (A2 - C)) ∧ 
    0 < t ∧ t < 1))) := by sorry

end circle1_correct_circle2_correct_l3385_338550


namespace green_marbles_count_l3385_338589

theorem green_marbles_count (G : ℕ) : 
  (2 / (2 + G : ℝ)) * (1 / (1 + G : ℝ)) = 0.1 → G = 3 := by
sorry

end green_marbles_count_l3385_338589


namespace original_selling_price_l3385_338505

theorem original_selling_price (P : ℝ) (S : ℝ) (S_new : ℝ) : 
  S = 1.1 * P →
  S_new = 1.3 * (0.9 * P) →
  S_new = S + 35 →
  S = 550 := by
sorry

end original_selling_price_l3385_338505


namespace doubled_roots_quadratic_l3385_338525

theorem doubled_roots_quadratic (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 5 * x₁ - 8 = 0 ∧ 2 * x₂^2 - 5 * x₂ - 8 = 0) →
  ((2 * x₁)^2 - 5 * (2 * x₁) - 16 = 0 ∧ (2 * x₂)^2 - 5 * (2 * x₂) - 16 = 0) :=
by sorry

end doubled_roots_quadratic_l3385_338525


namespace triangle_perimeter_l3385_338515

/-- A triangle with side lengths x, x+1, and x-1 has a perimeter of 21 if and only if x = 7 -/
theorem triangle_perimeter (x : ℝ) : 
  x > 0 ∧ x + 1 > 0 ∧ x - 1 > 0 → 
  (x + (x + 1) + (x - 1) = 21 ↔ x = 7) := by
sorry

end triangle_perimeter_l3385_338515


namespace line_parallel_to_plane_l3385_338533

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicularLine : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)

-- State the theorem
theorem line_parallel_to_plane
  (α β : Plane) (m : Line)
  (h1 : perpendicular α β)
  (h2 : perpendicularLine m β)
  (h3 : ¬ contains α m) :
  parallel m α :=
sorry

end line_parallel_to_plane_l3385_338533


namespace equation_solution_l3385_338554

theorem equation_solution : ∃ y : ℚ, (40 / 60 = Real.sqrt (y / 60)) ∧ y = 80 / 3 := by
  sorry

end equation_solution_l3385_338554


namespace quadratic_factorization_l3385_338521

theorem quadratic_factorization :
  ∃ (x : ℝ), x^2 + 6*x - 2 = 0 ↔ ∃ (x : ℝ), (x + 3)^2 = 11 :=
by sorry

end quadratic_factorization_l3385_338521


namespace product_mod_500_l3385_338585

theorem product_mod_500 : (1502 * 2021) % 500 = 42 := by
  sorry

end product_mod_500_l3385_338585


namespace count_propositions_with_connectives_l3385_338561

-- Define a proposition type
inductive Proposition
| feb14_2010 : Proposition
| multiple_10_5 : Proposition
| trapezoid_rectangle : Proposition

-- Define a function to check if a proposition uses a logical connective
def uses_logical_connective (p : Proposition) : Bool :=
  match p with
  | Proposition.feb14_2010 => true  -- Uses "and"
  | Proposition.multiple_10_5 => false
  | Proposition.trapezoid_rectangle => true  -- Uses "not"

-- Define the list of propositions
def propositions : List Proposition :=
  [Proposition.feb14_2010, Proposition.multiple_10_5, Proposition.trapezoid_rectangle]

-- Theorem statement
theorem count_propositions_with_connectives :
  (propositions.filter uses_logical_connective).length = 2 := by
  sorry

end count_propositions_with_connectives_l3385_338561


namespace similar_triangle_perimeter_l3385_338586

/-- Given two similar right triangles, where one has side lengths 6, 8, and 10,
    and the other has its shortest side equal to 15,
    prove that the perimeter of the larger triangle is 60. -/
theorem similar_triangle_perimeter :
  ∀ (a b c d e f : ℝ),
  a = 6 ∧ b = 8 ∧ c = 10 ∧  -- First triangle side lengths
  d = 15 ∧                  -- Shortest side of the similar triangle
  a^2 + b^2 = c^2 ∧         -- Pythagorean theorem for the first triangle
  (d / a) * b = e ∧         -- Similar triangles proportion for the second side
  (d / a) * c = f →         -- Similar triangles proportion for the third side
  d + e + f = 60 :=
by
  sorry


end similar_triangle_perimeter_l3385_338586


namespace train_platform_crossing_time_l3385_338517

/-- Given a train of length 1200 meters that takes 120 seconds to pass a point,
    the time required for this train to completely pass a platform of length 700 meters is 190 seconds. -/
theorem train_platform_crossing_time
  (train_length : ℝ)
  (point_crossing_time : ℝ)
  (platform_length : ℝ)
  (h1 : train_length = 1200)
  (h2 : point_crossing_time = 120)
  (h3 : platform_length = 700) :
  (train_length + platform_length) / (train_length / point_crossing_time) = 190 :=
sorry

end train_platform_crossing_time_l3385_338517


namespace average_transformation_l3385_338576

theorem average_transformation (t b c : ℝ) :
  (t + b + c + 14 + 15) / 5 = 12 →
  (t + b + c + 29) / 4 = 15 := by
sorry

end average_transformation_l3385_338576


namespace emily_age_l3385_338569

theorem emily_age :
  ∀ (e m : ℕ),
  e = m - 18 →
  e + m = 54 →
  e = 18 :=
by
  sorry

end emily_age_l3385_338569


namespace min_consecutive_even_numbers_divisible_by_384_l3385_338542

-- Define a function that checks if a number is even
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define a function that generates a list of consecutive even numbers
def consecutiveEvenNumbers (start : ℕ) (count : ℕ) : List ℕ :=
  List.range count |>.map (λ i => start + 2 * i)

-- Define a function that calculates the product of a list of numbers
def productOfList (l : List ℕ) : ℕ :=
  l.foldl (·*·) 1

-- The main theorem
theorem min_consecutive_even_numbers_divisible_by_384 :
  ∀ n : ℕ, n ≥ 7 →
    ∀ start : ℕ, isEven start →
      384 ∣ productOfList (consecutiveEvenNumbers start n) ∧
      ∀ m : ℕ, m < 7 →
        ∃ s : ℕ, isEven s ∧ ¬(384 ∣ productOfList (consecutiveEvenNumbers s m)) :=
by sorry


end min_consecutive_even_numbers_divisible_by_384_l3385_338542


namespace tv_cost_l3385_338501

def original_savings : ℚ := 600

def furniture_fraction : ℚ := 2/4

theorem tv_cost (savings : ℚ) (frac : ℚ) (h1 : savings = original_savings) (h2 : frac = furniture_fraction) : 
  savings * (1 - frac) = 300 := by
  sorry

end tv_cost_l3385_338501


namespace ratio_problem_l3385_338503

theorem ratio_problem (a b c d : ℝ) 
  (h1 : b / a = 3)
  (h2 : c / b = 4)
  (h3 : d = 5 * b)
  : (a + b + d) / (b + c + d) = 19 / 30 := by
  sorry

end ratio_problem_l3385_338503


namespace isosceles_triangle_perimeter_l3385_338597

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 9*x + 18 = 0

-- Define the isosceles triangle
structure IsoscelesTriangle :=
  (base : ℝ)
  (leg : ℝ)
  (base_is_root : quadratic_equation base)
  (leg_is_root : quadratic_equation leg)

-- Theorem statement
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : 
  t.base + 2 * t.leg = 15 := by
  sorry

end isosceles_triangle_perimeter_l3385_338597


namespace min_value_expression_l3385_338590

theorem min_value_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1/(a*b) + 1/(a*(a-b)) ≥ 4 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > b₀ ∧ b₀ > 0 ∧ a₀^2 + 1/(a₀*b₀) + 1/(a₀*(a₀-b₀)) = 4 :=
sorry

end min_value_expression_l3385_338590


namespace soccer_match_ratio_l3385_338577

def soccer_match (kickers_second_period : ℕ) : Prop :=
  let kickers_first_period : ℕ := 2
  let spiders_first_period : ℕ := kickers_first_period / 2
  let spiders_second_period : ℕ := 2 * kickers_second_period
  let total_goals : ℕ := 15
  (kickers_first_period + kickers_second_period + spiders_first_period + spiders_second_period = total_goals) ∧
  (kickers_second_period : ℚ) / (kickers_first_period : ℚ) = 2 / 1

theorem soccer_match_ratio : ∃ (kickers_second_period : ℕ), soccer_match kickers_second_period := by
  sorry

end soccer_match_ratio_l3385_338577


namespace jerome_money_problem_l3385_338537

theorem jerome_money_problem (certain_amount : ℕ) : 
  (2 * certain_amount - (8 + 3 * 8) = 54) → 
  certain_amount = 43 := by
  sorry

end jerome_money_problem_l3385_338537


namespace c_k_value_l3385_338516

/-- Arithmetic sequence with first term 1 and common difference d -/
def arithmetic_seq (d : ℕ) (n : ℕ) : ℕ :=
  1 + (n - 1) * d

/-- Geometric sequence with first term 1 and common ratio r -/
def geometric_seq (r : ℕ) (n : ℕ) : ℕ :=
  r ^ (n - 1)

/-- Sum of arithmetic and geometric sequences -/
def c_seq (d r : ℕ) (n : ℕ) : ℕ :=
  arithmetic_seq d n + geometric_seq r n

theorem c_k_value (d r k : ℕ) :
  c_seq d r (k - 1) = 50 ∧ c_seq d r (k + 1) = 500 → c_seq d r k = 78 := by
  sorry

end c_k_value_l3385_338516


namespace midnight_temperature_l3385_338591

/-- Calculates the final temperature given initial temperature and temperature changes --/
def finalTemperature (initial : Int) (noonChange : Int) (midnightChange : Int) : Int :=
  initial + noonChange - midnightChange

/-- Theorem stating that the final temperature at midnight is -4°C --/
theorem midnight_temperature :
  finalTemperature (-2) 6 8 = -4 := by
  sorry

end midnight_temperature_l3385_338591


namespace absolute_value_inequality_l3385_338566

theorem absolute_value_inequality (x : ℝ) : 
  x ≠ 3 → (|(3 * x + 2) / (x - 3)| < 4 ↔ (10/7 < x ∧ x < 3) ∨ (3 < x ∧ x < 14)) :=
by sorry

end absolute_value_inequality_l3385_338566


namespace fred_has_four_dimes_l3385_338552

/-- The number of dimes Fred has after his sister borrowed some -/
def fred_remaining_dimes (initial : ℕ) (borrowed : ℕ) : ℕ :=
  initial - borrowed

/-- Theorem stating that Fred has 4 dimes after his sister borrowed 3 from his initial 7 -/
theorem fred_has_four_dimes :
  fred_remaining_dimes 7 3 = 4 := by
  sorry

end fred_has_four_dimes_l3385_338552


namespace factorization_equality_l3385_338512

theorem factorization_equality (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2*b) * (a - 2*b) := by
  sorry

end factorization_equality_l3385_338512


namespace planes_parallel_if_perpendicular_to_same_line_l3385_338545

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_same_line 
  (m : Line) (α β : Plane) (h1 : α ≠ β) :
  perpendicular m α → perpendicular m β → parallel α β :=
sorry

end planes_parallel_if_perpendicular_to_same_line_l3385_338545


namespace y_coordinate_relationship_l3385_338581

/-- A quadratic function of the form y = -(x-2)² + h -/
def f (h : ℝ) (x : ℝ) : ℝ := -(x - 2)^2 + h

/-- Theorem stating the relationship between y-coordinates of three points on the quadratic function -/
theorem y_coordinate_relationship (h : ℝ) (y₁ y₂ y₃ : ℝ) 
  (hA : f h (-1/2) = y₁)
  (hB : f h 1 = y₂)
  (hC : f h 2 = y₃) :
  y₁ < y₂ ∧ y₂ < y₃ := by
  sorry

end y_coordinate_relationship_l3385_338581


namespace hyperbola_parameters_l3385_338592

/-- Prove that for a hyperbola with given properties, its parameters satisfy specific values -/
theorem hyperbola_parameters (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a^2 + b^2) / a^2 = 4 →  -- eccentricity is 2
  (a * b / Real.sqrt (a^2 + b^2))^2 = 3 →  -- asymptote is tangent to the circle
  a^2 = 4 ∧ b^2 = 12 := by sorry

end hyperbola_parameters_l3385_338592


namespace least_months_to_triple_l3385_338546

theorem least_months_to_triple (rate : ℝ) (triple : ℝ) : ∃ (n : ℕ), n > 0 ∧ (1 + rate)^n > triple ∧ ∀ (m : ℕ), m > 0 → m < n → (1 + rate)^m ≤ triple :=
  by
  -- Let rate be 0.06 (6%) and triple be 3
  have h1 : rate = 0.06 := by sorry
  have h2 : triple = 3 := by sorry
  
  -- The answer is 19
  use 19
  
  sorry -- Skip the proof

end least_months_to_triple_l3385_338546


namespace students_in_both_clubs_l3385_338570

/-- The number of students in both drama and science clubs at Lincoln High School -/
theorem students_in_both_clubs 
  (total_students : ℕ)
  (drama_club : ℕ)
  (science_club : ℕ)
  (either_club : ℕ)
  (h1 : total_students = 250)
  (h2 : drama_club = 100)
  (h3 : science_club = 130)
  (h4 : either_club = 210) :
  drama_club + science_club - either_club = 20 := by
sorry

end students_in_both_clubs_l3385_338570


namespace centripetal_acceleration_proportionality_l3385_338551

/-- Centripetal acceleration proportionality -/
theorem centripetal_acceleration_proportionality
  (a v r ω T : ℝ) (h1 : a = v^2 / r) (h2 : a = r * ω^2) (h3 : a = 4 * Real.pi^2 * r / T^2) :
  (∃ k1 : ℝ, a = k1 * (v^2 / r)) ∧
  (∃ k2 : ℝ, a = k2 * (r * ω^2)) ∧
  (∃ k3 : ℝ, a = k3 * (r / T^2)) :=
by sorry

end centripetal_acceleration_proportionality_l3385_338551


namespace evaluate_F_of_f_l3385_338506

-- Define the functions f and F
def f (a : ℝ) : ℝ := a^2 - 1
def F (a b : ℝ) : ℝ := b^3 - a

-- State the theorem
theorem evaluate_F_of_f : F 2 (f 3) = 510 := by
  sorry

end evaluate_F_of_f_l3385_338506


namespace street_length_calculation_l3385_338518

/-- Proves that the length of a street is 1800 meters, given that a person crosses it in 12 minutes at a speed of 9 km per hour. -/
theorem street_length_calculation (crossing_time : ℝ) (speed_kmh : ℝ) :
  crossing_time = 12 →
  speed_kmh = 9 →
  (speed_kmh * 1000 / 60) * crossing_time = 1800 := by
sorry

end street_length_calculation_l3385_338518


namespace r_plus_s_equals_12_l3385_338565

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = -1/2 * x + 8

-- Define points P and Q
def P : ℝ × ℝ := (16, 0)
def Q : ℝ × ℝ := (0, 8)

-- Define point T
def T (r s : ℝ) : ℝ × ℝ := (r, s)

-- Define that T is on line segment PQ
def T_on_PQ (r s : ℝ) : Prop :=
  line_equation r s ∧ 0 ≤ r ∧ r ≤ 16

-- Define the area of triangle POQ
def area_POQ : ℝ := 64

-- Define the area of triangle TOP
def area_TOP (s : ℝ) : ℝ := 8 * s

-- Theorem statement
theorem r_plus_s_equals_12 (r s : ℝ) :
  T_on_PQ r s → area_POQ = 2 * area_TOP s → r + s = 12 :=
sorry

end r_plus_s_equals_12_l3385_338565


namespace merry_go_round_time_l3385_338541

/-- The time taken for the second horse to travel the same distance as the first horse -/
theorem merry_go_round_time (r₁ r₂ : ℝ) (rev : ℕ) (v₁ v₂ : ℝ) : 
  r₁ = 30 → r₂ = 15 → rev = 40 → v₁ = 3 → v₂ = 6 → 
  (2 * Real.pi * r₂ * (rev * 2 * Real.pi * r₁) / v₂) = (400 * Real.pi) := by
  sorry

#check merry_go_round_time

end merry_go_round_time_l3385_338541


namespace samuel_coaching_fee_l3385_338548

/-- Calculates the number of days in a month, assuming a non-leap year -/
def daysInMonth (month : Nat) : Nat :=
  match month with
  | 1 | 3 | 5 | 7 | 8 | 10 | 12 => 31
  | 4 | 6 | 9 | 11 => 30
  | 2 => 28
  | _ => 0

/-- Calculates the total number of days from January 1 to a given date -/
def daysFromNewYear (month : Nat) (day : Nat) : Nat :=
  (List.range (month - 1)).foldl (fun acc m => acc + daysInMonth (m + 1)) day

/-- Represents the coaching period and daily fee -/
structure CoachingData where
  startMonth : Nat
  startDay : Nat
  endMonth : Nat
  endDay : Nat
  dailyFee : Nat

/-- Calculates the total coaching fee -/
def totalCoachingFee (data : CoachingData) : Nat :=
  let totalDays := daysFromNewYear data.endMonth data.endDay - daysFromNewYear data.startMonth data.startDay + 1
  totalDays * data.dailyFee

/-- Theorem: The total coaching fee for Samuel is 7084 dollars -/
theorem samuel_coaching_fee :
  let data : CoachingData := {
    startMonth := 1,
    startDay := 1,
    endMonth := 11,
    endDay := 4,
    dailyFee := 23
  }
  totalCoachingFee data = 7084 := by
  sorry


end samuel_coaching_fee_l3385_338548


namespace quadratic_function_properties_l3385_338532

/-- A quadratic function with a non-zero leading coefficient -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The function value at a given point -/
def QuadraticFunction.value (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- The derivative of the quadratic function -/
def QuadraticFunction.derivative (f : QuadraticFunction) (x : ℝ) : ℝ :=
  2 * f.a * x + f.b

theorem quadratic_function_properties (f : QuadraticFunction) 
  (h1 : f.derivative 1 = 0)
  (h2 : f.value 1 = 3)
  (h3 : f.value 2 = 8) :
  f.value (-1) ≠ 0 := by
  sorry

end quadratic_function_properties_l3385_338532


namespace digits_1198_to_1200_form_473_l3385_338563

/-- A function that generates the list of positive integers with first digit 1 or 2 -/
def firstDigitOneOrTwo : ℕ → Bool := sorry

/-- The number of digits written before reaching a given position in the list -/
def digitCount (n : ℕ) : ℕ := sorry

/-- The number at a given position in the list -/
def numberAtPosition (n : ℕ) : ℕ := sorry

theorem digits_1198_to_1200_form_473 :
  let pos := 1198
  ∃ (n : ℕ), 
    firstDigitOneOrTwo n ∧ 
    digitCount n ≤ pos ∧ 
    digitCount (n + 1) > pos + 2 ∧
    numberAtPosition n = 473 := by sorry

end digits_1198_to_1200_form_473_l3385_338563


namespace even_monotone_increasing_range_l3385_338520

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem even_monotone_increasing_range 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_mono : monotone_increasing_on f (Set.Ici 0)) :
  {x : ℝ | f x < f 1} = Set.Ioo (-1) 1 := by
  sorry

end even_monotone_increasing_range_l3385_338520


namespace g_of_two_eq_zero_l3385_338553

/-- The function g(x) = x^2 - 4x + 4 -/
def g (x : ℝ) : ℝ := x^2 - 4*x + 4

/-- Theorem: g(2) = 0 -/
theorem g_of_two_eq_zero : g 2 = 0 := by
  sorry

end g_of_two_eq_zero_l3385_338553


namespace greatest_value_l3385_338575

theorem greatest_value (p : ℝ) (a b c d : ℝ) 
  (h1 : a + 1 = p) 
  (h2 : b - 2 = p) 
  (h3 : c + 3 = p) 
  (h4 : d - 4 = p) : 
  d > a ∧ d > b ∧ d > c :=
by sorry

end greatest_value_l3385_338575


namespace arithmetic_mean_two_digit_multiples_of_8_l3385_338543

/-- The smallest positive two-digit multiple of 8 -/
def smallest_multiple : ℕ := 16

/-- The largest positive two-digit multiple of 8 -/
def largest_multiple : ℕ := 96

/-- The count of positive two-digit multiples of 8 -/
def count_multiples : ℕ := 11

/-- The sum of all positive two-digit multiples of 8 -/
def sum_multiples : ℕ := 616

/-- Theorem stating that the arithmetic mean of all positive two-digit multiples of 8 is 56 -/
theorem arithmetic_mean_two_digit_multiples_of_8 :
  (sum_multiples : ℚ) / count_multiples = 56 := by sorry

end arithmetic_mean_two_digit_multiples_of_8_l3385_338543


namespace range_of_m_for_trig_equation_l3385_338593

theorem range_of_m_for_trig_equation :
  ∀ α m : ℝ,
  (∃ α, Real.cos α - Real.sqrt 3 * Real.sin α = (4 * m - 6) / (4 - m)) →
  -1 ≤ m ∧ m ≤ 7/3 :=
by sorry

end range_of_m_for_trig_equation_l3385_338593


namespace triangle_angle_b_is_pi_third_l3385_338558

theorem triangle_angle_b_is_pi_third 
  (A B C : Real) 
  (a b c : Real) 
  (h1 : b^2 = a*c) 
  (h2 : Real.sin A + Real.sin C = 2 * Real.sin B) 
  (h3 : A + B + C = π) 
  (h4 : a > 0 ∧ b > 0 ∧ c > 0) : 
  B = π/3 := by
  sorry

end triangle_angle_b_is_pi_third_l3385_338558


namespace boisjoli_farm_egg_boxes_l3385_338510

/-- The number of boxes filled with eggs in a week -/
def boxes_filled_per_week (num_hens : ℕ) (days_per_week : ℕ) (eggs_per_box : ℕ) : ℕ :=
  (num_hens * days_per_week) / eggs_per_box

/-- Theorem stating that 270 hens laying eggs for 7 days, packed in boxes of 6, results in 315 boxes per week -/
theorem boisjoli_farm_egg_boxes :
  boxes_filled_per_week 270 7 6 = 315 := by
  sorry

end boisjoli_farm_egg_boxes_l3385_338510


namespace chess_game_probability_l3385_338584

theorem chess_game_probability (p_not_losing p_draw : ℝ) 
  (h1 : p_not_losing = 0.8)
  (h2 : p_draw = 0.5) :
  p_not_losing - p_draw = 0.3 := by
sorry

end chess_game_probability_l3385_338584


namespace binomial_1500_1_l3385_338500

theorem binomial_1500_1 : Nat.choose 1500 1 = 1500 := by
  sorry

end binomial_1500_1_l3385_338500


namespace f_is_quadratic_l3385_338587

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 + 5x = 0 -/
def f (x : ℝ) : ℝ := x^2 + 5*x

/-- Theorem: f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f :=
sorry

end f_is_quadratic_l3385_338587


namespace cubic_inequality_solution_l3385_338549

theorem cubic_inequality_solution (x : ℝ) :
  x^3 - 12*x^2 + 36*x > 0 ↔ (x > 0 ∧ x < 6) ∨ x > 6 := by
  sorry

end cubic_inequality_solution_l3385_338549


namespace problem_statement_l3385_338507

theorem problem_statement (n b : ℝ) : 
  n = 2^(7/3) → n^(3*b + 5) = 256 → b = -11/21 := by
  sorry

end problem_statement_l3385_338507


namespace chocolate_ratio_simplification_l3385_338514

theorem chocolate_ratio_simplification :
  let white_chocolate : ℕ := 20
  let dark_chocolate : ℕ := 15
  let gcd := Nat.gcd white_chocolate dark_chocolate
  (white_chocolate / gcd : ℚ) / (dark_chocolate / gcd : ℚ) = 4 / 3 := by
  sorry

end chocolate_ratio_simplification_l3385_338514


namespace exists_perfect_square_2022_not_perfect_square_for_a_2_l3385_338530

-- Part (a)
theorem exists_perfect_square_2022 : ∃ n : ℕ, ∃ k : ℕ, n * (n + 2022) + 2 = k^2 := by
  sorry

-- Part (b)
theorem not_perfect_square_for_a_2 : ∀ n : ℕ, ¬∃ k : ℕ, n * (n + 2) + 2 = k^2 := by
  sorry

end exists_perfect_square_2022_not_perfect_square_for_a_2_l3385_338530


namespace biotechnology_graduates_l3385_338557

theorem biotechnology_graduates (total : ℕ) (job : ℕ) (second_degree : ℕ) (neither : ℕ) :
  total = 73 →
  job = 32 →
  second_degree = 45 →
  neither = 9 →
  ∃ (both : ℕ), both = 13 ∧ job + second_degree - both = total - neither :=
by sorry

end biotechnology_graduates_l3385_338557


namespace equation_solutions_l3385_338539

theorem equation_solutions :
  let f (x : ℝ) := 3 / (Real.sqrt (x - 5) - 7) + 2 / (Real.sqrt (x - 5) - 3) +
                   9 / (Real.sqrt (x - 5) + 3) + 15 / (Real.sqrt (x - 5) + 7)
  ∀ x : ℝ, f x = 0 ↔ x = 54 ∨ x = 846 / 29 := by
  sorry

end equation_solutions_l3385_338539


namespace rectangle_area_l3385_338523

theorem rectangle_area (x : ℝ) : 
  (2 * (x + 4) + 2 * (x - 2) = 56) → 
  ((x + 4) * (x - 2) = 187) :=
by sorry

end rectangle_area_l3385_338523


namespace combined_transformation_correct_l3385_338513

/-- A dilation centered at the origin with scale factor k -/
def dilation (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.diagonal (λ _ => k)

/-- A reflection across the x-axis -/
def reflectionX : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.diagonal (λ i => if i = 0 then 1 else -1)

/-- The combined transformation matrix -/
def combinedTransformation : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.diagonal (λ i => if i = 0 then 5 else -5)

theorem combined_transformation_correct :
  combinedTransformation = reflectionX * dilation 5 := by
  sorry


end combined_transformation_correct_l3385_338513


namespace sufficient_not_necessary_l3385_338574

def M : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem sufficient_not_necessary : 
  (∀ a, a ∈ M → a ∈ N) ∧ (∃ a, a ∈ N ∧ a ∉ M) := by sorry

end sufficient_not_necessary_l3385_338574


namespace x_gt_y_necessary_not_sufficient_l3385_338536

theorem x_gt_y_necessary_not_sufficient (x y : ℝ) (hx : x > 0) :
  (∀ y, x > |y| → x > y) ∧ ¬(∀ y, x > y → x > |y|) := by
  sorry

end x_gt_y_necessary_not_sufficient_l3385_338536


namespace absolute_value_inequality_l3385_338583

theorem absolute_value_inequality (x : ℝ) : 
  abs (x - 1) + abs (x + 2) < 5 ↔ -3 < x ∧ x < 2 := by sorry

end absolute_value_inequality_l3385_338583


namespace marble_arrangement_remainder_l3385_338522

/-- Represents the number of green marbles --/
def green_marbles : ℕ := 7

/-- Represents the minimum number of red marbles required --/
def min_red_marbles : ℕ := green_marbles + 1

/-- Represents the maximum number of additional red marbles that can be added --/
def max_additional_reds : ℕ := min_red_marbles

/-- Represents the total number of spaces where additional red marbles can be placed --/
def total_spaces : ℕ := green_marbles + 1

/-- Represents the number of ways to arrange the marbles --/
def arrangement_count : ℕ := Nat.choose (max_additional_reds + total_spaces - 1) (total_spaces - 1)

theorem marble_arrangement_remainder :
  arrangement_count % 1000 = 435 := by sorry

end marble_arrangement_remainder_l3385_338522


namespace earnings_difference_l3385_338580

/-- Calculates the difference in earnings between two sets of tasks with different pay rates -/
theorem earnings_difference (low_tasks : ℕ) (low_rate : ℚ) (high_tasks : ℕ) (high_rate : ℚ) :
  low_tasks = 400 →
  low_rate = 1/4 →
  high_tasks = 5 →
  high_rate = 2 →
  (low_tasks : ℚ) * low_rate - (high_tasks : ℚ) * high_rate = 90 :=
by sorry

end earnings_difference_l3385_338580


namespace compound_weight_l3385_338531

/-- The atomic weight of Aluminum-27 in atomic mass units -/
def aluminum_weight : ℕ := 27

/-- The atomic weight of Iodine-127 in atomic mass units -/
def iodine_weight : ℕ := 127

/-- The atomic weight of Oxygen-16 in atomic mass units -/
def oxygen_weight : ℕ := 16

/-- The number of Aluminum-27 atoms in the compound -/
def aluminum_count : ℕ := 1

/-- The number of Iodine-127 atoms in the compound -/
def iodine_count : ℕ := 3

/-- The number of Oxygen-16 atoms in the compound -/
def oxygen_count : ℕ := 2

/-- The molecular weight of the compound -/
def molecular_weight : ℕ := 
  aluminum_count * aluminum_weight + 
  iodine_count * iodine_weight + 
  oxygen_count * oxygen_weight

theorem compound_weight : molecular_weight = 440 := by
  sorry

end compound_weight_l3385_338531


namespace solution_satisfies_equation_l3385_338556

/-- The general solution to the differential equation (4y - 3x - 5)y' + 7x - 3y + 2 = 0 -/
def general_solution (x y : ℝ) (C : ℝ) : Prop :=
  2 * y^2 - 3 * x * y + (7/2) * x^2 + 2 * x - 5 * y = C

/-- The differential equation (4y - 3x - 5)y' + 7x - 3y + 2 = 0 -/
def differential_equation (x y : ℝ) (y' : ℝ → ℝ) : Prop :=
  (4 * y - 3 * x - 5) * (y' x) + 7 * x - 3 * y + 2 = 0

theorem solution_satisfies_equation :
  ∀ (x y : ℝ) (C : ℝ),
  general_solution x y C →
  ∃ (y' : ℝ → ℝ), differential_equation x y y' :=
sorry

end solution_satisfies_equation_l3385_338556


namespace cos_difference_x1_x2_l3385_338519

theorem cos_difference_x1_x2 (x₁ x₂ : ℝ) 
  (h1 : 0 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < π)
  (h4 : Real.sin (2 * x₁ - π / 3) = 4 / 5)
  (h5 : Real.sin (2 * x₂ - π / 3) = 4 / 5) :
  Real.cos (x₁ - x₂) = 3 / 5 := by
  sorry

end cos_difference_x1_x2_l3385_338519


namespace betty_height_in_feet_betty_is_three_feet_tall_l3385_338567

/-- Given a dog's height, Carter's height relative to the dog, and Betty's height relative to Carter,
    calculate Betty's height in feet. -/
theorem betty_height_in_feet (dog_height : ℕ) (carter_ratio : ℕ) (betty_diff : ℕ) : ℕ :=
  let carter_height := dog_height * carter_ratio
  let betty_height_inches := carter_height - betty_diff
  betty_height_inches / 12

/-- Prove that Betty is 3 feet tall given the specific conditions. -/
theorem betty_is_three_feet_tall :
  betty_height_in_feet 24 2 12 = 3 := by
  sorry

end betty_height_in_feet_betty_is_three_feet_tall_l3385_338567


namespace james_total_spending_l3385_338595

def club_entry_fee : ℕ := 20
def friends_count : ℕ := 5
def rounds_for_friends : ℕ := 2
def james_drinks : ℕ := 6
def drink_cost : ℕ := 6
def food_cost : ℕ := 14
def tip_percentage : ℚ := 30 / 100

def total_drinks : ℕ := friends_count * rounds_for_friends + james_drinks

def order_cost : ℕ := total_drinks * drink_cost + food_cost

def tip_amount : ℚ := (order_cost : ℚ) * tip_percentage

def total_spending : ℚ := (club_entry_fee : ℚ) + (order_cost : ℚ) + tip_amount

theorem james_total_spending :
  total_spending = 163 := by sorry

end james_total_spending_l3385_338595


namespace box_edge_length_and_capacity_l3385_338555

/-- Given a cubical box that can contain 999.9999999999998 cubes of 10 cm edge length,
    prove that its edge length is 1 meter and it can contain 1000 cubes. -/
theorem box_edge_length_and_capacity (box_capacity : ℝ) 
  (h1 : box_capacity = 999.9999999999998) : ∃ (edge_length : ℝ),
  edge_length = 1 ∧ 
  (edge_length * 100 / 10)^3 = 1000 := by
  sorry

end box_edge_length_and_capacity_l3385_338555


namespace largest_divisor_of_60_36_divisible_by_3_l3385_338535

theorem largest_divisor_of_60_36_divisible_by_3 :
  ∃ (n : ℕ), n > 0 ∧ n ∣ 60 ∧ n ∣ 36 ∧ 3 ∣ n ∧
  ∀ (m : ℕ), m > n → (m ∣ 60 ∧ m ∣ 36 ∧ 3 ∣ m) → False :=
by
  use 12
  sorry

end largest_divisor_of_60_36_divisible_by_3_l3385_338535


namespace inheritance_satisfies_tax_conditions_inheritance_uniqueness_l3385_338524

/-- The inheritance amount that satisfies the tax conditions -/
def inheritance : ℝ := 41379

/-- The total tax paid -/
def total_tax : ℝ := 15000

/-- Federal tax rate -/
def federal_tax_rate : ℝ := 0.25

/-- State tax rate -/
def state_tax_rate : ℝ := 0.15

/-- Theorem stating that the inheritance amount satisfies the tax conditions -/
theorem inheritance_satisfies_tax_conditions :
  federal_tax_rate * inheritance + 
  state_tax_rate * (inheritance - federal_tax_rate * inheritance) = 
  total_tax := by sorry

/-- Theorem stating that the inheritance amount is unique -/
theorem inheritance_uniqueness (x : ℝ) :
  federal_tax_rate * x + 
  state_tax_rate * (x - federal_tax_rate * x) = 
  total_tax →
  x = inheritance := by sorry

end inheritance_satisfies_tax_conditions_inheritance_uniqueness_l3385_338524


namespace quadratic_linear_intersection_l3385_338547

/-- Quadratic function -/
def y1 (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Linear function -/
def y2 (a b x : ℝ) : ℝ := a * x + b

/-- Theorem stating the main results -/
theorem quadratic_linear_intersection 
  (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : y1 a b c 1 = 0) 
  (t : ℤ) 
  (h4 : t % 2 = 1) 
  (h5 : y1 a b c (t : ℝ) = 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ y1 a b c x1 = y2 a b x1 ∧ y1 a b c x2 = y2 a b x2) ∧ 
  (t = 1 ∨ t = -1) ∧
  (∀ A1 B1 : ℝ, y1 a b c A1 = y2 a b A1 → y1 a b c B1 = y2 a b B1 → 
    3/2 < |A1 - B1| ∧ |A1 - B1| < Real.sqrt 3) := by
  sorry

end quadratic_linear_intersection_l3385_338547


namespace volume_conversion_l3385_338526

-- Define conversion factors
def feet_to_meters : ℝ := 0.3048
def meters_to_yards : ℝ := 1.09361

-- Define the volume in cubic feet
def volume_cubic_feet : ℝ := 216

-- Define the conversion function from cubic feet to cubic meters
def cubic_feet_to_cubic_meters (v : ℝ) : ℝ := v * (feet_to_meters ^ 3)

-- Define the conversion function from cubic meters to cubic yards
def cubic_meters_to_cubic_yards (v : ℝ) : ℝ := v * (meters_to_yards ^ 3)

-- Theorem statement
theorem volume_conversion :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |cubic_meters_to_cubic_yards (cubic_feet_to_cubic_meters volume_cubic_feet) - 8| < ε :=
sorry

end volume_conversion_l3385_338526


namespace log_one_over_twentyfive_base_five_l3385_338578

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_one_over_twentyfive_base_five : log 5 (1 / 25) = -2 := by
  sorry

end log_one_over_twentyfive_base_five_l3385_338578


namespace extremum_point_implies_a_value_max_min_values_l3385_338559

-- Define the function f(x) = x^3 - ax
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

-- Theorem 1: If x=1 is an extremum point of f(x), then a = 3
theorem extremum_point_implies_a_value (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1 ∨ f a x ≥ f a 1) →
  a = 3 :=
sorry

-- Theorem 2: For f(x) = x^3 - 3x and x ∈ [0, 2], the maximum value is 2 and the minimum value is -2
theorem max_min_values :
  (∀ x ∈ Set.Icc 0 2, f 3 x ≤ 2) ∧
  (∀ x ∈ Set.Icc 0 2, f 3 x ≥ -2) ∧
  (∃ x ∈ Set.Icc 0 2, f 3 x = 2) ∧
  (∃ x ∈ Set.Icc 0 2, f 3 x = -2) :=
sorry

end extremum_point_implies_a_value_max_min_values_l3385_338559


namespace travelers_checks_average_l3385_338502

theorem travelers_checks_average (total_checks : ℕ) (total_worth : ℕ) 
  (spent_checks : ℕ) (h1 : total_checks = 30) (h2 : total_worth = 1800) 
  (h3 : spent_checks = 18) :
  let fifty_checks := (2 * total_worth - 100 * total_checks) / 50
  let hundred_checks := total_checks - fifty_checks
  let remaining_fifty := fifty_checks - spent_checks
  let remaining_total := remaining_fifty + hundred_checks
  let remaining_worth := 50 * remaining_fifty + 100 * hundred_checks
  remaining_worth / remaining_total = 75 := by
sorry

end travelers_checks_average_l3385_338502


namespace magnitude_of_3_minus_4i_l3385_338540

theorem magnitude_of_3_minus_4i : Complex.abs (3 - 4 * Complex.I) = 5 := by
  sorry

end magnitude_of_3_minus_4i_l3385_338540


namespace system_solution_range_l3385_338528

theorem system_solution_range (x y a : ℝ) :
  (2 * x + y = 3 - a) →
  (x + 2 * y = 4 + 2 * a) →
  (x + y < 1) →
  (a < -4) :=
by sorry

end system_solution_range_l3385_338528


namespace carrie_tshirt_purchase_l3385_338582

/-- The number of t-shirts Carrie bought -/
def num_tshirts : ℕ := 24

/-- The cost of each t-shirt in dollars -/
def cost_per_tshirt : ℚ := 9.95

/-- The total amount Carrie spent in dollars -/
def total_spent : ℚ := 248

/-- Theorem stating that the number of t-shirts Carrie bought is correct -/
theorem carrie_tshirt_purchase : 
  (↑num_tshirts : ℚ) * cost_per_tshirt ≤ total_spent ∧ 
  (↑(num_tshirts + 1) : ℚ) * cost_per_tshirt > total_spent :=
by sorry

end carrie_tshirt_purchase_l3385_338582


namespace ellipse_inscribed_triangle_uniqueness_l3385_338508

/-- Represents an ellipse with semi-major axis a and semi-minor axis 1 -/
def Ellipse (a : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + p.2^2 = 1}

/-- Represents a right-angled isosceles triangle inscribed in the ellipse -/
def InscribedTriangle (a : ℝ) := 
  {triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) | 
    let (A, B, C) := triangle
    B = (0, 1) ∧ 
    A ∈ Ellipse a ∧ 
    C ∈ Ellipse a ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 ∧
    (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0}

/-- The main theorem -/
theorem ellipse_inscribed_triangle_uniqueness (a : ℝ) 
  (h1 : a > 1) 
  (h2 : ∃! triangle, triangle ∈ InscribedTriangle a) : 
  1 < a ∧ a ≤ Real.sqrt 3 :=
by sorry

end ellipse_inscribed_triangle_uniqueness_l3385_338508


namespace green_pill_cost_proof_l3385_338560

/-- The cost of a green pill in dollars -/
def green_pill_cost : ℚ := 41 / 3

/-- The cost of a pink pill in dollars -/
def pink_pill_cost : ℚ := green_pill_cost - 1

/-- The number of days in the treatment period -/
def treatment_days : ℕ := 21

/-- The total cost of medication for the treatment period -/
def total_cost : ℚ := 819

theorem green_pill_cost_proof :
  (green_pill_cost + 2 * pink_pill_cost) * treatment_days = total_cost :=
sorry

end green_pill_cost_proof_l3385_338560


namespace marble_problem_l3385_338529

theorem marble_problem (atticus jensen cruz harper : ℕ) : 
  4 * (atticus + jensen + cruz + harper) = 120 →
  atticus = jensen / 2 →
  atticus = 4 →
  jensen = 2 * harper →
  cruz = 14 := by
  sorry

end marble_problem_l3385_338529


namespace smallest_valid_n_l3385_338527

def is_valid_n (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  2 * n = 100 * c + 10 * b + a + 5

theorem smallest_valid_n :
  ∃ (n : ℕ), is_valid_n n ∧ ∀ m, is_valid_n m → n ≤ m :=
sorry

end smallest_valid_n_l3385_338527


namespace subscription_savings_l3385_338562

def category_A_cost : ℝ := 520
def category_B_cost : ℝ := 860
def category_C_cost : ℝ := 620

def category_A_cut_percentage : ℝ := 0.25
def category_B_cut_percentage : ℝ := 0.35
def category_C_cut_percentage : ℝ := 0.30

def total_savings : ℝ :=
  category_A_cost * category_A_cut_percentage +
  category_B_cost * category_B_cut_percentage +
  category_C_cost * category_C_cut_percentage

theorem subscription_savings : total_savings = 617 := by
  sorry

end subscription_savings_l3385_338562


namespace gcd_consecutive_b_terms_bound_l3385_338564

def b (n : ℕ) : ℕ := (2 * n).factorial + n^2

theorem gcd_consecutive_b_terms_bound (n : ℕ) (h : n ≥ 1) :
  Nat.gcd (b n) (b (n + 1)) ≤ 1 := by
  sorry

end gcd_consecutive_b_terms_bound_l3385_338564


namespace simplify_expression_l3385_338599

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^4 + b^4 = a^2 + b^2) :
  a/b + b/a - 1/(a*b) = 3 := by
  sorry

end simplify_expression_l3385_338599


namespace negation_equivalence_l3385_338594

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - 4*x₀ + 1 < 0) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 4*x + 1 ≥ 0) := by sorry

end negation_equivalence_l3385_338594


namespace favorite_fruit_oranges_l3385_338573

theorem favorite_fruit_oranges (total students_pears students_apples students_strawberries : ℕ) 
  (h_total : total = 450)
  (h_pears : students_pears = 120)
  (h_apples : students_apples = 147)
  (h_strawberries : students_strawberries = 113) :
  total - (students_pears + students_apples + students_strawberries) = 70 := by
  sorry

end favorite_fruit_oranges_l3385_338573


namespace cone_volume_l3385_338511

/-- The volume of a cone with slant height 5 and lateral area 20π is 16π -/
theorem cone_volume (s : ℝ) (lateral_area : ℝ) (h : s = 5) (h' : lateral_area = 20 * Real.pi) :
  (1 / 3 : ℝ) * Real.pi * (lateral_area / (Real.pi * s))^2 * Real.sqrt (s^2 - (lateral_area / (Real.pi * s))^2) = 16 * Real.pi :=
by sorry

end cone_volume_l3385_338511


namespace smallest_m_correct_l3385_338509

/-- The smallest positive integer m for which 10x^2 - mx + 420 = 0 has integral solutions -/
def smallest_m : ℕ := 130

/-- Predicate to check if a quadratic equation has integral solutions -/
def has_integral_solutions (a b c : ℤ) : Prop :=
  ∃ x : ℤ, a * x^2 + b * x + c = 0

theorem smallest_m_correct :
  (∀ m : ℕ, m < smallest_m → ¬ has_integral_solutions 10 (-m) 420) ∧
  has_integral_solutions 10 (-smallest_m) 420 :=
sorry

end smallest_m_correct_l3385_338509


namespace percentage_difference_l3385_338571

theorem percentage_difference : (45 / 100 * 60) - (35 / 100 * 40) = 13 := by
  sorry

end percentage_difference_l3385_338571


namespace pyramid_intersection_theorem_l3385_338572

structure Pyramid where
  base : Rectangle
  side_edge : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Given a pyramid with a rectangular base and equal side edges, when a plane intersects
    the side edges cutting off segments a, b, c, and d, the equation 1/a + 1/c = 1/b + 1/d holds. -/
theorem pyramid_intersection_theorem (p : Pyramid) (ha : p.a > 0) (hb : p.b > 0) (hc : p.c > 0) (hd : p.d > 0) :
  1 / p.a + 1 / p.c = 1 / p.b + 1 / p.d := by
  sorry

end pyramid_intersection_theorem_l3385_338572


namespace first_part_speed_l3385_338596

theorem first_part_speed (total_distance : ℝ) (first_part_distance : ℝ) (second_part_speed : ℝ) (average_speed : ℝ) 
  (h1 : total_distance = 60)
  (h2 : first_part_distance = 12)
  (h3 : second_part_speed = 48)
  (h4 : average_speed = 40)
  (h5 : total_distance = first_part_distance + (total_distance - first_part_distance))
  (h6 : average_speed = total_distance / (first_part_distance / v + (total_distance - first_part_distance) / second_part_speed)) :
  v = 24 := by
  sorry

end first_part_speed_l3385_338596


namespace sunglasses_sold_l3385_338544

/-- Proves that the number of pairs of sunglasses sold is 10 -/
theorem sunglasses_sold (selling_price cost_price sign_cost : ℕ) 
  (h1 : selling_price = 30)
  (h2 : cost_price = 26)
  (h3 : sign_cost = 20) :
  (sign_cost * 2) / (selling_price - cost_price) = 10 := by
  sorry

end sunglasses_sold_l3385_338544


namespace petya_vasya_meeting_l3385_338598

/-- The number of street lamps along the alley -/
def num_lamps : ℕ := 100

/-- The lamp number where Petya is observed -/
def petya_observed : ℕ := 22

/-- The lamp number where Vasya is observed -/
def vasya_observed : ℕ := 88

/-- Calculates the meeting point of Petya and Vasya -/
def meeting_point : ℕ := 64

/-- Theorem stating that Petya and Vasya meet at the calculated meeting point -/
theorem petya_vasya_meeting :
  ∀ (petya_speed vasya_speed : ℚ),
  petya_speed > 0 ∧ vasya_speed > 0 →
  (petya_speed * (meeting_point - 1) = vasya_speed * (num_lamps - meeting_point)) ∧
  (petya_speed * (petya_observed - 1) = vasya_speed * (num_lamps - vasya_observed)) :=
by sorry

#check petya_vasya_meeting

end petya_vasya_meeting_l3385_338598


namespace highest_power_of_three_for_concatenated_range_l3385_338579

def concatenate_range (a b : ℕ) : ℕ := sorry

def highest_power_of_three (n : ℕ) : ℕ := sorry

theorem highest_power_of_three_for_concatenated_range :
  let N := concatenate_range 31 73
  highest_power_of_three N = 1 := by sorry

end highest_power_of_three_for_concatenated_range_l3385_338579
