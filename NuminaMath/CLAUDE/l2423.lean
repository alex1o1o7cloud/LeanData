import Mathlib

namespace audrey_sleep_time_l2423_242336

/-- Given that Audrey dreamed for 2/5 of her sleep time and was not dreaming for 6 hours,
    prove that she was asleep for 10 hours. -/
theorem audrey_sleep_time :
  ∀ (total_sleep : ℝ),
  (2 / 5 : ℝ) * total_sleep + 6 = total_sleep →
  total_sleep = 10 :=
by
  sorry

end audrey_sleep_time_l2423_242336


namespace binomial_1293_2_l2423_242397

theorem binomial_1293_2 : Nat.choose 1293 2 = 835218 := by sorry

end binomial_1293_2_l2423_242397


namespace union_of_sets_l2423_242311

theorem union_of_sets : 
  let P : Set ℕ := {1, 2}
  let Q : Set ℕ := {2, 3}
  P ∪ Q = {1, 2, 3} := by sorry

end union_of_sets_l2423_242311


namespace strategy2_is_cheaper_l2423_242376

def original_price : ℝ := 12000

def strategy1_cost (price : ℝ) : ℝ :=
  price * (1 - 0.30) * (1 - 0.15) * (1 - 0.05)

def strategy2_cost (price : ℝ) : ℝ :=
  price * (1 - 0.45) * (1 - 0.10) * (1 - 0.10) + 150

theorem strategy2_is_cheaper :
  strategy2_cost original_price < strategy1_cost original_price :=
by sorry

end strategy2_is_cheaper_l2423_242376


namespace volume_is_zero_l2423_242390

def S : Set (ℝ × ℝ) := {(x, y) | |6 - x| + y ≤ 8 ∧ 2*y - x ≥ 10}

def revolution_axis : Set (ℝ × ℝ) := {(x, y) | 2*y - x = 10}

def volume_of_revolution (region : Set (ℝ × ℝ)) (axis : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem volume_is_zero :
  volume_of_revolution S revolution_axis = 0 := by sorry

end volume_is_zero_l2423_242390


namespace tan_equality_implies_negative_thirty_l2423_242363

theorem tan_equality_implies_negative_thirty (n : ℤ) :
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1230 * π / 180) →
  n = -30 := by sorry

end tan_equality_implies_negative_thirty_l2423_242363


namespace faculty_reduction_l2423_242362

theorem faculty_reduction (initial_faculty : ℕ) : 
  (initial_faculty : ℝ) * 0.85 * 0.80 = 195 → 
  initial_faculty = 287 := by
  sorry

end faculty_reduction_l2423_242362


namespace planes_perpendicular_l2423_242378

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular 
  (l m : Line) (α β : Plane)
  (h1 : subset l α)
  (h2 : parallel l m)
  (h3 : perpendicular m β) :
  perp_planes α β :=
sorry

end planes_perpendicular_l2423_242378


namespace problem_solution_l2423_242361

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1|

theorem problem_solution :
  (∀ x : ℝ, f 2 x + g x ≤ 7 ↔ -1/2 ≤ x ∧ x ≤ 2) ∧
  (∀ a : ℝ, (∀ x : ℝ, g x ≤ 5 → f a x ≤ 6) ↔ a ≤ 1) :=
sorry

end problem_solution_l2423_242361


namespace distance_for_boy_problem_l2423_242300

/-- Calculates the distance covered given time in minutes and speed in meters per second -/
def distance_covered (time_minutes : ℕ) (speed_meters_per_second : ℕ) : ℕ :=
  time_minutes * 60 * speed_meters_per_second

/-- Proves that given 30 minutes and a speed of 1 meter per second, the distance covered is 1800 meters -/
theorem distance_for_boy_problem : distance_covered 30 1 = 1800 := by
  sorry

#eval distance_covered 30 1

end distance_for_boy_problem_l2423_242300


namespace atomic_weight_Al_l2423_242384

/-- The atomic weight of oxygen -/
def atomic_weight_O : ℝ := 16

/-- The molecular weight of Al2O3 -/
def molecular_weight_Al2O3 : ℝ := 102

/-- The number of aluminum atoms in Al2O3 -/
def num_Al_atoms : ℕ := 2

/-- The number of oxygen atoms in Al2O3 -/
def num_O_atoms : ℕ := 3

/-- Theorem stating that the atomic weight of Al is 27 -/
theorem atomic_weight_Al :
  (molecular_weight_Al2O3 - num_O_atoms * atomic_weight_O) / num_Al_atoms = 27 := by
  sorry

end atomic_weight_Al_l2423_242384


namespace smallest_three_digit_integer_l2423_242356

theorem smallest_three_digit_integer (n : ℕ) : 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (45 * n ≡ 135 [MOD 280]) ∧ 
  (n ≡ 3 [MOD 7]) →
  n ≥ 115 :=
by sorry

end smallest_three_digit_integer_l2423_242356


namespace vector_equation_l2423_242327

variable {V : Type*} [AddCommGroup V]

theorem vector_equation (O A B C : V) :
  (A - O) - (B - O) + (C - A) = C - B := by sorry

end vector_equation_l2423_242327


namespace factorial_ratio_l2423_242375

theorem factorial_ratio : Nat.factorial 15 / (Nat.factorial 6 * Nat.factorial 9) = 5005 := by
  sorry

end factorial_ratio_l2423_242375


namespace sheep_in_wilderness_l2423_242340

/-- Given that 90% of sheep are in a pen and there are 81 sheep in the pen,
    prove that there are 9 sheep in the wilderness. -/
theorem sheep_in_wilderness (total : ℕ) (in_pen : ℕ) (h1 : in_pen = 81) 
    (h2 : in_pen = (90 : ℕ) * total / 100) : total - in_pen = 9 := by
  sorry

end sheep_in_wilderness_l2423_242340


namespace not_octal_7857_l2423_242316

def is_octal_digit (d : Nat) : Prop := d ≤ 7

def is_octal_number (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 8 → is_octal_digit d

theorem not_octal_7857 : ¬ is_octal_number 7857 := by
  sorry

end not_octal_7857_l2423_242316


namespace derivative_sin_2x_l2423_242320

theorem derivative_sin_2x (x : ℝ) : 
  deriv (fun x => Real.sin (2 * x)) x = 2 * Real.cos (2 * x) := by
sorry

end derivative_sin_2x_l2423_242320


namespace swim_club_members_l2423_242352

theorem swim_club_members :
  ∀ (total_members : ℕ) 
    (passed_test : ℕ) 
    (not_passed_with_course : ℕ) 
    (not_passed_without_course : ℕ),
  passed_test = (30 * total_members) / 100 →
  not_passed_with_course = 5 →
  not_passed_without_course = 30 →
  total_members = passed_test + not_passed_with_course + not_passed_without_course →
  total_members = 50 := by
sorry

end swim_club_members_l2423_242352


namespace polynomial_sequence_problem_l2423_242345

theorem polynomial_sequence_problem (a b x y : ℝ) 
  (eq1 : a * x + b * y = 3)
  (eq2 : a * x^2 + b * y^2 = 7)
  (eq3 : a * x^3 + b * y^3 = 16)
  (eq4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := by
sorry

end polynomial_sequence_problem_l2423_242345


namespace regular_polygon_exterior_angle_l2423_242357

theorem regular_polygon_exterior_angle (n : ℕ) (h : (n - 2) * 180 = 1800) :
  360 / n = 30 := by
  sorry

end regular_polygon_exterior_angle_l2423_242357


namespace complement_M_in_U_l2423_242369

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define the set M
def M : Set ℝ := {x | x > 1}

-- State the theorem
theorem complement_M_in_U : 
  (U \ M) = {x | 0 < x ∧ x ≤ 1} := by sorry

end complement_M_in_U_l2423_242369


namespace frans_original_seat_l2423_242315

/-- Represents the seats in the theater --/
inductive Seat
  | one
  | two
  | three
  | four
  | five
  | six

/-- Represents the friends --/
inductive Friend
  | ada
  | bea
  | ceci
  | dee
  | edie
  | fran

/-- Represents the direction of movement --/
inductive Direction
  | left
  | right

/-- Represents a movement of a friend --/
structure Movement where
  friend : Friend
  distance : Nat
  direction : Direction

/-- The initial seating arrangement --/
def initialSeating : Friend → Seat := sorry

/-- The movements of the friends --/
def friendMovements : List Movement := [
  ⟨Friend.ada, 3, Direction.right⟩,
  ⟨Friend.bea, 2, Direction.left⟩,
  ⟨Friend.ceci, 0, Direction.right⟩,
  ⟨Friend.dee, 0, Direction.right⟩,
  ⟨Friend.edie, 1, Direction.right⟩
]

/-- Function to apply movements and get the final seating arrangement --/
def applyMovements (initial : Friend → Seat) (movements : List Movement) : Friend → Seat := sorry

/-- Function to find the vacant seat after movements --/
def findVacantSeat (seating : Friend → Seat) : Seat := sorry

/-- Theorem stating Fran's original seat --/
theorem frans_original_seat :
  initialSeating Friend.fran = Seat.three ∧
  (findVacantSeat (applyMovements initialSeating friendMovements) = Seat.one ∨
   findVacantSeat (applyMovements initialSeating friendMovements) = Seat.six) := by
  sorry

end frans_original_seat_l2423_242315


namespace hyperbola_equation_l2423_242380

-- Define the hyperbola
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the standard equation of a hyperbola
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  (17 * x^2) / 4 - (17 * y^2) / 64 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop :=
  y = 4 * x

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem hyperbola_equation (h : Hyperbola) :
  (∃ x y : ℝ, asymptote_equation x y) ∧
  (h.c = parabola_focus.1) →
  ∀ x y : ℝ, standard_equation h x y :=
by sorry

end hyperbola_equation_l2423_242380


namespace range_of_a_l2423_242349

-- Define the conditions
def p (x : ℝ) : Prop := x > 1 ∨ x < -3
def q (x a : ℝ) : Prop := x > a

-- State the theorem
theorem range_of_a (h : ∀ x a : ℝ, q x a → p x) :
  ∀ a : ℝ, (∀ x : ℝ, q x a → p x) ↔ a ≥ 1 := by sorry

end range_of_a_l2423_242349


namespace range_of_f_l2423_242364

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 4*x - 6

-- Define the domain
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 5 }

-- Theorem statement
theorem range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | -11 ≤ y ∧ y ≤ -2 } := by
  sorry

end range_of_f_l2423_242364


namespace units_digit_of_sum_units_digit_of_42_4_plus_24_4_l2423_242381

theorem units_digit_of_sum (a b : ℕ) : (a^4 + b^4) % 10 = ((a^4 % 10) + (b^4 % 10)) % 10 := by sorry

theorem units_digit_of_42_4_plus_24_4 : (42^4 + 24^4) % 10 = 2 := by
  have h1 : 42^4 % 10 = 6 := by sorry
  have h2 : 24^4 % 10 = 6 := by sorry
  have h3 : (6 + 6) % 10 = 2 := by sorry
  
  calc
    (42^4 + 24^4) % 10 = ((42^4 % 10) + (24^4 % 10)) % 10 := by apply units_digit_of_sum
    _ = (6 + 6) % 10 := by rw [h1, h2]
    _ = 2 := by rw [h3]

end units_digit_of_sum_units_digit_of_42_4_plus_24_4_l2423_242381


namespace quadratic_expression_value_l2423_242353

theorem quadratic_expression_value (x : ℝ) (h : x^2 + 3*x - 5 = 0) : 2*x^2 + 6*x - 3 = 7 := by
  sorry

end quadratic_expression_value_l2423_242353


namespace min_distance_PM_l2423_242302

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the curve C
def C (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 16

-- Define a point P on l₁
structure Point_P where
  x : ℝ
  y : ℝ
  on_l₁ : l₁ x y

-- Define a line l₂ passing through P
structure Line_l₂ (P : Point_P) where
  slope : ℝ
  passes_through_P : True  -- This is a simplification, as we don't need the specific equation

-- Define the intersection point M
structure Point_M (P : Point_P) (l₂ : Line_l₂ P) where
  x : ℝ
  y : ℝ
  on_C : C x y
  on_l₂ : True  -- This is a simplification, as we don't need the specific condition

-- State the theorem
theorem min_distance_PM (P : Point_P) (l₂ : Line_l₂ P) (M : Point_M P l₂) :
  ∃ (d : ℝ), d = 4 ∧ ∀ (M' : Point_M P l₂), Real.sqrt ((M'.x - P.x)^2 + (M'.y - P.y)^2) ≥ d :=
sorry

end min_distance_PM_l2423_242302


namespace sum_of_x_and_y_l2423_242306

theorem sum_of_x_and_y (x y : ℝ) (h1 : y - x = 1) (h2 : y^2 = x^2 + 6) : x + y = 6 := by
  sorry

end sum_of_x_and_y_l2423_242306


namespace average_difference_theorem_l2423_242388

/-- Represents the enrollment of a class -/
structure ClassEnrollment where
  students : ℕ

/-- Represents a school with students, teachers, and class enrollments -/
structure School where
  totalStudents : ℕ
  totalTeachers : ℕ
  classEnrollments : List ClassEnrollment

/-- Calculates the average number of students per teacher -/
def averageStudentsPerTeacher (school : School) : ℚ :=
  school.totalStudents / school.totalTeachers

/-- Calculates the average number of students per student -/
def averageStudentsPerStudent (school : School) : ℚ :=
  (school.classEnrollments.map (λ c => c.students * c.students)).sum / school.totalStudents

/-- The main theorem to prove -/
theorem average_difference_theorem (school : School) 
  (h1 : school.totalStudents = 120)
  (h2 : school.totalTeachers = 6)
  (h3 : school.classEnrollments = [⟨60⟩, ⟨30⟩, ⟨20⟩, ⟨5⟩, ⟨3⟩, ⟨2⟩])
  (h4 : (school.classEnrollments.map (λ c => c.students)).sum = school.totalStudents) :
  averageStudentsPerTeacher school - averageStudentsPerStudent school = -21 := by
  sorry

end average_difference_theorem_l2423_242388


namespace f_composition_three_roots_l2423_242386

-- Define the function f
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 6*x + c

-- State the theorem
theorem f_composition_three_roots (c : ℝ) :
  (∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧
    (∀ x : ℝ, f c (f c x) = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃)) ↔
  c = (11 - Real.sqrt 13) / 2 :=
sorry

end f_composition_three_roots_l2423_242386


namespace triple_addichiffrer_1998_power_l2423_242389

/-- The addichiffrer function adds all digits of a natural number. -/
def addichiffrer (n : ℕ) : ℕ := sorry

/-- Apply addichiffrer process three times to a given number. -/
def triple_addichiffrer (n : ℕ) : ℕ := 
  addichiffrer (addichiffrer (addichiffrer n))

/-- Theorem stating that applying addichiffrer three times to 1998^1998 results in 9. -/
theorem triple_addichiffrer_1998_power : triple_addichiffrer (1998^1998) = 9 := by sorry

end triple_addichiffrer_1998_power_l2423_242389


namespace hyperbola_equation_l2423_242360

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The point (2,2) lies on the hyperbola -/
def point_on_hyperbola (h : Hyperbola) : Prop :=
  4 / h.a^2 - 4 / h.b^2 = 1

/-- The distance from the foci to the asymptotes equals the length of the real axis -/
def foci_distance_condition (h : Hyperbola) : Prop :=
  h.b = 2 * h.a

theorem hyperbola_equation (h : Hyperbola) 
  (h_point : point_on_hyperbola h) 
  (h_distance : foci_distance_condition h) : 
  h.a = Real.sqrt 3 ∧ h.b = 2 * Real.sqrt 3 := by
  sorry

end hyperbola_equation_l2423_242360


namespace largest_whole_number_solution_l2423_242344

theorem largest_whole_number_solution : 
  (∀ n : ℕ, n > 3 → ¬(1/4 + n/5 < 9/10)) ∧ 
  (1/4 + 3/5 < 9/10) := by
sorry

end largest_whole_number_solution_l2423_242344


namespace complex_power_sum_l2423_242330

theorem complex_power_sum (z : ℂ) (h : z + 1 / z = 2 * Real.cos (5 * π / 180)) :
  z^12 + 1 / z^12 = 1 := by
  sorry

end complex_power_sum_l2423_242330


namespace stating_max_perpendicular_diagonals_correct_l2423_242355

/-- 
Given a regular n-gon with n ≥ 3, this function returns the maximum number of diagonals
that can be drawn such that any intersecting pair is perpendicular.
-/
def maxPerpendicularDiagonals (n : ℕ) : ℕ :=
  if n % 2 = 0 then n - 2 else n - 3

/-- 
Theorem stating that maxPerpendicularDiagonals correctly computes the maximum number
of diagonals in a regular n-gon (n ≥ 3) such that any intersecting pair is perpendicular.
-/
theorem max_perpendicular_diagonals_correct (n : ℕ) (h : n ≥ 3) :
  maxPerpendicularDiagonals n = 
    if n % 2 = 0 then n - 2 else n - 3 :=
by sorry

end stating_max_perpendicular_diagonals_correct_l2423_242355


namespace evaluate_expression_l2423_242341

theorem evaluate_expression : 5^4 + 5^4 + 5^4 - 5^4 = 1250 := by
  sorry

end evaluate_expression_l2423_242341


namespace log_inequality_implies_greater_l2423_242317

theorem log_inequality_implies_greater (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  Real.log a > Real.log b → a > b := by
  sorry

end log_inequality_implies_greater_l2423_242317


namespace john_journey_distance_l2423_242347

/-- Calculates the total distance traveled given two journey segments -/
def total_distance (speed1 speed2 : ℝ) (time1 time2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2

/-- Theorem stating that the total distance of John's journey is 240 miles -/
theorem john_journey_distance :
  total_distance 45 50 2 3 = 240 := by
  sorry

end john_journey_distance_l2423_242347


namespace allen_pizza_change_l2423_242322

def pizza_order (num_boxes : ℕ) (price_per_box : ℚ) (tip_fraction : ℚ) (payment : ℚ) : ℚ :=
  let total_cost := num_boxes * price_per_box
  let tip := total_cost * tip_fraction
  let total_spent := total_cost + tip
  payment - total_spent

theorem allen_pizza_change : 
  pizza_order 5 7 (1/7) 100 = 60 := by
  sorry

end allen_pizza_change_l2423_242322


namespace rocking_chair_legs_count_l2423_242326

/-- Represents the number of legs on the rocking chair -/
def rocking_chair_legs : ℕ := 2

/-- Represents the total number of legs in the room -/
def total_legs : ℕ := 40

/-- Represents the number of four-legged tables -/
def four_leg_tables : ℕ := 4

/-- Represents the number of sofas -/
def sofas : ℕ := 1

/-- Represents the number of four-legged chairs -/
def four_leg_chairs : ℕ := 2

/-- Represents the number of three-legged tables -/
def three_leg_tables : ℕ := 3

/-- Represents the number of one-legged tables -/
def one_leg_tables : ℕ := 1

theorem rocking_chair_legs_count : 
  rocking_chair_legs = 
    total_legs - 
    (4 * four_leg_tables + 
     4 * sofas + 
     4 * four_leg_chairs + 
     3 * three_leg_tables + 
     1 * one_leg_tables) :=
by sorry

end rocking_chair_legs_count_l2423_242326


namespace car_speed_problem_l2423_242334

/-- Given two cars traveling in opposite directions, prove that the speed of one car is 52 mph -/
theorem car_speed_problem (v : ℝ) : 
  v > 0 →  -- Assuming positive speed
  3.5 * v + 3.5 * 58 = 385 → 
  v = 52 := by
sorry

end car_speed_problem_l2423_242334


namespace cos_4theta_l2423_242318

theorem cos_4theta (θ : ℝ) (h : Complex.exp (θ * Complex.I) = (1 + Complex.I * Real.sqrt 7) / 4) : 
  Real.cos (4 * θ) = 1 / 32 := by
sorry

end cos_4theta_l2423_242318


namespace museum_artifacts_l2423_242379

theorem museum_artifacts (total_wings : Nat) 
  (painting_wings : Nat) (large_painting_wings : Nat) 
  (small_painting_wings : Nat) (paintings_per_small_wing : Nat) 
  (artifact_multiplier : Nat) :
  total_wings = 8 →
  painting_wings = 3 →
  large_painting_wings = 1 →
  small_painting_wings = 2 →
  paintings_per_small_wing = 12 →
  artifact_multiplier = 4 →
  let total_paintings := large_painting_wings + small_painting_wings * paintings_per_small_wing
  let total_artifacts := total_paintings * artifact_multiplier
  let artifact_wings := total_wings - painting_wings
  ∀ wing, wing ≤ artifact_wings → 
    (total_artifacts / artifact_wings : Nat) = 20 := by
  sorry

#check museum_artifacts

end museum_artifacts_l2423_242379


namespace barbaras_selling_price_l2423_242325

/-- Proves that Barbara's selling price for each stuffed animal is $2 --/
theorem barbaras_selling_price : 
  ∀ (barbara_price : ℚ),
  (9 : ℚ) * barbara_price + (2 * 9 : ℚ) * (3/2 : ℚ) = 45 →
  barbara_price = 2 := by
sorry

end barbaras_selling_price_l2423_242325


namespace coefficient_x_10_in_expansion_l2423_242374

theorem coefficient_x_10_in_expansion : ∃ (c : ℤ), c = -11 ∧ 
  (∀ (x : ℝ), (x - 1)^11 = c * x^10 + (λ (y : ℝ) => (y - 1)^11 - c * y^10) x) := by
sorry

end coefficient_x_10_in_expansion_l2423_242374


namespace division_problem_l2423_242382

theorem division_problem (a b q : ℕ) 
  (h1 : a - b = 1345)
  (h2 : a = 1596)
  (h3 : a = b * q + 15) :
  q = 6 := by
sorry

end division_problem_l2423_242382


namespace f_even_implies_specific_points_l2423_242348

/-- A function f on the real numbers. -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2 * a - b

/-- The domain of f is [2a-1, a^2+1] -/
def domain (a : ℝ) : Set ℝ := Set.Icc (2 * a - 1) (a^2 + 1)

/-- f is an even function -/
def is_even (a b : ℝ) : Prop := ∀ x ∈ domain a, f a b x = f a b (-x)

/-- The theorem stating that given the conditions, (a, b) can only be (0, 0) or (-2, 0) -/
theorem f_even_implies_specific_points :
  ∀ a b : ℝ, is_even a b → (a = 0 ∧ b = 0) ∨ (a = -2 ∧ b = 0) :=
sorry

end f_even_implies_specific_points_l2423_242348


namespace megan_seashell_count_l2423_242350

/-- The number of seashells Megan needs to add to her collection -/
def additional_shells : ℕ := 6

/-- The total number of seashells Megan wants in her collection -/
def target_shells : ℕ := 25

/-- Megan's current number of seashells -/
def current_shells : ℕ := target_shells - additional_shells

theorem megan_seashell_count : current_shells = 19 := by
  sorry

end megan_seashell_count_l2423_242350


namespace smallest_base_for_82_five_satisfies_condition_five_is_smallest_base_l2423_242366

theorem smallest_base_for_82 : 
  ∀ b : ℕ, b > 0 → (b^2 ≤ 82 ∧ 82 < b^3) → b ≥ 5 :=
by
  sorry

theorem five_satisfies_condition : 
  5^2 ≤ 82 ∧ 82 < 5^3 :=
by
  sorry

theorem five_is_smallest_base : 
  ∀ b : ℕ, b > 0 → b^2 ≤ 82 ∧ 82 < b^3 → b = 5 :=
by
  sorry

end smallest_base_for_82_five_satisfies_condition_five_is_smallest_base_l2423_242366


namespace james_initial_balance_l2423_242307

def ticket_cost_1 : ℚ := 150
def ticket_cost_2 : ℚ := 150
def ticket_cost_3 : ℚ := ticket_cost_1 / 3
def total_cost : ℚ := ticket_cost_1 + ticket_cost_2 + ticket_cost_3
def james_share : ℚ := total_cost / 2
def remaining_balance : ℚ := 325

theorem james_initial_balance :
  ∀ x : ℚ, x - james_share = remaining_balance → x = 500 :=
by sorry

end james_initial_balance_l2423_242307


namespace geometric_sequence_ratio_l2423_242323

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : is_geometric_sequence a)
  (h_sum : a 2 + a 6 = 10)
  (h_prod : a 3 * a 5 = 16) :
  ∃ q : ℝ, (q = Real.sqrt 2 ∨ q = Real.sqrt 2 / 2) ∧ 
    ∀ n : ℕ, a (n + 1) = a n * q :=
sorry

end geometric_sequence_ratio_l2423_242323


namespace cans_first_day_correct_l2423_242371

/-- The number of cans collected on the first day, given the conditions of the problem -/
def cans_first_day : ℕ := 20

/-- The number of days cans are collected -/
def collection_days : ℕ := 5

/-- The daily increase in the number of cans collected -/
def daily_increase : ℕ := 5

/-- The total number of cans collected over the collection period -/
def total_cans : ℕ := 150

/-- Theorem stating that the number of cans collected on the first day is correct -/
theorem cans_first_day_correct : 
  cans_first_day * collection_days + 
  (daily_increase * (collection_days - 1) * collection_days / 2) = total_cans := by
  sorry

end cans_first_day_correct_l2423_242371


namespace tim_weekly_reading_time_l2423_242392

/-- Tim's daily meditation time in hours -/
def daily_meditation_time : ℝ := 1

/-- Tim's daily reading time in hours -/
def daily_reading_time : ℝ := 2 * daily_meditation_time

/-- Number of days in a week -/
def days_in_week : ℕ := 7

/-- Tim's weekly reading time in hours -/
def weekly_reading_time : ℝ := daily_reading_time * days_in_week

theorem tim_weekly_reading_time :
  weekly_reading_time = 14 := by sorry

end tim_weekly_reading_time_l2423_242392


namespace boat_speed_in_still_water_l2423_242354

/-- Proves that a boat traveling 12 km downstream in 2 hours and 12 km upstream in 3 hours has a speed of 5 km/h in still water. -/
theorem boat_speed_in_still_water (downstream_distance : ℝ) (upstream_distance : ℝ)
  (downstream_time : ℝ) (upstream_time : ℝ) (h1 : downstream_distance = 12)
  (h2 : upstream_distance = 12) (h3 : downstream_time = 2) (h4 : upstream_time = 3) :
  ∃ (boat_speed : ℝ) (stream_speed : ℝ),
    boat_speed = 5 ∧
    downstream_distance / downstream_time = boat_speed + stream_speed ∧
    upstream_distance / upstream_time = boat_speed - stream_speed := by
  sorry

end boat_speed_in_still_water_l2423_242354


namespace average_speed_is_25_l2423_242342

def initial_reading : ℕ := 45654
def final_reading : ℕ := 45854
def total_time : ℕ := 8

def distance : ℕ := final_reading - initial_reading
def average_speed : ℚ := distance / total_time

theorem average_speed_is_25 : average_speed = 25 := by
  sorry

end average_speed_is_25_l2423_242342


namespace village_population_l2423_242351

theorem village_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) :
  percentage = 80 / 100 →
  partial_population = 23040 →
  (percentage : ℚ) * total_population = partial_population →
  total_population = 28800 := by
sorry

end village_population_l2423_242351


namespace practice_time_difference_l2423_242314

/-- Represents the practice schedule for Carlo's music recital --/
structure PracticeSchedule where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Theorem stating the difference in practice time between Wednesday and Thursday --/
theorem practice_time_difference (schedule : PracticeSchedule) : 
  schedule.monday = 2 * schedule.tuesday →
  schedule.tuesday = schedule.wednesday - 10 →
  schedule.wednesday > schedule.thursday →
  schedule.thursday = 50 →
  schedule.friday = 60 →
  schedule.monday + schedule.tuesday + schedule.wednesday + schedule.thursday + schedule.friday = 300 →
  schedule.wednesday - schedule.thursday = 5 := by
  sorry

end practice_time_difference_l2423_242314


namespace sin_cos_sum_equals_half_l2423_242335

theorem sin_cos_sum_equals_half : 
  Real.sin (21 * π / 180) * Real.cos (9 * π / 180) + 
  Real.sin (69 * π / 180) * Real.sin (9 * π / 180) = 1/2 := by
  sorry

end sin_cos_sum_equals_half_l2423_242335


namespace parade_team_size_l2423_242337

theorem parade_team_size : 
  ∃ n : ℕ, 
    n % 5 = 0 ∧ 
    n ≥ 1000 ∧ 
    n % 4 = 3 ∧ 
    n % 3 = 2 ∧ 
    n % 2 = 1 ∧ 
    n = 1045 ∧ 
    ∀ m : ℕ, 
      (m % 5 = 0 ∧ 
       m ≥ 1000 ∧ 
       m % 4 = 3 ∧ 
       m % 3 = 2 ∧ 
       m % 2 = 1) → 
      m ≥ n :=
by sorry

end parade_team_size_l2423_242337


namespace right_triangle_leg_square_l2423_242308

theorem right_triangle_leg_square (a b c : ℝ) (h1 : c = a + 2) (h2 : a^2 + b^2 = c^2) : b^2 = 4*a + 4 := by
  sorry

end right_triangle_leg_square_l2423_242308


namespace smallest_integer_with_remainders_l2423_242331

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧
  (n % 2 = 1) ∧
  (n % 3 ≠ 0) ∧
  (n % 4 = 3) ∧
  (n % 10 = 9) ∧
  (∀ m : ℕ, m > 0 → m % 2 = 1 → m % 3 ≠ 0 → m % 4 = 3 → m % 10 = 9 → m ≥ n) ∧
  n = 59 :=
by sorry

end smallest_integer_with_remainders_l2423_242331


namespace intersection_complement_equality_l2423_242303

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | ∃ y, y = Real.sqrt x}
def B : Set ℝ := {y | ∃ x, y = -x^2}

-- State the theorem
theorem intersection_complement_equality : A ∩ (U \ B) = {x | x > 0} := by sorry

end intersection_complement_equality_l2423_242303


namespace total_cards_l2423_242377

theorem total_cards (brenda janet mara : ℕ) : 
  janet = brenda + 9 →
  mara = 2 * janet →
  mara = 150 - 40 →
  brenda + janet + mara = 211 := by
  sorry

end total_cards_l2423_242377


namespace problem_statement_l2423_242358

noncomputable section

def f (x : ℝ) := Real.exp x * Real.sin x - Real.cos x
def g (x : ℝ) := x * Real.cos x - Real.sqrt 2 * Real.exp x

theorem problem_statement :
  (∀ m > -1 - Real.sqrt 2, ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 (Real.pi / 2) ∧ 
    x₂ ∈ Set.Icc 0 (Real.pi / 2) ∧ f x₁ + g x₂ < m) ∧
  (∀ x > -1, f x - g x > 0) := by
  sorry

end

end problem_statement_l2423_242358


namespace christina_weekly_distance_l2423_242395

/-- The total distance Christina covers in a week -/
def total_distance (school_distance : ℕ) (days : ℕ) (extra_distance : ℕ) : ℕ :=
  2 * school_distance * days + 2 * extra_distance

/-- Theorem stating the total distance Christina covered in a week -/
theorem christina_weekly_distance :
  total_distance 7 5 2 = 74 := by
  sorry

end christina_weekly_distance_l2423_242395


namespace percentage_calculation_l2423_242370

theorem percentage_calculation (x : ℝ) : 
  (70 / 100 * 600 : ℝ) = (x / 100 * 1050 : ℝ) → x = 40 := by
  sorry

end percentage_calculation_l2423_242370


namespace baseball_card_value_l2423_242343

def initialValue : ℝ := 100

def yearlyChanges : List ℝ := [-0.10, 0.12, -0.08, 0.05, -0.07]

def applyChange (value : ℝ) (change : ℝ) : ℝ := value * (1 + change)

def finalValue : ℝ := yearlyChanges.foldl applyChange initialValue

theorem baseball_card_value : 
  ∃ ε > 0, |finalValue - 90.56| < ε :=
sorry

end baseball_card_value_l2423_242343


namespace min_length_MN_l2423_242398

-- Define the circle
def circle_center : ℝ × ℝ := (1, 1)

-- Define the property of being tangent to x and y axes
def tangent_to_axes (c : ℝ × ℝ) : Prop :=
  c.1 = c.2 ∧ c.1 > 0

-- Define the line MN
def line_MN (m n : ℝ × ℝ) : Prop :=
  m.2 = 0 ∧ n.1 = 0 ∧ m.1 > 0 ∧ n.2 > 0

-- Define the property of MN being tangent to the circle
def tangent_to_circle (m n : ℝ × ℝ) (c : ℝ × ℝ) : Prop :=
  ∃ p : ℝ × ℝ, (p.1 - c.1)^2 + (p.2 - c.2)^2 = 1 ∧
              (n.2 - m.2) * (p.1 - m.1) = (n.1 - m.1) * (p.2 - m.2)

-- Theorem statement
theorem min_length_MN :
  tangent_to_axes circle_center →
  ∀ m n : ℝ × ℝ, line_MN m n →
  tangent_to_circle m n circle_center →
  ∃ min_length : ℝ, min_length = 2 * Real.sqrt 2 - 2 ∧
  ∀ m' n' : ℝ × ℝ, line_MN m' n' → tangent_to_circle m' n' circle_center →
  Real.sqrt ((m'.1 - n'.1)^2 + (m'.2 - n'.2)^2) ≥ min_length :=
sorry

end min_length_MN_l2423_242398


namespace product_satisfies_X_l2423_242328

/-- Condition X: every positive integer less than m is a sum of distinct divisors of m -/
def condition_X (m : ℕ+) : Prop :=
  ∀ k < m, ∃ (S : Finset ℕ), (∀ d ∈ S, d ∣ m) ∧ (Finset.sum S id = k)

theorem product_satisfies_X (m n : ℕ+) (hm : condition_X m) (hn : condition_X n) :
  condition_X (m * n) :=
sorry

end product_satisfies_X_l2423_242328


namespace symmetric_circle_correct_l2423_242372

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  (x + 5)^2 + (y - 6)^2 = 16

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x - y = 0

-- Define the symmetric circle C
def symmetric_circle (x y : ℝ) : Prop :=
  (x - 6)^2 + (y + 5)^2 = 16

-- Theorem stating that the symmetric circle C is correct
theorem symmetric_circle_correct :
  ∀ (x y : ℝ),
  (∃ (x₀ y₀ : ℝ), original_circle x₀ y₀ ∧ 
   symmetry_line ((x + x₀) / 2) ((y + y₀) / 2)) →
  symmetric_circle x y :=
sorry

end symmetric_circle_correct_l2423_242372


namespace cubic_root_equation_solutions_l2423_242383

theorem cubic_root_equation_solutions :
  ∀ x : ℝ, 
    (x^(1/3) - 4 / (x^(1/3) + 4) = 0) ↔ 
    (x = (-2 + 2 * Real.sqrt 2)^3 ∨ x = (-2 - 2 * Real.sqrt 2)^3) :=
by sorry

end cubic_root_equation_solutions_l2423_242383


namespace math_club_members_l2423_242319

/-- The number of female members in the Math club -/
def female_members : ℕ := 6

/-- The ratio of male to female members in the Math club -/
def male_to_female_ratio : ℕ := 2

/-- The total number of members in the Math club -/
def total_members : ℕ := female_members * (male_to_female_ratio + 1)

theorem math_club_members :
  total_members = 18 := by
  sorry

end math_club_members_l2423_242319


namespace calculate_expression_l2423_242312

theorem calculate_expression : (8 * 2.25 - 5 * 0.85 / 2.5) = 16.3 := by
  sorry

end calculate_expression_l2423_242312


namespace leaf_decrease_l2423_242373

theorem leaf_decrease (green_yesterday red_yesterday yellow_yesterday 
                       green_today yellow_today red_today : ℕ) :
  green_yesterday = red_yesterday →
  yellow_yesterday = 7 * red_yesterday →
  green_today = yellow_today →
  red_today = 7 * yellow_today →
  green_today + yellow_today + red_today ≤ (green_yesterday + red_yesterday + yellow_yesterday) / 4 :=
by sorry

end leaf_decrease_l2423_242373


namespace max_value_on_ellipse_l2423_242324

theorem max_value_on_ellipse :
  ∃ (M : ℝ), M = Real.sqrt 22 ∧
  (∀ x y : ℝ, 2 * x^2 + 3 * y^2 = 1 → x + 2*y ≤ M) ∧
  (∃ x y : ℝ, 2 * x^2 + 3 * y^2 = 1 ∧ x + 2*y = M) := by
  sorry

end max_value_on_ellipse_l2423_242324


namespace parabola_focus_directrix_distance_l2423_242321

/-- Given a parabola x² = 2py with p > 0, if there exists a point on the parabola
    with ordinate l such that its distance to the focus is 3,
    then the distance from the focus to the directrix is 4. -/
theorem parabola_focus_directrix_distance (p : ℝ) (l : ℝ) (h1 : p > 0) :
  (∃ x : ℝ, x^2 = 2*p*l) →  -- point (x,l) is on the parabola
  (l + p/2)^2 + (p/2)^2 = 3^2 →  -- distance from (x,l) to focus (0,p/2) is 3
  p = 4 :=  -- distance from focus to directrix
by sorry

end parabola_focus_directrix_distance_l2423_242321


namespace cheese_division_possible_l2423_242365

/-- Represents the state of the cheese pieces -/
structure CheeseState where
  piece1 : ℕ
  piece2 : ℕ
  piece3 : ℕ

/-- Represents a single cut operation -/
inductive Cut
  | cut12 : Cut  -- Cut 1g from piece1 and piece2
  | cut13 : Cut  -- Cut 1g from piece1 and piece3
  | cut23 : Cut  -- Cut 1g from piece2 and piece3

/-- Applies a single cut to a CheeseState -/
def applyCut (state : CheeseState) (cut : Cut) : CheeseState :=
  match cut with
  | Cut.cut12 => ⟨state.piece1 - 1, state.piece2 - 1, state.piece3⟩
  | Cut.cut13 => ⟨state.piece1 - 1, state.piece2, state.piece3 - 1⟩
  | Cut.cut23 => ⟨state.piece1, state.piece2 - 1, state.piece3 - 1⟩

/-- Checks if all pieces in a CheeseState are equal -/
def allEqual (state : CheeseState) : Prop :=
  state.piece1 = state.piece2 ∧ state.piece2 = state.piece3

/-- The theorem to be proved -/
theorem cheese_division_possible : ∃ (cuts : List Cut), 
  let finalState := cuts.foldl applyCut ⟨5, 8, 11⟩
  allEqual finalState ∧ finalState.piece1 ≥ 0 := by
  sorry


end cheese_division_possible_l2423_242365


namespace quadratic_inequality_implication_l2423_242339

theorem quadratic_inequality_implication (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + a > 0) → a > 1 := by
  sorry

end quadratic_inequality_implication_l2423_242339


namespace unbounded_function_l2423_242367

def IsUnbounded (f : ℝ → ℝ) : Prop :=
  ∀ M : ℝ, ∃ x : ℝ, f x > M

theorem unbounded_function (f : ℝ → ℝ) 
  (h_pos : ∀ x, 0 < f x) 
  (h_ineq : ∀ x y, 0 < x → 0 < y → (f (x + f y))^2 ≥ f x * (f (x + f y) + f y)) : 
  IsUnbounded f := by
  sorry

end unbounded_function_l2423_242367


namespace g_five_l2423_242338

/-- A function satisfying g(x+y) = g(x) + g(y) for all real x and y, and g(1) = 2 -/
def g : ℝ → ℝ :=
  fun x => sorry

/-- The functional equation property of g -/
axiom g_add (x y : ℝ) : g (x + y) = g x + g y

/-- The value of g at 1 -/
axiom g_one : g 1 = 2

/-- The theorem stating that g(5) = 10 -/
theorem g_five : g 5 = 10 := by sorry

end g_five_l2423_242338


namespace parabola_vertex_l2423_242385

/-- A parabola is defined by the equation y = (x - 2)^2 - 1 -/
def parabola (x y : ℝ) : Prop := y = (x - 2)^2 - 1

/-- The vertex of a parabola is the point where it reaches its minimum or maximum -/
def is_vertex (x y : ℝ) : Prop :=
  parabola x y ∧ ∀ x' y', parabola x' y' → y ≤ y'

theorem parabola_vertex :
  is_vertex 2 (-1) :=
sorry

end parabola_vertex_l2423_242385


namespace cube_surface_area_with_holes_eq_222_l2423_242329

/-- Calculates the entire surface area of a cube with holes, including inside surfaces -/
def cubeSurfaceAreaWithHoles (cubeEdge : ℝ) (holeEdge : ℝ) : ℝ :=
  let originalSurface := 6 * cubeEdge^2
  let holeArea := 6 * holeEdge^2
  let newExposedArea := 6 * 4 * holeEdge^2
  originalSurface - holeArea + newExposedArea

/-- The entire surface area of the cube with holes is 222 square meters -/
theorem cube_surface_area_with_holes_eq_222 :
  cubeSurfaceAreaWithHoles 5 2 = 222 := by
  sorry

end cube_surface_area_with_holes_eq_222_l2423_242329


namespace factor_sum_l2423_242332

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 7) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 54 := by
sorry

end factor_sum_l2423_242332


namespace cube_root_of_four_l2423_242387

theorem cube_root_of_four (x : ℝ) : x^3 = 4 → x = 4^(1/3) := by sorry

end cube_root_of_four_l2423_242387


namespace ellipse_max_angle_ratio_l2423_242396

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x - y + 10 = 0

-- Define the angle F₁PF₂
def angle_F₁PF₂ (P : ℝ × ℝ) : ℝ := sorry

-- Define the ratio PF₁/PF₂
def ratio_PF₁_PF₂ (P : ℝ × ℝ) : ℝ := sorry

theorem ellipse_max_angle_ratio :
  ∀ a b : ℝ, a > 0 → b > 0 →
  ∀ P : ℝ × ℝ,
  ellipse a b P.1 P.2 →
  line_l P.1 P.2 →
  (∀ Q : ℝ × ℝ, ellipse a b Q.1 Q.2 → line_l Q.1 Q.2 → angle_F₁PF₂ P ≥ angle_F₁PF₂ Q) →
  ratio_PF₁_PF₂ P = -1 := by
  sorry

end ellipse_max_angle_ratio_l2423_242396


namespace complex_power_sum_l2423_242305

open Complex

theorem complex_power_sum (z : ℂ) (h : z + 1 / z = 2 * Real.cos (5 * π / 180)) :
  z^1000 + 1 / z^1000 = 2 * Real.cos (40 * π / 180) := by
  sorry

end complex_power_sum_l2423_242305


namespace computer_price_2004_l2423_242333

/-- The yearly decrease rate of the computer price -/
def yearly_decrease_rate : ℚ := 1 / 3

/-- The initial price of the computer in 2000 -/
def initial_price : ℚ := 8100

/-- The number of years between 2000 and 2004 -/
def years : ℕ := 4

/-- The price of the computer in 2004 -/
def price_2004 : ℚ := initial_price * (1 - yearly_decrease_rate) ^ years

theorem computer_price_2004 : price_2004 = 1600 := by
  sorry

end computer_price_2004_l2423_242333


namespace paint_required_for_similar_statues_l2423_242359

/-- The amount of paint required for similar statues with different heights and thicknesses -/
theorem paint_required_for_similar_statues 
  (original_height : ℝ) 
  (original_paint : ℝ) 
  (new_height : ℝ) 
  (num_statues : ℕ) 
  (thickness_factor : ℝ)
  (h1 : original_height > 0)
  (h2 : original_paint > 0)
  (h3 : new_height > 0)
  (h4 : thickness_factor > 0) :
  let surface_area_ratio := (new_height / original_height) ^ 2
  let paint_per_new_statue := original_paint * surface_area_ratio * thickness_factor
  let total_paint := paint_per_new_statue * num_statues
  total_paint = 28.8 :=
by
  sorry

#check paint_required_for_similar_statues 10 1 2 360 2

end paint_required_for_similar_statues_l2423_242359


namespace arithmetic_sequence_sum_l2423_242310

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Theorem: For an arithmetic sequence, if S_9 = 54 and S_8 - S_5 = 30, then S_11 = 88 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
    (h1 : seq.S 9 = 54)
    (h2 : seq.S 8 - seq.S 5 = 30) :
    seq.S 11 = 88 := by
  sorry

end arithmetic_sequence_sum_l2423_242310


namespace no_solution_in_interval_l2423_242399

theorem no_solution_in_interval (a : ℝ) : 
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/2), (2 - a) * (x - 1) - 2 * Real.log x ≠ 0) ↔ 
  a ∈ Set.Ici (2 - 4 * Real.log 2) := by
  sorry

end no_solution_in_interval_l2423_242399


namespace stratified_sample_teachers_l2423_242393

def total_teachers : ℕ := 300
def senior_teachers : ℕ := 90
def intermediate_teachers : ℕ := 150
def junior_teachers : ℕ := 60
def sample_size : ℕ := 40

def stratified_sample (total : ℕ) (strata : List ℕ) (sample : ℕ) : List ℕ :=
  strata.map (λ stratum => (stratum * sample) / total)

theorem stratified_sample_teachers :
  stratified_sample total_teachers [senior_teachers, intermediate_teachers, junior_teachers] sample_size = [12, 20, 8] := by
  sorry

end stratified_sample_teachers_l2423_242393


namespace girls_points_in_checkers_tournament_l2423_242313

theorem girls_points_in_checkers_tournament (x : ℕ) : 
  x > 0 →  -- number of girls is positive
  2 * x * (10 * x - 1) = 18 →  -- derived equation for girls' points
  ∃ (total_games : ℕ) (total_points : ℕ),
    -- total number of games
    total_games = (10 * x) * (10 * x - 1) / 2 ∧
    -- total points distributed
    total_points = 2 * total_games ∧
    -- boys' points are 4 times girls' points
    4 * (2 * x * (10 * x - 1)) = total_points - (2 * x * (10 * x - 1)) :=
by
  sorry

#check girls_points_in_checkers_tournament

end girls_points_in_checkers_tournament_l2423_242313


namespace right_triangle_hypotenuse_l2423_242304

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  (a^2 - 5*a + 6 = 0) → 
  (b^2 - 5*b + 6 = 0) → 
  (a ≠ b) →
  (c^2 = a^2 + b^2) →
  c = Real.sqrt 13 := by
sorry

end right_triangle_hypotenuse_l2423_242304


namespace eighth_term_value_l2423_242309

def is_arithmetic_sequence (s : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, s (n + 1) - s n = d

theorem eighth_term_value (a : ℕ → ℚ) 
  (h1 : a 2 = 3)
  (h2 : a 5 = 1)
  (h3 : is_arithmetic_sequence (fun n ↦ 1 / (a n + 1))) :
  a 8 = 1/3 := by
  sorry

end eighth_term_value_l2423_242309


namespace inequality_proof_l2423_242368

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_one : a + b + c + d = 1) : 
  b * c * d / (1 - a)^2 + a * c * d / (1 - b)^2 + 
  a * b * d / (1 - c)^2 + a * b * c / (1 - d)^2 ≤ 1/9 ∧ 
  (b * c * d / (1 - a)^2 + a * c * d / (1 - b)^2 + 
   a * b * d / (1 - c)^2 + a * b * c / (1 - d)^2 = 1/9 ↔ 
   a = 1/4 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 1/4) :=
sorry

end inequality_proof_l2423_242368


namespace initial_bottles_count_l2423_242346

/-- The number of bottles Maria drank -/
def maria_drank : ℝ := 14.0

/-- The number of bottles Maria's sister drank -/
def sister_drank : ℝ := 8.0

/-- The number of bottles left in the fridge -/
def bottles_left : ℕ := 23

/-- The initial number of bottles in Maria's fridge -/
def initial_bottles : ℝ := maria_drank + sister_drank + bottles_left

theorem initial_bottles_count : initial_bottles = 45.0 := by
  sorry

end initial_bottles_count_l2423_242346


namespace min_toothpicks_removal_l2423_242301

/-- Represents a triangular figure constructed with toothpicks -/
structure TriangularFigure where
  total_toothpicks : ℕ
  upward_triangles : ℕ
  downward_triangles : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (figure : TriangularFigure) : ℕ :=
  figure.upward_triangles

/-- Theorem stating the minimum number of toothpicks to remove -/
theorem min_toothpicks_removal (figure : TriangularFigure) 
  (h1 : figure.total_toothpicks = 45)
  (h2 : figure.upward_triangles = 15)
  (h3 : figure.downward_triangles = 10) :
  min_toothpicks_to_remove figure = 15 := by
  sorry

#check min_toothpicks_removal

end min_toothpicks_removal_l2423_242301


namespace infinite_equal_pairs_l2423_242391

-- Define the sequence type
def InfiniteSequence := ℤ → ℝ

-- Define the property that each term is 1/4 of the sum of its neighbors
def NeighborSumProperty (a : InfiniteSequence) :=
  ∀ n : ℤ, a n = (1 / 4) * (a (n - 1) + a (n + 1))

-- Define the existence of two equal terms
def HasEqualTerms (a : InfiniteSequence) :=
  ∃ i j : ℤ, i ≠ j ∧ a i = a j

-- Define the existence of infinitely many pairs of equal terms
def HasInfiniteEqualPairs (a : InfiniteSequence) :=
  ∀ N : ℕ, ∃ i j : ℤ, i ≠ j ∧ |i - j| > N ∧ a i = a j

-- The main theorem
theorem infinite_equal_pairs
  (a : InfiniteSequence)
  (h1 : NeighborSumProperty a)
  (h2 : HasEqualTerms a) :
  HasInfiniteEqualPairs a :=
sorry

end infinite_equal_pairs_l2423_242391


namespace rectangular_garden_length_l2423_242394

/-- Calculates the length of a rectangular garden given its perimeter and breadth. -/
theorem rectangular_garden_length 
  (perimeter : ℝ) 
  (breadth : ℝ) 
  (h1 : perimeter = 480) 
  (h2 : breadth = 100) : 
  perimeter / 2 - breadth = 140 := by
  sorry

end rectangular_garden_length_l2423_242394
