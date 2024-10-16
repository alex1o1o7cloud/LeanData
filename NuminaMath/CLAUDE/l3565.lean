import Mathlib

namespace NUMINAMATH_CALUDE_kangaroo_hop_distance_l3565_356513

theorem kangaroo_hop_distance :
  let a : ℚ := 1/4  -- first term
  let r : ℚ := 3/4  -- common ratio
  let n : ℕ := 6    -- number of hops
  (a * (1 - r^n)) / (1 - r) = 3367/4096 := by
sorry

end NUMINAMATH_CALUDE_kangaroo_hop_distance_l3565_356513


namespace NUMINAMATH_CALUDE_olya_candies_l3565_356507

theorem olya_candies 
  (total : ℕ)
  (pasha masha tolya olya : ℕ)
  (h_total : pasha + masha + tolya + olya = total)
  (h_total_val : total = 88)
  (h_masha_tolya : masha + tolya = 57)
  (h_pasha_most : pasha > masha ∧ pasha > tolya ∧ pasha > olya)
  (h_at_least_one : pasha ≥ 1 ∧ masha ≥ 1 ∧ tolya ≥ 1 ∧ olya ≥ 1) :
  olya = 1 := by
sorry

end NUMINAMATH_CALUDE_olya_candies_l3565_356507


namespace NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l3565_356525

theorem unique_solution_for_exponential_equation :
  ∀ (a b : ℕ+), 1 + 5^(a : ℕ) = 6^(b : ℕ) ↔ a = 1 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l3565_356525


namespace NUMINAMATH_CALUDE_probability_of_meeting_l3565_356539

/-- Two people arrive independently and uniformly at random within a 2-hour interval -/
def arrival_interval : ℝ := 2

/-- Each person stays for 30 minutes after arrival -/
def stay_duration : ℝ := 0.5

/-- The maximum arrival time for each person is 30 minutes before the end of the 2-hour interval -/
def max_arrival_time : ℝ := arrival_interval - stay_duration

/-- The probability of two people seeing each other given the conditions -/
theorem probability_of_meeting :
  let total_area : ℝ := arrival_interval ^ 2
  let overlap_area : ℝ := total_area - 2 * (stay_duration ^ 2 / 2)
  overlap_area / total_area = 15 / 16 := by sorry

end NUMINAMATH_CALUDE_probability_of_meeting_l3565_356539


namespace NUMINAMATH_CALUDE_unique_triple_solution_l3565_356504

theorem unique_triple_solution : ∃! (a b c : ℕ+), 5^(a.val) + 3^(b.val) - 2^(c.val) = 32 ∧ a = 2 ∧ b = 2 ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l3565_356504


namespace NUMINAMATH_CALUDE_tan_pi_four_minus_theta_l3565_356583

theorem tan_pi_four_minus_theta (θ : Real) (h : (Real.tan θ) = -2) :
  Real.tan (π / 4 - θ) = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_four_minus_theta_l3565_356583


namespace NUMINAMATH_CALUDE_biology_marks_l3565_356560

def marks_english : ℕ := 91
def marks_mathematics : ℕ := 65
def marks_physics : ℕ := 82
def marks_chemistry : ℕ := 67
def average_marks : ℕ := 78
def total_subjects : ℕ := 5

theorem biology_marks :
  let total_marks := average_marks * total_subjects
  let known_marks := marks_english + marks_mathematics + marks_physics + marks_chemistry
  total_marks - known_marks = 85 := by sorry

end NUMINAMATH_CALUDE_biology_marks_l3565_356560


namespace NUMINAMATH_CALUDE_quarters_percentage_theorem_l3565_356565

def num_dimes : ℕ := 70
def num_quarters : ℕ := 30
def num_nickels : ℕ := 15

def value_dime : ℕ := 10
def value_quarter : ℕ := 25
def value_nickel : ℕ := 5

def total_value : ℕ := num_dimes * value_dime + num_quarters * value_quarter + num_nickels * value_nickel
def quarters_value : ℕ := num_quarters * value_quarter

theorem quarters_percentage_theorem : 
  (quarters_value : ℚ) / (total_value : ℚ) * 100 = 750 / 1525 * 100 := by
  sorry

end NUMINAMATH_CALUDE_quarters_percentage_theorem_l3565_356565


namespace NUMINAMATH_CALUDE_right_triangle_sine_calculation_l3565_356551

theorem right_triangle_sine_calculation (D E F : ℝ) :
  0 < D ∧ D < π/2 →
  0 < E ∧ E < π/2 →
  0 < F ∧ F < π/2 →
  D + E + F = π →
  Real.sin D = 5/13 →
  Real.sin E = 1 →
  Real.sin F = 12/13 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sine_calculation_l3565_356551


namespace NUMINAMATH_CALUDE_polygon_count_l3565_356572

theorem polygon_count (n : ℕ) (h : n = 15) : 
  2^n - (n.choose 0 + n.choose 1 + n.choose 2 + n.choose 3) = 32192 :=
by sorry

end NUMINAMATH_CALUDE_polygon_count_l3565_356572


namespace NUMINAMATH_CALUDE_max_value_f_l3565_356514

/-- Given positive real numbers x, y, z satisfying xyz = 1, 
    the maximum value of f(x, y, z) = (1 - yz + z)(1 - zx + x)(1 - xy + y) is 1, 
    and this maximum is achieved when x = y = z = 1. -/
theorem max_value_f (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  let f := fun (a b c : ℝ) => (1 - b*c + c) * (1 - c*a + a) * (1 - a*b + b)
  (∀ a b c, a > 0 → b > 0 → c > 0 → a * b * c = 1 → f a b c ≤ 1) ∧ 
  f x y z ≤ 1 ∧
  f 1 1 1 = 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_f_l3565_356514


namespace NUMINAMATH_CALUDE_inequality_proof_l3565_356503

theorem inequality_proof (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  Real.sqrt ((a + c) * (b + d)) ≥ Real.sqrt (a * b) + Real.sqrt (c * d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3565_356503


namespace NUMINAMATH_CALUDE_multiple_of_99_sum_of_digits_l3565_356556

theorem multiple_of_99_sum_of_digits (A B : ℕ) : 
  A ≤ 9 → B ≤ 9 → 
  (100000 * A + 15000 + 100 * B + 94) % 99 = 0 →
  A + B = 8 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_99_sum_of_digits_l3565_356556


namespace NUMINAMATH_CALUDE_intersection_length_tangent_line_m_range_l3565_356574

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Define circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 - 4 = 0

-- Define circle C
def circle_C (a x y : ℝ) : Prop := x^2 + y^2 + 2*a*x - 2*a*y + 2*a^2 - 4*a = 0

-- Theorem for part 1
theorem intersection_length :
  ∀ (x1 y1 x2 y2 : ℝ),
  circle_O x1 y1 ∧ circle_O x2 y2 ∧
  circle_C 3 x1 y1 ∧ circle_C 3 x2 y2 →
  ((x1 - x2)^2 + (y1 - y2)^2)^(1/2) = Real.sqrt 94 / 3 := by sorry

-- Theorem for part 2
theorem tangent_line_m_range :
  ∀ (a m : ℝ),
  0 < a ∧ a ≤ 4 ∧
  (∃ (x y : ℝ), line_l m x y ∧ circle_C a x y) ∧
  (∀ (x y : ℝ), line_l m x y → (x + a)^2 + (y - a)^2 ≥ 4*a) →
  -1 ≤ m ∧ m ≤ 8 - 4*Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_intersection_length_tangent_line_m_range_l3565_356574


namespace NUMINAMATH_CALUDE_passing_percentage_l3565_356524

theorem passing_percentage (marks_obtained : ℕ) (marks_short : ℕ) (total_marks : ℕ) :
  marks_obtained = 125 →
  marks_short = 40 →
  total_marks = 500 →
  (((marks_obtained + marks_short : ℚ) / total_marks) * 100 : ℚ) = 33 := by
  sorry

end NUMINAMATH_CALUDE_passing_percentage_l3565_356524


namespace NUMINAMATH_CALUDE_hotdogs_day1_proof_l3565_356541

/-- Represents the price of a hamburger in dollars -/
def hamburger_price : ℚ := 2

/-- Represents the price of a hot dog in dollars -/
def hotdog_price : ℚ := 1

/-- Represents the number of hamburgers bought on day 1 -/
def hamburgers_day1 : ℕ := 3

/-- Represents the number of hamburgers bought on day 2 -/
def hamburgers_day2 : ℕ := 2

/-- Represents the number of hot dogs bought on day 2 -/
def hotdogs_day2 : ℕ := 3

/-- Represents the total cost of purchases on day 1 in dollars -/
def total_cost_day1 : ℚ := 10

/-- Represents the total cost of purchases on day 2 in dollars -/
def total_cost_day2 : ℚ := 7

/-- Calculates the number of hot dogs bought on day 1 -/
def hotdogs_day1 : ℕ := 4

theorem hotdogs_day1_proof : 
  hamburgers_day1 * hamburger_price + hotdogs_day1 * hotdog_price = total_cost_day1 ∧
  hamburgers_day2 * hamburger_price + hotdogs_day2 * hotdog_price = total_cost_day2 :=
by sorry

end NUMINAMATH_CALUDE_hotdogs_day1_proof_l3565_356541


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3565_356598

theorem max_value_sqrt_sum (x : ℝ) (h : -36 ≤ x ∧ x ≤ 36) :
  Real.sqrt (36 + x) + Real.sqrt (36 - x) + x / 6 ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3565_356598


namespace NUMINAMATH_CALUDE_ion_relationship_l3565_356527

/-- Represents an ion with atomic number and charge -/
structure Ion where
  atomic_number : ℕ
  charge : ℤ

/-- Two ions have the same electron shell structure -/
def same_electron_shell (x y : Ion) : Prop :=
  x.atomic_number + x.charge = y.atomic_number - y.charge

theorem ion_relationship {a b n m : ℕ} (X Y : Ion)
  (hX : X.atomic_number = a ∧ X.charge = -n)
  (hY : Y.atomic_number = b ∧ Y.charge = m)
  (h_same_shell : same_electron_shell X Y) :
  a + m = b - n := by
  sorry


end NUMINAMATH_CALUDE_ion_relationship_l3565_356527


namespace NUMINAMATH_CALUDE_max_sum_on_parabola_l3565_356505

theorem max_sum_on_parabola :
  ∃ (max : ℝ), max = 13/4 ∧ 
  ∀ (m n : ℝ), n = -m^2 + 3 → m + n ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_sum_on_parabola_l3565_356505


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3565_356512

theorem complex_equation_solution (a b c : ℤ) : 
  (a * (3 - Complex.I)^4 + b * (3 - Complex.I)^3 + c * (3 - Complex.I)^2 + b * (3 - Complex.I) + a = 0) →
  (Int.gcd a (Int.gcd b c) = 1) →
  (abs c = 109) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3565_356512


namespace NUMINAMATH_CALUDE_wheel_speed_l3565_356593

/-- The speed (in miles per hour) of a wheel with a 10-foot circumference -/
def r : ℝ := sorry

/-- The time (in hours) for one complete rotation of the wheel -/
def t : ℝ := sorry

/-- Relation between speed, time, and distance for one rotation -/
axiom speed_time_relation : r * t = (10 / 5280)

/-- Relation between original and new speed and time -/
axiom speed_time_change : (r + 5) * (t - 1 / (3 * 3600)) = (10 / 5280)

theorem wheel_speed : r = 10 := by sorry

end NUMINAMATH_CALUDE_wheel_speed_l3565_356593


namespace NUMINAMATH_CALUDE_rectangle_length_proof_l3565_356516

theorem rectangle_length_proof (w : ℝ) (h1 : w > 0) : 
  let l := 2 * w
  let new_area := (l - 5) * (w + 5)
  new_area = l * w + 75 → l = 40 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_proof_l3565_356516


namespace NUMINAMATH_CALUDE_computer_contract_probability_l3565_356577

theorem computer_contract_probability (p_hardware : ℝ) (p_at_least_one : ℝ) (p_both : ℝ) :
  p_hardware = 3/4 →
  p_at_least_one = 5/6 →
  p_both = 0.31666666666666654 →
  1 - (p_at_least_one - p_hardware + p_both) = 0.6 :=
by sorry

end NUMINAMATH_CALUDE_computer_contract_probability_l3565_356577


namespace NUMINAMATH_CALUDE_min_k_existence_l3565_356510

open Real

theorem min_k_existence (k : ℕ) : (∃ x₀ : ℝ, x₀ > 2 ∧ k * (x₀ - 2) > x₀ * (log x₀ + 1)) ↔ k ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_min_k_existence_l3565_356510


namespace NUMINAMATH_CALUDE_complex_multiplication_l3565_356595

theorem complex_multiplication (z : ℂ) (i : ℂ) : 
  z.re = 1 ∧ z.im = 1 ∧ i * i = -1 → z * (1 - i) = 2 := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3565_356595


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3565_356581

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (1 + x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₁ + a₃ = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3565_356581


namespace NUMINAMATH_CALUDE_johns_house_nails_l3565_356553

/-- The number of nails needed for a house wall -/
def total_nails (large_planks small_planks large_nails small_nails : ℕ) : ℕ :=
  large_nails + small_nails

/-- Theorem stating the total number of nails needed for John's house wall -/
theorem johns_house_nails :
  total_nails 12 10 15 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_johns_house_nails_l3565_356553


namespace NUMINAMATH_CALUDE_sin_plus_sin_alpha_nonperiodic_l3565_356533

/-- A function f is periodic if there exists a non-zero real number T such that f(x + T) = f(x) for all x -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f (x + T) = f x

/-- The main theorem: for any positive irrational α, the function f(x) = sin x + sin(αx) is non-periodic -/
theorem sin_plus_sin_alpha_nonperiodic (α : ℝ) (h_pos : α > 0) (h_irr : Irrational α) :
  ¬IsPeriodic (fun x ↦ Real.sin x + Real.sin (α * x)) := by
  sorry


end NUMINAMATH_CALUDE_sin_plus_sin_alpha_nonperiodic_l3565_356533


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3565_356538

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3565_356538


namespace NUMINAMATH_CALUDE_less_expensive_coat_cost_l3565_356557

/-- Represents the cost of a coat and its lifespan in years -/
structure Coat where
  cost : ℕ
  lifespan : ℕ

/-- Calculates the total cost of a coat over a given period -/
def totalCost (coat : Coat) (period : ℕ) : ℕ :=
  (period / coat.lifespan) * coat.cost

theorem less_expensive_coat_cost (expensive_coat less_expensive_coat : Coat) : 
  expensive_coat.cost = 300 →
  expensive_coat.lifespan = 15 →
  less_expensive_coat.lifespan = 5 →
  totalCost expensive_coat 30 + 120 = totalCost less_expensive_coat 30 →
  less_expensive_coat.cost = 120 := by
sorry

end NUMINAMATH_CALUDE_less_expensive_coat_cost_l3565_356557


namespace NUMINAMATH_CALUDE_process_time_per_picture_l3565_356564

/-- Given a total number of pictures and total processing time in hours,
    calculate the time required to process each picture in minutes. -/
def time_per_picture (total_pictures : ℕ) (total_hours : ℕ) : ℚ :=
  (total_hours * 60) / total_pictures

/-- Theorem: Given 960 pictures and a total processing time of 32 hours,
    the time required to process each picture is 2 minutes. -/
theorem process_time_per_picture :
  time_per_picture 960 32 = 2 := by
  sorry

end NUMINAMATH_CALUDE_process_time_per_picture_l3565_356564


namespace NUMINAMATH_CALUDE_exists_monochromatic_triangle_l3565_356535

-- Define the vertices of the hexagon
inductive Vertex : Type
  | A | B | C | D | E | F

-- Define the colors
inductive Color : Type
  | Blue | Yellow

-- Define an edge as a pair of vertices
def Edge : Type := Vertex × Vertex

-- Function to get the color of an edge
def edge_color : Edge → Color := sorry

-- Define the hexagon
def hexagon : Set Edge := sorry

-- Theorem statement
theorem exists_monochromatic_triangle :
  ∃ (v1 v2 v3 : Vertex) (c : Color),
    v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧
    edge_color (v1, v2) = c ∧
    edge_color (v2, v3) = c ∧
    edge_color (v1, v3) = c :=
  sorry

end NUMINAMATH_CALUDE_exists_monochromatic_triangle_l3565_356535


namespace NUMINAMATH_CALUDE_science_club_membership_l3565_356587

theorem science_club_membership (total : ℕ) (biology : ℕ) (chemistry : ℕ) (both : ℕ) 
  (h1 : total = 60)
  (h2 : biology = 40)
  (h3 : chemistry = 35)
  (h4 : both = 25) :
  total - (biology + chemistry - both) = 10 := by
  sorry

end NUMINAMATH_CALUDE_science_club_membership_l3565_356587


namespace NUMINAMATH_CALUDE_contrapositive_is_true_l3565_356517

-- Define the original proposition
def original_proposition (x y : ℝ) : Prop :=
  (x = 2 ∧ y = 3) → (x + y = 5)

-- Define the contrapositive of the original proposition
def contrapositive (x y : ℝ) : Prop :=
  (x + y ≠ 5) → (x ≠ 2 ∨ y ≠ 3)

-- Theorem stating that the contrapositive is true
theorem contrapositive_is_true : ∀ x y : ℝ, contrapositive x y :=
by
  sorry

end NUMINAMATH_CALUDE_contrapositive_is_true_l3565_356517


namespace NUMINAMATH_CALUDE_equation_solution_l3565_356582

theorem equation_solution :
  ∀ x : ℝ, 3 * (x - 3) = (x - 3)^2 ↔ x = 3 ∨ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3565_356582


namespace NUMINAMATH_CALUDE_morning_snowfall_l3565_356590

theorem morning_snowfall (total : ℝ) (afternoon : ℝ) (h1 : total = 0.63) (h2 : afternoon = 0.5) :
  total - afternoon = 0.13 := by
  sorry

end NUMINAMATH_CALUDE_morning_snowfall_l3565_356590


namespace NUMINAMATH_CALUDE_polynomial_exists_for_non_squares_l3565_356567

-- Define the polynomial P(x,y,z)
def P (x y z : ℕ) : ℤ :=
  (1 - 2013 * (z - 1) * (z - 2)) * ((x + y - 1)^2 + 2*y - 2 + z)

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^2

-- State the theorem
theorem polynomial_exists_for_non_squares :
  ∀ n : ℕ, n > 0 →
    (¬ is_perfect_square n ↔ ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ P x y z = n) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_exists_for_non_squares_l3565_356567


namespace NUMINAMATH_CALUDE_weight_difference_l3565_356580

def john_weight : ℕ := 81
def roy_weight : ℕ := 4

theorem weight_difference : john_weight - roy_weight = 77 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l3565_356580


namespace NUMINAMATH_CALUDE_inequality_must_hold_l3565_356522

theorem inequality_must_hold (a b c : ℝ) (h : a > b ∧ b > c) : a - |c| > b - |c| := by
  sorry

end NUMINAMATH_CALUDE_inequality_must_hold_l3565_356522


namespace NUMINAMATH_CALUDE_special_function_at_2021_l3565_356500

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧ 
  (∀ x y, x > y ∧ y > 0 → f (x - y) = Real.sqrt (f (x * y) + 2))

/-- The main theorem stating that any function satisfying SpecialFunction has f(2021) = 2 -/
theorem special_function_at_2021 (f : ℝ → ℝ) (h : SpecialFunction f) : f 2021 = 2 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_2021_l3565_356500


namespace NUMINAMATH_CALUDE_cyclist_distance_l3565_356519

/-- The distance traveled by a cyclist moving between two people walking towards each other -/
theorem cyclist_distance (distance : ℝ) (speed_vasya : ℝ) (speed_roma : ℝ) 
  (h1 : distance > 0)
  (h2 : speed_vasya > 0)
  (h3 : speed_roma > 0) :
  let speed_dima := speed_vasya + speed_roma
  let time := distance / (speed_vasya + speed_roma)
  speed_dima * time = distance :=
by sorry

end NUMINAMATH_CALUDE_cyclist_distance_l3565_356519


namespace NUMINAMATH_CALUDE_butcher_purchase_cost_l3565_356561

/-- Calculates the total cost of a butcher's purchase given the weights and prices of various items. -/
theorem butcher_purchase_cost (steak_weight : ℚ) (steak_price : ℚ)
                               (chicken_weight : ℚ) (chicken_price : ℚ)
                               (sausage_weight : ℚ) (sausage_price : ℚ)
                               (pork_weight : ℚ) (pork_price : ℚ)
                               (bacon_weight : ℚ) (bacon_price : ℚ)
                               (salmon_weight : ℚ) (salmon_price : ℚ) :
  steak_weight = 3/2 ∧ steak_price = 15 ∧
  chicken_weight = 3/2 ∧ chicken_price = 8 ∧
  sausage_weight = 2 ∧ sausage_price = 13/2 ∧
  pork_weight = 7/2 ∧ pork_price = 10 ∧
  bacon_weight = 1/2 ∧ bacon_price = 9 ∧
  salmon_weight = 1/4 ∧ salmon_price = 30 →
  steak_weight * steak_price +
  chicken_weight * chicken_price +
  sausage_weight * sausage_price +
  pork_weight * pork_price +
  bacon_weight * bacon_price +
  salmon_weight * salmon_price = 189/2 := by
sorry


end NUMINAMATH_CALUDE_butcher_purchase_cost_l3565_356561


namespace NUMINAMATH_CALUDE_david_average_marks_l3565_356585

def david_marks : List ℕ := [86, 85, 82, 87, 85]

theorem david_average_marks :
  (List.sum david_marks) / (List.length david_marks) = 85 := by
  sorry

end NUMINAMATH_CALUDE_david_average_marks_l3565_356585


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l3565_356547

noncomputable section

def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m
def g (x : ℝ) : ℝ := 6 * Real.log x - 4 * x

theorem tangent_line_intersection (m : ℝ) : 
  (∃ a : ℝ, a > 0 ∧ 
    f m a = g a ∧ 
    (deriv (f m)) a = (deriv g) a) → 
  m = -5 :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_intersection_l3565_356547


namespace NUMINAMATH_CALUDE_milk_price_decrease_is_60_percent_l3565_356596

/-- Represents the price change of milk powder and coffee from June to July -/
structure PriceChange where
  june_price : ℝ  -- Price of both milk powder and coffee in June
  coffee_increase : ℝ  -- Percentage increase in coffee price
  july_mixture_price : ℝ  -- Price of 3 lbs mixture in July
  july_milk_price : ℝ  -- Price of milk powder per pound in July

/-- Calculates the percentage decrease in milk powder price -/
def milk_price_decrease (pc : PriceChange) : ℝ :=
  -- We'll implement the calculation here
  sorry

/-- Theorem stating that given the conditions, the milk price decrease is 60% -/
theorem milk_price_decrease_is_60_percent (pc : PriceChange) 
  (h1 : pc.coffee_increase = 200)
  (h2 : pc.july_mixture_price = 5.1)
  (h3 : pc.july_milk_price = 0.4) : 
  milk_price_decrease pc = 60 := by
  sorry

end NUMINAMATH_CALUDE_milk_price_decrease_is_60_percent_l3565_356596


namespace NUMINAMATH_CALUDE_handshakes_eight_couples_l3565_356570

/-- Represents the number of handshakes in a group of couples with one injured person --/
def handshakes_in_couples_group (num_couples : ℕ) : ℕ :=
  let total_people := 2 * num_couples
  let shaking_people := total_people - 1
  let handshakes_per_person := total_people - 3
  (shaking_people * handshakes_per_person) / 2

/-- Theorem stating that in a group of 8 married couples where everyone shakes hands
    with each other except their spouse and one person doesn't shake hands at all,
    the total number of handshakes is 90 --/
theorem handshakes_eight_couples :
  handshakes_in_couples_group 8 = 90 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_eight_couples_l3565_356570


namespace NUMINAMATH_CALUDE_valid_trapezoid_iff_s_gt_8r_l3565_356569

/-- A right-angled tangential trapezoid with an inscribed circle -/
structure RightAngledTangentialTrapezoid where
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Perimeter of the trapezoid -/
  s : ℝ
  /-- r is positive -/
  r_pos : r > 0
  /-- s is positive -/
  s_pos : s > 0

/-- Theorem: A valid right-angled tangential trapezoid exists iff s > 8r -/
theorem valid_trapezoid_iff_s_gt_8r (t : RightAngledTangentialTrapezoid) :
  ∃ (trapezoid : RightAngledTangentialTrapezoid), trapezoid.r = t.r ∧ trapezoid.s = t.s ↔ t.s > 8 * t.r :=
by sorry

end NUMINAMATH_CALUDE_valid_trapezoid_iff_s_gt_8r_l3565_356569


namespace NUMINAMATH_CALUDE_a_1998_value_l3565_356520

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  (∀ n m, n < m → a n < a m) ∧ 
  (∀ k : ℕ, ∃! (i j k : ℕ), k = a i + 2 * a j + 4 * a k)

theorem a_1998_value (a : ℕ → ℕ) (h : is_valid_sequence a) : a 1998 = 1227096648 := by
  sorry

end NUMINAMATH_CALUDE_a_1998_value_l3565_356520


namespace NUMINAMATH_CALUDE_set_A_equals_zero_to_three_l3565_356530

def A : Set ℤ := {x | x^2 - 3*x - 4 < 0}

theorem set_A_equals_zero_to_three : A = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_set_A_equals_zero_to_three_l3565_356530


namespace NUMINAMATH_CALUDE_rectangle_area_is_72_l3565_356591

-- Define the radius of the circles
def circle_radius : ℝ := 3

-- Define the rectangle
structure Rectangle where
  length : ℝ
  width : ℝ

-- Define the property that the circles touch each other and the rectangle sides
def circles_touch_rectangle_and_each_other (r : Rectangle) : Prop :=
  r.length = 4 * circle_radius ∧ r.width = 2 * circle_radius

-- Theorem statement
theorem rectangle_area_is_72 (r : Rectangle) 
  (h : circles_touch_rectangle_and_each_other r) : r.length * r.width = 72 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_72_l3565_356591


namespace NUMINAMATH_CALUDE_problem_statement_l3565_356550

theorem problem_statement : 
  let A := (16 * Real.sqrt 2) ^ (1/3 : ℝ)
  let B := Real.sqrt (9 * 9 ^ (1/3 : ℝ))
  let C := ((2 ^ (1/5 : ℝ)) ^ 2) ^ 2
  A ^ 2 + B ^ 3 + C ^ 5 = 105 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3565_356550


namespace NUMINAMATH_CALUDE_max_area_difference_l3565_356558

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the left vertex A and left focus F
def A : ℝ × ℝ := (-2, 0)
def F : ℝ × ℝ := (-1, 0)

-- Define a line passing through F
def line_through_F (k : ℝ) (x y : ℝ) : Prop := x = k*y - 1

-- Define the intersection points C and D
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | E p.1 p.2 ∧ line_through_F k p.1 p.2}

-- Define the area difference function
def area_difference (C D : ℝ × ℝ) : ℝ :=
  |C.2 + D.2|

-- Theorem statement
theorem max_area_difference :
  ∃ (max_diff : ℝ), max_diff = Real.sqrt 3 / 2 ∧
  ∀ (k : ℝ) (C D : ℝ × ℝ),
    C ∈ intersection_points k → D ∈ intersection_points k →
    area_difference C D ≤ max_diff :=
sorry

end NUMINAMATH_CALUDE_max_area_difference_l3565_356558


namespace NUMINAMATH_CALUDE_range_of_p_exists_point_C_l3565_356584

-- Define the parabola L: x^2 = 2py
def L (p : ℝ) := {(x, y) : ℝ × ℝ | x^2 = 2*p*y ∧ p > 0}

-- Define point M
def M : ℝ × ℝ := (2, 2)

-- Define the condition for points A and B
def satisfies_condition (A B : ℝ × ℝ) (p : ℝ) :=
  A ∈ L p ∧ B ∈ L p ∧ A ≠ B ∧ 
  (A.1 - M.1, A.2 - M.2) = (-B.1 + M.1, -B.2 + M.2)

-- Theorem 1: Range of p
theorem range_of_p (p : ℝ) :
  (∃ A B, satisfies_condition A B p) → p > 1 :=
sorry

-- Define the circle through three points
def circle_through (A B C : ℝ × ℝ) := 
  {(x, y) : ℝ × ℝ | (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
                    (x - A.1)^2 + (y - A.2)^2 = (x - C.1)^2 + (y - C.2)^2}

-- Define the tangent line to the parabola at a point
def tangent_line (p : ℝ) (C : ℝ × ℝ) :=
  {(x, y) : ℝ × ℝ | y - C.2 = (C.1 / (2*p)) * (x - C.1)}

-- Theorem 2: Existence of point C when p = 2
theorem exists_point_C :
  ∃ C, C ∈ L 2 ∧ C ≠ (0, 0) ∧ C ≠ (4, 4) ∧
       C.1 = -2 ∧ C.2 = 1 ∧
       (∀ x y, (x, y) ∈ circle_through (0, 0) (4, 4) C →
               (x, y) ∈ tangent_line 2 C) :=
sorry

end NUMINAMATH_CALUDE_range_of_p_exists_point_C_l3565_356584


namespace NUMINAMATH_CALUDE_simplify_expression_l3565_356589

theorem simplify_expression (x : ℝ) 
  (h1 : x ≠ 1) 
  (h2 : x ≠ -1) 
  (h3 : x ≠ (-1 + Real.sqrt 5) / 2) 
  (h4 : x ≠ (-1 - Real.sqrt 5) / 2) : 
  1 - (1 / (1 + x / (x^2 - 1))) = x / (x^2 + x - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3565_356589


namespace NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l3565_356515

theorem fraction_equality_implies_numerator_equality 
  (x y b : ℝ) (hb : b ≠ 0) : x / b = y / b → x = y :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l3565_356515


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l3565_356548

theorem area_between_concentric_circles (r₁ r₂ chord_length : ℝ) 
  (h₁ : r₁ = 60) 
  (h₂ : r₂ = 40) 
  (h₃ : chord_length = 100) 
  (h₄ : r₁ > r₂) 
  (h₅ : chord_length / 2 > r₂) : 
  (r₁^2 - r₂^2) * π = 2500 * π := by
  sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l3565_356548


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l3565_356599

theorem sum_of_x_solutions_is_zero (x y : ℝ) : 
  y = 9 → x^2 + y^2 = 169 → ∃ x₁ x₂ : ℝ, x₁ + x₂ = 0 ∧ x₁^2 + y^2 = 169 ∧ x₂^2 + y^2 = 169 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l3565_356599


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l3565_356566

theorem least_positive_integer_multiple_of_53 :
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → ¬(53 ∣ (3*y)^2 + 2*41*3*y + 41^2)) ∧
  (53 ∣ (3*x)^2 + 2*41*3*x + 41^2) ∧
  x = 4 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l3565_356566


namespace NUMINAMATH_CALUDE_set_B_proof_l3565_356521

open Set

theorem set_B_proof (U : Set ℕ) (A B : Set ℕ) :
  U = {1, 2, 3, 4, 5, 6, 7, 8, 9} →
  (U \ (A ∪ B)) = {1, 3} →
  ((U \ A) ∩ B) = {2, 4} →
  B = {5, 6, 7, 8, 9} :=
by sorry

end NUMINAMATH_CALUDE_set_B_proof_l3565_356521


namespace NUMINAMATH_CALUDE_time_to_reach_ticket_window_l3565_356528

-- Define Kit's movement and remaining distance
def initial_distance : ℝ := 90 -- feet
def initial_time : ℝ := 30 -- minutes
def remaining_distance : ℝ := 100 -- yards

-- Define conversion factor
def yards_to_feet : ℝ := 3 -- feet per yard

-- Theorem to prove
theorem time_to_reach_ticket_window : 
  (remaining_distance * yards_to_feet) / (initial_distance / initial_time) = 100 := by
  sorry

end NUMINAMATH_CALUDE_time_to_reach_ticket_window_l3565_356528


namespace NUMINAMATH_CALUDE_reciprocal_plus_two_product_l3565_356586

theorem reciprocal_plus_two_product (x y : ℝ) : 
  x ≠ y → x = 1/x + 2 → y = 1/y + 2 → x * y = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_plus_two_product_l3565_356586


namespace NUMINAMATH_CALUDE_no_upper_limit_for_q_q_determines_side_ratio_l3565_356592

/-- Represents a rectangle with side lengths a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ
  h : 0 < a ∧ 0 < b

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.a * r.b

/-- The combined area of two rotated congruent rectangles -/
noncomputable def combined_area (r : Rectangle) : ℝ :=
  if r.b / r.a ≥ Real.sqrt 2 - 1 then
    (1 - 1 / Real.sqrt 2) * (r.a + r.b)^2
  else
    2 * r.a * r.b - Real.sqrt 2 * r.b^2

/-- The ratio of combined area to single rectangle area -/
noncomputable def area_ratio (r : Rectangle) : ℝ :=
  combined_area r / r.area

theorem no_upper_limit_for_q :
  ∀ M : ℝ, ∃ r : Rectangle, area_ratio r > M :=
sorry

theorem q_determines_side_ratio {r : Rectangle} (h : Real.sqrt 2 ≤ area_ratio r ∧ area_ratio r < 2) :
  r.b / r.a = (2 - area_ratio r) / Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_no_upper_limit_for_q_q_determines_side_ratio_l3565_356592


namespace NUMINAMATH_CALUDE_sons_age_l3565_356559

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 16 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 14 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l3565_356559


namespace NUMINAMATH_CALUDE_pencils_taken_l3565_356509

theorem pencils_taken (initial_pencils : ℕ) (pencils_left : ℕ) (h1 : initial_pencils = 34) (h2 : pencils_left = 12) :
  initial_pencils - pencils_left = 22 := by
sorry

end NUMINAMATH_CALUDE_pencils_taken_l3565_356509


namespace NUMINAMATH_CALUDE_number_equation_proof_l3565_356540

theorem number_equation_proof (x : ℤ) : 
  x - (28 - (37 - (15 - 15))) = 54 → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_proof_l3565_356540


namespace NUMINAMATH_CALUDE_distribute_six_balls_four_boxes_l3565_356555

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes,
    with at least one ball in each box. -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 2 ways to distribute 6 indistinguishable balls into 4 indistinguishable boxes,
    with at least one ball in each box. -/
theorem distribute_six_balls_four_boxes :
  distribute_balls 6 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_four_boxes_l3565_356555


namespace NUMINAMATH_CALUDE_problem_distribution_l3565_356532

theorem problem_distribution (n m : ℕ) (hn : n = 10) (hm : m = 7) :
  (Nat.choose n m * Nat.factorial m * n^(n - m) : ℕ) = 712800000 :=
by sorry

end NUMINAMATH_CALUDE_problem_distribution_l3565_356532


namespace NUMINAMATH_CALUDE_acid_dilution_l3565_356576

/-- Proves that adding 30 ounces of pure water to 50 ounces of a 40% acid solution results in a 25% acid solution -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (water_added : ℝ) (final_concentration : ℝ) :
  initial_volume = 50 →
  initial_concentration = 0.4 →
  water_added = 30 →
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by
  sorry

#check acid_dilution

end NUMINAMATH_CALUDE_acid_dilution_l3565_356576


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3565_356506

theorem imaginary_part_of_z : Complex.im ((2 : ℂ) - Complex.I) ^ 2 = -4 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3565_356506


namespace NUMINAMATH_CALUDE_bumper_car_line_theorem_l3565_356588

/-- The number of people initially in line for bumper cars -/
def initial_people : ℕ := 9

/-- The number of people who left the line -/
def people_left : ℕ := 6

/-- The number of people who joined the line -/
def people_joined : ℕ := 3

/-- The final number of people in line -/
def final_people : ℕ := 6

/-- Theorem stating that the initial number of people satisfies the given conditions -/
theorem bumper_car_line_theorem :
  initial_people - people_left + people_joined = final_people :=
by sorry

end NUMINAMATH_CALUDE_bumper_car_line_theorem_l3565_356588


namespace NUMINAMATH_CALUDE_dividend_proof_l3565_356526

theorem dividend_proof : ∃ (a b : ℕ), 
  (11 * 10^5 + a * 10^3 + 7 * 10^2 + 7 * 10 + b) / 12 = 999809 → 
  11 * 10^5 + a * 10^3 + 7 * 10^2 + 7 * 10 + b = 11997708 :=
by
  sorry

end NUMINAMATH_CALUDE_dividend_proof_l3565_356526


namespace NUMINAMATH_CALUDE_gcd_288_123_l3565_356534

theorem gcd_288_123 : Nat.gcd 288 123 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_288_123_l3565_356534


namespace NUMINAMATH_CALUDE_workshop_total_workers_l3565_356597

/-- Represents the workshop scenario with workers and their salaries -/
structure Workshop where
  avgSalary : ℕ
  technicianCount : ℕ
  technicianAvgSalary : ℕ
  supervisorAvgSalary : ℕ
  laborerAvgSalary : ℕ
  supervisorLaborerTotalSalary : ℕ

/-- Theorem stating that the total number of workers in the workshop is 38 -/
theorem workshop_total_workers (w : Workshop)
  (h1 : w.avgSalary = 9000)
  (h2 : w.technicianCount = 6)
  (h3 : w.technicianAvgSalary = 12000)
  (h4 : w.supervisorAvgSalary = 15000)
  (h5 : w.laborerAvgSalary = 6000)
  (h6 : w.supervisorLaborerTotalSalary = 270000) :
  ∃ (supervisorCount laborerCount : ℕ),
    w.technicianCount + supervisorCount + laborerCount = 38 :=
by sorry

end NUMINAMATH_CALUDE_workshop_total_workers_l3565_356597


namespace NUMINAMATH_CALUDE_common_tangents_count_l3565_356511

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 4*y + 4 = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 4 = 0

-- Define the function to count common tangents
def count_common_tangents (C1 C2 : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem statement
theorem common_tangents_count :
  count_common_tangents circle_C1 circle_C2 = 3 := by sorry

end NUMINAMATH_CALUDE_common_tangents_count_l3565_356511


namespace NUMINAMATH_CALUDE_cos_two_alpha_plus_two_beta_l3565_356571

theorem cos_two_alpha_plus_two_beta (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3)
  (h2 : Real.cos α * Real.sin β = 1/6) : 
  Real.cos (2*α + 2*β) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_plus_two_beta_l3565_356571


namespace NUMINAMATH_CALUDE_customs_waiting_time_l3565_356544

/-- The time Jack waited to get through customs, given total waiting time and quarantine days. -/
theorem customs_waiting_time (total_hours quarantine_days : ℕ) : 
  total_hours = 356 ∧ quarantine_days = 14 → 
  total_hours - (quarantine_days * 24) = 20 := by
  sorry

end NUMINAMATH_CALUDE_customs_waiting_time_l3565_356544


namespace NUMINAMATH_CALUDE_complex_point_on_line_l3565_356537

theorem complex_point_on_line (a : ℝ) : 
  let z : ℂ := (a - Complex.I) / (1 + Complex.I)
  let x : ℝ := z.re
  let y : ℝ := z.im
  (x - y + 1 = 0) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_point_on_line_l3565_356537


namespace NUMINAMATH_CALUDE_two_colonies_limit_time_l3565_356563

/-- Represents the growth of a bacteria colony -/
structure BacteriaColony where
  initialSize : ℕ
  doubleTime : ℕ
  habitatLimit : ℕ

/-- The time it takes for a single colony to reach the habitat limit -/
def singleColonyLimitTime (colony : BacteriaColony) : ℕ := 20

/-- The size of a colony after a given number of days -/
def colonySize (colony : BacteriaColony) (days : ℕ) : ℕ :=
  colony.initialSize * 2^days

/-- Predicate to check if a colony has reached the habitat limit -/
def hasReachedLimit (colony : BacteriaColony) (days : ℕ) : Prop :=
  colonySize colony days ≥ colony.habitatLimit

/-- Theorem: Two colonies reach the habitat limit in the same time as a single colony -/
theorem two_colonies_limit_time (colony1 colony2 : BacteriaColony) :
  (∃ t : ℕ, hasReachedLimit colony1 t ∧ hasReachedLimit colony2 t) →
  (∃ t : ℕ, t = singleColonyLimitTime colony1 ∧ hasReachedLimit colony1 t ∧ hasReachedLimit colony2 t) :=
sorry

end NUMINAMATH_CALUDE_two_colonies_limit_time_l3565_356563


namespace NUMINAMATH_CALUDE_point_coordinates_sum_l3565_356549

/-- Given points A, B, C in a plane rectangular coordinate system,
    where AB is parallel to the x-axis and AC is parallel to the y-axis,
    prove that a + b = -1 -/
theorem point_coordinates_sum (a b : ℝ) : 
  (∃ (A B C : ℝ × ℝ),
    A = (a, -1) ∧
    B = (2, 3 - b) ∧
    C = (-5, 4) ∧
    A.2 = B.2 ∧  -- AB is parallel to x-axis
    A.1 = C.1    -- AC is parallel to y-axis
  ) →
  a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_sum_l3565_356549


namespace NUMINAMATH_CALUDE_sum_of_ages_l3565_356575

theorem sum_of_ages (bella_age : ℕ) (age_difference : ℕ) : 
  bella_age = 5 → 
  age_difference = 9 → 
  bella_age + (bella_age + age_difference) = 19 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l3565_356575


namespace NUMINAMATH_CALUDE_election_votes_l3565_356502

theorem election_votes (marcy barry joey : ℕ) : 
  marcy = 3 * barry → 
  barry = 2 * (joey + 3) → 
  marcy = 66 → 
  joey = 8 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l3565_356502


namespace NUMINAMATH_CALUDE_friends_who_bought_is_five_l3565_356578

/-- The number of pencils in one color box -/
def pencils_per_box : ℕ := 7

/-- The total number of pencils -/
def total_pencils : ℕ := 42

/-- The number of color boxes Chloe has -/
def chloe_boxes : ℕ := 1

/-- Calculate the number of friends who bought the color box -/
def friends_who_bought : ℕ :=
  (total_pencils - chloe_boxes * pencils_per_box) / pencils_per_box

theorem friends_who_bought_is_five : friends_who_bought = 5 := by
  sorry

end NUMINAMATH_CALUDE_friends_who_bought_is_five_l3565_356578


namespace NUMINAMATH_CALUDE_count_prime_in_sequence_count_1973_in_sequence_l3565_356579

def generate_sequence (steps : Nat) : List Nat :=
  sorry

def count_occurrences (n : Nat) (list : List Nat) : Nat :=
  sorry

def is_prime (n : Nat) : Prop :=
  sorry

theorem count_prime_in_sequence (p : Nat) (h : is_prime p) :
  count_occurrences p (generate_sequence 1973) = p - 1 :=
sorry

theorem count_1973_in_sequence :
  count_occurrences 1973 (generate_sequence 1973) = 1972 :=
sorry

end NUMINAMATH_CALUDE_count_prime_in_sequence_count_1973_in_sequence_l3565_356579


namespace NUMINAMATH_CALUDE_points_four_units_from_negative_two_l3565_356573

def distance (x y : ℝ) : ℝ := |x - y|

theorem points_four_units_from_negative_two : 
  {x : ℝ | distance x (-2) = 4} = {2, -6} := by
  sorry

end NUMINAMATH_CALUDE_points_four_units_from_negative_two_l3565_356573


namespace NUMINAMATH_CALUDE_four_fish_guarantee_l3565_356552

/-- A coloring of vertices of a regular polygon -/
def Coloring (n : ℕ) := Fin n → Bool

/-- The number of perpendicular segments with same-colored endpoints for a given diagonal -/
def sameColorSegments (n : ℕ) (c : Coloring n) (d : Fin n) : ℕ :=
  sorry

theorem four_fish_guarantee (c : Coloring 20) :
  ∃ d : Fin 20, sameColorSegments 20 c d ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_four_fish_guarantee_l3565_356552


namespace NUMINAMATH_CALUDE_fraction_simplification_l3565_356594

theorem fraction_simplification (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (8 * a^4 * b^2 * c) / (4 * a^3 * b) = 2 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3565_356594


namespace NUMINAMATH_CALUDE_candle_height_at_half_time_l3565_356536

/-- Calculates the total burning time for a candle of given height -/
def totalBurningTime (height : ℕ) : ℕ :=
  10 * (height * (height + 1) * (2 * height + 1)) / 6

/-- Calculates the height of the candle after a given time -/
def heightAfterTime (initialHeight : ℕ) (elapsedTime : ℕ) : ℕ :=
  initialHeight - (Finset.filter (fun k => 10 * k * k ≤ elapsedTime) (Finset.range initialHeight)).card

theorem candle_height_at_half_time (initialHeight : ℕ) (halfTimeHeight : ℕ) :
  initialHeight = 150 →
  halfTimeHeight = heightAfterTime initialHeight (totalBurningTime initialHeight / 2) →
  halfTimeHeight = 80 := by
  sorry

#eval heightAfterTime 150 (totalBurningTime 150 / 2)

end NUMINAMATH_CALUDE_candle_height_at_half_time_l3565_356536


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_120_l3565_356554

theorem last_three_digits_of_7_to_120 : 7^120 % 1000 = 681 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_120_l3565_356554


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3565_356508

/-- Given a real number m and a complex number z defined as z = (2m² + m - 1) + (-m² - 3m - 2)i,
    if z is purely imaginary, then m = 1/2. -/
theorem purely_imaginary_complex_number (m : ℝ) :
  let z : ℂ := Complex.mk (2 * m^2 + m - 1) (-m^2 - 3*m - 2)
  (z.re = 0 ∧ z.im ≠ 0) → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3565_356508


namespace NUMINAMATH_CALUDE_prob_at_least_one_red_l3565_356545

/-- Probability of drawing at least one red ball in three independent draws with replacement -/
theorem prob_at_least_one_red : 
  let total_balls : ℕ := 2
  let red_balls : ℕ := 1
  let num_draws : ℕ := 3
  let prob_blue : ℚ := 1 / 2
  let prob_all_blue : ℚ := prob_blue ^ num_draws
  prob_all_blue = 1 / 8 ∧ (1 : ℚ) - prob_all_blue = 7 / 8 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_red_l3565_356545


namespace NUMINAMATH_CALUDE_work_increase_proof_l3565_356562

theorem work_increase_proof (W p : ℝ) (h_p_pos : p > 0) : 
  let original_work_per_person := W / p
  let remaining_persons := (7 / 8) * p
  let new_work_per_person := W / remaining_persons
  new_work_per_person - original_work_per_person = W / (7 * p) :=
by sorry

end NUMINAMATH_CALUDE_work_increase_proof_l3565_356562


namespace NUMINAMATH_CALUDE_merry_go_round_revolutions_l3565_356568

theorem merry_go_round_revolutions (outer_radius inner_radius : ℝ) 
  (outer_revolutions : ℕ) (h1 : outer_radius = 30) (h2 : inner_radius = 10) 
  (h3 : outer_revolutions = 25) : 
  ∃ inner_revolutions : ℕ, 
    inner_revolutions * inner_radius * 2 * Real.pi = outer_revolutions * outer_radius * 2 * Real.pi ∧ 
    inner_revolutions = 75 := by
  sorry

end NUMINAMATH_CALUDE_merry_go_round_revolutions_l3565_356568


namespace NUMINAMATH_CALUDE_unique_solution_for_B_l3565_356542

theorem unique_solution_for_B : ∃! B : ℕ, 
  B < 10 ∧ (∃ A : ℕ, A < 10 ∧ 500 + 10 * A + 8 - (100 * B + 14) = 364) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_for_B_l3565_356542


namespace NUMINAMATH_CALUDE_census_population_scientific_notation_l3565_356523

/-- 
Given a positive integer n, its scientific notation is a representation of the form a × 10^b, 
where 1 ≤ a < 10 and b is an integer.
-/
def scientific_notation (n : ℕ+) : ℝ × ℤ := sorry

theorem census_population_scientific_notation :
  scientific_notation 932700 = (9.327, 5) := by sorry

end NUMINAMATH_CALUDE_census_population_scientific_notation_l3565_356523


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l3565_356546

theorem scientific_notation_equivalence : 
  8200000 = 8.2 * (10 : ℝ) ^ 6 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l3565_356546


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l3565_356529

theorem square_garden_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) :
  area = 200 →
  area = side ^ 2 →
  perimeter = 4 * side →
  perimeter = 40 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l3565_356529


namespace NUMINAMATH_CALUDE_benny_comic_books_l3565_356531

theorem benny_comic_books (x : ℚ) : 
  (3/4 * (2/5 * x + 12) + 18 = 72) → x = 150 := by
  sorry

end NUMINAMATH_CALUDE_benny_comic_books_l3565_356531


namespace NUMINAMATH_CALUDE_apples_for_juice_l3565_356518

/-- Given that 36 apples make 27 liters of apple juice, prove that 12 apples make 9 liters of apple juice -/
theorem apples_for_juice (apples : ℕ) (juice : ℕ) (h : 36 * juice = 27 * apples) : 
  12 * juice = 9 * apples :=
by sorry

end NUMINAMATH_CALUDE_apples_for_juice_l3565_356518


namespace NUMINAMATH_CALUDE_point_movement_l3565_356501

def number_line_move (start : ℤ) (move : ℤ) : ℤ :=
  start + move

theorem point_movement :
  let point_A : ℤ := -3
  let movement : ℤ := 6
  let point_B : ℤ := number_line_move point_A movement
  point_B = 3 := by sorry

end NUMINAMATH_CALUDE_point_movement_l3565_356501


namespace NUMINAMATH_CALUDE_cos_shift_equals_sin_l3565_356543

theorem cos_shift_equals_sin (x : ℝ) : 
  Real.cos (x + π/3) = Real.sin (x + 5*π/6) := by
  sorry

end NUMINAMATH_CALUDE_cos_shift_equals_sin_l3565_356543
