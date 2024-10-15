import Mathlib

namespace NUMINAMATH_CALUDE_cricket_average_l3705_370576

theorem cricket_average (innings : ℕ) (next_runs : ℕ) (increase : ℕ) (current_average : ℕ) : 
  innings = 10 → 
  next_runs = 79 → 
  increase = 4 → 
  (innings * current_average + next_runs) / (innings + 1) = current_average + increase →
  current_average = 35 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_l3705_370576


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_minimum_a_for_f_geq_two_l3705_370580

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a*x - 1| + |x + 2|

-- Part I
theorem solution_set_when_a_is_one :
  ∀ x : ℝ, f 1 x ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 :=
sorry

-- Part II
theorem minimum_a_for_f_geq_two :
  (∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f a x ≥ 2) ∧
  (∀ ε > 0, ∃ a x : ℝ, 0 < a ∧ a < 1/2 + ε ∧ f a x < 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_minimum_a_for_f_geq_two_l3705_370580


namespace NUMINAMATH_CALUDE_fraction_equality_l3705_370507

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 40)
  (h2 : p / n = 5)
  (h3 : p / q = 1 / 8) :
  m / q = 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3705_370507


namespace NUMINAMATH_CALUDE_eq_length_is_40_l3705_370514

/-- Represents a trapezoid with a circle inscribed in it -/
structure InscribedCircleTrapezoid where
  -- Lengths of the trapezoid sides
  ef : ℝ
  fg : ℝ
  gh : ℝ
  he : ℝ
  -- Ensure EF is parallel to GH (represented by their lengths being different)
  ef_parallel_gh : ef ≠ gh
  -- Circle center Q is on EF
  eq : ℝ
  -- Circle is tangent to FG and HE (implicitly assumed by the structure)

/-- The specific trapezoid from the problem -/
def problemTrapezoid : InscribedCircleTrapezoid where
  ef := 100
  fg := 60
  gh := 22
  he := 80
  ef_parallel_gh := by norm_num
  eq := 40  -- This is what we want to prove

/-- The main theorem: EQ = 40 in the given trapezoid -/
theorem eq_length_is_40 : problemTrapezoid.eq = 40 := by
  sorry

#eval problemTrapezoid.eq  -- Should output 40

end NUMINAMATH_CALUDE_eq_length_is_40_l3705_370514


namespace NUMINAMATH_CALUDE_fermat_prime_condition_l3705_370577

theorem fermat_prime_condition (n : ℕ) :
  Nat.Prime (2^n + 1) → (n = 0 ∨ ∃ α : ℕ, n = 2^α) :=
by sorry

end NUMINAMATH_CALUDE_fermat_prime_condition_l3705_370577


namespace NUMINAMATH_CALUDE_cost_price_proof_l3705_370553

/-- The cost price of a ball in rupees -/
def cost_price : ℝ := 90

/-- The number of balls sold -/
def balls_sold : ℕ := 13

/-- The selling price of all balls in rupees -/
def selling_price : ℝ := 720

/-- The number of balls whose cost price equals the loss -/
def loss_balls : ℕ := 5

theorem cost_price_proof :
  cost_price * balls_sold = selling_price + cost_price * loss_balls :=
sorry

end NUMINAMATH_CALUDE_cost_price_proof_l3705_370553


namespace NUMINAMATH_CALUDE_second_subject_grade_l3705_370596

/-- Represents the grade of a student in a subject as a percentage -/
def Grade := Fin 101

/-- Calculates the average of three grades -/
def average (g1 g2 g3 : Grade) : ℚ :=
  (g1.val + g2.val + g3.val) / 3

theorem second_subject_grade 
  (g1 g3 : Grade) 
  (h1 : g1.val = 60) 
  (h3 : g3.val = 80) :
  ∃ (g2 : Grade), average g1 g2 g3 = 70 ∧ g2.val = 70 := by
  sorry

end NUMINAMATH_CALUDE_second_subject_grade_l3705_370596


namespace NUMINAMATH_CALUDE_bacteria_growth_example_l3705_370522

/-- The time needed for bacteria to reach a certain population -/
def bacteria_growth_time (initial_count : ℕ) (final_count : ℕ) (growth_factor : ℕ) (growth_time : ℕ) : ℕ :=
  let growth_cycles := (final_count / initial_count).log growth_factor
  growth_cycles * growth_time

/-- Theorem: The time needed for 200 bacteria to reach 145,800 bacteria, 
    given that they triple every 5 hours, is 30 hours. -/
theorem bacteria_growth_example : bacteria_growth_time 200 145800 3 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_example_l3705_370522


namespace NUMINAMATH_CALUDE_function_machine_output_l3705_370530

/-- Function machine operation -/
def function_machine (input : ℕ) : ℕ :=
  let doubled := input * 2
  if doubled ≤ 15 then
    doubled * 3
  else
    doubled * 3

/-- Theorem: The function machine outputs 90 for an input of 15 -/
theorem function_machine_output : function_machine 15 = 90 := by
  sorry

end NUMINAMATH_CALUDE_function_machine_output_l3705_370530


namespace NUMINAMATH_CALUDE_corral_area_ratio_l3705_370584

/-- The side length of a small equilateral triangular corral -/
def small_side : ℝ := sorry

/-- The side length of the large equilateral triangular corral -/
def large_side : ℝ := 3 * small_side

/-- The area of a single small equilateral triangular corral -/
def small_area : ℝ := sorry

/-- The area of the large equilateral triangular corral -/
def large_area : ℝ := sorry

/-- The total area of all nine small equilateral triangular corrals -/
def total_small_area : ℝ := 9 * small_area

theorem corral_area_ratio : total_small_area = large_area := by
  sorry

end NUMINAMATH_CALUDE_corral_area_ratio_l3705_370584


namespace NUMINAMATH_CALUDE_sam_puppies_l3705_370531

theorem sam_puppies (initial_puppies : Float) (given_away : Float) :
  initial_puppies = 6.0 → given_away = 2.0 → initial_puppies - given_away = 4.0 := by
  sorry

end NUMINAMATH_CALUDE_sam_puppies_l3705_370531


namespace NUMINAMATH_CALUDE_additional_people_needed_l3705_370549

/-- The number of person-hours required to mow the lawn -/
def personHours : ℕ := 32

/-- The initial number of people who can mow the lawn in 8 hours -/
def initialPeople : ℕ := 4

/-- The desired time to mow the lawn -/
def desiredTime : ℕ := 3

/-- The total number of people needed to mow the lawn in the desired time -/
def totalPeopleNeeded : ℕ := (personHours + desiredTime - 1) / desiredTime

theorem additional_people_needed :
  totalPeopleNeeded - initialPeople = 7 :=
by sorry

end NUMINAMATH_CALUDE_additional_people_needed_l3705_370549


namespace NUMINAMATH_CALUDE_probability_two_non_defective_pens_l3705_370510

theorem probability_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h1 : total_pens = 10) (h2 : defective_pens = 3) :
  let non_defective := total_pens - defective_pens
  let prob_first := non_defective / total_pens
  let prob_second := (non_defective - 1) / (total_pens - 1)
  prob_first * prob_second = 7 / 15 := by sorry

end NUMINAMATH_CALUDE_probability_two_non_defective_pens_l3705_370510


namespace NUMINAMATH_CALUDE_divisibility_conditions_l3705_370521

theorem divisibility_conditions (n : ℤ) :
  (∃ k : ℤ, 3 ∣ (5*n^2 + 10*n + 8) ↔ n = 2 + 3*k) ∧
  (∃ k : ℤ, 4 ∣ (5*n^2 + 10*n + 8) ↔ n = 2*k) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_conditions_l3705_370521


namespace NUMINAMATH_CALUDE_binomial_18_6_l3705_370512

theorem binomial_18_6 : (Nat.choose 18 6) = 18564 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_6_l3705_370512


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3705_370506

theorem cubic_equation_roots :
  ∃ (pos_roots : ℕ), 
    (pos_roots = 1 ∨ pos_roots = 3) ∧
    (∀ x : ℝ, x^3 - 3*x^2 + 4*x - 12 = 0 → x > 0) ∧
    (¬∃ x : ℝ, x < 0 ∧ x^3 - 3*x^2 + 4*x - 12 = 0) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3705_370506


namespace NUMINAMATH_CALUDE_number_of_players_is_64_l3705_370550

/-- The cost of a pair of shoes in dollars -/
def shoe_cost : ℕ := 12

/-- The cost of a jersey in dollars -/
def jersey_cost : ℕ := shoe_cost + 8

/-- The cost of a cap in dollars -/
def cap_cost : ℕ := jersey_cost / 2

/-- The total cost for one player's equipment in dollars -/
def player_cost : ℕ := 2 * (shoe_cost + jersey_cost) + cap_cost

/-- The total expenses for all players' equipment in dollars -/
def total_expenses : ℕ := 4760

theorem number_of_players_is_64 : 
  ∃ n : ℕ, n * player_cost = total_expenses ∧ n = 64 := by
  sorry

end NUMINAMATH_CALUDE_number_of_players_is_64_l3705_370550


namespace NUMINAMATH_CALUDE_extreme_value_and_monotonicity_l3705_370537

/-- The function f(x) = x³ + ax² + bx + a² -/
def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f(x) -/
def f' (x a b : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extreme_value_and_monotonicity (a b : ℝ) :
  (f 1 a b = 10 ∧ f' 1 a b = 0) →
  (a = 4 ∧ b = -11) ∧
  (∀ x : ℝ, 
    (b = -a^2 → 
      (a > 0 → 
        ((x < -a ∨ x > a/3) → f' x a (-a^2) > 0) ∧
        ((-a < x ∧ x < a/3) → f' x a (-a^2) < 0)) ∧
      (a < 0 → 
        ((x < a/3 ∨ x > -a) → f' x a (-a^2) > 0) ∧
        ((a/3 < x ∧ x < -a) → f' x a (-a^2) < 0)) ∧
      (a = 0 → f' x a (-a^2) > 0))) :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_and_monotonicity_l3705_370537


namespace NUMINAMATH_CALUDE_conference_room_arrangements_l3705_370590

/-- The number of distinct arrangements of seats in a conference room -/
theorem conference_room_arrangements (n m : ℕ) (hn : n = 12) (hm : m = 4) :
  (Nat.choose n m) = 495 := by sorry

end NUMINAMATH_CALUDE_conference_room_arrangements_l3705_370590


namespace NUMINAMATH_CALUDE_mixed_doubles_pairings_l3705_370568

theorem mixed_doubles_pairings (n_men : Nat) (n_women : Nat) : 
  n_men = 5 → n_women = 4 → (n_men.choose 2) * (n_women.choose 2) * 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_mixed_doubles_pairings_l3705_370568


namespace NUMINAMATH_CALUDE_largest_n_divisible_equality_l3705_370571

def divisibleCount (n : ℕ) (d : ℕ) : ℕ := (n / d : ℕ)

def divisibleBy5or7 (n : ℕ) : ℕ :=
  divisibleCount n 5 + divisibleCount n 7 - divisibleCount n 35

theorem largest_n_divisible_equality : ∀ n : ℕ, n > 65 →
  (divisibleCount n 3 ≠ divisibleBy5or7 n) ∧
  (divisibleCount 65 3 = divisibleBy5or7 65) := by
  sorry

#eval divisibleCount 65 3  -- Expected: 21
#eval divisibleBy5or7 65   -- Expected: 21

end NUMINAMATH_CALUDE_largest_n_divisible_equality_l3705_370571


namespace NUMINAMATH_CALUDE_divisibility_by_30_l3705_370504

theorem divisibility_by_30 :
  (∃ p : ℕ, p.Prime ∧ p ≥ 7 ∧ 30 ∣ (p^2 - 1)) ∧
  (∃ q : ℕ, q.Prime ∧ q ≥ 7 ∧ ¬(30 ∣ (q^2 - 1))) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_30_l3705_370504


namespace NUMINAMATH_CALUDE_min_yz_minus_xy_l3705_370509

/-- Represents a triangle with integer side lengths -/
structure Triangle :=
  (xy yz xz : ℕ)

/-- The perimeter of the triangle -/
def Triangle.perimeter (t : Triangle) : ℕ := t.xy + t.yz + t.xz

/-- Predicate for a valid triangle satisfying the given conditions -/
def isValidTriangle (t : Triangle) : Prop :=
  t.xy < t.yz ∧ t.yz ≤ t.xz ∧
  t.perimeter = 2010 ∧
  t.xy + t.yz > t.xz ∧ t.xy + t.xz > t.yz ∧ t.yz + t.xz > t.xy

theorem min_yz_minus_xy (t : Triangle) (h : isValidTriangle t) :
  ∀ (t' : Triangle), isValidTriangle t' → t'.yz - t'.xy ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_min_yz_minus_xy_l3705_370509


namespace NUMINAMATH_CALUDE_even_function_solution_set_l3705_370527

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The domain of f is symmetric about the origin -/
def HasSymmetricDomain (f : ℝ → ℝ) (a : ℝ) : Prop :=
  Set.Icc (1 + a) 1 = Set.Icc (-1) 1

theorem even_function_solution_set
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h1 : f = fun x ↦ a * x^2 + b * x + 2)
  (h2 : IsEven f)
  (h3 : HasSymmetricDomain f a) :
  {x : ℝ | f x > 0} = Set.Ioo (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_solution_set_l3705_370527


namespace NUMINAMATH_CALUDE_equation_system_solution_l3705_370534

theorem equation_system_solution (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 5)
  (h2 : a / x + b / y + c / z = 0) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l3705_370534


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3705_370592

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x - f y) = f (f y) + x * f y + f x - 1

/-- The theorem stating that the only function satisfying the functional equation is f(z) = 1 - z²/2 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ z : ℝ, f z = 1 - z^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3705_370592


namespace NUMINAMATH_CALUDE_planes_perpendicular_from_parallel_lines_l3705_370575

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- A line is contained in a plane -/
def contained_in (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line is perpendicular to a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Two planes are perpendicular -/
def perpendicular_plane_plane (p1 p2 : Plane3D) : Prop :=
  sorry

theorem planes_perpendicular_from_parallel_lines
  (m n : Line3D) (α β : Plane3D)
  (h1 : parallel m n)
  (h2 : contained_in m α)
  (h3 : perpendicular_line_plane n β) :
  perpendicular_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_from_parallel_lines_l3705_370575


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l3705_370529

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : principal = 10000)
  (h2 : rate = 0.05)
  (h3 : time = 1) :
  principal * rate * time = 500 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l3705_370529


namespace NUMINAMATH_CALUDE_fish_ratio_l3705_370544

/-- The number of fish in Billy's aquarium -/
def billy_fish : ℕ := 10

/-- The number of fish in Tony's aquarium -/
def tony_fish : ℕ := billy_fish * 3

/-- The number of fish in Sarah's aquarium -/
def sarah_fish : ℕ := tony_fish + 5

/-- The number of fish in Bobby's aquarium -/
def bobby_fish : ℕ := sarah_fish * 2

/-- The total number of fish in all aquariums -/
def total_fish : ℕ := 145

theorem fish_ratio : 
  billy_fish = 10 ∧ 
  tony_fish = billy_fish * 3 ∧ 
  sarah_fish = tony_fish + 5 ∧ 
  bobby_fish = sarah_fish * 2 ∧ 
  bobby_fish + sarah_fish + tony_fish + billy_fish = total_fish → 
  tony_fish / billy_fish = 3 := by
  sorry

end NUMINAMATH_CALUDE_fish_ratio_l3705_370544


namespace NUMINAMATH_CALUDE_clownfish_in_display_tank_l3705_370505

theorem clownfish_in_display_tank 
  (total_fish : ℕ)
  (clownfish blowfish : ℕ)
  (blowfish_in_own_tank : ℕ)
  (h1 : total_fish = 100)
  (h2 : clownfish = blowfish)
  (h3 : blowfish_in_own_tank = 26)
  (h4 : total_fish = clownfish + blowfish) :
  let blowfish_in_display := blowfish - blowfish_in_own_tank
  let initial_clownfish_in_display := blowfish_in_display
  let final_clownfish_in_display := initial_clownfish_in_display - initial_clownfish_in_display / 3
  final_clownfish_in_display = 16 := by
sorry

end NUMINAMATH_CALUDE_clownfish_in_display_tank_l3705_370505


namespace NUMINAMATH_CALUDE_num_small_orders_l3705_370552

def large_order_weight : ℕ := 200
def small_order_weight : ℕ := 50
def total_weight_used : ℕ := 800
def num_large_orders : ℕ := 3

theorem num_small_orders : 
  (total_weight_used - num_large_orders * large_order_weight) / small_order_weight = 4 := by
  sorry

end NUMINAMATH_CALUDE_num_small_orders_l3705_370552


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l3705_370508

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_to_scientific_notation :
  toScientificNotation (87.65 * 10^9) = ScientificNotation.mk 8.765 10 sorry := by
  sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l3705_370508


namespace NUMINAMATH_CALUDE_mean_twice_mode_iff_x_21_l3705_370591

def is_valid_list (x : ℕ) : Prop :=
  x > 0 ∧ x ≤ 100

def mean_of_list (x : ℕ) : ℚ :=
  (31 + 58 + 98 + 3 * x) / 6

def mode_of_list (x : ℕ) : ℕ := x

theorem mean_twice_mode_iff_x_21 :
  ∀ x : ℕ, is_valid_list x →
    (mean_of_list x = 2 * mode_of_list x) ↔ x = 21 := by
  sorry

end NUMINAMATH_CALUDE_mean_twice_mode_iff_x_21_l3705_370591


namespace NUMINAMATH_CALUDE_overall_gain_percentage_l3705_370539

/-- Calculate the overall gain percentage for three items --/
theorem overall_gain_percentage
  (cycle_cp cycle_sp scooter_cp scooter_sp skateboard_cp skateboard_sp : ℚ)
  (h_cycle_cp : cycle_cp = 900)
  (h_cycle_sp : cycle_sp = 1170)
  (h_scooter_cp : scooter_cp = 15000)
  (h_scooter_sp : scooter_sp = 18000)
  (h_skateboard_cp : skateboard_cp = 2000)
  (h_skateboard_sp : skateboard_sp = 2400) :
  let total_cp := cycle_cp + scooter_cp + skateboard_cp
  let total_sp := cycle_sp + scooter_sp + skateboard_sp
  let gain_percentage := (total_sp - total_cp) / total_cp * 100
  ∃ (ε : ℚ), abs (gain_percentage - 20.50) < ε ∧ ε > 0 ∧ ε < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_overall_gain_percentage_l3705_370539


namespace NUMINAMATH_CALUDE_square_perimeter_l3705_370559

/-- Given a square with area 625 cm², prove its perimeter is 100 cm -/
theorem square_perimeter (s : ℝ) (h_area : s^2 = 625) : 4 * s = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3705_370559


namespace NUMINAMATH_CALUDE_max_boxes_per_delivery_l3705_370513

/-- Represents the maximum capacity of each truck in pounds -/
def truckCapacity : ℕ := 2000

/-- Represents the weight of a light box in pounds -/
def lightBoxWeight : ℕ := 10

/-- Represents the weight of a heavy box in pounds -/
def heavyBoxWeight : ℕ := 40

/-- Represents the number of trucks available for each delivery -/
def numberOfTrucks : ℕ := 3

/-- Theorem stating the maximum number of boxes that can be shipped in each delivery -/
theorem max_boxes_per_delivery :
  ∃ (n : ℕ), n = numberOfTrucks * truckCapacity / (lightBoxWeight + heavyBoxWeight) * 2 ∧ n = 240 := by
  sorry

end NUMINAMATH_CALUDE_max_boxes_per_delivery_l3705_370513


namespace NUMINAMATH_CALUDE_system_solution_l3705_370519

/-- The function φ(t) = 2t^3 + t - 2 -/
def φ (t : ℝ) : ℝ := 2 * t^3 + t - 2

/-- The system of equations -/
def satisfies_system (x y z : ℝ) : Prop :=
  x^5 = φ y ∧ y^5 = φ z ∧ z^5 = φ x

theorem system_solution (x y z : ℝ) (h : satisfies_system x y z) :
  x = y ∧ y = z ∧ φ x = x^5 := by
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l3705_370519


namespace NUMINAMATH_CALUDE_lawrence_walk_l3705_370501

/-- The distance traveled given a constant speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating that given a speed of 3 km/h and a time of 1.33 hours, 
    the distance traveled is 3.99 km -/
theorem lawrence_walk : distance 3 1.33 = 3.99 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_walk_l3705_370501


namespace NUMINAMATH_CALUDE_inequality_proof_l3705_370569

theorem inequality_proof (a b c : ℝ) (h1 : a < b) (h2 : b < c) :
  a^2 * b + b^2 * c + c^2 * a < a^2 * c + b^2 * a + c^2 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3705_370569


namespace NUMINAMATH_CALUDE_divisible_by_thirteen_l3705_370593

theorem divisible_by_thirteen (n : ℤ) : 
  13 ∣ (n^2 - 6*n - 4) ↔ n ≡ 3 [ZMOD 13] := by
sorry

end NUMINAMATH_CALUDE_divisible_by_thirteen_l3705_370593


namespace NUMINAMATH_CALUDE_line_properties_l3705_370581

/-- Represents a line in the form ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the slope angle of a line in degrees --/
def slope_angle (l : Line) : ℝ :=
  sorry

/-- Calculates the y-intercept of a line --/
def y_intercept (l : Line) : ℝ :=
  sorry

/-- The line x + y + 1 = 0 --/
def line : Line :=
  { a := 1, b := 1, c := 1 }

theorem line_properties :
  slope_angle line = 135 ∧ y_intercept line = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_properties_l3705_370581


namespace NUMINAMATH_CALUDE_parallel_line_through_midpoint_l3705_370595

/-- Given two points A and B in ℝ², and a line L, 
    prove that the line passing through the midpoint of AB 
    and parallel to L has the equation 3x + y + 3 = 0 -/
theorem parallel_line_through_midpoint 
  (A B : ℝ × ℝ) 
  (hA : A = (-5, 2)) 
  (hB : B = (1, 4)) 
  (L : ℝ → ℝ) 
  (hL : ∀ x y, L x = y ↔ 3 * x + y - 2 = 0) :
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ∀ x y, 3 * x + y + 3 = 0 ↔ 
    ∃ k, y - M.2 = k * (x - M.1) ∧ 
         ∀ x' y', L x' = y' → y' - M.2 = k * (x' - M.1) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_midpoint_l3705_370595


namespace NUMINAMATH_CALUDE_unique_solution_ceiling_equation_l3705_370503

theorem unique_solution_ceiling_equation :
  ∃! b : ℝ, b + ⌈b⌉ = 19.6 := by sorry

end NUMINAMATH_CALUDE_unique_solution_ceiling_equation_l3705_370503


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_players_l3705_370554

theorem cricket_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (h1 : total_players = 120)
  (h2 : throwers = 55)
  (h3 : 2 * (total_players - throwers) = 5 * (total_players - throwers - (total_players - throwers - throwers)))
  (h4 : throwers ≤ total_players) :
  throwers + (total_players - throwers - (2 * (total_players - throwers) / 5)) = 94 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_right_handed_players_l3705_370554


namespace NUMINAMATH_CALUDE_tom_video_game_spending_l3705_370516

/-- The amount Tom spent on the Batman game -/
def batman_game_cost : ℚ := 13.6

/-- The amount Tom spent on the Superman game -/
def superman_game_cost : ℚ := 5.06

/-- The total amount Tom spent on video games -/
def total_spent : ℚ := batman_game_cost + superman_game_cost

theorem tom_video_game_spending :
  total_spent = 18.66 := by sorry

end NUMINAMATH_CALUDE_tom_video_game_spending_l3705_370516


namespace NUMINAMATH_CALUDE_apple_boxes_count_apple_boxes_count_specific_l3705_370547

theorem apple_boxes_count (apples_per_crate : ℕ) (crates_delivered : ℕ) 
  (rotten_apples : ℕ) (apples_per_box : ℕ) : ℕ :=
  let total_apples := apples_per_crate * crates_delivered
  let remaining_apples := total_apples - rotten_apples
  remaining_apples / apples_per_box

theorem apple_boxes_count_specific : 
  apple_boxes_count 180 12 160 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_apple_boxes_count_apple_boxes_count_specific_l3705_370547


namespace NUMINAMATH_CALUDE_no_distributive_laws_hold_l3705_370588

-- Define the # operation
def hash (a b : ℝ) : ℝ := a + 2 * b

-- Theorem stating that none of the distributive laws hold
theorem no_distributive_laws_hold :
  ¬(∀ (x y z : ℝ), hash x (y + z) = hash x y + hash x z) ∧
  ¬(∀ (x y z : ℝ), x + hash y z = hash (x + y) (x + z)) ∧
  ¬(∀ (x y z : ℝ), hash x (hash y z) = hash (hash x y) (hash x z)) :=
sorry

end NUMINAMATH_CALUDE_no_distributive_laws_hold_l3705_370588


namespace NUMINAMATH_CALUDE_clock_distance_theorem_l3705_370545

/-- Represents a clock on the table -/
structure Clock where
  center : ℝ × ℝ
  radius : ℝ

/-- The state of all clocks at a given time -/
def ClockState := List Clock

/-- Calculate the position of the minute hand at a given time -/
def minuteHandPosition (clock : Clock) (time : ℝ) : ℝ × ℝ :=
  sorry

/-- Calculate the sum of distances from the table center to clock centers -/
def sumDistancesToCenters (clocks : ClockState) : ℝ :=
  sorry

/-- Calculate the sum of distances from the table center to minute hand ends at a given time -/
def sumDistancesToMinuteHands (clocks : ClockState) (time : ℝ) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem clock_distance_theorem (clocks : ClockState) (h : clocks.length = 50) :
  ∃ t : ℝ, sumDistancesToMinuteHands clocks t > sumDistancesToCenters clocks :=
sorry

end NUMINAMATH_CALUDE_clock_distance_theorem_l3705_370545


namespace NUMINAMATH_CALUDE_no_unique_five_day_august_l3705_370515

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a calendar month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Given a month, returns the number of occurrences of each day of the week -/
def countDaysInMonth (m : Month) : DayOfWeek → Nat :=
  sorry

/-- July has five Tuesdays and 30 days -/
def july : Month :=
  { days := 30,
    firstDay := sorry }

/-- August follows July and has 30 days -/
def august : Month :=
  { days := 30,
    firstDay := sorry }

/-- There is no unique day that occurs five times in August -/
theorem no_unique_five_day_august :
  ¬ ∃! (d : DayOfWeek), countDaysInMonth august d = 5 :=
sorry

end NUMINAMATH_CALUDE_no_unique_five_day_august_l3705_370515


namespace NUMINAMATH_CALUDE_custom_op_example_l3705_370511

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := 5 * a + 2 * b

-- State the theorem
theorem custom_op_example : custom_op 4 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l3705_370511


namespace NUMINAMATH_CALUDE_satisfying_polynomial_form_l3705_370585

/-- A polynomial that satisfies the given equation for all real numbers a, b, c 
    such that ab + bc + ca = 0 -/
def SatisfyingPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, a * b + b * c + c * a = 0 → 
    P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)

/-- Theorem stating the form of polynomials satisfying the equation -/
theorem satisfying_polynomial_form (P : ℝ → ℝ) :
  SatisfyingPolynomial P →
  ∃ a b : ℝ, ∀ x : ℝ, P x = a * x^4 + b * x^2 := by
  sorry

end NUMINAMATH_CALUDE_satisfying_polynomial_form_l3705_370585


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3705_370589

theorem inequality_solution_set (x : ℝ) :
  (1 / (x^2 + 1) > 4/x + 19/10) ↔ (-2 < x ∧ x < 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3705_370589


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3705_370564

theorem quadratic_function_property (a m : ℝ) (ha : a > 0) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + x + a
  f m < 0 → f (m + 1) > 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3705_370564


namespace NUMINAMATH_CALUDE_range_of_h_sign_and_inequality_smallest_upper_bound_l3705_370543

-- Define sets A and M_n
def A : Set (ℝ → ℝ) := {f | ∃ k, ∀ x > 0, f x < k}
def M (n : ℕ) : Set (ℝ → ℝ) := {f | ∀ x y, 0 < x ∧ x < y → (f x / x^n) < (f y / y^n)}

-- Statement 1
theorem range_of_h (h : ℝ) :
  (fun x => x^3 + h) ∈ M 1 ↔ h ≤ 0 :=
sorry

-- Statement 2
theorem sign_and_inequality (f : ℝ → ℝ) (a b d : ℝ) 
  (hf : f ∈ M 1) (hab : 0 < a ∧ a < b) (hd : f a = d ∧ f b = d) :
  d < 0 ∧ f (a + b) > 2 * d :=
sorry

-- Statement 3
theorem smallest_upper_bound (m : ℝ) :
  (∀ f ∈ A ∩ M 2, ∀ x > 0, f x < m) ↔ m ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_h_sign_and_inequality_smallest_upper_bound_l3705_370543


namespace NUMINAMATH_CALUDE_m_range_l3705_370579

/-- The function g(x) = mx + 2 -/
def g (m : ℝ) (x : ℝ) : ℝ := m * x + 2

/-- The function f(x) = x^2 - 2x -/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The closed interval [-1, 2] -/
def I : Set ℝ := Set.Icc (-1) 2

theorem m_range :
  (∀ m : ℝ, (∀ x₁ ∈ I, ∃ x₀ ∈ I, g m x₁ = f x₀) ↔ m ∈ Set.Icc (-1) (1/2)) :=
sorry

end NUMINAMATH_CALUDE_m_range_l3705_370579


namespace NUMINAMATH_CALUDE_divisibility_and_expression_l3705_370574

theorem divisibility_and_expression (k : ℕ) : 
  30^k ∣ 929260 → 3^k - k^3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_and_expression_l3705_370574


namespace NUMINAMATH_CALUDE_band_members_proof_l3705_370578

/-- Represents the price per set of costumes based on the quantity purchased -/
def price_per_set (quantity : ℕ) : ℕ :=
  if quantity ≤ 39 then 80
  else if quantity ≤ 79 then 70
  else 60

theorem band_members_proof :
  ∀ (x y : ℕ),
    x + y = 75 →
    x ≥ 40 →
    price_per_set x * x + price_per_set y * y = 5600 →
    x = 40 ∧ y = 35 := by
  sorry

end NUMINAMATH_CALUDE_band_members_proof_l3705_370578


namespace NUMINAMATH_CALUDE_solve_banana_cost_l3705_370599

def banana_cost_problem (initial_amount remaining_amount pears_cost asparagus_cost chicken_cost : ℕ) 
  (num_banana_packs : ℕ) : Prop :=
  let total_spent := initial_amount - remaining_amount
  let other_items_cost := pears_cost + asparagus_cost + chicken_cost
  let banana_total_cost := total_spent - other_items_cost
  banana_total_cost / num_banana_packs = 4

theorem solve_banana_cost :
  banana_cost_problem 55 28 2 6 11 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_banana_cost_l3705_370599


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_specific_m_value_diameter_circle_equation_l3705_370573

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem circle_intersection_theorem :
  ∀ m : ℝ,
  (∃ x y : ℝ, circle_equation x y m) ↔ m < 5 :=
sorry

theorem specific_m_value :
  ∀ x1 y1 x2 y2 : ℝ,
  circle_equation x1 y1 (8/5) ∧
  circle_equation x2 y2 (8/5) ∧
  line_equation x1 y1 ∧
  line_equation x2 y2 ∧
  perpendicular x1 y1 x2 y2 →
  (8/5 : ℝ) = 8/5 :=
sorry

theorem diameter_circle_equation :
  ∀ x1 y1 x2 y2 : ℝ,
  circle_equation x1 y1 (8/5) ∧
  circle_equation x2 y2 (8/5) ∧
  line_equation x1 y1 ∧
  line_equation x2 y2 ∧
  perpendicular x1 y1 x2 y2 →
  ∀ x y : ℝ,
  x^2 + y^2 - (8/5)*x - (16/5)*y = 0 ↔
  (x - x1) * (x - x2) + (y - y1) * (y - y2) = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_specific_m_value_diameter_circle_equation_l3705_370573


namespace NUMINAMATH_CALUDE_ball_probabilities_l3705_370555

structure BallBag where
  red_balls : ℕ
  white_balls : ℕ

def initial_bag : BallBag := ⟨3, 2⟩

def total_balls (bag : BallBag) : ℕ := bag.red_balls + bag.white_balls

def P_A1 (bag : BallBag) : ℚ := bag.red_balls / total_balls bag
def P_A2 (bag : BallBag) : ℚ := bag.white_balls / total_balls bag

def P_B (bag : BallBag) : ℚ :=
  (P_A1 bag * (bag.red_balls - 1) / (total_balls bag - 1)) +
  (P_A2 bag * (bag.white_balls - 1) / (total_balls bag - 1))

def P_C_given_A2 (bag : BallBag) : ℚ := bag.red_balls / (total_balls bag - 1)

theorem ball_probabilities (bag : BallBag) :
  P_A1 bag + P_A2 bag = 1 ∧
  P_B initial_bag = 2/5 ∧
  P_C_given_A2 initial_bag = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_l3705_370555


namespace NUMINAMATH_CALUDE_simplify_expression_l3705_370563

theorem simplify_expression (a : ℝ) (h : 2 < a ∧ a < 3) :
  Real.sqrt ((a - Real.pi) ^ 2) + |a - 2| = Real.pi - 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3705_370563


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l3705_370540

/-- Calculates the average speed of a round trip given the following conditions:
  * Distance traveled one way is 5280 feet (1 mile)
  * Speed northward is 3 minutes per mile
  * Rest time is 10 minutes
  * Speed southward is 3 miles per minute
-/
theorem round_trip_average_speed :
  let distance_feet : ℝ := 5280
  let distance_miles : ℝ := distance_feet / 5280
  let speed_north : ℝ := 1 / 3  -- miles per minute
  let speed_south : ℝ := 3  -- miles per minute
  let rest_time : ℝ := 10  -- minutes
  let time_north : ℝ := distance_miles / speed_north
  let time_south : ℝ := distance_miles / speed_south
  let total_time : ℝ := time_north + time_south + rest_time
  let total_distance : ℝ := 2 * distance_miles
  let avg_speed : ℝ := total_distance / (total_time / 60)
  avg_speed = 9 := by
  sorry


end NUMINAMATH_CALUDE_round_trip_average_speed_l3705_370540


namespace NUMINAMATH_CALUDE_elementary_to_kindergarten_ratio_is_two_to_one_l3705_370525

/-- Represents the purchase of dinosaur models by a school --/
structure ModelPurchase where
  regular_price : ℕ  -- Regular price of each model in dollars
  kindergarten_models : ℕ  -- Number of models for kindergarten library
  total_paid : ℕ  -- Total amount paid in dollars
  discount_percent : ℕ  -- Discount percentage applied

/-- Calculates the ratio of elementary library models to kindergarten library models --/
def elementary_to_kindergarten_ratio (purchase : ModelPurchase) : ℚ :=
  let discounted_price := purchase.regular_price * (100 - purchase.discount_percent) / 100
  let kindergarten_cost := purchase.kindergarten_models * purchase.regular_price
  let elementary_cost := purchase.total_paid - kindergarten_cost
  let elementary_models := elementary_cost / discounted_price
  elementary_models / purchase.kindergarten_models

/-- Theorem stating the ratio of elementary to kindergarten models is 2:1 --/
theorem elementary_to_kindergarten_ratio_is_two_to_one 
  (purchase : ModelPurchase)
  (h1 : purchase.regular_price = 100)
  (h2 : purchase.kindergarten_models = 2)
  (h3 : purchase.total_paid = 570)
  (h4 : purchase.discount_percent = 5)
  (h5 : purchase.kindergarten_models + 
        (purchase.total_paid - purchase.kindergarten_models * purchase.regular_price) / 
        (purchase.regular_price * (100 - purchase.discount_percent) / 100) > 5) :
  elementary_to_kindergarten_ratio purchase = 2 := by
  sorry

end NUMINAMATH_CALUDE_elementary_to_kindergarten_ratio_is_two_to_one_l3705_370525


namespace NUMINAMATH_CALUDE_white_balls_in_box_l3705_370566

theorem white_balls_in_box (orange_balls black_balls : ℕ) 
  (prob_not_orange_or_white : ℚ) (white_balls : ℕ) : 
  orange_balls = 8 → 
  black_balls = 7 → 
  prob_not_orange_or_white = 38095238095238093 / 100000000000000000 →
  (black_balls : ℚ) / (orange_balls + black_balls + white_balls : ℚ) = prob_not_orange_or_white →
  white_balls = 3 := by
sorry

end NUMINAMATH_CALUDE_white_balls_in_box_l3705_370566


namespace NUMINAMATH_CALUDE_root_sum_square_problem_l3705_370582

theorem root_sum_square_problem (α β : ℝ) : 
  (α^2 + 2*α - 2025 = 0) → 
  (β^2 + 2*β - 2025 = 0) → 
  α^2 + 3*α + β = 2023 := by
sorry

end NUMINAMATH_CALUDE_root_sum_square_problem_l3705_370582


namespace NUMINAMATH_CALUDE_sprint_race_losing_distance_l3705_370546

/-- Represents a sprint race between Kelly and Abel -/
structure SprintRace where
  raceLength : ℝ
  headStart : ℝ
  extraDistanceToOvertake : ℝ

/-- Calculates the distance by which Abel lost the race to Kelly -/
def losingDistance (race : SprintRace) : ℝ :=
  race.headStart + race.extraDistanceToOvertake

theorem sprint_race_losing_distance : 
  let race : SprintRace := {
    raceLength := 100,
    headStart := 3,
    extraDistanceToOvertake := 19.9
  }
  losingDistance race = 22.9 := by sorry

end NUMINAMATH_CALUDE_sprint_race_losing_distance_l3705_370546


namespace NUMINAMATH_CALUDE_closest_whole_number_to_expression_l3705_370583

theorem closest_whole_number_to_expression : 
  ∃ (n : ℕ), n = 1000 ∧ 
  ∀ (m : ℕ), |((10^2010 + 5 * 10^2012) : ℚ) / ((2 * 10^2011 + 3 * 10^2011) : ℚ) - n| ≤ 
             |((10^2010 + 5 * 10^2012) : ℚ) / ((2 * 10^2011 + 3 * 10^2011) : ℚ) - m| :=
by sorry

end NUMINAMATH_CALUDE_closest_whole_number_to_expression_l3705_370583


namespace NUMINAMATH_CALUDE_mango_problem_l3705_370500

theorem mango_problem (alexis dilan ashley : ℕ) : 
  alexis = 4 * (dilan + ashley) →
  ashley = 2 * dilan →
  alexis = 60 →
  alexis + dilan + ashley = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_mango_problem_l3705_370500


namespace NUMINAMATH_CALUDE_slope_product_is_two_l3705_370526

/-- Given two lines with slopes m and n, where one line makes twice the angle
    with the horizontal as the other, has 4 times the slope, and is not horizontal,
    prove that the product of their slopes is 2. -/
theorem slope_product_is_two (m n : ℝ) : 
  (∃ θ₁ θ₂ : ℝ, θ₁ = 2 * θ₂ ∧ m = Real.tan θ₁ ∧ n = Real.tan θ₂) →  -- L₁ makes twice the angle
  m = 4 * n →                                                      -- L₁ has 4 times the slope
  m ≠ 0 →                                                          -- L₁ is not horizontal
  m * n = 2 := by
  sorry

end NUMINAMATH_CALUDE_slope_product_is_two_l3705_370526


namespace NUMINAMATH_CALUDE_periodic_function_l3705_370533

/-- A function f is periodic if there exists a non-zero real number p such that
    f(x + p) = f(x) for all x in the domain of f. -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x

/-- The given conditions on function f -/
structure FunctionConditions (f : ℝ → ℝ) (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  (sum_neq : a₁ + b₁ ≠ a₂ + b₂)
  (cond : ∀ x : ℝ, (f (a₁ + x) = f (b₁ - x) ∧ f (a₂ + x) = f (b₂ - x)) ∨
                   (f (a₁ + x) = -f (b₁ - x) ∧ f (a₂ + x) = -f (b₂ - x)))

/-- The main theorem stating that f is periodic with the given period -/
theorem periodic_function (f : ℝ → ℝ) (a₁ b₁ a₂ b₂ : ℝ)
    (h : FunctionConditions f a₁ b₁ a₂ b₂) :
    IsPeriodic f ∧ ∃ p : ℝ, p = |((a₂ + b₂) - (a₁ + b₁))| ∧
    ∀ x : ℝ, f (x + p) = f x :=
  sorry


end NUMINAMATH_CALUDE_periodic_function_l3705_370533


namespace NUMINAMATH_CALUDE_special_cone_volume_l3705_370561

/-- A cone with circumscribed and inscribed spheres sharing the same center -/
structure SpecialCone where
  /-- The radius of the circumscribed sphere -/
  r_circum : ℝ
  /-- The circumscribed and inscribed spheres have the same center -/
  spheres_same_center : Bool

/-- The volume of the special cone -/
noncomputable def volume (cone : SpecialCone) : ℝ := sorry

/-- Theorem: The volume of the special cone is 3π when the radius of the circumscribed sphere is 2 -/
theorem special_cone_volume (cone : SpecialCone) 
  (h1 : cone.r_circum = 2) 
  (h2 : cone.spheres_same_center = true) : 
  volume cone = 3 * Real.pi := by sorry

end NUMINAMATH_CALUDE_special_cone_volume_l3705_370561


namespace NUMINAMATH_CALUDE_paperclip_theorem_l3705_370536

/-- The day of the week when Jasmine first has more than 500 paperclips -/
theorem paperclip_theorem : ∃ k : ℕ, k > 0 ∧ 
  (∀ j : ℕ, j < k → 5 * 3^j ≤ 500) ∧ 
  5 * 3^k > 500 ∧
  k = 5 := by
  sorry

end NUMINAMATH_CALUDE_paperclip_theorem_l3705_370536


namespace NUMINAMATH_CALUDE_max_corner_sum_l3705_370594

/-- Represents a face of the cube -/
structure Face where
  value : Nat
  inv_value : Nat
  sum_eq_eight : value + inv_value = 8
  value_in_range : 1 ≤ value ∧ value ≤ 6

/-- Represents a cube with six faces -/
structure Cube where
  faces : Fin 6 → Face
  distinct : ∀ i j, i ≠ j → (faces i).value ≠ (faces j).value

/-- Represents a corner of the cube -/
structure Corner where
  cube : Cube
  face1 : Fin 6
  face2 : Fin 6
  face3 : Fin 6
  distinct : face1 ≠ face2 ∧ face2 ≠ face3 ∧ face1 ≠ face3
  adjacent : ¬ ((cube.faces face1).inv_value = (cube.faces face2).value ∨
                (cube.faces face1).inv_value = (cube.faces face3).value ∨
                (cube.faces face2).inv_value = (cube.faces face3).value)

/-- The sum of values at a corner -/
def cornerSum (c : Corner) : Nat :=
  (c.cube.faces c.face1).value + (c.cube.faces c.face2).value + (c.cube.faces c.face3).value

/-- The theorem to be proved -/
theorem max_corner_sum (c : Cube) : 
  ∀ corner : Corner, corner.cube = c → cornerSum corner ≤ 15 :=
sorry

end NUMINAMATH_CALUDE_max_corner_sum_l3705_370594


namespace NUMINAMATH_CALUDE_decagon_ratio_l3705_370597

/-- Represents a decagon with specific properties -/
structure Decagon where
  unit_squares : ℕ
  triangles : ℕ
  triangle_base : ℝ
  bottom_square : ℕ
  bottom_area : ℝ

/-- Theorem statement for the decagon problem -/
theorem decagon_ratio 
  (d : Decagon)
  (h1 : d.unit_squares = 12)
  (h2 : d.triangles = 2)
  (h3 : d.triangle_base = 3)
  (h4 : d.bottom_square = 1)
  (h5 : d.bottom_area = 6)
  : ∃ (xq yq : ℝ), xq / yq = 1 ∧ xq + yq = 3 := by
  sorry

end NUMINAMATH_CALUDE_decagon_ratio_l3705_370597


namespace NUMINAMATH_CALUDE_valuable_files_after_three_rounds_l3705_370565

def first_round_files : ℕ := 1200
def first_round_delete_percent : ℚ := 80 / 100
def second_round_files : ℕ := 600
def second_round_irrelevant_fraction : ℚ := 4 / 5
def final_round_files : ℕ := 700
def final_round_not_pertinent_percent : ℚ := 65 / 100

theorem valuable_files_after_three_rounds :
  let first_round_valuable := first_round_files - (first_round_files * first_round_delete_percent).floor
  let second_round_valuable := second_round_files - (second_round_files * second_round_irrelevant_fraction).floor
  let final_round_valuable := final_round_files - (final_round_files * final_round_not_pertinent_percent).floor
  first_round_valuable + second_round_valuable + final_round_valuable = 605 := by
sorry


end NUMINAMATH_CALUDE_valuable_files_after_three_rounds_l3705_370565


namespace NUMINAMATH_CALUDE_man_rowing_speed_l3705_370587

/-- The speed of a man rowing in still water, given his speeds with wind influence -/
theorem man_rowing_speed 
  (upstream_speed : ℝ) 
  (downstream_speed : ℝ) 
  (wind_speed : ℝ) 
  (h1 : upstream_speed = 25) 
  (h2 : downstream_speed = 65) 
  (h3 : wind_speed = 5) : 
  (upstream_speed + downstream_speed) / 2 = 45 := by
sorry


end NUMINAMATH_CALUDE_man_rowing_speed_l3705_370587


namespace NUMINAMATH_CALUDE_number_percentage_problem_l3705_370558

theorem number_percentage_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 35 → (40/100 : ℝ) * N = 420 := by
  sorry

end NUMINAMATH_CALUDE_number_percentage_problem_l3705_370558


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3705_370518

/-- The length of the major axis of an ellipse formed by the intersection of a plane and a right circular cylinder -/
def majorAxisLength (cylinderRadius : ℝ) (majorAxisRatio : ℝ) : ℝ :=
  2 * cylinderRadius * (1 + majorAxisRatio)

/-- Theorem: The length of the major axis is 7 for the given conditions -/
theorem ellipse_major_axis_length :
  majorAxisLength 2 0.75 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3705_370518


namespace NUMINAMATH_CALUDE_A_3_2_l3705_370572

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2 : A 3 2 = 12 := by sorry

end NUMINAMATH_CALUDE_A_3_2_l3705_370572


namespace NUMINAMATH_CALUDE_initially_calculated_average_of_class_l3705_370538

/-- The initially calculated average height of a class of boys -/
def initially_calculated_average (num_boys : ℕ) (actual_average : ℚ) (initial_error : ℕ) : ℚ :=
  actual_average + (initial_error : ℚ) / num_boys

/-- Theorem stating the initially calculated average height -/
theorem initially_calculated_average_of_class (num_boys : ℕ) (actual_average : ℚ) (initial_error : ℕ) 
  (h1 : num_boys = 35)
  (h2 : actual_average = 178)
  (h3 : initial_error = 50) :
  initially_calculated_average num_boys actual_average initial_error = 179 + 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_initially_calculated_average_of_class_l3705_370538


namespace NUMINAMATH_CALUDE_bottle_cap_distribution_l3705_370586

theorem bottle_cap_distribution (total_caps : ℕ) (num_groups : ℕ) (caps_per_group : ℕ) : 
  total_caps = 35 → num_groups = 7 → caps_per_group = total_caps / num_groups → caps_per_group = 5 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_distribution_l3705_370586


namespace NUMINAMATH_CALUDE_angle_theta_trig_values_l3705_370517

/-- An angle θ with vertex at the origin, initial side along positive x-axis, and terminal side on y = 2x -/
structure AngleTheta where
  terminal_side : ∀ (x y : ℝ), y = 2 * x

theorem angle_theta_trig_values (θ : AngleTheta) :
  ∃ (s c : ℝ),
    s^2 + c^2 = 1 ∧
    |s| = 2 * Real.sqrt 5 / 5 ∧
    |c| = Real.sqrt 5 / 5 ∧
    s / c = 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_theta_trig_values_l3705_370517


namespace NUMINAMATH_CALUDE_range_of_a_l3705_370532

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | (4*x - 3)^2 ≤ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ a + 1}

-- Define the property that ¬p is a necessary but not sufficient condition for ¬q
def necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, x ∉ B a → x ∉ A) ∧ ¬(∀ x, x ∉ A → x ∉ B a)

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, necessary_not_sufficient a ↔ 0 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3705_370532


namespace NUMINAMATH_CALUDE_circle_sections_theorem_l3705_370560

-- Define the circle and its sections
def Circle (r : ℝ) := { x : ℝ × ℝ | x.1^2 + x.2^2 = r^2 }

structure Section (r : ℝ) where
  area : ℝ
  perimeter : ℝ

-- Define the theorem
theorem circle_sections_theorem (r : ℝ) (h : r > 0) :
  ∃ (s1 s2 s3 : Section r),
    -- Areas are equal and sum to the circle's area
    s1.area = s2.area ∧ s2.area = s3.area ∧
    s1.area + s2.area + s3.area = π * r^2 ∧
    -- Each section's area is r²π/3
    s1.area = (π * r^2) / 3 ∧
    -- Perimeters are equal to the circle's perimeter
    s1.perimeter = s2.perimeter ∧ s2.perimeter = s3.perimeter ∧
    s1.perimeter = 2 * π * r :=
by
  sorry


end NUMINAMATH_CALUDE_circle_sections_theorem_l3705_370560


namespace NUMINAMATH_CALUDE_product_mod_seven_l3705_370535

theorem product_mod_seven : (2009 * 2010 * 2011 * 2012) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l3705_370535


namespace NUMINAMATH_CALUDE_solution_absolute_value_equation_l3705_370548

theorem solution_absolute_value_equation (x : ℝ) : 5 * x + 2 * |x| = 3 * x → x ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_absolute_value_equation_l3705_370548


namespace NUMINAMATH_CALUDE_total_raisins_l3705_370557

def yellow_raisins : ℝ := 0.3
def black_raisins : ℝ := 0.4

theorem total_raisins : yellow_raisins + black_raisins = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_total_raisins_l3705_370557


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_range_l3705_370570

theorem quadratic_inequality_implies_range (x : ℝ) : 
  x^2 - 7*x + 12 < 0 → 42 < x^2 + 7*x + 12 ∧ x^2 + 7*x + 12 < 56 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_range_l3705_370570


namespace NUMINAMATH_CALUDE_parabola_properties_l3705_370562

-- Define the parabola
structure Parabola where
  a : ℝ
  equation : ℝ → ℝ → Prop := fun x y ↦ y^2 = 2 * a * x

-- Define the properties of the parabola C
def C : Parabola where
  a := 1  -- This makes the equation y² = 2x

-- Theorem statement
theorem parabola_properties :
  -- The parabola passes through (2,2)
  C.equation 2 2 ∧
  -- The focus is on the x-axis at (1/2, 0)
  C.equation (1/2) 0 ∧
  -- The intersection with x - y - 1 = 0 gives |MN| = 2√6
  ∃ (x₁ x₂ : ℝ),
    C.equation x₁ (x₁ - 1) ∧
    C.equation x₂ (x₂ - 1) ∧
    x₁ ≠ x₂ ∧
    (x₂ - x₁)^2 + ((x₂ - 1) - (x₁ - 1))^2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l3705_370562


namespace NUMINAMATH_CALUDE_dice_roll_probability_l3705_370520

/-- The probability of rolling a number less than four on a six-sided die -/
def prob_first_die : ℚ := 1 / 2

/-- The probability of rolling a number greater than five on an eight-sided die -/
def prob_second_die : ℚ := 3 / 8

/-- The probability of both events occurring -/
def prob_both : ℚ := prob_first_die * prob_second_die

theorem dice_roll_probability : prob_both = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l3705_370520


namespace NUMINAMATH_CALUDE_third_number_value_l3705_370542

theorem third_number_value : ∃ x : ℝ, 3 + 33 + x + 3.33 = 369.63 ∧ x = 330.30 := by
  sorry

end NUMINAMATH_CALUDE_third_number_value_l3705_370542


namespace NUMINAMATH_CALUDE_husband_towel_usage_l3705_370502

/-- The number of bath towels used by Kylie in a month -/
def kylie_towels : ℕ := 3

/-- The number of bath towels used by Kylie's daughters in a month -/
def daughters_towels : ℕ := 6

/-- The number of bath towels that fit in one load of laundry -/
def towels_per_load : ℕ := 4

/-- The number of loads of laundry needed to clean all used towels -/
def loads_of_laundry : ℕ := 3

/-- The number of bath towels used by the husband in a month -/
def husband_towels : ℕ := 3

theorem husband_towel_usage :
  kylie_towels + daughters_towels + husband_towels = towels_per_load * loads_of_laundry :=
by sorry

end NUMINAMATH_CALUDE_husband_towel_usage_l3705_370502


namespace NUMINAMATH_CALUDE_gcf_40_48_l3705_370551

theorem gcf_40_48 : Nat.gcd 40 48 = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcf_40_48_l3705_370551


namespace NUMINAMATH_CALUDE_differential_y_differential_F_differential_z_dz_at_zero_l3705_370567

noncomputable section

-- Function definitions
def y (x : ℝ) := x^3 - 3^x
def F (φ : ℝ) := Real.cos (φ/3) + Real.sin (3/φ)
def z (x : ℝ) := Real.log (1 + Real.exp (10*x)) + Real.arctan (Real.exp (5*x))⁻¹

-- Theorem statements
theorem differential_y (x : ℝ) :
  deriv y x = 3*x^2 - 3^x * Real.log 3 :=
sorry

theorem differential_F (φ : ℝ) (h : φ ≠ 0) :
  deriv F φ = -1/3 * Real.sin (φ/3) - 3 * Real.cos (3/φ) / φ^2 :=
sorry

theorem differential_z (x : ℝ) :
  deriv z x = (5 * Real.exp (5*x) * (2 * Real.exp (5*x) - 1)) / (1 + Real.exp (10*x)) :=
sorry

theorem dz_at_zero :
  (deriv z 0) * 0.1 = 0.25 :=
sorry

end

end NUMINAMATH_CALUDE_differential_y_differential_F_differential_z_dz_at_zero_l3705_370567


namespace NUMINAMATH_CALUDE_expand_expression_l3705_370524

theorem expand_expression (x : ℝ) : (7 * x^2 - 3) * 5 * x^3 = 35 * x^5 - 15 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3705_370524


namespace NUMINAMATH_CALUDE_hannah_friday_distance_l3705_370556

/-- The distance Hannah ran on Monday in kilometers -/
def monday_km : ℝ := 9

/-- The distance Hannah ran on Wednesday in meters -/
def wednesday_m : ℝ := 4816

/-- The additional distance Hannah ran on Monday compared to Wednesday and Friday combined, in meters -/
def additional_m : ℝ := 2089

/-- Conversion factor from kilometers to meters -/
def km_to_m : ℝ := 1000

theorem hannah_friday_distance :
  ∃ (friday_m : ℝ),
    (monday_km * km_to_m = wednesday_m + friday_m + additional_m) ∧
    friday_m = 2095 := by
  sorry

end NUMINAMATH_CALUDE_hannah_friday_distance_l3705_370556


namespace NUMINAMATH_CALUDE_inscribed_circle_triangle_sides_l3705_370541

/-- A triangle with an inscribed circle --/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle --/
  r : ℝ
  /-- The length of the first segment on one side --/
  s1 : ℝ
  /-- The length of the second segment on one side --/
  s2 : ℝ
  /-- The length of the second side --/
  a : ℝ
  /-- The length of the third side --/
  b : ℝ
  /-- Ensure all lengths are positive --/
  r_pos : r > 0
  s1_pos : s1 > 0
  s2_pos : s2 > 0
  a_pos : a > 0
  b_pos : b > 0

/-- Theorem about a specific triangle with an inscribed circle --/
theorem inscribed_circle_triangle_sides (t : InscribedCircleTriangle)
  (h1 : t.r = 4)
  (h2 : t.s1 = 6)
  (h3 : t.s2 = 8) :
  t.a = 13 ∧ t.b = 15 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_triangle_sides_l3705_370541


namespace NUMINAMATH_CALUDE_boom_boom_language_size_l3705_370528

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 6

/-- The length of a word -/
def word_length : ℕ := 6

/-- The number of words with at least two identical letters -/
def num_words_with_repeats : ℕ := alphabet_size ^ word_length - Nat.factorial alphabet_size

theorem boom_boom_language_size :
  num_words_with_repeats = 45936 :=
sorry

end NUMINAMATH_CALUDE_boom_boom_language_size_l3705_370528


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3705_370523

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x + k = 0 ∧ y^2 + 2*y + k = 0) ↔ k < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3705_370523


namespace NUMINAMATH_CALUDE_other_root_of_complex_equation_l3705_370598

theorem other_root_of_complex_equation (z : ℂ) :
  z ^ 2 = -72 + 27 * I →
  (-6 + 3 * I) ^ 2 = -72 + 27 * I →
  ∃ w : ℂ, w ^ 2 = -72 + 27 * I ∧ w ≠ -6 + 3 * I ∧ w = 6 - 3 * I :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_complex_equation_l3705_370598
