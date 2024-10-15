import Mathlib

namespace NUMINAMATH_CALUDE_g_50_eq_zero_l3725_372586

-- Define φ(n) as the number of positive integers not exceeding n that are coprime to n
def phi (n : ℕ) : ℕ := sorry

-- Define g(n) to satisfy the condition that for any positive integer n, 
-- the sum of g(d) over all positive divisors d of n equals φ(n)
def g (n : ℕ) : ℤ :=
  sorry

-- Theorem to prove
theorem g_50_eq_zero : g 50 = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_50_eq_zero_l3725_372586


namespace NUMINAMATH_CALUDE_calculate_walking_speed_l3725_372549

/-- Given two people walking towards each other, this theorem calculates the speed of one person given the total distance, the speed of the other person, and the distance traveled by the first person. -/
theorem calculate_walking_speed 
  (total_distance : ℝ) 
  (brad_speed : ℝ) 
  (maxwell_distance : ℝ) 
  (h1 : total_distance = 40) 
  (h2 : brad_speed = 5) 
  (h3 : maxwell_distance = 15) : 
  ∃ (maxwell_speed : ℝ), maxwell_speed = 3 := by
  sorry

#check calculate_walking_speed

end NUMINAMATH_CALUDE_calculate_walking_speed_l3725_372549


namespace NUMINAMATH_CALUDE_salary_increase_after_reduction_l3725_372501

theorem salary_increase_after_reduction (original_salary : ℝ) (h : original_salary > 0) :
  let reduced_salary := original_salary * (1 - 0.35)
  let increase_factor := (1 / 0.65) - 1
  reduced_salary * (1 + increase_factor) = original_salary := by
  sorry

#eval (1 / 0.65 - 1) * 100 -- To show the approximate percentage increase

end NUMINAMATH_CALUDE_salary_increase_after_reduction_l3725_372501


namespace NUMINAMATH_CALUDE_chimney_bricks_count_l3725_372531

-- Define the number of bricks in the chimney
def chimney_bricks : ℕ := 148

-- Define Brenda's time to build the chimney alone
def brenda_time : ℕ := 7

-- Define Brandon's time to build the chimney alone
def brandon_time : ℕ := 8

-- Define the time they take to build the chimney together
def combined_time : ℕ := 6

-- Define the productivity drop when working together
def productivity_drop : ℕ := 15

-- Theorem statement
theorem chimney_bricks_count :
  -- Individual rates
  let brenda_rate := chimney_bricks / brenda_time
  let brandon_rate := chimney_bricks / brandon_time
  -- Combined rate without drop
  let combined_rate := brenda_rate + brandon_rate
  -- Actual combined rate with productivity drop
  let actual_combined_rate := combined_rate - productivity_drop
  -- The work completed matches the number of bricks
  actual_combined_rate * combined_time = chimney_bricks := by
  sorry

end NUMINAMATH_CALUDE_chimney_bricks_count_l3725_372531


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l3725_372573

/-- Given two plane vectors a and b, with the angle between them being 60°,
    a = (2,0), and |b| = 1, prove that |a + 2b| = 2√3 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  let angle := Real.pi / 3  -- 60° in radians
  a = (2, 0) ∧ 
  ‖b‖ = 1 ∧ 
  a.1 * b.1 + a.2 * b.2 = ‖a‖ * ‖b‖ * Real.cos angle →
  ‖a + 2 • b‖ = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l3725_372573


namespace NUMINAMATH_CALUDE_max_carlson_jars_l3725_372541

/-- Represents the initial state of jam jars for Carlson and Baby -/
structure JamState where
  carlson_weight : ℕ  -- Total weight of Carlson's jars
  baby_weight : ℕ    -- Total weight of Baby's jars
  carlson_jars : ℕ   -- Number of Carlson's jars

/-- Conditions of the jam problem -/
def jam_conditions (state : JamState) : Prop :=
  state.carlson_weight = 13 * state.baby_weight ∧
  ∃ (lightest : ℕ), 
    lightest > 0 ∧
    (state.carlson_weight - lightest) = 8 * (state.baby_weight + lightest)

/-- The theorem to be proved -/
theorem max_carlson_jars : 
  ∀ (state : JamState), 
    jam_conditions state → 
    state.carlson_jars ≤ 23 :=
sorry

end NUMINAMATH_CALUDE_max_carlson_jars_l3725_372541


namespace NUMINAMATH_CALUDE_sequence_problem_l3725_372528

theorem sequence_problem (b : ℕ → ℝ) 
  (h1 : ∀ n ≥ 2, b n = b (n - 1) * b (n + 1))
  (h2 : b 1 = 3 + Real.sqrt 11)
  (h3 : b 1987 = 17 + Real.sqrt 11) :
  b 2015 = (3 - Real.sqrt 11) / 8 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l3725_372528


namespace NUMINAMATH_CALUDE_angle_CBO_is_20_degrees_l3725_372546

-- Define the triangle ABC
variable (A B C O : Point) (ABC : Triangle A B C)

-- Define the angles
def angle (P Q R : Point) : ℝ := sorry

-- State the theorem
theorem angle_CBO_is_20_degrees 
  (h1 : angle B A O = angle C A O)
  (h2 : angle C B O = angle A B O)
  (h3 : angle A C O = angle B C O)
  (h4 : angle A O C = 110)
  (h5 : ∀ P Q R : Point, angle P Q R + angle Q R P + angle R P Q = 180) :
  angle C B O = 20 := by sorry

end NUMINAMATH_CALUDE_angle_CBO_is_20_degrees_l3725_372546


namespace NUMINAMATH_CALUDE_Q_trajectory_equation_l3725_372519

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line on which point P moves -/
def line_P (p : Point) : Prop :=
  2 * p.x - p.y + 3 = 0

/-- The fixed point M -/
def M : Point :=
  ⟨-1, 2⟩

/-- Q is on the extension of PM and PM = MQ -/
def Q_position (p q : Point) : Prop :=
  q.x - M.x = M.x - p.x ∧ q.y - M.y = M.y - p.y

/-- The trajectory of point Q -/
def Q_trajectory (q : Point) : Prop :=
  2 * q.x - q.y + 5 = 0

/-- Theorem: The trajectory of Q satisfies the equation 2x - y + 5 = 0 -/
theorem Q_trajectory_equation :
  ∀ p q : Point, line_P p → Q_position p q → Q_trajectory q :=
by sorry

end NUMINAMATH_CALUDE_Q_trajectory_equation_l3725_372519


namespace NUMINAMATH_CALUDE_m_value_l3725_372599

def A (m : ℝ) : Set ℝ := {-1, 2, 2*m-1}
def B (m : ℝ) : Set ℝ := {2, m^2}

theorem m_value (m : ℝ) : B m ⊆ A m → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_m_value_l3725_372599


namespace NUMINAMATH_CALUDE_problem_solution_l3725_372511

theorem problem_solution (x y : ℝ) (n : ℝ) : 
  x = 3 → y = 1 → n = x - y^(x-(y+1)) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3725_372511


namespace NUMINAMATH_CALUDE_only_cylinder_not_polyhedron_l3725_372569

-- Define the set of given figures
inductive Figure
  | ObliquePrism
  | Cube
  | Cylinder
  | Tetrahedron

-- Define what a polyhedron is
def isPolyhedron (f : Figure) : Prop :=
  match f with
  | Figure.ObliquePrism => true
  | Figure.Cube => true
  | Figure.Cylinder => false
  | Figure.Tetrahedron => true

-- Theorem statement
theorem only_cylinder_not_polyhedron :
  ∀ f : Figure, ¬(isPolyhedron f) ↔ f = Figure.Cylinder :=
sorry

end NUMINAMATH_CALUDE_only_cylinder_not_polyhedron_l3725_372569


namespace NUMINAMATH_CALUDE_stating_max_wickets_theorem_l3725_372548

/-- Represents the maximum number of wickets a bowler can take in one over -/
def max_wickets_per_over : ℕ := 3

/-- Represents the number of overs bowled by the bowler in an innings -/
def overs_bowled : ℕ := 6

/-- Represents the number of players in a cricket team -/
def players_per_team : ℕ := 11

/-- Represents the maximum number of wickets that can be taken in an innings -/
def max_wickets_in_innings : ℕ := players_per_team - 1

/-- 
Theorem stating that the maximum number of wickets a bowler can take in an innings
is the minimum of the theoretical maximum (max_wickets_per_over * overs_bowled) 
and the actual maximum (max_wickets_in_innings)
-/
theorem max_wickets_theorem : 
  min (max_wickets_per_over * overs_bowled) max_wickets_in_innings = max_wickets_in_innings := by
  sorry

end NUMINAMATH_CALUDE_stating_max_wickets_theorem_l3725_372548


namespace NUMINAMATH_CALUDE_function_lower_bound_l3725_372522

noncomputable def f (a b x : ℝ) : ℝ := (a * x - 3/4) * Real.exp x - (b * Real.exp x) / (Real.exp x + 1)

theorem function_lower_bound (a : ℝ) :
  (∀ x ∈ Set.Ici (-2 : ℝ), f a 1 x ≥ -5/4) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_lower_bound_l3725_372522


namespace NUMINAMATH_CALUDE_function_property_l3725_372518

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def periodic_neg (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f x

def half_x_on_unit (f : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 1/2 * x

theorem function_property (f : ℝ → ℝ) 
  (h1 : is_odd f) 
  (h2 : periodic_neg f) 
  (h3 : half_x_on_unit f) : 
  {x : ℝ | f x = -1/2} = {x : ℝ | ∃ k : ℤ, x = 4 * k - 1} := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3725_372518


namespace NUMINAMATH_CALUDE_compound_interest_problem_l3725_372513

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- Total amount calculation -/
def total_amount (principal : ℝ) (interest : ℝ) : ℝ :=
  principal + interest

theorem compound_interest_problem (P : ℝ) :
  compound_interest P 0.06 2 = 370.80 →
  total_amount P 370.80 = 3370.80 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l3725_372513


namespace NUMINAMATH_CALUDE_brothers_identity_l3725_372556

-- Define the types for brothers and card suits
inductive Brother
| First
| Second

inductive Suit
| Black
| Red

-- Define the statements made by each brother
def firstBrotherStatement (secondBrotherName : String) (secondBrotherSuit : Suit) : Prop :=
  secondBrotherName = "Tweedledee" ∧ secondBrotherSuit = Suit.Black

def secondBrotherStatement (firstBrotherName : String) (firstBrotherSuit : Suit) : Prop :=
  firstBrotherName = "Tweedledum" ∧ firstBrotherSuit = Suit.Red

-- Define the theorem
theorem brothers_identity :
  ∃ (firstBrotherName secondBrotherName : String) 
    (firstBrotherSuit secondBrotherSuit : Suit),
    (firstBrotherName = "Tweedledee" ∧ secondBrotherName = "Tweedledum") ∧
    (firstBrotherSuit = Suit.Black ∧ secondBrotherSuit = Suit.Red) ∧
    (firstBrotherStatement secondBrotherName secondBrotherSuit ≠ 
     secondBrotherStatement firstBrotherName firstBrotherSuit) :=
by
  sorry

end NUMINAMATH_CALUDE_brothers_identity_l3725_372556


namespace NUMINAMATH_CALUDE_lcm_problem_l3725_372567

theorem lcm_problem (n : ℕ+) 
  (h1 : Nat.lcm 40 n = 120) 
  (h2 : Nat.lcm n 45 = 180) : 
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l3725_372567


namespace NUMINAMATH_CALUDE_valid_numbers_l3725_372560

def is_valid (n : ℕ+) : Prop :=
  ∀ a : ℕ+, (a ≤ 1 + Real.sqrt n.val) → (Nat.gcd a.val n.val = 1) →
    ∃ x : ℤ, (a.val : ℤ) ≡ x^2 [ZMOD n.val]

theorem valid_numbers : {n : ℕ+ | is_valid n} = {1, 2, 12} := by sorry

end NUMINAMATH_CALUDE_valid_numbers_l3725_372560


namespace NUMINAMATH_CALUDE_trout_division_l3725_372516

theorem trout_division (total_trout : ℕ) (num_people : ℕ) (trout_per_person : ℕ) :
  total_trout = 18 →
  num_people = 2 →
  trout_per_person = total_trout / num_people →
  trout_per_person = 9 :=
by sorry

end NUMINAMATH_CALUDE_trout_division_l3725_372516


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3725_372570

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ ∀ m : ℕ, 0 < m → m < n → ¬(537 * m ≡ 1073 * m [ZMOD 30])) → 
  (537 * n ≡ 1073 * n [ZMOD 30]) → 
  n = 15 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3725_372570


namespace NUMINAMATH_CALUDE_triangle_area_fraction_l3725_372547

/-- The area of a triangle given its three vertices -/
def triangleArea (x1 y1 x2 y2 x3 y3 : ℚ) : ℚ :=
  (1/2) * abs (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

/-- The theorem stating that the area of the given triangle divided by the area of the grid equals 13/96 -/
theorem triangle_area_fraction :
  let a_x := 2
  let a_y := 4
  let b_x := 7
  let b_y := 2
  let c_x := 6
  let c_y := 5
  let grid_width := 8
  let grid_height := 6
  (triangleArea a_x a_y b_x b_y c_x c_y) / (grid_width * grid_height) = 13 / 96 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_fraction_l3725_372547


namespace NUMINAMATH_CALUDE_mass_percentage_Cl_is_66_04_l3725_372575

/-- The mass percentage of Cl in a certain compound -/
def mass_percentage_Cl : ℝ := 66.04

/-- Theorem stating that the mass percentage of Cl is 66.04% -/
theorem mass_percentage_Cl_is_66_04 : mass_percentage_Cl = 66.04 := by
  sorry

end NUMINAMATH_CALUDE_mass_percentage_Cl_is_66_04_l3725_372575


namespace NUMINAMATH_CALUDE_exp_two_ln_two_equals_four_l3725_372596

theorem exp_two_ln_two_equals_four : Real.exp (2 * Real.log 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_exp_two_ln_two_equals_four_l3725_372596


namespace NUMINAMATH_CALUDE_perfect_power_l3725_372520

theorem perfect_power (M a b r : ℕ) (f : ℤ → ℤ) 
  (h_a : a ≥ 2) 
  (h_r : r ≥ 2) 
  (h_comp : ∀ n : ℤ, (f^[r]) n = a * n + b) 
  (h_nonneg : ∀ n : ℤ, n ≥ M → f n ≥ 0) 
  (h_div : ∀ n m : ℤ, n > m → m > M → (n - m) ∣ (f n - f m)) :
  ∃ k : ℕ, a = k^r := by
sorry

end NUMINAMATH_CALUDE_perfect_power_l3725_372520


namespace NUMINAMATH_CALUDE_january_salary_l3725_372580

/-- Given the average salaries for two sets of four months and the salary for May,
    prove that the salary for January is 4100. -/
theorem january_salary (jan feb mar apr may : ℕ)
  (h1 : (jan + feb + mar + apr) / 4 = 8000)
  (h2 : (feb + mar + apr + may) / 4 = 8600)
  (h3 : may = 6500) :
  jan = 4100 := by
  sorry

end NUMINAMATH_CALUDE_january_salary_l3725_372580


namespace NUMINAMATH_CALUDE_bart_mixtape_length_l3725_372524

/-- Calculates the total length of a mixtape in minutes -/
def mixtape_length (first_side_songs : ℕ) (second_side_songs : ℕ) (song_length : ℕ) : ℕ :=
  (first_side_songs + second_side_songs) * song_length

/-- Proves that the total length of Bart's mixtape is 40 minutes -/
theorem bart_mixtape_length :
  mixtape_length 6 4 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_bart_mixtape_length_l3725_372524


namespace NUMINAMATH_CALUDE_cone_lateral_area_l3725_372542

/-- The lateral area of a cone with height 3 and slant height 5 is 20π. -/
theorem cone_lateral_area (h : ℝ) (s : ℝ) (r : ℝ) :
  h = 3 →
  s = 5 →
  r^2 + h^2 = s^2 →
  (1/2 : ℝ) * (2 * π * r) * s = 20 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l3725_372542


namespace NUMINAMATH_CALUDE_uber_cost_is_22_l3725_372593

/-- The cost of a taxi ride --/
def taxi_cost : ℝ := 15

/-- The cost of a Lyft ride --/
def lyft_cost : ℝ := taxi_cost + 4

/-- The cost of an Uber ride --/
def uber_cost : ℝ := lyft_cost + 3

/-- The total cost of a taxi ride including a 20% tip --/
def taxi_total_cost : ℝ := taxi_cost * 1.2

theorem uber_cost_is_22 :
  (taxi_total_cost = 18) →
  (uber_cost = 22) :=
by
  sorry

#eval uber_cost

end NUMINAMATH_CALUDE_uber_cost_is_22_l3725_372593


namespace NUMINAMATH_CALUDE_square_field_side_length_l3725_372584

theorem square_field_side_length (area : ℝ) (side_length : ℝ) :
  area = 100 ∧ area = side_length ^ 2 → side_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_field_side_length_l3725_372584


namespace NUMINAMATH_CALUDE_skaters_meeting_distance_l3725_372507

/-- Represents the meeting point of two skaters --/
structure MeetingPoint where
  time : ℝ
  distance_allie : ℝ
  distance_billie : ℝ

/-- Calculates the meeting point of two skaters --/
def calculate_meeting_point (speed_allie speed_billie distance_ab angle : ℝ) : MeetingPoint :=
  sorry

/-- The theorem to be proved --/
theorem skaters_meeting_distance 
  (speed_allie : ℝ)
  (speed_billie : ℝ)
  (distance_ab : ℝ)
  (angle : ℝ)
  (h1 : speed_allie = 8)
  (h2 : speed_billie = 7)
  (h3 : distance_ab = 100)
  (h4 : angle = π / 3) -- 60 degrees in radians
  : 
  let meeting := calculate_meeting_point speed_allie speed_billie distance_ab angle
  meeting.distance_allie = 160 :=
by
  sorry

end NUMINAMATH_CALUDE_skaters_meeting_distance_l3725_372507


namespace NUMINAMATH_CALUDE_completing_square_transform_l3725_372553

theorem completing_square_transform (x : ℝ) :
  (2 * x^2 - 4 * x - 3 = 0) ↔ ((x - 1)^2 - 5/2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transform_l3725_372553


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l3725_372504

theorem wire_cutting_problem (total_length : ℝ) (ratio : ℝ) : 
  total_length = 140 → ratio = 2/5 → 
  ∃ (shorter_piece longer_piece : ℝ), 
    shorter_piece + longer_piece = total_length ∧ 
    shorter_piece = ratio * longer_piece ∧
    shorter_piece = 40 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l3725_372504


namespace NUMINAMATH_CALUDE_correct_committee_count_l3725_372574

/-- Represents a department in the division of mathematical sciences --/
inductive Department
| Mathematics
| Statistics
| ComputerScience

/-- Represents the gender of a professor --/
inductive Gender
| Male
| Female

/-- Represents the composition of professors in a department --/
structure DepartmentComposition where
  department : Department
  maleCount : Nat
  femaleCount : Nat

/-- Represents the requirements for forming a committee --/
structure CommitteeRequirements where
  totalSize : Nat
  femaleCount : Nat
  maleCount : Nat
  mathDepartmentCount : Nat
  minDepartmentsRepresented : Nat

def divisionComposition : List DepartmentComposition := [
  ⟨Department.Mathematics, 3, 3⟩,
  ⟨Department.Statistics, 2, 3⟩,
  ⟨Department.ComputerScience, 2, 3⟩
]

def committeeReqs : CommitteeRequirements := {
  totalSize := 7,
  femaleCount := 4,
  maleCount := 3,
  mathDepartmentCount := 2,
  minDepartmentsRepresented := 3
}

/-- Calculates the number of possible committees given the division composition and requirements --/
def countPossibleCommittees (composition : List DepartmentComposition) (reqs : CommitteeRequirements) : Nat :=
  sorry

theorem correct_committee_count :
  countPossibleCommittees divisionComposition committeeReqs = 1050 :=
sorry

end NUMINAMATH_CALUDE_correct_committee_count_l3725_372574


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l3725_372509

theorem gcd_of_polynomial_and_multiple (x : ℤ) : 
  (∃ k : ℤ, x = 11739 * k) → 
  Nat.gcd ((3*x + 4)*(5*x + 3)*(11*x + 5)*(x + 11)).natAbs x.natAbs = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_l3725_372509


namespace NUMINAMATH_CALUDE_officer_election_proof_l3725_372538

def total_candidates : ℕ := 18
def past_officers : ℕ := 8
def positions_available : ℕ := 6

theorem officer_election_proof :
  (Nat.choose total_candidates positions_available) -
  (Nat.choose (total_candidates - past_officers) positions_available) -
  (Nat.choose past_officers 1 * Nat.choose (total_candidates - past_officers) (positions_available - 1)) = 16338 := by
  sorry

end NUMINAMATH_CALUDE_officer_election_proof_l3725_372538


namespace NUMINAMATH_CALUDE_leah_lost_money_proof_l3725_372578

def leah_lost_money (initial_earnings : ℝ) (milkshake_fraction : ℝ) (comic_book_fraction : ℝ) (savings_fraction : ℝ) (not_shredded_fraction : ℝ) : ℝ :=
  let remaining_after_milkshake := initial_earnings - milkshake_fraction * initial_earnings
  let remaining_after_comic := remaining_after_milkshake - comic_book_fraction * remaining_after_milkshake
  let remaining_after_savings := remaining_after_comic - savings_fraction * remaining_after_comic
  let not_shredded := not_shredded_fraction * remaining_after_savings
  remaining_after_savings - not_shredded

theorem leah_lost_money_proof :
  leah_lost_money 28 (1/7) (1/5) (3/8) 0.1 = 10.80 := by
  sorry

end NUMINAMATH_CALUDE_leah_lost_money_proof_l3725_372578


namespace NUMINAMATH_CALUDE_motorcycle_price_is_correct_l3725_372568

/-- Represents the factory's production and profit information -/
structure FactoryInfo where
  car_material_cost : ℕ
  cars_produced : ℕ
  car_price : ℕ
  motorcycle_material_cost : ℕ
  motorcycles_produced : ℕ
  profit_difference : ℕ

/-- Calculates the price per motorcycle based on the given factory information -/
def calculate_motorcycle_price (info : FactoryInfo) : ℕ :=
  (info.profit_difference + (info.car_material_cost + info.motorcycle_material_cost - info.cars_produced * info.car_price) + info.motorcycle_material_cost) / info.motorcycles_produced

/-- Theorem stating that the calculated motorcycle price is correct -/
theorem motorcycle_price_is_correct (info : FactoryInfo) 
  (h1 : info.car_material_cost = 100)
  (h2 : info.cars_produced = 4)
  (h3 : info.car_price = 50)
  (h4 : info.motorcycle_material_cost = 250)
  (h5 : info.motorcycles_produced = 8)
  (h6 : info.profit_difference = 50) :
  calculate_motorcycle_price info = 50 := by
  sorry

end NUMINAMATH_CALUDE_motorcycle_price_is_correct_l3725_372568


namespace NUMINAMATH_CALUDE_scarves_per_box_l3725_372590

theorem scarves_per_box (num_boxes : ℕ) (mittens_per_box : ℕ) (total_items : ℕ) : 
  num_boxes = 4 → 
  mittens_per_box = 6 → 
  total_items = 32 → 
  (total_items - num_boxes * mittens_per_box) / num_boxes = 2 := by
  sorry

end NUMINAMATH_CALUDE_scarves_per_box_l3725_372590


namespace NUMINAMATH_CALUDE_junior_fraction_l3725_372544

theorem junior_fraction (J S : ℕ) (h1 : J > 0) (h2 : S > 0) (h3 : J * 3 = S * 4) :
  J / (J + S) = 4 / 7 :=
by sorry

end NUMINAMATH_CALUDE_junior_fraction_l3725_372544


namespace NUMINAMATH_CALUDE_no_rational_roots_for_three_digit_prime_quadratic_l3725_372591

def is_three_digit_prime (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ Nat.Prime n

def digits_of_three_digit_number (n : ℕ) : ℕ × ℕ × ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  (a, b, c)

theorem no_rational_roots_for_three_digit_prime_quadratic :
  ∀ A : ℕ, is_three_digit_prime A →
    let (a, b, c) := digits_of_three_digit_number A
    ∀ x : ℚ, a * x^2 + b * x + c ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_no_rational_roots_for_three_digit_prime_quadratic_l3725_372591


namespace NUMINAMATH_CALUDE_linos_shells_l3725_372594

/-- The number of shells Lino picked up -/
def shells_picked_up : ℝ := 324.0

/-- The number of shells Lino put back -/
def shells_put_back : ℝ := 292.00

/-- The number of shells Lino has in all -/
def shells_remaining : ℝ := shells_picked_up - shells_put_back

/-- Theorem stating that the number of shells Lino has in all is 32.0 -/
theorem linos_shells : shells_remaining = 32.0 := by sorry

end NUMINAMATH_CALUDE_linos_shells_l3725_372594


namespace NUMINAMATH_CALUDE_brownie_pieces_count_l3725_372534

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents the tray of brownies -/
def tray : Dimensions := ⟨24, 20⟩

/-- Represents a single piece of brownie -/
def piece : Dimensions := ⟨3, 2⟩

/-- Theorem stating that the tray can be divided into exactly 80 pieces -/
theorem brownie_pieces_count : (area tray) / (area piece) = 80 := by
  sorry

end NUMINAMATH_CALUDE_brownie_pieces_count_l3725_372534


namespace NUMINAMATH_CALUDE_fourth_number_in_list_l3725_372577

theorem fourth_number_in_list (known_numbers : List ℕ) (average : ℕ) : 
  known_numbers = [55, 48, 507, 684, 42] →
  average = 223 →
  ∃ x : ℕ, (List.sum known_numbers + x) / 6 = average ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_fourth_number_in_list_l3725_372577


namespace NUMINAMATH_CALUDE_ladder_problem_l3725_372581

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 15)
  (h2 : height = 9) :
  ∃ (base : ℝ), base^2 + height^2 = ladder_length^2 ∧ base = 12 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l3725_372581


namespace NUMINAMATH_CALUDE_phillip_and_paula_numbers_l3725_372563

theorem phillip_and_paula_numbers (a b : ℚ) 
  (h1 : a = b + 12)
  (h2 : a^2 + b^2 = 169/2)
  (h3 : a^4 - b^4 = 5070) : 
  a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_phillip_and_paula_numbers_l3725_372563


namespace NUMINAMATH_CALUDE_fraction_before_lunch_l3725_372530

/-- Proves that the fraction of distance driven before lunch is 1/4 given the problem conditions --/
theorem fraction_before_lunch (total_distance : ℝ) (total_time : ℝ) (lunch_time : ℝ) 
  (h1 : total_distance = 200)
  (h2 : total_time = 5)
  (h3 : lunch_time = 1)
  (h4 : total_time ≥ lunch_time) :
  let f := (total_time - lunch_time) / 4 / (total_time - lunch_time)
  f = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_before_lunch_l3725_372530


namespace NUMINAMATH_CALUDE_coat_price_reduction_l3725_372506

/-- Given a coat with an original price and a price reduction, 
    calculate the percent reduction. -/
theorem coat_price_reduction 
  (original_price : ℝ) 
  (price_reduction : ℝ) 
  (h1 : original_price = 500) 
  (h2 : price_reduction = 150) : 
  (price_reduction / original_price) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_reduction_l3725_372506


namespace NUMINAMATH_CALUDE_inequality_proof_l3725_372529

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.sqrt (a + b + c) + Real.sqrt a) / (b + c) +
  (Real.sqrt (a + b + c) + Real.sqrt b) / (c + a) +
  (Real.sqrt (a + b + c) + Real.sqrt c) / (a + b) ≥
  (9 + 3 * Real.sqrt 3) / (2 * Real.sqrt (a + b + c)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3725_372529


namespace NUMINAMATH_CALUDE_xy_max_value_l3725_372587

theorem xy_max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2 * y = 8) :
  x * y ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_xy_max_value_l3725_372587


namespace NUMINAMATH_CALUDE_reader_collection_pages_l3725_372533

def book1_chapters : List Nat := [24, 32, 40, 20]
def book2_chapters : List Nat := [48, 52, 36]
def book3_chapters : List Nat := [16, 28, 44, 22, 34]

def total_pages (chapters : List Nat) : Nat :=
  chapters.sum

theorem reader_collection_pages :
  total_pages book1_chapters +
  total_pages book2_chapters +
  total_pages book3_chapters = 396 := by
  sorry

end NUMINAMATH_CALUDE_reader_collection_pages_l3725_372533


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l3725_372557

theorem opposite_of_negative_five :
  ∀ x : ℤ, ((-5 : ℤ) + x = 0) → x = 5 := by
sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l3725_372557


namespace NUMINAMATH_CALUDE_fans_with_all_items_l3725_372523

def arena_capacity : ℕ := 5000
def tshirt_interval : ℕ := 100
def cap_interval : ℕ := 40
def brochure_interval : ℕ := 60

theorem fans_with_all_items : 
  (arena_capacity / (Nat.lcm (Nat.lcm tshirt_interval cap_interval) brochure_interval) : ℕ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_fans_with_all_items_l3725_372523


namespace NUMINAMATH_CALUDE_least_number_remainder_l3725_372572

theorem least_number_remainder : ∃ (r : ℕ), r > 0 ∧ r < 3 ∧ r < 38 ∧ 115 % 38 = r ∧ 115 % 3 = r := by
  sorry

end NUMINAMATH_CALUDE_least_number_remainder_l3725_372572


namespace NUMINAMATH_CALUDE_flag_designs_count_l3725_372552

/-- The number of colors available for the flag design. -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag. -/
def num_stripes : ℕ := 3

/-- Calculate the number of possible flag designs. -/
def num_flag_designs : ℕ := num_colors ^ num_stripes

/-- Theorem stating that the number of possible flag designs is 27. -/
theorem flag_designs_count : num_flag_designs = 27 := by
  sorry

end NUMINAMATH_CALUDE_flag_designs_count_l3725_372552


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l3725_372595

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ s : ℝ, s > 0 ∧ s^3 = 8*x ∧ 6*s^2 = x) → x = 13824 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l3725_372595


namespace NUMINAMATH_CALUDE_quadratic_equation_problem_l3725_372588

theorem quadratic_equation_problem (m : ℤ) (a : ℝ) 
  (h1 : ∃ x y : ℝ, x ≠ y ∧ (m^2 - m) * x^2 - 2*m*x + 1 = 0 ∧ (m^2 - m) * y^2 - 2*m*y + 1 = 0)
  (h2 : m < 3)
  (h3 : (m^2 - m) * a^2 - 2*m*a + 1 = 0) :
  m = 2 ∧ (2*a^2 - 3*a - 3 = (-6 + Real.sqrt 2) / 2 ∨ 2*a^2 - 3*a - 3 = (-6 - Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_problem_l3725_372588


namespace NUMINAMATH_CALUDE_tv_production_average_l3725_372539

theorem tv_production_average (total_days : ℕ) (first_period : ℕ) (first_avg : ℝ) (total_avg : ℝ) :
  total_days = 30 →
  first_period = 25 →
  first_avg = 50 →
  total_avg = 45 →
  (total_days * total_avg - first_period * first_avg) / (total_days - first_period) = 20 := by
sorry

end NUMINAMATH_CALUDE_tv_production_average_l3725_372539


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_molecular_weight_proof_l3725_372583

/-- Given the molecular weight of 10 moles of a substance, 
    calculate the molecular weight of x moles of the same substance. -/
theorem molecular_weight_calculation 
  (mw_10_moles : ℝ)  -- molecular weight of 10 moles
  (x : ℝ)            -- number of moles we want to calculate
  (h : mw_10_moles > 0) -- assumption that molecular weight is positive
  : ℝ :=
  (mw_10_moles / 10) * x

/-- Prove that the molecular weight calculation is correct -/
theorem molecular_weight_proof 
  (mw_10_moles : ℝ)  -- molecular weight of 10 moles
  (x : ℝ)            -- number of moles we want to calculate
  (h : mw_10_moles > 0) -- assumption that molecular weight is positive
  : molecular_weight_calculation mw_10_moles x h = (mw_10_moles / 10) * x :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_molecular_weight_proof_l3725_372583


namespace NUMINAMATH_CALUDE_percentage_increase_problem_l3725_372564

theorem percentage_increase_problem (a b x m : ℝ) (k : ℝ) (h1 : a > 0) (h2 : b > 0) :
  a = 4 * k ∧ b = 5 * k ∧ k > 0 →
  (∃ p : ℝ, x = a * (1 + p / 100)) →
  m = b * 0.4 →
  m / x = 0.4 →
  ∃ p : ℝ, x = a * (1 + p / 100) ∧ p = 25 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_problem_l3725_372564


namespace NUMINAMATH_CALUDE_lilliputian_matchboxes_in_gulliverian_l3725_372582

/-- Represents the scale factor between Lilliput and Gulliver's homeland -/
def scale_factor : ℝ := 12

/-- Calculates the volume of a matchbox given its dimensions -/
def matchbox_volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The number of Lilliputian matchboxes that fit into one Gulliverian matchbox is 1728 -/
theorem lilliputian_matchboxes_in_gulliverian (l w h : ℝ) (hl : l > 0) (hw : w > 0) (hh : h > 0) :
  (matchbox_volume l w h) / (matchbox_volume (l / scale_factor) (w / scale_factor) (h / scale_factor)) = 1728 := by
  sorry

#check lilliputian_matchboxes_in_gulliverian

end NUMINAMATH_CALUDE_lilliputian_matchboxes_in_gulliverian_l3725_372582


namespace NUMINAMATH_CALUDE_trigonometric_problem_l3725_372551

theorem trigonometric_problem (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : -Real.pi / 2 < β ∧ β < 0)
  (h3 : Real.cos (Real.pi / 4 + α) = 1 / 3)
  (h4 : Real.cos (Real.pi / 4 - β / 2) = Real.sqrt 3 / 3) :
  (Real.cos α = (Real.sqrt 2 + 4) / 6) ∧
  (Real.cos (α + β / 2) = 5 * Real.sqrt 3 / 9) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l3725_372551


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_range_l3725_372579

theorem quadratic_inequality_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → x^2 + 2*a*x + 1 ≥ 0) → a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_range_l3725_372579


namespace NUMINAMATH_CALUDE_log_equation_solution_l3725_372559

theorem log_equation_solution (x : ℝ) :
  x > 0 → (4 * Real.log x / Real.log 3 = Real.log (6 * x) / Real.log 3) ↔ x = (6 : ℝ) ^ (1/3) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3725_372559


namespace NUMINAMATH_CALUDE_price_per_book_is_two_l3725_372525

/-- Represents the sale of books with given conditions -/
def BookSale (total_books : ℕ) (price_per_book : ℚ) : Prop :=
  (2 : ℚ) / 3 * total_books + 36 = total_books ∧
  (2 : ℚ) / 3 * total_books * price_per_book = 144

/-- Theorem stating that the price per book is $2 given the conditions -/
theorem price_per_book_is_two :
  ∃ (total_books : ℕ), BookSale total_books 2 := by
  sorry

end NUMINAMATH_CALUDE_price_per_book_is_two_l3725_372525


namespace NUMINAMATH_CALUDE_smallest_S_value_l3725_372566

def is_valid_arrangement (a b c d : Fin 4 → ℕ) : Prop :=
  ∀ i : Fin 16, ∃! j : Fin 4, ∃! k : Fin 4,
    i.val + 1 = a j ∨ i.val + 1 = b j ∨ i.val + 1 = c j ∨ i.val + 1 = d j

def S (a b c d : Fin 4 → ℕ) : ℕ :=
  (a 0) * (a 1) * (a 2) * (a 3) +
  (b 0) * (b 1) * (b 2) * (b 3) +
  (c 0) * (c 1) * (c 2) * (c 3) +
  (d 0) * (d 1) * (d 2) * (d 3)

theorem smallest_S_value :
  ∀ a b c d : Fin 4 → ℕ, is_valid_arrangement a b c d → S a b c d ≥ 2074 :=
by sorry

end NUMINAMATH_CALUDE_smallest_S_value_l3725_372566


namespace NUMINAMATH_CALUDE_cyrus_remaining_pages_l3725_372515

/-- Calculates the remaining pages Cyrus needs to write -/
def remaining_pages (total_pages first_day second_day third_day fourth_day : ℕ) : ℕ :=
  total_pages - (first_day + second_day + third_day + fourth_day)

/-- Theorem stating that Cyrus needs to write 315 more pages -/
theorem cyrus_remaining_pages :
  remaining_pages 500 25 (2 * 25) (2 * (2 * 25)) 10 = 315 := by
  sorry

end NUMINAMATH_CALUDE_cyrus_remaining_pages_l3725_372515


namespace NUMINAMATH_CALUDE_female_democrats_count_l3725_372510

def meeting_participants (total_participants : ℕ) 
  (female_participants : ℕ) (male_participants : ℕ) : Prop :=
  female_participants + male_participants = total_participants

def democrat_ratio (female_democrats : ℕ) (male_democrats : ℕ) 
  (female_participants : ℕ) (male_participants : ℕ) : Prop :=
  female_democrats = female_participants / 2 ∧ 
  male_democrats = male_participants / 4

def total_democrats (female_democrats : ℕ) (male_democrats : ℕ) 
  (total_participants : ℕ) : Prop :=
  female_democrats + male_democrats = total_participants / 3

theorem female_democrats_count : 
  ∀ (total_participants female_participants male_participants 
     female_democrats male_democrats : ℕ),
  total_participants = 990 →
  meeting_participants total_participants female_participants male_participants →
  democrat_ratio female_democrats male_democrats female_participants male_participants →
  total_democrats female_democrats male_democrats total_participants →
  female_democrats = 165 := by
  sorry

end NUMINAMATH_CALUDE_female_democrats_count_l3725_372510


namespace NUMINAMATH_CALUDE_right_to_left_eval_equals_56_over_9_l3725_372597

def right_to_left_eval : ℚ := by
  -- Define the operations
  let square (x : ℚ) := x * x
  let divide (x y : ℚ) := x / y
  let add (x y : ℚ) := x + y
  let multiply (x y : ℚ) := x * y

  -- Evaluate from right to left
  let step1 := square 6
  let step2 := divide 4 step1
  let step3 := add 3 step2
  let step4 := multiply 2 step3

  exact step4

-- Theorem statement
theorem right_to_left_eval_equals_56_over_9 : 
  right_to_left_eval = 56 / 9 := by
  sorry

end NUMINAMATH_CALUDE_right_to_left_eval_equals_56_over_9_l3725_372597


namespace NUMINAMATH_CALUDE_time_ratio_third_to_first_l3725_372540

-- Define the distances and speed ratios
def distance_first : ℝ := 60
def distance_second : ℝ := 240
def distance_third : ℝ := 180
def speed_ratio_second : ℝ := 4
def speed_ratio_third : ℝ := 2

-- Define the theorem
theorem time_ratio_third_to_first :
  let time_first := distance_first / (distance_first / time_first)
  let time_third := distance_third / (speed_ratio_third * (distance_first / time_first))
  time_third / time_first = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_time_ratio_third_to_first_l3725_372540


namespace NUMINAMATH_CALUDE_two_ice_cream_cones_cost_l3725_372558

/-- The cost of an ice cream cone in cents -/
def ice_cream_cost : ℕ := 99

/-- The number of ice cream cones -/
def num_cones : ℕ := 2

/-- Theorem: The cost of 2 ice cream cones is 198 cents -/
theorem two_ice_cream_cones_cost : 
  ice_cream_cost * num_cones = 198 := by
  sorry

end NUMINAMATH_CALUDE_two_ice_cream_cones_cost_l3725_372558


namespace NUMINAMATH_CALUDE_decimal_89_to_base5_l3725_372503

/-- Converts a natural number to its base-5 representation --/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Checks if a list of digits is a valid base-5 representation --/
def isValidBase5 (l : List ℕ) : Prop :=
  l.all (· < 5)

theorem decimal_89_to_base5 :
  let base5_representation := toBase5 89
  isValidBase5 base5_representation ∧ base5_representation = [4, 2, 3] :=
by sorry

end NUMINAMATH_CALUDE_decimal_89_to_base5_l3725_372503


namespace NUMINAMATH_CALUDE_solve_for_k_l3725_372543

theorem solve_for_k : ∀ k : ℝ, (2 * k * 1 - (-7) = -1) → k = -4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l3725_372543


namespace NUMINAMATH_CALUDE_max_consecutive_sum_30_l3725_372545

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of n consecutive integers starting from 2 -/
def sum_from_2 (n : ℕ) : ℕ := sum_first_n (n + 1) - 1

/-- 30 is the maximum number of consecutive positive integers 
    starting from 2 that can be added together without exceeding 500 -/
theorem max_consecutive_sum_30 :
  (∀ k : ℕ, k ≤ 30 → sum_from_2 k ≤ 500) ∧
  (∀ k : ℕ, k > 30 → sum_from_2 k > 500) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_30_l3725_372545


namespace NUMINAMATH_CALUDE_student_calculation_l3725_372537

theorem student_calculation (x : ℕ) (h : x = 125) : 2 * x - 138 = 112 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_l3725_372537


namespace NUMINAMATH_CALUDE_basketball_team_combinations_l3725_372592

theorem basketball_team_combinations :
  let total_players : ℕ := 15
  let team_size : ℕ := 6
  let must_include : ℕ := 2
  let remaining_slots : ℕ := team_size - must_include
  let remaining_players : ℕ := total_players - must_include
  Nat.choose remaining_players remaining_slots = 715 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_combinations_l3725_372592


namespace NUMINAMATH_CALUDE_expression_bounds_l3725_372565

theorem expression_bounds (a b c d e : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) (he : 0 ≤ e ∧ e ≤ 1) : 
  2 * Real.sqrt 2 ≤ Real.sqrt (e^2 + a^2) + Real.sqrt (a^2 + b^2) + 
    Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (d^2 + e^2) ∧
  Real.sqrt (e^2 + a^2) + Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + 
    Real.sqrt (c^2 + d^2) + Real.sqrt (d^2 + e^2) ≤ 5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l3725_372565


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3725_372505

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 1) ↔ (∃ x₀ : ℝ, x₀^2 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3725_372505


namespace NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l3725_372500

/-- Given a quadratic function f(x) = ax^2 + bx + c with vertex (3, 7) and one x-intercept at (-2, 0),
    the x-coordinate of the other x-intercept is 8. -/
theorem other_x_intercept_of_quadratic (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = 7 + a * (x - 3)^2) →  -- Vertex form of quadratic with vertex (3, 7)
  (a * (-2)^2 + b * (-2) + c = 0) →                 -- (-2, 0) is an x-intercept
  ∃ x, x ≠ -2 ∧ a * x^2 + b * x + c = 0 ∧ x = 8 :=  -- Other x-intercept exists and equals 8
by sorry

end NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l3725_372500


namespace NUMINAMATH_CALUDE_bridge_weight_is_88_ounces_l3725_372517

/-- The weight of a toy bridge given the number of full soda cans, 
    the weight of soda in each can, the weight of an empty can, 
    and the number of additional empty cans. -/
def bridge_weight (full_cans : ℕ) (soda_weight : ℕ) (empty_can_weight : ℕ) (additional_empty_cans : ℕ) : ℕ :=
  (full_cans * (soda_weight + empty_can_weight)) + (additional_empty_cans * empty_can_weight)

/-- Theorem stating that the bridge must hold 88 ounces given the specified conditions. -/
theorem bridge_weight_is_88_ounces : 
  bridge_weight 6 12 2 2 = 88 := by
  sorry

end NUMINAMATH_CALUDE_bridge_weight_is_88_ounces_l3725_372517


namespace NUMINAMATH_CALUDE_exchange_rate_problem_l3725_372571

theorem exchange_rate_problem (x : ℕ) : 
  (8 * x / 5 : ℚ) - 80 = x →
  (x / 100 + (x % 100) / 10 + x % 10 : ℕ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_exchange_rate_problem_l3725_372571


namespace NUMINAMATH_CALUDE_infinite_coprime_pairs_l3725_372527

theorem infinite_coprime_pairs (m : ℕ+) :
  ∃ (seq : ℕ → ℕ × ℕ), ∀ n : ℕ,
    let (x, y) := seq n
    Int.gcd x y = 1 ∧
    x > 0 ∧ y > 0 ∧
    (y^2 + m.val) % x = 0 ∧
    (x^2 + m.val) % y = 0 ∧
    (∀ k < n, seq k ≠ seq n) :=
sorry

end NUMINAMATH_CALUDE_infinite_coprime_pairs_l3725_372527


namespace NUMINAMATH_CALUDE_range_of_ratio_l3725_372535

theorem range_of_ratio (x y : ℝ) : 
  x^2 - 8*x + y^2 - 4*y + 16 ≤ 0 → 
  0 ≤ y/x ∧ y/x ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_ratio_l3725_372535


namespace NUMINAMATH_CALUDE_cheerleaders_size2_l3725_372521

/-- The number of cheerleaders needing size 2 uniforms -/
def size2 (total size6 : ℕ) : ℕ :=
  total - (size6 + size6 / 2)

/-- Theorem stating that 4 cheerleaders need size 2 uniforms -/
theorem cheerleaders_size2 :
  size2 19 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_cheerleaders_size2_l3725_372521


namespace NUMINAMATH_CALUDE_equation_solution_l3725_372532

theorem equation_solution :
  ∃ x : ℚ, (0.05 * x + 0.12 * (30 + x) = 18) ∧ (x = 144 / 17) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3725_372532


namespace NUMINAMATH_CALUDE_roots_expression_value_l3725_372598

theorem roots_expression_value (m n : ℝ) : 
  m^2 + 2*m - 2027 = 0 → 
  n^2 + 2*n - 2027 = 0 → 
  2*m - m*n + 2*n = 2023 := by
sorry

end NUMINAMATH_CALUDE_roots_expression_value_l3725_372598


namespace NUMINAMATH_CALUDE_equal_numbers_product_l3725_372512

theorem equal_numbers_product (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 15 →
  a = 10 →
  b = 18 →
  c = d →
  c * d = 256 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l3725_372512


namespace NUMINAMATH_CALUDE_expression_equals_500_l3725_372555

theorem expression_equals_500 : 88 * 4 + 37 * 4 = 500 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_500_l3725_372555


namespace NUMINAMATH_CALUDE_nonnegative_solutions_count_l3725_372526

theorem nonnegative_solutions_count : 
  ∃! (n : ℕ), ∃ (x : ℝ), x ≥ 0 ∧ x^2 = -6*x ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_nonnegative_solutions_count_l3725_372526


namespace NUMINAMATH_CALUDE_cards_per_layer_in_house_of_cards_l3725_372576

/-- Proves that given 16 decks of 52 cards each, and a house of cards with 32 layers
    where each layer has the same number of cards, the number of cards per layer is 26. -/
theorem cards_per_layer_in_house_of_cards 
  (num_decks : ℕ) 
  (cards_per_deck : ℕ) 
  (num_layers : ℕ) 
  (h1 : num_decks = 16) 
  (h2 : cards_per_deck = 52) 
  (h3 : num_layers = 32) : 
  (num_decks * cards_per_deck) / num_layers = 26 := by
  sorry

#eval (16 * 52) / 32  -- Expected output: 26

end NUMINAMATH_CALUDE_cards_per_layer_in_house_of_cards_l3725_372576


namespace NUMINAMATH_CALUDE_three_color_theorem_l3725_372508

/-- Represents a country on the island -/
structure Country where
  vertices : Fin 3 → ℝ × ℝ

/-- Represents the entire island -/
structure Island where
  countries : List Country
  adjacent : Country → Country → Bool

/-- A coloring of the island -/
def Coloring := Country → Fin 3

theorem three_color_theorem (I : Island) :
  ∃ (c : Coloring), ∀ (x y : Country),
    I.adjacent x y → c x ≠ c y :=
sorry

end NUMINAMATH_CALUDE_three_color_theorem_l3725_372508


namespace NUMINAMATH_CALUDE_inequality_proof_l3725_372502

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (a^2 + b^2 + c^2) * (a / (b + c) + b / (c + a) + c / (a + b)) ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3725_372502


namespace NUMINAMATH_CALUDE_inequality_proof_l3725_372514

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3725_372514


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3725_372536

-- Define the inverse relationship between two quantities
def inverse_relation (x y : ℝ) (k : ℝ) : Prop := x * y = k

-- Define the given conditions
def conditions : Prop :=
  ∃ (k m : ℝ),
    inverse_relation 1500 0.4 k ∧
    inverse_relation 1500 2.5 m ∧
    inverse_relation 3000 0.2 k ∧
    inverse_relation 3000 1.25 m

-- State the theorem
theorem inverse_variation_problem :
  conditions → (∃ (s t : ℝ), s = 0.2 ∧ t = 1.25) :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3725_372536


namespace NUMINAMATH_CALUDE_polygon_interior_angle_sum_not_polygon_interior_angle_sum_l3725_372561

theorem polygon_interior_angle_sum (n : ℕ) (sum : ℕ) : sum = (n - 2) * 180 → n ≥ 3 :=
by sorry

theorem not_polygon_interior_angle_sum : ¬ ∃ (n : ℕ), 800 = (n - 2) * 180 ∧ n ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_polygon_interior_angle_sum_not_polygon_interior_angle_sum_l3725_372561


namespace NUMINAMATH_CALUDE_invisibility_elixir_combinations_l3725_372550

/-- The number of valid combinations for the invisibility elixir. -/
def valid_combinations (roots : ℕ) (minerals : ℕ) (incompatible : ℕ) : ℕ :=
  roots * minerals - incompatible

/-- Theorem: Given 4 roots, 6 minerals, and 3 incompatible combinations,
    the number of valid combinations for the invisibility elixir is 21. -/
theorem invisibility_elixir_combinations :
  valid_combinations 4 6 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_invisibility_elixir_combinations_l3725_372550


namespace NUMINAMATH_CALUDE_choir_average_age_l3725_372562

theorem choir_average_age (female_count : ℕ) (male_count : ℕ) (children_count : ℕ)
  (female_avg_age : ℝ) (male_avg_age : ℝ) (children_avg_age : ℝ)
  (h_female_count : female_count = 12)
  (h_male_count : male_count = 18)
  (h_children_count : children_count = 10)
  (h_female_avg : female_avg_age = 28)
  (h_male_avg : male_avg_age = 36)
  (h_children_avg : children_avg_age = 10) :
  let total_count := female_count + male_count + children_count
  let total_age := female_count * female_avg_age + male_count * male_avg_age + children_count * children_avg_age
  total_age / total_count = 27.1 := by
  sorry

end NUMINAMATH_CALUDE_choir_average_age_l3725_372562


namespace NUMINAMATH_CALUDE_average_increase_is_five_l3725_372554

/-- Represents a batsman's performance -/
structure BatsmanPerformance where
  innings : ℕ
  totalRuns : ℕ
  lastInningRuns : ℕ

/-- Calculates the average runs per inning -/
def average (bp : BatsmanPerformance) : ℚ :=
  bp.totalRuns / bp.innings

/-- Theorem: The increase in average is 5 runs -/
theorem average_increase_is_five (bp : BatsmanPerformance) 
  (h1 : bp.innings = 11)
  (h2 : bp.lastInningRuns = 85)
  (h3 : average bp = 35) :
  average bp - average { bp with 
    innings := bp.innings - 1,
    totalRuns := bp.totalRuns - bp.lastInningRuns
  } = 5 := by
  sorry

#check average_increase_is_five

end NUMINAMATH_CALUDE_average_increase_is_five_l3725_372554


namespace NUMINAMATH_CALUDE_midpoint_sum_coordinates_l3725_372585

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (2, 3) and (8, 15) is 14. -/
theorem midpoint_sum_coordinates : 
  let x₁ : ℝ := 2
  let y₁ : ℝ := 3
  let x₂ : ℝ := 8
  let y₂ : ℝ := 15
  let midpoint_x : ℝ := (x₁ + x₂) / 2
  let midpoint_y : ℝ := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = 14 := by
  sorry


end NUMINAMATH_CALUDE_midpoint_sum_coordinates_l3725_372585


namespace NUMINAMATH_CALUDE_four_friends_same_group_probability_l3725_372589

/-- The total number of students -/
def total_students : ℕ := 900

/-- The number of groups -/
def num_groups : ℕ := 4

/-- The number of students in each group -/
def students_per_group : ℕ := total_students / num_groups

/-- The probability of a single student being assigned to a specific group -/
def prob_single_student : ℚ := 1 / num_groups

theorem four_friends_same_group_probability :
  (prob_single_student ^ 3 : ℚ) = 1 / 64 :=
sorry

end NUMINAMATH_CALUDE_four_friends_same_group_probability_l3725_372589
