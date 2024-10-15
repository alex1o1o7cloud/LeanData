import Mathlib

namespace NUMINAMATH_CALUDE_square_roots_of_four_l1762_176209

theorem square_roots_of_four :
  {y : ℝ | y ^ 2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_roots_of_four_l1762_176209


namespace NUMINAMATH_CALUDE_desired_average_l1762_176264

theorem desired_average (numbers : List ℕ) (h1 : numbers = [6, 16, 8, 22]) : 
  (numbers.sum / numbers.length : ℚ) = 13 := by
  sorry

end NUMINAMATH_CALUDE_desired_average_l1762_176264


namespace NUMINAMATH_CALUDE_amanda_final_pay_l1762_176265

/-- Calculate Amanda's final pay after deductions and penalties --/
theorem amanda_final_pay 
  (regular_wage : ℝ) 
  (regular_hours : ℝ) 
  (overtime_rate : ℝ) 
  (overtime_hours : ℝ) 
  (commission : ℝ) 
  (tax_rate : ℝ) 
  (insurance_rate : ℝ) 
  (other_expenses : ℝ) 
  (penalty_rate : ℝ) 
  (h1 : regular_wage = 50)
  (h2 : regular_hours = 8)
  (h3 : overtime_rate = 1.5)
  (h4 : overtime_hours = 2)
  (h5 : commission = 150)
  (h6 : tax_rate = 0.15)
  (h7 : insurance_rate = 0.05)
  (h8 : other_expenses = 40)
  (h9 : penalty_rate = 0.2) :
  let total_earnings := regular_wage * regular_hours + 
                        regular_wage * overtime_rate * overtime_hours + 
                        commission
  let deductions := total_earnings * tax_rate + 
                    total_earnings * insurance_rate + 
                    other_expenses
  let earnings_after_deductions := total_earnings - deductions
  let penalty := earnings_after_deductions * penalty_rate
  let final_pay := earnings_after_deductions - penalty
  final_pay = 416 := by sorry

end NUMINAMATH_CALUDE_amanda_final_pay_l1762_176265


namespace NUMINAMATH_CALUDE_largest_gcd_of_four_integers_l1762_176290

theorem largest_gcd_of_four_integers (a b c d : ℕ+) : 
  a + b + c + d = 1105 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∃ k : ℕ, k > 1 ∧ k ∣ a ∧ k ∣ b) ∧
  (∃ k : ℕ, k > 1 ∧ k ∣ a ∧ k ∣ c) ∧
  (∃ k : ℕ, k > 1 ∧ k ∣ a ∧ k ∣ d) ∧
  (∃ k : ℕ, k > 1 ∧ k ∣ b ∧ k ∣ c) ∧
  (∃ k : ℕ, k > 1 ∧ k ∣ b ∧ k ∣ d) ∧
  (∃ k : ℕ, k > 1 ∧ k ∣ c ∧ k ∣ d) →
  (∀ g : ℕ, g ∣ a ∧ g ∣ b ∧ g ∣ c ∧ g ∣ d → g ≤ 221) ∧
  (∃ g : ℕ, g = 221 ∧ g ∣ a ∧ g ∣ b ∧ g ∣ c ∧ g ∣ d) :=
by sorry

end NUMINAMATH_CALUDE_largest_gcd_of_four_integers_l1762_176290


namespace NUMINAMATH_CALUDE_digit_sum_problem_l1762_176240

theorem digit_sum_problem (P Q R S : ℕ) : 
  P < 10 → Q < 10 → R < 10 → S < 10 →
  P * 100 + 45 + Q * 10 + R + S = 654 →
  P + Q + R + S = 15 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l1762_176240


namespace NUMINAMATH_CALUDE_sector_area_special_case_l1762_176219

/-- The area of a sector with central angle 2π/3 and radius √3 is equal to π. -/
theorem sector_area_special_case :
  let central_angle : ℝ := 2 * Real.pi / 3
  let radius : ℝ := Real.sqrt 3
  let sector_area : ℝ := (1 / 2) * radius^2 * central_angle
  sector_area = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_area_special_case_l1762_176219


namespace NUMINAMATH_CALUDE_square_dissection_ratio_l1762_176214

/-- A square dissection problem -/
theorem square_dissection_ratio (A B E F G X Y W Z : ℝ × ℝ) : 
  let square_side : ℝ := 4
  let AE : ℝ := 1
  let BF : ℝ := 4
  let EF : ℝ := 2
  let AG : ℝ := 4
  let BG : ℝ := Real.sqrt 17
  -- AG perpendicular to BF
  (AG * BF = 0) →
  -- Area preservation
  (square_side * square_side = XY * WZ) →
  -- XY equals AG
  (XY = AG) →
  -- Ratio calculation
  (XY / WZ = 1) := by
  sorry

end NUMINAMATH_CALUDE_square_dissection_ratio_l1762_176214


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_range_l1762_176232

theorem quadratic_always_nonnegative_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) → a ∈ Set.Icc (-2) 2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_range_l1762_176232


namespace NUMINAMATH_CALUDE_fraction_of_woodwind_and_brass_players_l1762_176281

theorem fraction_of_woodwind_and_brass_players (total_students : ℝ) : 
  let woodwind_last_year := (1 / 2 : ℝ) * total_students
  let brass_last_year := (2 / 5 : ℝ) * total_students
  let percussion_last_year := (1 / 10 : ℝ) * total_students
  let woodwind_this_year := (1 / 2 : ℝ) * woodwind_last_year
  let brass_this_year := (3 / 4 : ℝ) * brass_last_year
  let percussion_this_year := percussion_last_year
  let total_this_year := woodwind_this_year + brass_this_year + percussion_this_year
  (woodwind_this_year + brass_this_year) / total_this_year = (11 / 20 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_woodwind_and_brass_players_l1762_176281


namespace NUMINAMATH_CALUDE_z_power_2000_eq_one_l1762_176200

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- The complex number (3 + 4i) / (4 - 3i) -/
noncomputable def z : ℂ := (3 + 4 * i) / (4 - 3 * i)

/-- Theorem stating that z^2000 = 1 -/
theorem z_power_2000_eq_one : z ^ 2000 = 1 := by sorry

end NUMINAMATH_CALUDE_z_power_2000_eq_one_l1762_176200


namespace NUMINAMATH_CALUDE_hotel_towels_l1762_176249

theorem hotel_towels (num_rooms : ℕ) (people_per_room : ℕ) (total_towels : ℕ) : 
  num_rooms = 10 →
  people_per_room = 3 →
  total_towels = 60 →
  total_towels / (num_rooms * people_per_room) = 2 := by
sorry

end NUMINAMATH_CALUDE_hotel_towels_l1762_176249


namespace NUMINAMATH_CALUDE_point_on_y_axis_l1762_176277

/-- A point M with coordinates (t-3, 5-t) is on the y-axis if and only if its coordinates are (0, 2) -/
theorem point_on_y_axis (t : ℝ) :
  (t - 3 = 0 ∧ (t - 3, 5 - t) = (0, 2)) ↔ (t - 3, 5 - t).1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l1762_176277


namespace NUMINAMATH_CALUDE_percentage_relation_l1762_176261

theorem percentage_relation (A B C x y : ℝ) : 
  A > C ∧ C > B ∧ B > 0 →
  C = B * (1 + y / 100) →
  A = C * (1 + x / 100) →
  x = 100 * ((100 * (A - B)) / (100 + y)) :=
by sorry

end NUMINAMATH_CALUDE_percentage_relation_l1762_176261


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1762_176229

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ∈ Set.Ici (0 : ℝ) → x^3 + 2*x ≥ 0) ↔
  (∃ x : ℝ, x ∈ Set.Ici (0 : ℝ) ∧ x^3 + 2*x < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1762_176229


namespace NUMINAMATH_CALUDE_ad_broadcast_solution_l1762_176286

/-- Represents the number of ads remaining after the k-th broadcasting -/
def remaining_ads (m : ℕ) (k : ℕ) : ℚ :=
  if k = 0 then m
  else (7/8 : ℚ) * remaining_ads m (k-1) - (7/8 : ℚ) * k

/-- The total number of ads broadcast up to and including the k-th insert -/
def ads_broadcast (m : ℕ) (k : ℕ) : ℚ :=
  m - remaining_ads m k

theorem ad_broadcast_solution (n : ℕ) (m : ℕ) (h1 : n > 1) 
  (h2 : ads_broadcast m n = m) 
  (h3 : ∀ k < n, ads_broadcast m k < m) :
  n = 7 ∧ m = 49 := by
  sorry


end NUMINAMATH_CALUDE_ad_broadcast_solution_l1762_176286


namespace NUMINAMATH_CALUDE_inequality_solution_l1762_176282

theorem inequality_solution (x : ℝ) :
  0 < x ∧ x < Real.pi →
  ((8 / (3 * Real.sin x - Real.sin (3 * x))) + 3 * (Real.sin x)^2 ≤ 5) ↔
  x = Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1762_176282


namespace NUMINAMATH_CALUDE_y_divisibility_l1762_176260

def y : ℕ := 54 + 108 + 162 + 216 + 648 + 810 + 972

theorem y_divisibility :
  (∃ k : ℕ, y = 2 * k) ∧
  (∃ k : ℕ, y = 3 * k) ∧
  (∃ k : ℕ, y = 6 * k) ∧
  (∃ k : ℕ, y = 9 * k) :=
by sorry

end NUMINAMATH_CALUDE_y_divisibility_l1762_176260


namespace NUMINAMATH_CALUDE_function_eventually_constant_l1762_176204

def is_eventually_constant (f : ℕ+ → ℕ+) : Prop :=
  ∃ m : ℕ+, ∀ x ≥ m, f x = f m

theorem function_eventually_constant
  (f : ℕ+ → ℕ+)
  (h1 : ∀ x : ℕ+, f x + f (x + 2) ≤ 2 * f (x + 1))
  (h2 : ∀ x : ℕ+, f x < 2000) :
  is_eventually_constant f :=
sorry

end NUMINAMATH_CALUDE_function_eventually_constant_l1762_176204


namespace NUMINAMATH_CALUDE_pseudo_periodic_minus_one_is_periodic_two_cos_pseudo_periodic_iff_omega_multiple_of_pi_l1762_176254

-- Define a pseudo-periodic function
def IsPseudoPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T ≠ 0 ∧ ∀ x, f (x + T) = T * f x

-- Theorem 1
theorem pseudo_periodic_minus_one_is_periodic_two (f : ℝ → ℝ) 
  (h : IsPseudoPeriodic f (-1)) : 
  ∀ x, f (x + 2) = f x := by sorry

-- Theorem 2
theorem cos_pseudo_periodic_iff_omega_multiple_of_pi (ω : ℝ) :
  IsPseudoPeriodic (λ x => Real.cos (ω * x)) T ↔ ∃ k : ℤ, ω = k * Real.pi := by sorry

end NUMINAMATH_CALUDE_pseudo_periodic_minus_one_is_periodic_two_cos_pseudo_periodic_iff_omega_multiple_of_pi_l1762_176254


namespace NUMINAMATH_CALUDE_parabola_shift_l1762_176263

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := 5 * x^2

-- Define the horizontal shift
def horizontal_shift : ℝ := 2

-- Define the vertical shift
def vertical_shift : ℝ := 3

-- Define the resulting parabola after shifts
def shifted_parabola (x : ℝ) : ℝ := 5 * (x + horizontal_shift)^2 + vertical_shift

-- Theorem statement
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = 5 * (x + 2)^2 + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l1762_176263


namespace NUMINAMATH_CALUDE_fourth_angle_measure_l1762_176292

-- Define a quadrilateral type
structure Quadrilateral :=
  (angle1 : ℝ)
  (angle2 : ℝ)
  (angle3 : ℝ)
  (angle4 : ℝ)

-- Define the property that the sum of angles in a quadrilateral is 360°
def sum_of_angles (q : Quadrilateral) : Prop :=
  q.angle1 + q.angle2 + q.angle3 + q.angle4 = 360

-- Theorem statement
theorem fourth_angle_measure (q : Quadrilateral) 
  (h1 : q.angle1 = 120)
  (h2 : q.angle2 = 85)
  (h3 : q.angle3 = 90)
  (h4 : sum_of_angles q) :
  q.angle4 = 65 := by
  sorry

end NUMINAMATH_CALUDE_fourth_angle_measure_l1762_176292


namespace NUMINAMATH_CALUDE_base9_multiplication_l1762_176231

/-- Converts a base 9 number represented as a list of digits to its decimal equivalent -/
def base9ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 9^i) 0

/-- Converts a decimal number to its base 9 representation as a list of digits -/
def decimalToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 9) ((m % 9) :: acc)
  aux n []

/-- Multiplies two base 9 numbers -/
def multiplyBase9 (a b : List Nat) : List Nat :=
  decimalToBase9 ((base9ToDecimal a) * (base9ToDecimal b))

theorem base9_multiplication (a b : List Nat) :
  multiplyBase9 [3, 5, 4] [1, 2] = [1, 2, 5, 1] := by
  sorry

#eval multiplyBase9 [3, 5, 4] [1, 2]

end NUMINAMATH_CALUDE_base9_multiplication_l1762_176231


namespace NUMINAMATH_CALUDE_cubes_to_add_l1762_176278

theorem cubes_to_add (small_cube_side : ℕ) (large_cube_side : ℕ) (add_cube_side : ℕ) : 
  small_cube_side = 8 →
  large_cube_side = 12 →
  add_cube_side = 2 →
  (large_cube_side^3 - small_cube_side^3) / add_cube_side^3 = 152 := by
  sorry

end NUMINAMATH_CALUDE_cubes_to_add_l1762_176278


namespace NUMINAMATH_CALUDE_problem_solution_l1762_176255

theorem problem_solution (x y : ℝ) 
  (h1 : 1/x + 1/y = 5)
  (h2 : x*y + 2*x + 2*y = 7) :
  x^2*y + x*y^2 = 245/121 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1762_176255


namespace NUMINAMATH_CALUDE_factorial_simplification_l1762_176230

theorem factorial_simplification : (13 * 12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) / 
  ((10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) + 3 * (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) = 1320 := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l1762_176230


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l1762_176226

theorem unique_solution_quadratic_inequality (a : ℝ) :
  (∃! x : ℝ, |x^2 - a*x + 4*a| ≤ 3) ↔ (a = 8 + 2*Real.sqrt 13 ∨ a = 8 - 2*Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l1762_176226


namespace NUMINAMATH_CALUDE_danny_wrappers_l1762_176299

/-- Represents Danny's collection of bottle caps and wrappers -/
structure Collection where
  initial_caps : ℕ
  found_caps : ℕ
  found_wrappers : ℕ
  total_caps : ℕ
  initial_wrappers : ℕ

/-- The theorem states that the number of wrappers Danny has now
    is equal to his initial number of wrappers plus the number of wrappers found -/
theorem danny_wrappers (c : Collection)
  (h1 : c.initial_caps = 6)
  (h2 : c.found_caps = 22)
  (h3 : c.found_wrappers = 8)
  (h4 : c.total_caps = 28)
  (h5 : c.total_caps = c.initial_caps + c.found_caps) :
  c.initial_wrappers + c.found_wrappers = c.initial_wrappers + 8 := by
  sorry


end NUMINAMATH_CALUDE_danny_wrappers_l1762_176299


namespace NUMINAMATH_CALUDE_smallest_n_for_divisible_by_20_l1762_176271

theorem smallest_n_for_divisible_by_20 :
  ∃ (n : ℕ), n = 9 ∧ n ≥ 4 ∧
  (∀ (S : Finset ℤ), S.card = n →
    ∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    20 ∣ (a + b - c - d)) ∧
  (∀ (m : ℕ), m < 9 → m ≥ 4 →
    ∃ (T : Finset ℤ), T.card = m ∧
    ∀ (a b c d : ℤ), a ∈ T → b ∈ T → c ∈ T → d ∈ T →
    a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
    ¬(20 ∣ (a + b - c - d))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_divisible_by_20_l1762_176271


namespace NUMINAMATH_CALUDE_no_solution_steers_cows_l1762_176291

theorem no_solution_steers_cows : ¬∃ (s c : ℕ), 
  30 * s + 32 * c = 1200 ∧ c > s ∧ s > 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_steers_cows_l1762_176291


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l1762_176275

theorem quadratic_root_transformation (k ℓ : ℝ) (r₁ r₂ : ℝ) : 
  (r₁^2 + k*r₁ + ℓ = 0) → 
  (r₂^2 + k*r₂ + ℓ = 0) → 
  ∃ v : ℝ, r₁^2^2 + (-k^2 + 2*ℓ)*r₁^2 + v = 0 ∧ r₂^2^2 + (-k^2 + 2*ℓ)*r₂^2 + v = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_transformation_l1762_176275


namespace NUMINAMATH_CALUDE_tan_sum_pi_24_and_7pi_24_l1762_176203

theorem tan_sum_pi_24_and_7pi_24 : 
  Real.tan (π / 24) + Real.tan (7 * π / 24) = 2 * Real.sqrt 6 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_24_and_7pi_24_l1762_176203


namespace NUMINAMATH_CALUDE_mia_darwin_money_multiple_l1762_176268

theorem mia_darwin_money_multiple (darwin_money mia_money : ℕ) (multiple : ℚ) : 
  darwin_money = 45 →
  mia_money = 110 →
  mia_money = multiple * darwin_money + 20 →
  multiple = 2 := by sorry

end NUMINAMATH_CALUDE_mia_darwin_money_multiple_l1762_176268


namespace NUMINAMATH_CALUDE_only_statement4_correct_l1762_176247

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define symmetry operations
def symmetryXAxis (p : Point3D) : Point3D := ⟨p.x, -p.y, -p.z⟩
def symmetryYOZPlane (p : Point3D) : Point3D := ⟨-p.x, p.y, p.z⟩
def symmetryYAxis (p : Point3D) : Point3D := ⟨-p.x, p.y, -p.z⟩
def symmetryOrigin (p : Point3D) : Point3D := ⟨-p.x, -p.y, -p.z⟩

-- Define the statements
def statement1 (p : Point3D) : Prop := symmetryXAxis p = ⟨p.x, -p.y, p.z⟩
def statement2 (p : Point3D) : Prop := symmetryYOZPlane p = ⟨p.x, -p.y, -p.z⟩
def statement3 (p : Point3D) : Prop := symmetryYAxis p = ⟨-p.x, p.y, p.z⟩
def statement4 (p : Point3D) : Prop := symmetryOrigin p = ⟨-p.x, -p.y, -p.z⟩

-- Theorem to prove
theorem only_statement4_correct (p : Point3D) :
  ¬(statement1 p) ∧ ¬(statement2 p) ∧ ¬(statement3 p) ∧ (statement4 p) :=
sorry

end NUMINAMATH_CALUDE_only_statement4_correct_l1762_176247


namespace NUMINAMATH_CALUDE_unknown_road_length_l1762_176253

/-- Represents a road network with four cities and five roads. -/
structure RoadNetwork where
  /-- The length of the first known road -/
  road1 : ℕ
  /-- The length of the second known road -/
  road2 : ℕ
  /-- The length of the third known road -/
  road3 : ℕ
  /-- The length of the fourth known road -/
  road4 : ℕ
  /-- The length of the unknown road -/
  x : ℕ

/-- The theorem stating that the only possible value for the unknown road length is 17 km. -/
theorem unknown_road_length (network : RoadNetwork) 
  (h1 : network.road1 = 10)
  (h2 : network.road2 = 5)
  (h3 : network.road3 = 8)
  (h4 : network.road4 = 21) :
  network.x = 17 := by
  sorry


end NUMINAMATH_CALUDE_unknown_road_length_l1762_176253


namespace NUMINAMATH_CALUDE_cow_count_is_ten_l1762_176227

/-- Represents the number of animals in a group -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs in the group -/
def totalLegs (group : AnimalGroup) : ℕ :=
  2 * group.ducks + 4 * group.cows

/-- Calculates the total number of heads in the group -/
def totalHeads (group : AnimalGroup) : ℕ :=
  group.ducks + group.cows

/-- Theorem stating that the number of cows is 10 -/
theorem cow_count_is_ten (group : AnimalGroup) 
    (h : totalLegs group = 2 * totalHeads group + 20) : 
    group.cows = 10 := by
  sorry

#check cow_count_is_ten

end NUMINAMATH_CALUDE_cow_count_is_ten_l1762_176227


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l1762_176285

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 6) % 12 = 0 ∧
  (n - 6) % 16 = 0 ∧
  (n - 6) % 18 = 0 ∧
  (n - 6) % 21 = 0 ∧
  (n - 6) % 28 = 0 ∧
  (n - 6) % 35 = 0 ∧
  (n - 6) % 39 = 0

theorem smallest_number_divisible_by_all :
  is_divisible_by_all 65526 ∧
  ∀ m : ℕ, m < 65526 → ¬is_divisible_by_all m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l1762_176285


namespace NUMINAMATH_CALUDE_f_max_min_range_l1762_176212

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a+6)*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a+6)

/-- Condition for f to have both a maximum and a minimum -/
def has_max_and_min (a : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ f' a x = 0 ∧ f' a y = 0

theorem f_max_min_range (a : ℝ) : has_max_and_min a → a < -3 ∨ a > 6 := by sorry

end NUMINAMATH_CALUDE_f_max_min_range_l1762_176212


namespace NUMINAMATH_CALUDE_boat_speed_specific_boat_speed_l1762_176221

/-- The speed of a boat in still water given its travel times with and against a current. -/
theorem boat_speed (distance : ℝ) (time_against : ℝ) (time_with : ℝ) :
  distance > 0 ∧ time_against > 0 ∧ time_with > 0 →
  ∃ (boat_speed current_speed : ℝ),
    (boat_speed - current_speed) * time_against = distance ∧
    (boat_speed + current_speed) * time_with = distance ∧
    boat_speed = 15.6 := by
  sorry

/-- The specific case mentioned in the problem -/
theorem specific_boat_speed :
  ∃ (boat_speed current_speed : ℝ),
    (boat_speed - current_speed) * 8 = 96 ∧
    (boat_speed + current_speed) * 5 = 96 ∧
    boat_speed = 15.6 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_specific_boat_speed_l1762_176221


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l1762_176295

theorem consecutive_integers_average (x : ℤ) : 
  (((x - 9) + (x - 7) + (x - 5) + (x - 3) + (x - 1) + (x + 1) + (x + 3) + (x + 5) + (x + 7) + (x + 9)) / 10 : ℚ) = 31/2 →
  ((x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9)) / 10 : ℚ) = 49/2 :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l1762_176295


namespace NUMINAMATH_CALUDE_sum_of_factors_l1762_176280

theorem sum_of_factors (m n p q : ℤ) : 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q →
  (5 - m) * (5 - n) * (5 - p) * (5 - q) = 9 →
  m + n + p + q = 20 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factors_l1762_176280


namespace NUMINAMATH_CALUDE_not_divisible_by_three_l1762_176222

theorem not_divisible_by_three (n : ℕ+) 
  (h : ∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / n.val = k) :
  ¬(3 ∣ n.val) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_three_l1762_176222


namespace NUMINAMATH_CALUDE_problem_solution_l1762_176207

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem problem_solution (m n : ℝ) 
  (h1 : 0 < m) (h2 : m < n) 
  (h3 : f m = f n) 
  (h4 : ∀ x ∈ Set.Icc (m^2) n, f x ≤ 2) 
  (h5 : ∃ x ∈ Set.Icc (m^2) n, f x = 2) : 
  n / m = Real.exp 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1762_176207


namespace NUMINAMATH_CALUDE_intersection_symmetry_l1762_176205

/-- Given a line y = kx that intersects the circle (x-1)^2 + y^2 = 1 at two points
    symmetric with respect to the line x - y + b = 0, prove that k = -1 and b = -1 -/
theorem intersection_symmetry (k b : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    -- The line intersects the circle at two points
    y₁ = k * x₁ ∧ (x₁ - 1)^2 + y₁^2 = 1 ∧
    y₂ = k * x₂ ∧ (x₂ - 1)^2 + y₂^2 = 1 ∧
    -- The points are distinct
    (x₁, y₁) ≠ (x₂, y₂) ∧
    -- The points are symmetric with respect to x - y + b = 0
    ∃ x₀ y₀ : ℝ, x₀ - y₀ + b = 0 ∧
    x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2) →
  k = -1 ∧ b = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_symmetry_l1762_176205


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1762_176245

universe u

def U : Finset ℕ := {1, 2, 3, 4, 5}
def M : Finset ℕ := {2, 4}
def N : Finset ℕ := {3, 5}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1762_176245


namespace NUMINAMATH_CALUDE_x_neq_zero_necessary_not_sufficient_l1762_176236

theorem x_neq_zero_necessary_not_sufficient :
  (∀ x : ℝ, x > 0 → x ≠ 0) ∧
  ¬(∀ x : ℝ, x ≠ 0 → x > 0) :=
by sorry

end NUMINAMATH_CALUDE_x_neq_zero_necessary_not_sufficient_l1762_176236


namespace NUMINAMATH_CALUDE_shoe_price_calculation_l1762_176262

theorem shoe_price_calculation (initial_money : ℝ) (sweater_price : ℝ) (tshirt_price : ℝ) (refund_percentage : ℝ) (final_money : ℝ) :
  initial_money = 74 →
  sweater_price = 9 →
  tshirt_price = 11 →
  refund_percentage = 0.9 →
  final_money = 51 →
  ∃ (shoe_price : ℝ),
    shoe_price = 30 ∧
    final_money = initial_money - sweater_price - tshirt_price - shoe_price + refund_percentage * shoe_price :=
by
  sorry

end NUMINAMATH_CALUDE_shoe_price_calculation_l1762_176262


namespace NUMINAMATH_CALUDE_window_width_theorem_l1762_176202

/-- Represents the dimensions of a window pane -/
structure Pane where
  width : ℝ
  height : ℝ
  ratio : height = 3 * width

/-- Represents the configuration of a window -/
structure Window where
  pane : Pane
  grid_width : Nat
  grid_height : Nat
  border_width : ℝ

/-- Calculates the total width of a window -/
def total_window_width (w : Window) : ℝ :=
  w.grid_width * w.pane.width + (w.grid_width + 1) * w.border_width

/-- Theorem stating the total width of the window -/
theorem window_width_theorem (w : Window) 
  (h1 : w.grid_width = 3)
  (h2 : w.grid_height = 3)
  (h3 : w.border_width = 3)
  : total_window_width w = 3 * w.pane.width + 12 := by
  sorry

#check window_width_theorem

end NUMINAMATH_CALUDE_window_width_theorem_l1762_176202


namespace NUMINAMATH_CALUDE_division_problem_l1762_176298

theorem division_problem (total : ℚ) (a_amt b_amt c_amt : ℚ) : 
  total = 544 →
  a_amt = (2/3) * b_amt →
  b_amt = (1/4) * c_amt →
  a_amt + b_amt + c_amt = total →
  b_amt = 96 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1762_176298


namespace NUMINAMATH_CALUDE_six_digit_divisibility_by_seven_l1762_176283

theorem six_digit_divisibility_by_seven (a b c d e f : ℕ) :
  (0 < a) →
  (a < 10) →
  (b < 10) →
  (c < 10) →
  (d < 10) →
  (e < 10) →
  (f < 10) →
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) % 7 = 0 →
  (100000 * f + 10000 * a + 1000 * b + 100 * c + 10 * d + e) % 7 = 0 := by
sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_by_seven_l1762_176283


namespace NUMINAMATH_CALUDE_trig_sum_equals_two_l1762_176284

theorem trig_sum_equals_two :
  Real.cos (0 : ℝ) ^ 4 +
  Real.cos (Real.pi / 2) ^ 4 +
  Real.sin (Real.pi / 4) ^ 4 +
  Real.sin (3 * Real.pi / 4) ^ 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_equals_two_l1762_176284


namespace NUMINAMATH_CALUDE_can_collection_increase_l1762_176296

/-- Proves that the daily increase in can collection is 5 cans --/
theorem can_collection_increase (initial_cans : ℕ) (days : ℕ) (total_cans : ℕ) 
  (h1 : initial_cans = 20)
  (h2 : days = 5)
  (h3 : total_cans = 150)
  (h4 : ∃ x : ℕ, total_cans = initial_cans * days + (days * (days - 1) / 2) * x) :
  ∃ x : ℕ, x = 5 ∧ total_cans = initial_cans * days + (days * (days - 1) / 2) * x := by
  sorry

end NUMINAMATH_CALUDE_can_collection_increase_l1762_176296


namespace NUMINAMATH_CALUDE_first_grade_sample_size_l1762_176239

/-- Represents the ratio of students in each grade -/
structure GradeRatio :=
  (first second third fourth : ℕ)

/-- Calculates the number of students to be sampled from the first grade
    given the total sample size and the grade ratio -/
def sampleFirstGrade (totalSample : ℕ) (ratio : GradeRatio) : ℕ :=
  (totalSample * ratio.first) / (ratio.first + ratio.second + ratio.third + ratio.fourth)

/-- Theorem stating that for a sample size of 300 and a grade ratio of 4:5:5:6,
    the number of students to be sampled from the first grade is 60 -/
theorem first_grade_sample_size :
  let totalSample : ℕ := 300
  let ratio : GradeRatio := { first := 4, second := 5, third := 5, fourth := 6 }
  sampleFirstGrade totalSample ratio = 60 := by
  sorry


end NUMINAMATH_CALUDE_first_grade_sample_size_l1762_176239


namespace NUMINAMATH_CALUDE_remainder_problem_l1762_176259

theorem remainder_problem (N : ℤ) (h : N % 1423 = 215) :
  (N - (N / 109)^2) % 109 = 106 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1762_176259


namespace NUMINAMATH_CALUDE_class_size_l1762_176233

/-- The number of people who like both baseball and football -/
def both : ℕ := 5

/-- The number of people who only like baseball -/
def only_baseball : ℕ := 2

/-- The number of people who only like football -/
def only_football : ℕ := 3

/-- The number of people who like neither baseball nor football -/
def neither : ℕ := 6

/-- The total number of people in the class -/
def total : ℕ := both + only_baseball + only_football + neither

theorem class_size : total = 16 := by sorry

end NUMINAMATH_CALUDE_class_size_l1762_176233


namespace NUMINAMATH_CALUDE_volleyball_lineup_count_l1762_176210

/-- The number of players in the volleyball team -/
def total_players : ℕ := 16

/-- The number of quadruplets in the team -/
def num_quadruplets : ℕ := 4

/-- The number of starters to be chosen -/
def num_starters : ℕ := 7

/-- The number of ways to choose starters with the given conditions -/
def valid_lineups : ℕ := Nat.choose total_players num_starters - Nat.choose (total_players - num_quadruplets) (num_starters - num_quadruplets)

theorem volleyball_lineup_count :
  valid_lineups = 11220 :=
sorry

end NUMINAMATH_CALUDE_volleyball_lineup_count_l1762_176210


namespace NUMINAMATH_CALUDE_sportswear_price_reduction_l1762_176237

/-- Given two equal percentage reductions that reduce a price from 560 to 315,
    prove that the equation 560(1-x)^2 = 315 holds true, where x is the decimal
    form of the percentage reduction. -/
theorem sportswear_price_reduction (x : ℝ) : 
  (∃ (x : ℝ), 0 < x ∧ x < 1 ∧ 560 * (1 - x)^2 = 315) :=
by sorry

end NUMINAMATH_CALUDE_sportswear_price_reduction_l1762_176237


namespace NUMINAMATH_CALUDE_quadrilateral_congruence_l1762_176256

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- The median line of a quadrilateral -/
def median_line (q : Quadrilateral) : ℝ × ℝ := sorry

/-- Two quadrilaterals are equal if their corresponding sides and median lines are equal -/
theorem quadrilateral_congruence (q1 q2 : Quadrilateral) :
  (q1.A = q2.A ∧ q1.B = q2.B ∧ q1.C = q2.C ∧ q1.D = q2.D) →
  median_line q1 = median_line q2 →
  q1 = q2 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_congruence_l1762_176256


namespace NUMINAMATH_CALUDE_dwarf_ice_cream_problem_l1762_176244

theorem dwarf_ice_cream_problem :
  ∀ (n : ℕ) (vanilla chocolate fruit : ℕ),
    n = 10 →
    vanilla = n →
    chocolate = n / 2 →
    fruit = 1 →
    ∃ (truthful : ℕ),
      truthful = 4 ∧
      truthful + (n - truthful) = n ∧
      truthful + 2 * (n - truthful) = vanilla + chocolate + fruit :=
by sorry

end NUMINAMATH_CALUDE_dwarf_ice_cream_problem_l1762_176244


namespace NUMINAMATH_CALUDE_decimal_multiplication_and_composition_l1762_176279

theorem decimal_multiplication_and_composition : 
  (35 * 0.01 = 0.35) ∧ (0.875 = 875 * 0.001) := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_and_composition_l1762_176279


namespace NUMINAMATH_CALUDE_fraction_equality_l1762_176218

theorem fraction_equality (a b : ℝ) (h1 : a * b ≠ 0) (h2 : 3 * b = 2 * a) :
  (2 * a + b) / b = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1762_176218


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1762_176269

/-- The parabola function y = x^2 - 6x + 8 -/
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- A square inscribed in the region bounded by the parabola and x-axis -/
structure InscribedSquare where
  center : ℝ
  sideLength : ℝ
  top_touches_parabola : parabola (center + sideLength/2) = sideLength
  bottom_on_x_axis : center - sideLength/2 ≥ 0

/-- The theorem stating the area of the inscribed square -/
theorem inscribed_square_area :
  ∀ (s : InscribedSquare), s.sideLength^2 = 12 - 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1762_176269


namespace NUMINAMATH_CALUDE_tennis_balls_order_l1762_176235

theorem tennis_balls_order (white yellow : ℕ) : 
  white = yellow →
  white / (yellow + 90) = 8 / 13 →
  white + yellow = 288 :=
by sorry

end NUMINAMATH_CALUDE_tennis_balls_order_l1762_176235


namespace NUMINAMATH_CALUDE_arithmetic_number_difference_l1762_176223

/-- A 4-digit number is arithmetic if its digits are distinct and form an arithmetic sequence. -/
def is_arithmetic (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  ∃ (a d : ℤ), 
    let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
    digits.map Int.ofNat = [a, a + d, a + 2*d, a + 3*d] ∧
    digits.toFinset.card = 4

/-- The largest arithmetic 4-digit number -/
def largest_arithmetic : ℕ := 9876

/-- The smallest arithmetic 4-digit number -/
def smallest_arithmetic : ℕ := 1234

theorem arithmetic_number_difference :
  is_arithmetic largest_arithmetic ∧
  is_arithmetic smallest_arithmetic ∧
  largest_arithmetic - smallest_arithmetic = 8642 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_number_difference_l1762_176223


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1762_176273

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 8 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_foci_distance (P : ℝ × ℝ) (h1 : is_on_ellipse P.1 P.2) 
  (h2 : distance P F1 = 2) : distance P F2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1762_176273


namespace NUMINAMATH_CALUDE_intersection_empty_implies_a_range_complement_intersection_when_a_minimum_l1762_176252

-- Define set A
def A (a : ℝ) : Set ℝ :=
  {y | y^2 - (a^2 + a + 1)*y + a*(a^2 + 1) > 0}

-- Define set B
def B : Set ℝ :=
  {y | ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ y = x^2 - x + 1}

-- Theorem 1
theorem intersection_empty_implies_a_range (a : ℝ) :
  A a ∩ B = ∅ → 1 ≤ a ∧ a ≤ 2 :=
sorry

-- Theorem 2
theorem complement_intersection_when_a_minimum :
  let a : ℝ := -2
  (∀ x : ℝ, x^2 + 1 ≥ a*x) →
  (Set.compl (A a) ∩ B) = {y : ℝ | 2 ≤ y ∧ y ≤ 4} :=
sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_a_range_complement_intersection_when_a_minimum_l1762_176252


namespace NUMINAMATH_CALUDE_bacteria_after_three_hours_l1762_176201

/-- Represents the number of bacteria after a given number of half-hour periods. -/
def bacteria_population (half_hours : ℕ) : ℕ := 2^half_hours

/-- Theorem stating that after 3 hours (6 half-hour periods), the bacteria population will be 64. -/
theorem bacteria_after_three_hours : bacteria_population 6 = 64 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_after_three_hours_l1762_176201


namespace NUMINAMATH_CALUDE_third_row_chairs_l1762_176274

def chair_sequence (n : ℕ) : ℕ → ℕ
  | 1 => 14
  | 2 => 23
  | 3 => n
  | 4 => 41
  | 5 => 50
  | 6 => 59
  | _ => 0

theorem third_row_chairs :
  ∃ n : ℕ, 
    chair_sequence n 2 - chair_sequence n 1 = 9 ∧
    chair_sequence n 4 - chair_sequence n 2 = 18 ∧
    chair_sequence n 5 - chair_sequence n 4 = 9 ∧
    chair_sequence n 6 - chair_sequence n 5 = 9 ∧
    n = 32 := by
  sorry

end NUMINAMATH_CALUDE_third_row_chairs_l1762_176274


namespace NUMINAMATH_CALUDE_only_one_divides_l1762_176211

theorem only_one_divides (n : ℕ+) : (n^2 + 1) ∣ (n + 1) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_only_one_divides_l1762_176211


namespace NUMINAMATH_CALUDE_cousins_distribution_l1762_176267

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers --/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- There are 4 rooms --/
def num_rooms : ℕ := 4

/-- There are 5 cousins --/
def num_cousins : ℕ := 5

/-- The theorem stating that there are 76 ways to distribute 5 cousins into 4 rooms --/
theorem cousins_distribution : distribute num_cousins num_rooms = 76 := by sorry

end NUMINAMATH_CALUDE_cousins_distribution_l1762_176267


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1762_176224

theorem decimal_to_fraction (d : ℚ) (h : d = 0.34) : 
  ∃ (n : ℕ) (d : ℕ), d ≠ 0 ∧ d.gcd n = 1 ∧ (n : ℚ) / d = 0.34 ∧ n = 17 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1762_176224


namespace NUMINAMATH_CALUDE_max_rooks_is_400_l1762_176225

/-- Represents a rectangular hole on a chessboard -/
structure Hole :=
  (x : Nat) (y : Nat) (width : Nat) (height : Nat)

/-- Represents a 300x300 chessboard with a hole -/
structure Board :=
  (hole : Hole)
  (is_valid : hole.x + hole.width < 300 ∧ hole.y + hole.height < 300)

/-- The maximum number of non-attacking rooks on a 300x300 board with a hole -/
def max_rooks (b : Board) : Nat :=
  sorry

/-- Theorem: The maximum number of non-attacking rooks is 400 for any valid hole -/
theorem max_rooks_is_400 (b : Board) : max_rooks b = 400 :=
  sorry

end NUMINAMATH_CALUDE_max_rooks_is_400_l1762_176225


namespace NUMINAMATH_CALUDE_remainder_of_power_700_l1762_176288

theorem remainder_of_power_700 (n : ℕ) (h : n^700 % 100 = 1) : n^700 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_power_700_l1762_176288


namespace NUMINAMATH_CALUDE_marshas_delivery_problem_l1762_176217

/-- Marsha's delivery problem -/
theorem marshas_delivery_problem (x : ℝ) : 
  (x + 28 + 14) * 2 = 104 → x = 10 := by sorry

end NUMINAMATH_CALUDE_marshas_delivery_problem_l1762_176217


namespace NUMINAMATH_CALUDE_zoo_tickets_problem_l1762_176246

/-- Proves that for a family of seven people buying zoo tickets, where adult tickets 
    cost $21 and children's tickets cost $14, if the total cost is $119, 
    then the number of adult tickets purchased is 3. -/
theorem zoo_tickets_problem (adult_cost children_cost total_cost : ℕ) 
  (family_size : ℕ) (num_adults : ℕ) :
  adult_cost = 21 →
  children_cost = 14 →
  total_cost = 119 →
  family_size = 7 →
  num_adults + (family_size - num_adults) = family_size →
  num_adults * adult_cost + (family_size - num_adults) * children_cost = total_cost →
  num_adults = 3 := by
  sorry

end NUMINAMATH_CALUDE_zoo_tickets_problem_l1762_176246


namespace NUMINAMATH_CALUDE_ratio_A_B_between_zero_and_one_l1762_176289

def A : ℕ := 1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28
def B : ℕ := 1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20

theorem ratio_A_B_between_zero_and_one : 0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_A_B_between_zero_and_one_l1762_176289


namespace NUMINAMATH_CALUDE_water_pollution_scientific_notation_l1762_176216

/-- The amount of water polluted by a button-sized waste battery in liters -/
def water_pollution : ℕ := 600000

/-- Scientific notation representation of water_pollution -/
def scientific_notation : ℝ := 6 * (10 ^ 5)

theorem water_pollution_scientific_notation :
  (water_pollution : ℝ) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_water_pollution_scientific_notation_l1762_176216


namespace NUMINAMATH_CALUDE_divisibility_problem_l1762_176213

theorem divisibility_problem (x y z : ℕ) (h1 : x = 987654) (h2 : y = 456) (h3 : z = 222) :
  (x + z) % y = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1762_176213


namespace NUMINAMATH_CALUDE_range_of_a_l1762_176297

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + (a-1)*x + 1 < 0

-- Define the theorem
theorem range_of_a (a : ℝ) 
  (h1 : p a ∨ q a) 
  (h2 : ¬(p a ∧ q a)) : 
  (a ∈ Set.Icc (-1) 1) ∨ (a > 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1762_176297


namespace NUMINAMATH_CALUDE_complex_arithmetic_result_l1762_176266

def B : ℂ := 5 - 2 * Complex.I
def N : ℂ := -5 + 2 * Complex.I
def T : ℂ := 2 * Complex.I
def Q : ℂ := 3

theorem complex_arithmetic_result : B - N + T - Q = 7 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_result_l1762_176266


namespace NUMINAMATH_CALUDE_range_of_a_l1762_176215

-- Define the conditions
def p (x : ℝ) : Prop := x^2 - 8*x - 20 < 0
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 ≤ 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x, p x → q x a) →
  (∃ x, ¬p x ∧ q x a) →
  a ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1762_176215


namespace NUMINAMATH_CALUDE_number_divided_by_3000_l1762_176241

theorem number_divided_by_3000 : 
  ∃ x : ℝ, x / 3000 = 0.008416666666666666 ∧ x = 25.25 :=
by sorry

end NUMINAMATH_CALUDE_number_divided_by_3000_l1762_176241


namespace NUMINAMATH_CALUDE_correct_probability_l1762_176294

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (card_count : cards.card = 52)

/-- Represents the suit of a card -/
inductive Suit
  | Clubs
  | Diamonds
  | Hearts
  | Spades

/-- Represents the rank of a card -/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten
  | Jack | Queen | King

/-- A card is a face card if it's a Jack, Queen, or King -/
def isFaceCard (r : Rank) : Bool :=
  match r with
  | Rank.Jack | Rank.Queen | Rank.King => true
  | _ => false

/-- The probability of drawing a club as the first card and a face card diamond as the second card -/
def consecutiveDrawProbability (d : Deck) : Rat :=
  (13 : Rat) / 884

theorem correct_probability (d : Deck) :
  consecutiveDrawProbability d = 13 / 884 := by
  sorry

end NUMINAMATH_CALUDE_correct_probability_l1762_176294


namespace NUMINAMATH_CALUDE_inequality_proof_l1762_176206

theorem inequality_proof (a b : ℝ) (h : a ≠ b) :
  a^4 + 6*a^2*b^2 + b^4 > 4*a*b*(a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1762_176206


namespace NUMINAMATH_CALUDE_test_score_calculation_l1762_176257

/-- The average test score for a portion of the class -/
def average_score (portion : ℝ) (score : ℝ) : ℝ := portion * score

/-- The overall class average -/
def class_average (score1 : ℝ) (score2 : ℝ) (score3 : ℝ) : ℝ :=
  average_score 0.45 0.95 + average_score 0.50 score2 + average_score 0.05 0.60

theorem test_score_calculation (score2 : ℝ) :
  class_average 0.95 score2 0.60 = 0.8475 → score2 = 0.78 := by
  sorry

end NUMINAMATH_CALUDE_test_score_calculation_l1762_176257


namespace NUMINAMATH_CALUDE_sum_of_disk_areas_l1762_176228

/-- The number of disks placed on the circle -/
def n : ℕ := 15

/-- The radius of the large circle -/
def R : ℝ := 1

/-- Represents the arrangement of disks on the circle -/
structure DiskArrangement where
  /-- The radius of each small disk -/
  r : ℝ
  /-- The disks cover the entire circle -/
  covers_circle : r > 0
  /-- The disks do not overlap -/
  no_overlap : 2 * n * r ≤ 2 * π * R
  /-- Each disk is tangent to its neighbors -/
  tangent_neighbors : 2 * n * r = 2 * π * R

/-- The theorem stating the sum of areas of the disks -/
theorem sum_of_disk_areas (arrangement : DiskArrangement) :
  n * π * arrangement.r^2 = 105 * π - 60 * π * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_disk_areas_l1762_176228


namespace NUMINAMATH_CALUDE_base_number_proof_l1762_176208

theorem base_number_proof (x n b : ℝ) 
  (h1 : n = x^(1/4))
  (h2 : n^b = 16)
  (h3 : b = 16.000000000000004) :
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_base_number_proof_l1762_176208


namespace NUMINAMATH_CALUDE_bob_bought_four_candies_l1762_176234

/-- The number of candies bought by each person -/
structure CandyPurchase where
  emily : ℕ
  jennifer : ℕ
  bob : ℕ

/-- The conditions of the candy purchase scenario -/
def candy_scenario (p : CandyPurchase) : Prop :=
  p.emily = 6 ∧
  p.jennifer = 2 * p.emily ∧
  p.jennifer = 3 * p.bob

/-- Theorem stating that Bob bought 4 candies -/
theorem bob_bought_four_candies :
  ∀ p : CandyPurchase, candy_scenario p → p.bob = 4 := by
  sorry

end NUMINAMATH_CALUDE_bob_bought_four_candies_l1762_176234


namespace NUMINAMATH_CALUDE_complex_square_simplify_l1762_176287

theorem complex_square_simplify :
  (4 - 3 * Complex.I) ^ 2 = 7 - 24 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplify_l1762_176287


namespace NUMINAMATH_CALUDE_qr_length_l1762_176272

/-- Given points P, Q, R on a line segment where Q is between P and R -/
structure LineSegment where
  P : ℝ
  Q : ℝ
  R : ℝ
  Q_between : P ≤ Q ∧ Q ≤ R

/-- The length of a line segment -/
def length (a b : ℝ) : ℝ := |b - a|

theorem qr_length (seg : LineSegment) 
  (h1 : length seg.P seg.R = 12)
  (h2 : length seg.P seg.Q = 3) : 
  length seg.Q seg.R = 9 := by
sorry

end NUMINAMATH_CALUDE_qr_length_l1762_176272


namespace NUMINAMATH_CALUDE_last_digit_power_difference_l1762_176258

def last_digit (n : ℤ) : ℕ := (n % 10).toNat

theorem last_digit_power_difference (x : ℤ) :
  last_digit (x^95 - 3^58) = 4 → last_digit (x^95) = 3 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_power_difference_l1762_176258


namespace NUMINAMATH_CALUDE_g36_values_product_l1762_176220

def is_valid_g (g : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, 3 * g (a^2 + b^2) = (g a)^2 + (g b)^2 + g a * g b

def possible_g36_values (g : ℕ → ℕ) : Set ℕ :=
  {x : ℕ | ∃ h : is_valid_g g, g 36 = x}

theorem g36_values_product (g : ℕ → ℕ) (h : is_valid_g g) :
  (Finset.card (Finset.image g {36})) * (Finset.sum (Finset.image g {36}) id) = 2 := by
  sorry

end NUMINAMATH_CALUDE_g36_values_product_l1762_176220


namespace NUMINAMATH_CALUDE_rowing_time_calculation_l1762_176250

-- Define the given constants
def man_speed : ℝ := 6
def river_speed : ℝ := 3
def total_distance : ℝ := 4.5

-- Define the theorem
theorem rowing_time_calculation :
  let upstream_speed := man_speed - river_speed
  let downstream_speed := man_speed + river_speed
  let one_way_distance := total_distance / 2
  let upstream_time := one_way_distance / upstream_speed
  let downstream_time := one_way_distance / downstream_speed
  let total_time := upstream_time + downstream_time
  total_time = 1 := by sorry

end NUMINAMATH_CALUDE_rowing_time_calculation_l1762_176250


namespace NUMINAMATH_CALUDE_polar_bear_daily_fish_consumption_l1762_176270

/-- The amount of trout eaten daily by the polar bear in buckets -/
def trout_amount : ℝ := 0.2

/-- The amount of salmon eaten daily by the polar bear in buckets -/
def salmon_amount : ℝ := 0.4

/-- The total amount of fish eaten daily by the polar bear in buckets -/
def total_fish_amount : ℝ := trout_amount + salmon_amount

theorem polar_bear_daily_fish_consumption :
  total_fish_amount = 0.6 := by sorry

end NUMINAMATH_CALUDE_polar_bear_daily_fish_consumption_l1762_176270


namespace NUMINAMATH_CALUDE_equation_solution_l1762_176243

theorem equation_solution : ∃! (x : ℝ), (81 : ℝ) ^ (x - 2) / (9 : ℝ) ^ (x - 1) = (729 : ℝ) ^ (3 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1762_176243


namespace NUMINAMATH_CALUDE_subtracted_number_for_perfect_square_l1762_176251

theorem subtracted_number_for_perfect_square : ∃ n : ℕ, (92555 : ℕ) - 139 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_for_perfect_square_l1762_176251


namespace NUMINAMATH_CALUDE_tan_theta_value_l1762_176242

open Complex

theorem tan_theta_value (θ : ℝ) :
  (↑(1 : ℂ) + I) * sin θ - (↑(1 : ℂ) + I * cos θ) ∈ {z : ℂ | z.re + z.im + 1 = 0} →
  tan θ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l1762_176242


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1762_176276

theorem inverse_variation_problem (p q : ℝ) (k : ℝ) (h1 : k > 0) :
  (∀ x y, x * y = k → x > 0 → y > 0) →  -- inverse variation definition
  (1500 * 0.5 = k) →                    -- initial condition
  (3000 * q = k) →                      -- new condition
  q = 0.250 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1762_176276


namespace NUMINAMATH_CALUDE_four_solutions_l1762_176238

/-- The number of integer solutions to x^4 + y^2 = 2y + 1 -/
def solution_count : ℕ := 4

/-- A function that checks if a pair of integers satisfies the equation -/
def satisfies_equation (x y : ℤ) : Prop :=
  x^4 + y^2 = 2*y + 1

/-- The theorem stating that there are exactly 4 integer solutions -/
theorem four_solutions :
  ∃! (solutions : Finset (ℤ × ℤ)), 
    solutions.card = solution_count ∧ 
    ∀ (x y : ℤ), (x, y) ∈ solutions ↔ satisfies_equation x y :=
sorry

end NUMINAMATH_CALUDE_four_solutions_l1762_176238


namespace NUMINAMATH_CALUDE_sin_cos_properties_l1762_176248

open Real

theorem sin_cos_properties : ¬(
  (∃ (T : ℝ), T > 0 ∧ T = π/2 ∧ ∀ (x : ℝ), sin (2*x) = sin (2*(x + T))) ∧
  (∀ (x : ℝ), cos x = cos (π - x))
) := by sorry

end NUMINAMATH_CALUDE_sin_cos_properties_l1762_176248


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_pi_3_l1762_176293

theorem cos_2alpha_plus_pi_3 (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin (α - π / 12) = 3 / 5) :
  Real.cos (2 * α + π / 3) = -24 / 25 := by
sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_pi_3_l1762_176293
