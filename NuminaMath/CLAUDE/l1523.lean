import Mathlib

namespace NUMINAMATH_CALUDE_dannys_travel_time_l1523_152316

theorem dannys_travel_time (danny_time steve_time halfway_danny halfway_steve : ℝ) 
  (h1 : steve_time = 2 * danny_time)
  (h2 : halfway_danny = danny_time / 2)
  (h3 : halfway_steve = steve_time / 2)
  (h4 : halfway_steve - halfway_danny = 12.5) :
  danny_time = 25 := by sorry

end NUMINAMATH_CALUDE_dannys_travel_time_l1523_152316


namespace NUMINAMATH_CALUDE_triangle_problem_l1523_152303

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  A = π / 6 →
  b = (4 + 2 * Real.sqrt 3) * a * Real.cos B →
  b = 1 →
  B = 5 * π / 12 ∧ 
  (1 / 2) * b * c * Real.sin A = 1 / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_problem_l1523_152303


namespace NUMINAMATH_CALUDE_inequality_proof_l1523_152339

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a + b) * (b + c) * (c + a) ≥ 4 * (a + b + c - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1523_152339


namespace NUMINAMATH_CALUDE_tub_drain_time_l1523_152374

/-- Given a tub that drains at a constant rate, this function calculates the additional time
    needed to empty the tub completely after a certain fraction has been drained. -/
def additional_drain_time (initial_fraction : ℚ) (initial_time : ℚ) : ℚ :=
  let remaining_fraction := 1 - initial_fraction
  let drain_rate := initial_fraction / initial_time
  remaining_fraction / drain_rate

/-- Theorem stating that for a tub draining 5/7 of its content in 4 minutes,
    it will take an additional 11.2 minutes to empty completely. -/
theorem tub_drain_time : additional_drain_time (5/7) 4 = 11.2 := by
  sorry

end NUMINAMATH_CALUDE_tub_drain_time_l1523_152374


namespace NUMINAMATH_CALUDE_quadratic_sum_l1523_152329

theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, 6 * x^2 + 72 * x + 500 = a * (x + b)^2 + c) → 
  a + b + c = 296 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1523_152329


namespace NUMINAMATH_CALUDE_distance_traveled_proof_l1523_152392

/-- Calculates the distance traveled given initial and final odometer readings -/
def distance_traveled (initial_reading final_reading : Real) : Real :=
  final_reading - initial_reading

/-- Theorem stating that the distance traveled is 159.7 miles -/
theorem distance_traveled_proof (initial_reading final_reading : Real) 
  (h1 : initial_reading = 212.3)
  (h2 : final_reading = 372.0) :
  distance_traveled initial_reading final_reading = 159.7 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_proof_l1523_152392


namespace NUMINAMATH_CALUDE_cauliflower_increase_l1523_152301

/-- Represents a square garden for growing cauliflowers -/
structure CauliflowerGarden where
  side : ℕ

/-- Calculates the number of cauliflowers in a garden -/
def cauliflowers (garden : CauliflowerGarden) : ℕ := garden.side * garden.side

/-- Theorem: If a square garden's cauliflower output increases by 401 while
    maintaining a square shape, the new total is 40,401 cauliflowers -/
theorem cauliflower_increase (old_garden new_garden : CauliflowerGarden) :
  cauliflowers new_garden - cauliflowers old_garden = 401 →
  cauliflowers new_garden = 40401 := by
  sorry


end NUMINAMATH_CALUDE_cauliflower_increase_l1523_152301


namespace NUMINAMATH_CALUDE_solution_set_of_even_decreasing_quadratic_l1523_152348

def f (a b x : ℝ) : ℝ := a * x^2 + (b - 2*a) * x - 2*b

theorem solution_set_of_even_decreasing_quadratic 
  (a b : ℝ) 
  (h_even : ∀ x, f a b x = f a b (-x))
  (h_decreasing : ∀ x y, 0 < x → x < y → f a b y < f a b x) :
  {x : ℝ | f a b x > 0} = {x : ℝ | -2 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_even_decreasing_quadratic_l1523_152348


namespace NUMINAMATH_CALUDE_trapezoid_area_increase_l1523_152373

/-- Represents a trapezoid with a given height -/
structure Trapezoid where
  height : ℝ

/-- Calculates the increase in area when both bases of a trapezoid are increased by a given amount -/
def area_increase (t : Trapezoid) (base_increase : ℝ) : ℝ :=
  t.height * base_increase

/-- Theorem: The area increase of a trapezoid with height 6 cm when both bases are increased by 4 cm is 24 square centimeters -/
theorem trapezoid_area_increase :
  let t : Trapezoid := { height := 6 }
  area_increase t 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_increase_l1523_152373


namespace NUMINAMATH_CALUDE_least_multiple_15_with_digit_product_multiple_15_l1523_152330

/-- Given a natural number, returns the product of its digits. -/
def digitProduct (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is a multiple of 15. -/
def isMultipleOf15 (n : ℕ) : Prop := ∃ k, n = 15 * k

theorem least_multiple_15_with_digit_product_multiple_15 :
  ∀ n : ℕ, n > 0 → isMultipleOf15 n → isMultipleOf15 (digitProduct n) →
  n ≥ 315 ∧ (n = 315 → isMultipleOf15 (digitProduct 315)) := by sorry

end NUMINAMATH_CALUDE_least_multiple_15_with_digit_product_multiple_15_l1523_152330


namespace NUMINAMATH_CALUDE_construction_workers_l1523_152313

theorem construction_workers (initial_workers : ℕ) (initial_days : ℕ) (remaining_days : ℕ)
  (initial_work_fraction : ℚ) (h1 : initial_workers = 60)
  (h2 : initial_days = 18) (h3 : remaining_days = 12)
  (h4 : initial_work_fraction = 1/3) :
  ∃ (additional_workers : ℕ),
    additional_workers = 60 ∧
    (additional_workers + initial_workers : ℚ) * remaining_days * initial_work_fraction =
    (1 - initial_work_fraction) * initial_workers * initial_days :=
by sorry

end NUMINAMATH_CALUDE_construction_workers_l1523_152313


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_sum_l1523_152390

theorem min_x_prime_factorization_sum (x y p q : ℕ+) (e f : ℕ) : 
  (∀ x' y' : ℕ+, 13 * x'^7 = 19 * y'^17 → x ≤ x') →
  13 * x^7 = 19 * y^17 →
  x = p^e * q^f →
  p.val.Prime ∧ q.val.Prime →
  p + q + e + f = 44 := by
  sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_sum_l1523_152390


namespace NUMINAMATH_CALUDE_inequality_proof_l1523_152395

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ((a^3) / (a^3 + 15*b*c*d))^(1/2) ≥ (a^(15/8)) / (a^(15/8) + b^(15/8) + c^(15/8) + d^(15/8)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1523_152395


namespace NUMINAMATH_CALUDE_fair_die_probabilities_l1523_152356

-- Define the sample space for a fair six-sided die
def Ω : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define event A: number of points ≥ 3
def A : Finset ℕ := {3, 4, 5, 6}

-- Define event B: number of points is odd
def B : Finset ℕ := {1, 3, 5}

-- Define the probability measure for a fair die
def P (S : Finset ℕ) : ℚ := (S ∩ Ω).card / Ω.card

-- Theorem statement
theorem fair_die_probabilities :
  P A = 2/3 ∧ P (A ∪ B) = 5/6 ∧ P (A ∩ B) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fair_die_probabilities_l1523_152356


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1523_152349

theorem tan_alpha_value (α : Real) (h1 : π < α ∧ α < 3*π/2) (h2 : Real.sin (α/2) = Real.sqrt 5 / 3) :
  Real.tan α = -4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1523_152349


namespace NUMINAMATH_CALUDE_hike_attendance_l1523_152342

/-- The number of cars used for the hike -/
def num_cars : ℕ := 3

/-- The number of taxis used for the hike -/
def num_taxis : ℕ := 6

/-- The number of vans used for the hike -/
def num_vans : ℕ := 2

/-- The number of people in each car -/
def people_per_car : ℕ := 4

/-- The number of people in each taxi -/
def people_per_taxi : ℕ := 6

/-- The number of people in each van -/
def people_per_van : ℕ := 5

/-- The total number of people who went on the hike -/
def total_people : ℕ := num_cars * people_per_car + num_taxis * people_per_taxi + num_vans * people_per_van

theorem hike_attendance : total_people = 58 := by
  sorry

end NUMINAMATH_CALUDE_hike_attendance_l1523_152342


namespace NUMINAMATH_CALUDE_number_of_valid_passwords_l1523_152328

/-- The number of digits in the password -/
def password_length : ℕ := 5

/-- The range of possible digits -/
def digit_range : ℕ := 10

/-- The number of passwords starting with the forbidden sequence -/
def forbidden_passwords : ℕ := 10

/-- Calculates the number of valid passwords -/
def valid_passwords : ℕ := digit_range ^ password_length - forbidden_passwords

/-- Theorem stating the number of valid passwords -/
theorem number_of_valid_passwords : valid_passwords = 99990 := by
  sorry

end NUMINAMATH_CALUDE_number_of_valid_passwords_l1523_152328


namespace NUMINAMATH_CALUDE_largest_quantity_l1523_152306

theorem largest_quantity (x y z w : ℝ) 
  (h : x + 5 = y - 3 ∧ x + 5 = z + 2 ∧ x + 5 = w - 4) : 
  w ≥ x ∧ w ≥ y ∧ w ≥ z := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l1523_152306


namespace NUMINAMATH_CALUDE_pyramid_cross_sections_l1523_152337

/-- Theorem about cross-sectional areas in a pyramid --/
theorem pyramid_cross_sections
  (S : ℝ) -- Base area of the pyramid
  (S₁ S₂ S₃ : ℝ) -- Cross-sectional areas
  (h₁ : S₁ = S / 4) -- S₁ bisects lateral edges
  (h₂ : S₂ = S / 2) -- S₂ bisects lateral surface area
  (h₃ : S₃ = S / (4 ^ (1/3))) -- S₃ bisects volume
  : S₁ < S₂ ∧ S₂ < S₃ := by
  sorry

end NUMINAMATH_CALUDE_pyramid_cross_sections_l1523_152337


namespace NUMINAMATH_CALUDE_congruence_theorem_l1523_152381

theorem congruence_theorem (x : ℤ) 
  (h1 : (8 + x) % 8 = 27 % 8)
  (h2 : (10 + x) % 27 = 16 % 27)
  (h3 : (13 + x) % 125 = 36 % 125) :
  x % 120 = 11 := by
sorry

end NUMINAMATH_CALUDE_congruence_theorem_l1523_152381


namespace NUMINAMATH_CALUDE_workshop_employees_l1523_152366

theorem workshop_employees :
  ∃ (n k1 k2 : ℕ),
    0 < n ∧ n < 60 ∧
    n = 8 * k1 + 5 ∧
    n = 6 * k2 + 3 ∧
    (n = 21 ∨ n = 45) :=
by sorry

end NUMINAMATH_CALUDE_workshop_employees_l1523_152366


namespace NUMINAMATH_CALUDE_xy_length_l1523_152331

/-- Represents a trapezoid WXYZ with specific properties -/
structure Trapezoid where
  -- W, X, Y, Z are points in the plane
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  -- WX is parallel to ZY
  wx_parallel_zy : (X.1 - W.1) * (Y.2 - Z.2) = (X.2 - W.2) * (Y.1 - Z.1)
  -- WY is perpendicular to ZY
  wy_perp_zy : (Y.1 - W.1) * (Y.1 - Z.1) + (Y.2 - W.2) * (Y.2 - Z.2) = 0
  -- YZ = 15
  yz_length : Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) = 15
  -- tan Z = 2
  tan_z : (W.2 - Z.2) / (Y.1 - Z.1) = 2
  -- tan X = 3/2
  tan_x : (W.2 - Y.2) / (X.1 - W.1) = 3/2

/-- The length of XY in the trapezoid is 10√13 -/
theorem xy_length (t : Trapezoid) : 
  Real.sqrt ((t.X.1 - t.Y.1)^2 + (t.X.2 - t.Y.2)^2) = 10 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_xy_length_l1523_152331


namespace NUMINAMATH_CALUDE_inheritance_tax_problem_l1523_152318

theorem inheritance_tax_problem (x : ℝ) : 
  let federal_tax := 0.25 * x
  let after_federal := x - federal_tax
  let state_tax := 0.15 * after_federal
  let after_state := after_federal - state_tax
  let luxury_tax := 0.05 * after_state
  let total_tax := federal_tax + state_tax + luxury_tax
  total_tax = 20000 → x = 50700 := by
sorry

end NUMINAMATH_CALUDE_inheritance_tax_problem_l1523_152318


namespace NUMINAMATH_CALUDE_house_wall_nails_l1523_152359

/-- The number of large planks used for the house wall. -/
def large_planks : ℕ := 13

/-- The number of nails needed for each large plank. -/
def nails_per_plank : ℕ := 17

/-- The number of additional nails needed for smaller planks. -/
def additional_nails : ℕ := 8

/-- The total number of nails needed for the house wall. -/
def total_nails : ℕ := large_planks * nails_per_plank + additional_nails

theorem house_wall_nails : total_nails = 229 := by
  sorry

end NUMINAMATH_CALUDE_house_wall_nails_l1523_152359


namespace NUMINAMATH_CALUDE_min_sum_m_n_min_sum_value_min_sum_achieved_l1523_152333

theorem min_sum_m_n (m n : ℕ+) (h : 98 * m = n ^ 3) : 
  ∀ (m' n' : ℕ+), 98 * m' = n' ^ 3 → m + n ≤ m' + n' :=
by
  sorry

theorem min_sum_value (m n : ℕ+) (h : 98 * m = n ^ 3) :
  m + n ≥ 42 :=
by
  sorry

theorem min_sum_achieved : 
  ∃ (m n : ℕ+), 98 * m = n ^ 3 ∧ m + n = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_m_n_min_sum_value_min_sum_achieved_l1523_152333


namespace NUMINAMATH_CALUDE_star_composition_l1523_152344

-- Define the star operations
def star_right (y : ℝ) : ℝ := 9 - y
def star_left (y : ℝ) : ℝ := y - 9

-- State the theorem
theorem star_composition : star_left (star_right 15) = -15 := by
  sorry

end NUMINAMATH_CALUDE_star_composition_l1523_152344


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_expansion_l1523_152307

theorem coefficient_x_cubed_expansion : 
  let expansion := (fun x => (x^2 + 1)^2 * (x - 1)^6)
  ∃ (a b c d e f g h : ℤ), 
    (∀ x, expansion x = a*x^8 + b*x^7 + c*x^6 + d*x^5 + e*x^4 + (-32)*x^3 + f*x^2 + g*x + h) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_expansion_l1523_152307


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l1523_152326

/-- Given a three-digit number abc where a, b, and c are non-zero digits,
    prove that abc = 425 if the sum of the other five three-digit numbers
    formed by rearranging a, b, c is 2017. -/
theorem three_digit_number_problem (a b c : ℕ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 →
  a < 10 → b < 10 → c < 10 →
  (100 * a + 10 * b + c) +
  (100 * a + 10 * c + b) +
  (100 * b + 10 * a + c) +
  (100 * b + 10 * c + a) +
  (100 * c + 10 * a + b) = 2017 →
  100 * a + 10 * b + c = 425 := by
sorry

end NUMINAMATH_CALUDE_three_digit_number_problem_l1523_152326


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_l1523_152387

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_l1523_152387


namespace NUMINAMATH_CALUDE_xyz_product_l1523_152370

theorem xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x * (y + z) = 360)
  (eq2 : y * (z + x) = 405)
  (eq3 : z * (x + y) = 450) :
  x * y * z = 2433 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l1523_152370


namespace NUMINAMATH_CALUDE_zero_in_M_l1523_152321

def M : Set Int := {-1, 0, 1}

theorem zero_in_M : 0 ∈ M := by sorry

end NUMINAMATH_CALUDE_zero_in_M_l1523_152321


namespace NUMINAMATH_CALUDE_isosceles_triangles_height_ratio_l1523_152315

/-- Two isosceles triangles with equal vertical angles and area ratio 16:49 have height ratio 4:7 -/
theorem isosceles_triangles_height_ratio (b₁ h₁ b₂ h₂ : ℝ) : 
  b₁ > 0 → h₁ > 0 → b₂ > 0 → h₂ > 0 →  -- Positive dimensions
  (1/2 * b₁ * h₁) / (1/2 * b₂ * h₂) = 16/49 →  -- Area ratio
  b₁ / b₂ = h₁ / h₂ →  -- Similar triangles condition
  h₁ / h₂ = 4/7 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangles_height_ratio_l1523_152315


namespace NUMINAMATH_CALUDE_pet_ownership_proof_l1523_152391

/-- The number of people owning only cats and dogs -/
def cats_and_dogs_owners : ℕ := 5

theorem pet_ownership_proof (total_owners : ℕ) (only_dogs : ℕ) (only_cats : ℕ) 
  (cats_dogs_snakes : ℕ) (total_snakes : ℕ) 
  (h1 : total_owners = 59)
  (h2 : only_dogs = 15)
  (h3 : only_cats = 10)
  (h4 : cats_dogs_snakes = 3)
  (h5 : total_snakes = 29) :
  cats_and_dogs_owners = total_owners - (only_dogs + only_cats + cats_dogs_snakes + (total_snakes - cats_dogs_snakes)) :=
by sorry

end NUMINAMATH_CALUDE_pet_ownership_proof_l1523_152391


namespace NUMINAMATH_CALUDE_xyz_sum_sqrt_l1523_152382

theorem xyz_sum_sqrt (x y z : ℝ) 
  (h1 : y + z = 15)
  (h2 : z + x = 16)
  (h3 : x + y = 17) :
  Real.sqrt (x * y * z * (x + y + z)) = 24 * Real.sqrt 42 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_sqrt_l1523_152382


namespace NUMINAMATH_CALUDE_chord_length_in_circle_l1523_152311

theorem chord_length_in_circle (r : ℝ) (h : r = 15) : 
  ∃ (c : ℝ), c = 26 ∧ 
  c^2 = 4 * (r^2 - (r/2)^2) ∧ 
  c > 0 := by
sorry

end NUMINAMATH_CALUDE_chord_length_in_circle_l1523_152311


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l1523_152343

theorem max_value_sum_of_roots (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a + b + 9 * c^2 = 1) :
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + 9 * z^2 = 1 ∧
    Real.sqrt x + Real.sqrt y + Real.sqrt 3 * z > Real.sqrt a + Real.sqrt b + Real.sqrt 3 * c) ∨
  Real.sqrt a + Real.sqrt b + Real.sqrt 3 * c = Real.sqrt (21 / 3) := by
  sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l1523_152343


namespace NUMINAMATH_CALUDE_ones_digit_of_3_to_26_l1523_152327

/-- The ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- The ones digit of 3^n for any natural number n -/
def onesDigitOf3Power (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0  -- This case is unreachable, but Lean requires it for exhaustiveness

theorem ones_digit_of_3_to_26 : onesDigit (3^26) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_3_to_26_l1523_152327


namespace NUMINAMATH_CALUDE_robert_gets_two_more_than_kate_l1523_152334

/-- The number of candy pieces each child receives. -/
structure CandyDistribution where
  robert : ℕ
  kate : ℕ
  bill : ℕ
  mary : ℕ

/-- The conditions of the candy distribution problem. -/
def ValidDistribution (d : CandyDistribution) : Prop :=
  d.robert + d.kate + d.bill + d.mary = 20 ∧
  d.robert > d.kate ∧
  d.bill = d.mary - 6 ∧
  d.mary = d.robert + 2 ∧
  d.kate = d.bill + 2 ∧
  d.kate = 4

theorem robert_gets_two_more_than_kate (d : CandyDistribution) 
  (h : ValidDistribution d) : d.robert - d.kate = 2 := by
  sorry

end NUMINAMATH_CALUDE_robert_gets_two_more_than_kate_l1523_152334


namespace NUMINAMATH_CALUDE_johns_age_doubles_l1523_152379

/-- Represents John's current age -/
def current_age : ℕ := 18

/-- Represents the number of years ago when John's age was half of a future age -/
def years_ago : ℕ := 5

/-- Represents the number of years until John's age is twice his age from five years ago -/
def years_until_double : ℕ := 8

/-- Theorem stating that in 8 years, John's age will be twice his age from five years ago -/
theorem johns_age_doubles : 
  2 * (current_age - years_ago) = current_age + years_until_double := by
  sorry

end NUMINAMATH_CALUDE_johns_age_doubles_l1523_152379


namespace NUMINAMATH_CALUDE_jeff_scores_mean_l1523_152350

def jeff_scores : List ℝ := [90, 93, 85, 97, 92, 88]

theorem jeff_scores_mean : 
  (jeff_scores.sum / jeff_scores.length : ℝ) = 90.8333 := by
  sorry

end NUMINAMATH_CALUDE_jeff_scores_mean_l1523_152350


namespace NUMINAMATH_CALUDE_union_equals_B_exists_union_equals_intersection_l1523_152399

-- Define the sets A, B, and C
def A : Set ℝ := {x | |x - 1| < 2}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x - 6 < 0}
def C : Set ℝ := {x | x^2 - 2*x - 15 < 0}

-- Statement 1
theorem union_equals_B (a : ℝ) : 
  A ∪ B a = B a ↔ a ∈ Set.Icc (-5 : ℝ) (-1 : ℝ) := by sorry

-- Statement 2
theorem exists_union_equals_intersection :
  ∃ a ∈ Set.Icc (-19/5 : ℝ) (-1 : ℝ), A ∪ B a = B a ∩ C := by sorry

end NUMINAMATH_CALUDE_union_equals_B_exists_union_equals_intersection_l1523_152399


namespace NUMINAMATH_CALUDE_bacteria_growth_l1523_152371

/-- The time interval between bacterial divisions in minutes -/
def division_interval : ℕ := 20

/-- The total observation time in hours -/
def total_time : ℕ := 3

/-- The number of divisions that occur in the total observation time -/
def num_divisions : ℕ := (total_time * 60) / division_interval

/-- The final number of bacteria after the total observation time -/
def final_bacteria_count : ℕ := 2^num_divisions

theorem bacteria_growth :
  final_bacteria_count = 512 :=
sorry

end NUMINAMATH_CALUDE_bacteria_growth_l1523_152371


namespace NUMINAMATH_CALUDE_yellow_chip_value_l1523_152305

theorem yellow_chip_value :
  ∀ (y : ℕ) (b : ℕ),
    y > 0 →
    b > 0 →
    y^4 * (4 * b)^b * (5 * b)^b = 16000 →
    y = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_chip_value_l1523_152305


namespace NUMINAMATH_CALUDE_largest_n_for_unique_k_l1523_152362

theorem largest_n_for_unique_k : 
  ∀ n : ℕ+, n ≤ 72 ↔ 
    ∃! k : ℤ, (9:ℚ)/17 < (n:ℚ)/(n + k) ∧ (n:ℚ)/(n + k) < 8/15 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_unique_k_l1523_152362


namespace NUMINAMATH_CALUDE_zilla_savings_theorem_l1523_152363

/-- Represents Zilla's monthly financial breakdown -/
structure ZillaFinances where
  earnings : ℝ
  rent_percent : ℝ
  groceries_percent : ℝ
  entertainment_percent : ℝ
  transportation_percent : ℝ
  rent_amount : ℝ

/-- Calculates Zilla's savings based on her financial breakdown -/
def calculate_savings (z : ZillaFinances) : ℝ :=
  z.earnings * (1 - z.rent_percent - z.groceries_percent - z.entertainment_percent - z.transportation_percent)

/-- Theorem stating that Zilla's savings are $589 given her financial breakdown -/
theorem zilla_savings_theorem (z : ZillaFinances) 
  (h1 : z.rent_percent = 0.07)
  (h2 : z.groceries_percent = 0.3)
  (h3 : z.entertainment_percent = 0.2)
  (h4 : z.transportation_percent = 0.12)
  (h5 : z.rent_amount = 133)
  (h6 : z.earnings * z.rent_percent = z.rent_amount) :
  calculate_savings z = 589 := by
  sorry


end NUMINAMATH_CALUDE_zilla_savings_theorem_l1523_152363


namespace NUMINAMATH_CALUDE_train_length_l1523_152352

/-- Given a train traveling at 45 kmph that passes a 140 m long bridge in 40 seconds,
    prove that the length of the train is 360 m. -/
theorem train_length (speed : ℝ) (bridge_length : ℝ) (time : ℝ) (train_length : ℝ) : 
  speed = 45 → bridge_length = 140 → time = 40 → 
  train_length = (speed * 1000 / 3600 * time) - bridge_length → 
  train_length = 360 := by
sorry

end NUMINAMATH_CALUDE_train_length_l1523_152352


namespace NUMINAMATH_CALUDE_f_five_not_unique_l1523_152398

/-- A function satisfying the given functional equation for all real x and y -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x + f (3 * x + y) + 3 * x * y = f (4 * x - y) + 3 * x^2 + 2

/-- The theorem stating that f(5) cannot be uniquely determined -/
theorem f_five_not_unique : 
  ¬ ∃ (a : ℝ), ∀ (f : ℝ → ℝ), FunctionalEquation f → f 5 = a :=
sorry

end NUMINAMATH_CALUDE_f_five_not_unique_l1523_152398


namespace NUMINAMATH_CALUDE_gumball_probability_l1523_152380

/-- Given a jar with pink and blue gumballs, if the probability of drawing two blue
    gumballs in a row with replacement is 16/36, then the probability of drawing
    a pink gumball is 1/3. -/
theorem gumball_probability (p_blue p_pink : ℝ) : 
  p_blue + p_pink = 1 →
  p_blue ^ 2 = 16 / 36 →
  p_pink = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_gumball_probability_l1523_152380


namespace NUMINAMATH_CALUDE_woo_jun_age_l1523_152341

theorem woo_jun_age :
  ∀ (w m : ℕ),
  w = m / 4 - 1 →
  m = 5 * w - 5 →
  w = 9 := by
sorry

end NUMINAMATH_CALUDE_woo_jun_age_l1523_152341


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l1523_152323

theorem negation_of_existence_proposition :
  ¬(∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) ↔ 
  (∀ a : ℝ, ∀ x : ℝ, a * x^2 + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l1523_152323


namespace NUMINAMATH_CALUDE_target_destruction_probability_l1523_152309

def prob_at_least_two (p1 p2 p3 : ℝ) : ℝ :=
  p1 * p2 * p3 +
  p1 * p2 * (1 - p3) +
  p1 * (1 - p2) * p3 +
  (1 - p1) * p2 * p3

theorem target_destruction_probability :
  prob_at_least_two 0.9 0.9 0.8 = 0.954 := by
  sorry

end NUMINAMATH_CALUDE_target_destruction_probability_l1523_152309


namespace NUMINAMATH_CALUDE_greatest_multiple_of_30_under_1000_l1523_152355

theorem greatest_multiple_of_30_under_1000 : 
  ∀ n : ℕ, n * 30 < 1000 → n * 30 ≤ 990 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_30_under_1000_l1523_152355


namespace NUMINAMATH_CALUDE_sports_cards_pages_l1523_152393

/-- Calculates the number of pages needed for a given number of cards and cards per page -/
def pagesNeeded (cards : ℕ) (cardsPerPage : ℕ) : ℕ :=
  (cards + cardsPerPage - 1) / cardsPerPage

theorem sports_cards_pages : 
  let baseballCards := 12
  let baseballCardsPerPage := 4
  let basketballCards := 14
  let basketballCardsPerPage := 3
  let soccerCards := 7
  let soccerCardsPerPage := 5
  (pagesNeeded baseballCards baseballCardsPerPage) +
  (pagesNeeded basketballCards basketballCardsPerPage) +
  (pagesNeeded soccerCards soccerCardsPerPage) = 10 := by
  sorry


end NUMINAMATH_CALUDE_sports_cards_pages_l1523_152393


namespace NUMINAMATH_CALUDE_apple_count_l1523_152368

theorem apple_count (red_apples green_apples total_apples : ℕ) : 
  red_apples = 16 →
  green_apples = red_apples + 12 →
  total_apples = red_apples + green_apples →
  total_apples = 44 := by
  sorry

end NUMINAMATH_CALUDE_apple_count_l1523_152368


namespace NUMINAMATH_CALUDE_aaron_cards_proof_l1523_152365

def aaron_final_cards (initial_aaron : ℕ) (found : ℕ) (lost : ℕ) (given : ℕ) : ℕ :=
  initial_aaron + found - lost - given

theorem aaron_cards_proof (initial_arthur : ℕ) (initial_aaron : ℕ) (found : ℕ) (lost : ℕ) (given : ℕ)
  (h1 : initial_arthur = 6)
  (h2 : initial_aaron = 5)
  (h3 : found = 62)
  (h4 : lost = 15)
  (h5 : given = 28) :
  aaron_final_cards initial_aaron found lost given = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_aaron_cards_proof_l1523_152365


namespace NUMINAMATH_CALUDE_function_and_inequality_l1523_152372

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 2|

-- Define the solution set condition
def solution_set (m : ℝ) : Set ℝ := {x | f m (x + 2) ≥ 0}

-- State the theorem
theorem function_and_inequality (m a b c : ℝ) : 
  (solution_set m = Set.Icc (-1) 1) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b + c = m) →
  (m = 1 ∧ a + 2*b + 3*c ≥ 9) := by
  sorry


end NUMINAMATH_CALUDE_function_and_inequality_l1523_152372


namespace NUMINAMATH_CALUDE_smallest_number_l1523_152388

theorem smallest_number (S : Set ℤ) : S = {-2, -1, 0, 1} → ∀ x ∈ S, -2 ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1523_152388


namespace NUMINAMATH_CALUDE_laundromat_cost_l1523_152364

def service_fee : ℝ := 3
def first_hour_cost : ℝ := 10
def additional_hour_cost : ℝ := 15
def usage_time : ℝ := 2.75
def discount_rate : ℝ := 0.1

def calculate_cost : ℝ :=
  let base_cost := first_hour_cost + (usage_time - 1) * additional_hour_cost
  let total_cost := base_cost + service_fee
  let discount := total_cost * discount_rate
  total_cost - discount

theorem laundromat_cost :
  calculate_cost = 35.32 := by sorry

end NUMINAMATH_CALUDE_laundromat_cost_l1523_152364


namespace NUMINAMATH_CALUDE_subtract_from_percentage_l1523_152340

theorem subtract_from_percentage (n : ℝ) : n = 300 → 0.3 * n - 70 = 20 := by
  sorry

end NUMINAMATH_CALUDE_subtract_from_percentage_l1523_152340


namespace NUMINAMATH_CALUDE_negation_equivalence_l1523_152336

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀^2 - x₀ - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1523_152336


namespace NUMINAMATH_CALUDE_equation_solution_l1523_152354

theorem equation_solution (x : ℝ) : 
  x ≠ 1 → x ≠ -6 → 
  ((3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1)) ↔ (x = -4 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1523_152354


namespace NUMINAMATH_CALUDE_unknown_number_in_set_l1523_152324

theorem unknown_number_in_set (x : ℝ) : 
  (20 + x + 60) / 3 = (10 + 70 + 13) / 3 + 9 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_in_set_l1523_152324


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1523_152345

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum1 : a 3 + a 4 + a 5 + a 6 + a 7 = 15)
  (h_sum2 : a 9 + a 10 + a 11 = 39) :
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1523_152345


namespace NUMINAMATH_CALUDE_range_of_a_l1523_152360

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem range_of_a (f : ℝ → ℝ) (h : DecreasingFunction f) :
  (∀ a : ℝ, f (3 * a) < f (-2 * a + 10)) →
  (∃ c : ℝ, c = 2 ∧ ∀ a : ℝ, a > c) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1523_152360


namespace NUMINAMATH_CALUDE_yard_length_22_trees_l1523_152357

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_distance : ℕ) : ℕ :=
  (num_trees - 1) * tree_distance

/-- Theorem: The length of a yard with 22 trees planted at equal distances,
    with one tree at each end and 21 metres between consecutive trees, is 441 metres. -/
theorem yard_length_22_trees : yard_length 22 21 = 441 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_22_trees_l1523_152357


namespace NUMINAMATH_CALUDE_train_length_l1523_152369

/-- Given a train that can cross an electric pole in 120 seconds while traveling at 90 km/h,
    prove that its length is 3000 meters. -/
theorem train_length (crossing_time : ℝ) (speed_kmh : ℝ) (length : ℝ) : 
  crossing_time = 120 →
  speed_kmh = 90 →
  length = speed_kmh * (1000 / 3600) * crossing_time →
  length = 3000 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1523_152369


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l1523_152396

def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

theorem monotonic_decreasing_interval_of_f :
  ∀ a b : ℝ, a = -1 ∧ b = 11 →
  (∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f y < f x) ∧
  ¬(∃ c d : ℝ, (c < a ∨ b < d) ∧
    (∀ x y : ℝ, c < x ∧ x < y ∧ y < d → f y < f x)) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l1523_152396


namespace NUMINAMATH_CALUDE_compound_interest_rate_l1523_152332

theorem compound_interest_rate (principal : ℝ) (final_amount : ℝ) (time : ℝ) 
  (h1 : principal = 400)
  (h2 : final_amount = 441)
  (h3 : time = 2) :
  ∃ (rate : ℝ), 
    final_amount = principal * (1 + rate) ^ time ∧ 
    rate = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l1523_152332


namespace NUMINAMATH_CALUDE_tan_ratio_equals_two_l1523_152338

theorem tan_ratio_equals_two (α β γ : ℝ) 
  (h : Real.sin (2 * (α + γ)) = 3 * Real.sin (2 * β)) : 
  Real.tan (α + β + γ) / Real.tan (α - β + γ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_equals_two_l1523_152338


namespace NUMINAMATH_CALUDE_extreme_values_and_range_l1523_152310

-- Define the function f
def f (a b c x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8 * c

-- Define the derivative of f
def f' (a b x : ℝ) : ℝ := 6 * x^2 + 6 * a * x + 3 * b

theorem extreme_values_and_range (a b c : ℝ) :
  (∀ x : ℝ, f' a b x = 0 ↔ x = 1 ∨ x = 2) →
  (∀ x : ℝ, x ∈ Set.Icc 0 3 → f a b c x < c^2) →
  (a = -3 ∧ b = 4) ∧ (c < -1 ∨ c > 9) := by
  sorry

#check extreme_values_and_range

end NUMINAMATH_CALUDE_extreme_values_and_range_l1523_152310


namespace NUMINAMATH_CALUDE_combined_cost_price_theorem_l1523_152312

/-- Calculates the cost price of a stock given its face value, discount/premium rate, and brokerage rate -/
def stockCostPrice (faceValue : ℝ) (discountRate : ℝ) (brokerageRate : ℝ) : ℝ :=
  let purchasePrice := faceValue * (1 + discountRate)
  let brokerageFee := purchasePrice * brokerageRate
  purchasePrice + brokerageFee

/-- Theorem stating the combined cost price of two stocks -/
theorem combined_cost_price_theorem :
  let stockA := stockCostPrice 100 (-0.02) 0.002
  let stockB := stockCostPrice 100 0.015 0.002
  stockA + stockB = 199.899 := by
  sorry


end NUMINAMATH_CALUDE_combined_cost_price_theorem_l1523_152312


namespace NUMINAMATH_CALUDE_binomial_coefficient_22_5_l1523_152325

theorem binomial_coefficient_22_5 (h1 : Nat.choose 20 3 = 1140)
                                  (h2 : Nat.choose 20 4 = 4845)
                                  (h3 : Nat.choose 20 5 = 15504) :
  Nat.choose 22 5 = 26334 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_22_5_l1523_152325


namespace NUMINAMATH_CALUDE_abs_a_minus_b_equals_eight_l1523_152377

theorem abs_a_minus_b_equals_eight (a b : ℚ) 
  (h : |a + b| + (b - 4)^2 = 0) : 
  |a - b| = 8 := by
sorry

end NUMINAMATH_CALUDE_abs_a_minus_b_equals_eight_l1523_152377


namespace NUMINAMATH_CALUDE_boys_fraction_l1523_152397

/-- In a class with boys and girls, prove that the fraction of boys is 2/3 given the conditions. -/
theorem boys_fraction (tall_boys : ℕ) (total_boys : ℕ) (girls : ℕ) 
  (h1 : tall_boys = 18)
  (h2 : tall_boys = 3 * total_boys / 4)
  (h3 : girls = 12) :
  total_boys / (total_boys + girls) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_boys_fraction_l1523_152397


namespace NUMINAMATH_CALUDE_composite_function_equality_l1523_152302

/-- Given two functions f and g, prove that f(g(f(3))) = 332 -/
theorem composite_function_equality (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = 4 * x + 4) 
  (hg : ∀ x, g x = 5 * x + 2) : 
  f (g (f 3)) = 332 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_equality_l1523_152302


namespace NUMINAMATH_CALUDE_circle_area_vs_circumference_probability_l1523_152300

theorem circle_area_vs_circumference_probability : 
  let die_roll : Set ℕ := {1, 2, 3, 4, 5, 6}
  let favorable_outcomes : Set ℕ := {n ∈ die_roll | n > 1}
  let probability := (Finset.card favorable_outcomes.toFinset) / (Finset.card die_roll.toFinset)
  let area (r : ℝ) := π * r^2
  let circumference (r : ℝ) := 2 * π * r
  (∀ r ∈ die_roll, area r > (1/2) * circumference r ↔ r > 1) →
  probability = 5/6 := by
sorry

end NUMINAMATH_CALUDE_circle_area_vs_circumference_probability_l1523_152300


namespace NUMINAMATH_CALUDE_product_of_xy_l1523_152353

theorem product_of_xy (x y : ℝ) (h : 3 * (2 * x * y + 9) = 51) : x * y = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_xy_l1523_152353


namespace NUMINAMATH_CALUDE_expression_evaluation_l1523_152308

theorem expression_evaluation : 3^2 / 3 - 4 * 2 + 2^3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1523_152308


namespace NUMINAMATH_CALUDE_goods_train_speed_calculation_l1523_152385

/-- The speed of the goods train in km/h -/
def goods_train_speed : ℝ := 36

/-- The speed of the express train in km/h -/
def express_train_speed : ℝ := 90

/-- The time difference between the departures of the two trains in hours -/
def time_difference : ℝ := 6

/-- The time taken by the express train to catch up with the goods train in hours -/
def catch_up_time : ℝ := 4

theorem goods_train_speed_calculation :
  goods_train_speed * (catch_up_time + time_difference) = express_train_speed * catch_up_time :=
sorry

end NUMINAMATH_CALUDE_goods_train_speed_calculation_l1523_152385


namespace NUMINAMATH_CALUDE_removed_triangles_area_l1523_152384

/-- Given a square with side length s, from which isosceles right triangles
    with equal sides of length x are removed from each corner to form a rectangle
    with longer side 16 units, the total area of the four removed triangles is 512 square units. -/
theorem removed_triangles_area (s x : ℝ) : 
  s > 0 ∧ x > 0 ∧ s - x = 16 ∧ 2 * x^2 = (s - 2*x)^2 → 4 * (1/2 * x^2) = 512 := by
  sorry

#check removed_triangles_area

end NUMINAMATH_CALUDE_removed_triangles_area_l1523_152384


namespace NUMINAMATH_CALUDE_emily_candy_duration_l1523_152346

/-- The number of days Emily's candy will last -/
def candy_duration (neighbors_candy : ℕ) (sister_candy : ℕ) (daily_consumption : ℕ) : ℕ :=
  (neighbors_candy + sister_candy) / daily_consumption

/-- Proof that Emily's candy will last for 2 days -/
theorem emily_candy_duration :
  candy_duration 5 13 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_emily_candy_duration_l1523_152346


namespace NUMINAMATH_CALUDE_tennis_tournament_theorem_l1523_152314

-- Define the number of women players
def n : ℕ := sorry

-- Define the total number of players
def total_players : ℕ := n + 3 * n

-- Define the total number of matches
def total_matches : ℕ := (total_players * (total_players - 1)) / 2

-- Define the number of matches won by women
def women_wins : ℕ := sorry

-- Define the number of matches won by men
def men_wins : ℕ := sorry

-- Theorem stating the conditions and the result to be proved
theorem tennis_tournament_theorem :
  -- Each player plays with every other player
  (∀ p : ℕ, p < total_players → (total_players - 1) * p = total_matches * 2) →
  -- No ties
  (women_wins + men_wins = total_matches) →
  -- Ratio of women's wins to men's wins is 3/2
  (3 * men_wins = 2 * women_wins) →
  -- n equals 4
  n = 4 := by sorry

end NUMINAMATH_CALUDE_tennis_tournament_theorem_l1523_152314


namespace NUMINAMATH_CALUDE_pens_for_friends_l1523_152358

/-- The number of friends who will receive pens from Kendra and Tony -/
def friends_receiving_pens (kendra_packs tony_packs pens_per_pack pens_kept_each : ℕ) : ℕ :=
  (kendra_packs + tony_packs) * pens_per_pack - 2 * pens_kept_each

/-- Theorem stating that Kendra and Tony will give pens to 14 friends -/
theorem pens_for_friends : 
  friends_receiving_pens 4 2 3 2 = 14 := by
  sorry

#eval friends_receiving_pens 4 2 3 2

end NUMINAMATH_CALUDE_pens_for_friends_l1523_152358


namespace NUMINAMATH_CALUDE_different_suit_card_combinations_l1523_152367

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def number_of_suits : ℕ := 4

/-- The number of cards per suit in a standard deck -/
def cards_per_suit : ℕ := standard_deck_size / number_of_suits

/-- The number of cards to be chosen -/
def cards_to_choose : ℕ := 4

-- Theorem statement
theorem different_suit_card_combinations :
  (number_of_suits.choose cards_to_choose) * (cards_per_suit ^ cards_to_choose) = 28561 := by
  sorry

end NUMINAMATH_CALUDE_different_suit_card_combinations_l1523_152367


namespace NUMINAMATH_CALUDE_rectangles_form_square_l1523_152319

/-- A rectangle represented by its width and height --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculate the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Check if a list of rectangles can form a square of side length 16 --/
def canFormSquare (rectangles : List Rectangle) : Prop :=
  ∃ (arrangement : List Rectangle), 
    arrangement.length = rectangles.length ∧
    (∀ r ∈ arrangement, r ∈ rectangles) ∧
    (∀ r ∈ rectangles, r ∈ arrangement) ∧
    arrangement.foldr (λ r acc => acc + r.width * r.height) 0 = 16 * 16

/-- The main theorem to prove --/
theorem rectangles_form_square :
  ∃ (rectangles : List Rectangle),
    rectangles.foldr (λ r acc => acc + perimeter r) 0 = 100 ∧
    canFormSquare rectangles := by sorry

end NUMINAMATH_CALUDE_rectangles_form_square_l1523_152319


namespace NUMINAMATH_CALUDE_circle_square_tangency_l1523_152394

theorem circle_square_tangency (r : ℝ) (s : ℝ) 
  (hr : r = 13) (hs : s = 18) : 
  let d := Real.sqrt (r^2 - (s - r)^2)
  (s - d = 1) ∧ d = 17 := by sorry

end NUMINAMATH_CALUDE_circle_square_tangency_l1523_152394


namespace NUMINAMATH_CALUDE_product_ratio_integer_l1523_152383

def divisible_count (seq : List Nat) (d : Nat) : Nat :=
  (seq.filter (fun x => x % d == 0)).length

theorem product_ratio_integer (m n : List Nat) :
  (∀ d : Nat, d > 1 → divisible_count m d ≥ divisible_count n d) →
  m.all (· > 0) →
  n.all (· > 0) →
  n.length > 0 →
  ∃ k : Nat, k > 0 ∧ (m.prod : Int) = k * (n.prod : Int) := by
  sorry

end NUMINAMATH_CALUDE_product_ratio_integer_l1523_152383


namespace NUMINAMATH_CALUDE_remaining_movie_time_l1523_152378

def movie_length : ℕ := 120
def session1 : ℕ := 35
def session2 : ℕ := 20
def session3 : ℕ := 15

theorem remaining_movie_time :
  movie_length - (session1 + session2 + session3) = 50 := by
  sorry

end NUMINAMATH_CALUDE_remaining_movie_time_l1523_152378


namespace NUMINAMATH_CALUDE_pumpkin_contest_theorem_l1523_152347

def pumpkin_contest (brad jessica betty carlos emily dave : ℝ) : Prop :=
  brad = 54 ∧
  jessica = brad / 2 ∧
  betty = 4 * jessica ∧
  carlos = 2.5 * (brad + jessica) ∧
  emily = 1.5 * (betty - brad) ∧
  dave = (jessica + betty) / 2 + 20 ∧
  max brad (max jessica (max betty (max carlos (max emily dave)))) -
  min brad (min jessica (min betty (min carlos (min emily dave)))) = 175.5

theorem pumpkin_contest_theorem :
  ∃ brad jessica betty carlos emily dave : ℝ,
    pumpkin_contest brad jessica betty carlos emily dave :=
by
  sorry

end NUMINAMATH_CALUDE_pumpkin_contest_theorem_l1523_152347


namespace NUMINAMATH_CALUDE_jogger_ahead_of_train_l1523_152304

/-- Calculates the distance a jogger is ahead of a train given their speeds and the time for the train to pass the jogger. -/
def jogger_distance_ahead (jogger_speed : Real) (train_speed : Real) (train_length : Real) (passing_time : Real) : Real :=
  (train_speed - jogger_speed) * passing_time - train_length

/-- Theorem stating the distance a jogger is ahead of a train under specific conditions. -/
theorem jogger_ahead_of_train (jogger_speed : Real) (train_speed : Real) (train_length : Real) (passing_time : Real)
  (h1 : jogger_speed = 9 * 1000 / 3600)
  (h2 : train_speed = 45 * 1000 / 3600)
  (h3 : train_length = 120)
  (h4 : passing_time = 40.00000000000001) :
  ∃ ε > 0, |jogger_distance_ahead jogger_speed train_speed train_length passing_time - 280| < ε :=
by sorry

end NUMINAMATH_CALUDE_jogger_ahead_of_train_l1523_152304


namespace NUMINAMATH_CALUDE_sum_product_inequality_l1523_152320

theorem sum_product_inequality (a b c : ℝ) (h : a + b + c = 1) : a * b + b * c + c * a ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l1523_152320


namespace NUMINAMATH_CALUDE_line_slope_range_l1523_152322

/-- A line passing through (1,1) with y-intercept in (0,2) has slope in (-1,1) -/
theorem line_slope_range (l : Set (ℝ × ℝ)) (y_intercept : ℝ) :
  (∀ p ∈ l, ∃ k : ℝ, p.2 - 1 = k * (p.1 - 1)) →  -- l is a line
  (1, 1) ∈ l →  -- l passes through (1,1)
  0 < y_intercept ∧ y_intercept < 2 →  -- y-intercept is in (0,2)
  (∃ b : ℝ, ∀ x y : ℝ, (x, y) ∈ l ↔ y = y_intercept + (y_intercept - 1) * (x - 1)) →
  ∃ k : ℝ, -1 < k ∧ k < 1 ∧ ∀ x y : ℝ, (x, y) ∈ l ↔ y - 1 = k * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_range_l1523_152322


namespace NUMINAMATH_CALUDE_max_three_digit_quotient_l1523_152376

theorem max_three_digit_quotient :
  ∃ (a b c : ℕ), 
    a > 5 ∧ b > 5 ∧ c > 5 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    ∀ (x y z : ℕ), 
      x > 5 ∧ y > 5 ∧ z > 5 ∧ 
      x ≠ y ∧ y ≠ z ∧ x ≠ z →
      (100 * a + 10 * b + c : ℚ) / (a + b + c) ≥ (100 * x + 10 * y + z : ℚ) / (x + y + z) ∧
    (100 * a + 10 * b + c : ℚ) / (a + b + c) = 41.125 := by
  sorry

end NUMINAMATH_CALUDE_max_three_digit_quotient_l1523_152376


namespace NUMINAMATH_CALUDE_new_person_weight_l1523_152375

/-- Proves that the weight of a new person is 65 kg given the conditions of the problem -/
theorem new_person_weight (initial_count : Nat) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 45 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 65 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1523_152375


namespace NUMINAMATH_CALUDE_new_concentration_after_replacement_l1523_152361

/-- Calculates the new concentration of a solution after partial replacement --/
def new_concentration (initial_conc : ℝ) (replacement_conc : ℝ) (fraction_replaced : ℝ) : ℝ :=
  (initial_conc * (1 - fraction_replaced) + replacement_conc * fraction_replaced)

/-- Theorem: New concentration after partial replacement --/
theorem new_concentration_after_replacement :
  new_concentration 0.4 0.25 (1/3) = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_new_concentration_after_replacement_l1523_152361


namespace NUMINAMATH_CALUDE_number_division_property_l1523_152386

theorem number_division_property : ∃ (n : ℕ), 
  let sum := 2468 + 1375
  let diff := 2468 - 1375
  n = 12609027 ∧
  n / sum = 3 * diff ∧
  n % sum = 150 ∧
  (n - 150) / sum = 5 * diff :=
by sorry

end NUMINAMATH_CALUDE_number_division_property_l1523_152386


namespace NUMINAMATH_CALUDE_smaller_square_area_percentage_l1523_152351

/-- Given a circle with radius 2√2 and a square inscribed in it with side length 4,
    prove that a smaller square with one side coinciding with the larger square
    and two vertices on the circle has an area that is 4% of the larger square's area. -/
theorem smaller_square_area_percentage (r : ℝ) (s : ℝ) (x : ℝ) :
  r = 2 * Real.sqrt 2 →
  s = 4 →
  (2 + 2*x)^2 + x^2 = r^2 →
  (2*x)^2 / s^2 = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_smaller_square_area_percentage_l1523_152351


namespace NUMINAMATH_CALUDE_kevin_watermelon_weight_l1523_152389

theorem kevin_watermelon_weight :
  let first_watermelon : ℝ := 9.91
  let second_watermelon : ℝ := 4.11
  let total_weight : ℝ := first_watermelon + second_watermelon
  total_weight = 14.02 := by sorry

end NUMINAMATH_CALUDE_kevin_watermelon_weight_l1523_152389


namespace NUMINAMATH_CALUDE_soda_cost_for_reunion_l1523_152317

/-- Calculates the cost per family member for soda at a family reunion --/
def soda_cost_per_family_member (attendees : ℕ) (cans_per_person : ℕ) (cans_per_box : ℕ) (cost_per_box : ℚ) (family_members : ℕ) : ℚ :=
  let total_cans := attendees * cans_per_person
  let boxes_needed := (total_cans + cans_per_box - 1) / cans_per_box  -- Ceiling division
  let total_cost := boxes_needed * cost_per_box
  total_cost / family_members

/-- Theorem stating the cost per family member for the given scenario --/
theorem soda_cost_for_reunion : 
  soda_cost_per_family_member (5 * 12) 2 10 2 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_for_reunion_l1523_152317


namespace NUMINAMATH_CALUDE_series_sum_l1523_152335

/-- The sum of a specific infinite series given positive real numbers a and b where a > 3b -/
theorem series_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > 3 * b) : 
  let series_term (n : ℕ) := 1 / (((3 * n - 6) * a - (n^2 - 5*n + 6) * b) * ((3 * n - 3) * a - (n^2 - 4*n + 3) * b))
  ∑' n, series_term n = 1 / (b * (a - b)) := by
sorry

end NUMINAMATH_CALUDE_series_sum_l1523_152335
