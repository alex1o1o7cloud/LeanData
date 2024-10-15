import Mathlib

namespace NUMINAMATH_CALUDE_dot_product_AB_BC_l3314_331446

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let CA := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = 7 ∧ BC = 5 ∧ CA = 6

-- Define the dot product of two 2D vectors
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

-- Theorem statement
theorem dot_product_AB_BC (A B C : ℝ × ℝ) :
  triangle_ABC A B C →
  dot_product (B.1 - A.1, B.2 - A.2) (C.1 - B.1, C.2 - B.2) = -19 :=
sorry

end NUMINAMATH_CALUDE_dot_product_AB_BC_l3314_331446


namespace NUMINAMATH_CALUDE_duty_arrangement_for_three_leaders_l3314_331422

/-- The number of ways to arrange n leaders for duty over d days, 
    with each leader on duty for m days. -/
def dutyArrangements (n d m : ℕ) : ℕ := sorry

/-- The number of combinations of n items taken k at a time. -/
def nCk (n k : ℕ) : ℕ := sorry

theorem duty_arrangement_for_three_leaders :
  dutyArrangements 3 6 2 = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_duty_arrangement_for_three_leaders_l3314_331422


namespace NUMINAMATH_CALUDE_unique_solution_cubic_equation_l3314_331477

theorem unique_solution_cubic_equation :
  ∃! x : ℝ, x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_equation_l3314_331477


namespace NUMINAMATH_CALUDE_apple_pear_equivalence_l3314_331470

theorem apple_pear_equivalence (apple_value pear_value : ℚ) :
  (3 / 4 : ℚ) * 12 * apple_value = 10 * pear_value →
  (3 / 5 : ℚ) * 15 * apple_value = 10 * pear_value :=
by
  sorry

end NUMINAMATH_CALUDE_apple_pear_equivalence_l3314_331470


namespace NUMINAMATH_CALUDE_min_value_perpendicular_vectors_l3314_331474

theorem min_value_perpendicular_vectors (x y : ℝ) :
  (x - 1) * 4 + 2 * y = 0 →
  ∃ (min : ℝ), min = 6 ∧ ∀ (z : ℝ), z = 9^x + 3^y → z ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_perpendicular_vectors_l3314_331474


namespace NUMINAMATH_CALUDE_will_chocolate_pieces_l3314_331417

theorem will_chocolate_pieces : 
  ∀ (total_boxes given_boxes pieces_per_box : ℕ),
  total_boxes = 7 →
  given_boxes = 3 →
  pieces_per_box = 4 →
  (total_boxes - given_boxes) * pieces_per_box = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_will_chocolate_pieces_l3314_331417


namespace NUMINAMATH_CALUDE_swimming_time_difference_l3314_331452

theorem swimming_time_difference 
  (distance : ℝ) 
  (jack_speed : ℝ) 
  (jill_speed : ℝ) 
  (h1 : distance = 1) 
  (h2 : jack_speed = 10) 
  (h3 : jill_speed = 4) : 
  (distance / jill_speed - distance / jack_speed) * 60 = 9 := by
  sorry

end NUMINAMATH_CALUDE_swimming_time_difference_l3314_331452


namespace NUMINAMATH_CALUDE_minimum_spotted_blueeyed_rabbits_l3314_331423

theorem minimum_spotted_blueeyed_rabbits 
  (total : ℕ) (spotted : ℕ) (blueeyed : ℕ) 
  (h_total : total = 100)
  (h_spotted : spotted = 53)
  (h_blueeyed : blueeyed = 73) :
  ∃ (both : ℕ), both ≥ 26 ∧ 
    ∀ (x : ℕ), x < 26 → spotted + blueeyed - x > total :=
by sorry

end NUMINAMATH_CALUDE_minimum_spotted_blueeyed_rabbits_l3314_331423


namespace NUMINAMATH_CALUDE_jensen_family_mileage_l3314_331443

/-- Represents the mileage problem for the Jensen family's road trip -/
theorem jensen_family_mileage
  (total_highway_miles : ℝ)
  (total_city_miles : ℝ)
  (highway_mpg : ℝ)
  (total_gallons : ℝ)
  (h1 : total_highway_miles = 210)
  (h2 : total_city_miles = 54)
  (h3 : highway_mpg = 35)
  (h4 : total_gallons = 9) :
  (total_city_miles / (total_gallons - total_highway_miles / highway_mpg)) = 18 :=
by sorry

end NUMINAMATH_CALUDE_jensen_family_mileage_l3314_331443


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l3314_331412

theorem geometric_sequence_middle_term (a b c : ℝ) : 
  (∃ r : ℝ, b = a * r ∧ c = b * r) →  -- a, b, c form a geometric sequence
  a = 5 + 2 * Real.sqrt 6 →
  c = 5 - 2 * Real.sqrt 6 →
  b = 1 ∨ b = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l3314_331412


namespace NUMINAMATH_CALUDE_definite_integral_x_squared_plus_sin_l3314_331406

open Real MeasureTheory

theorem definite_integral_x_squared_plus_sin : 
  ∫ x in (-1)..1, (x^2 + Real.sin x) = 2/3 := by sorry

end NUMINAMATH_CALUDE_definite_integral_x_squared_plus_sin_l3314_331406


namespace NUMINAMATH_CALUDE_four_fold_f_application_l3314_331402

-- Define the function f
noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z ^ 2 else -(z ^ 2)

-- State the theorem
theorem four_fold_f_application :
  f (f (f (f (2 + 2*I)))) = -16777216 := by
  sorry

end NUMINAMATH_CALUDE_four_fold_f_application_l3314_331402


namespace NUMINAMATH_CALUDE_set_equality_implies_m_zero_l3314_331455

theorem set_equality_implies_m_zero (m : ℝ) : 
  ({3, m} : Set ℝ) = ({3*m, 3} : Set ℝ) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_m_zero_l3314_331455


namespace NUMINAMATH_CALUDE_correct_stratified_sample_teaching_l3314_331458

/-- Represents the composition of staff in a school -/
structure SchoolStaff where
  total : ℕ
  administrative : ℕ
  teaching : ℕ
  support : ℕ

/-- Calculates the number of teaching staff to be included in a stratified sample -/
def stratifiedSampleTeaching (staff : SchoolStaff) (sampleSize : ℕ) : ℕ :=
  (staff.teaching * sampleSize) / staff.total

/-- Theorem stating the correct number of teaching staff in the stratified sample -/
theorem correct_stratified_sample_teaching (staff : SchoolStaff) (sampleSize : ℕ) :
  staff.total = 200 ∧ 
  staff.administrative = 24 ∧ 
  staff.teaching = 10 * staff.support ∧
  staff.teaching + staff.support + staff.administrative = staff.total ∧
  sampleSize = 50 →
  stratifiedSampleTeaching staff sampleSize = 40 := by
  sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_teaching_l3314_331458


namespace NUMINAMATH_CALUDE_carrot_count_l3314_331457

/-- The number of carrots initially on the scale -/
def initial_carrots : ℕ := 20

/-- The total weight of carrots in grams -/
def total_weight : ℕ := 3640

/-- The average weight of remaining carrots in grams -/
def avg_weight_remaining : ℕ := 180

/-- The average weight of removed carrots in grams -/
def avg_weight_removed : ℕ := 190

/-- The number of removed carrots -/
def removed_carrots : ℕ := 4

theorem carrot_count : 
  total_weight = (initial_carrots - removed_carrots) * avg_weight_remaining + 
                 removed_carrots * avg_weight_removed := by
  sorry

end NUMINAMATH_CALUDE_carrot_count_l3314_331457


namespace NUMINAMATH_CALUDE_total_blue_balloons_l3314_331459

theorem total_blue_balloons (joan_balloons sally_balloons jessica_balloons : ℕ) 
  (h1 : joan_balloons = 9)
  (h2 : sally_balloons = 5)
  (h3 : jessica_balloons = 2) :
  joan_balloons + sally_balloons + jessica_balloons = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_blue_balloons_l3314_331459


namespace NUMINAMATH_CALUDE_trapezoid_halving_line_iff_condition_l3314_331471

/-- A trapezoid with bases a and b, and legs c and d. -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d
  parallel_bases : a ≠ b → a < b

/-- The condition for a line to halve both perimeter and area of a trapezoid. -/
def halvingLineCondition (t : Trapezoid) : Prop :=
  (t.c + t.d) / 2 = (t.a + t.b) / 2 + Real.sqrt ((t.a^2 + t.b^2) / 2) ∨ t.a = t.b

/-- Theorem: A line parallel to the bases halves both perimeter and area of a trapezoid
    if and only if the halving line condition is satisfied. -/
theorem trapezoid_halving_line_iff_condition (t : Trapezoid) :
  ∃ (x : ℝ), 0 < x ∧ x < t.c ∧ x < t.d ∧
    (x + x + t.a + t.b = (t.a + t.b + t.c + t.d) / 2) ∧
    (x * (t.a + t.b) = (t.a + t.b) * t.c / 2) ↔
  halvingLineCondition t :=
sorry

end NUMINAMATH_CALUDE_trapezoid_halving_line_iff_condition_l3314_331471


namespace NUMINAMATH_CALUDE_total_amount_paid_l3314_331480

def grape_quantity : ℕ := 7
def grape_rate : ℕ := 70
def mango_quantity : ℕ := 9
def mango_rate : ℕ := 55

theorem total_amount_paid :
  grape_quantity * grape_rate + mango_quantity * mango_rate = 985 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_l3314_331480


namespace NUMINAMATH_CALUDE_point_on_graph_l3314_331484

/-- A point (x, y) lies on the graph of y = 2x - 1 -/
def lies_on_graph (x y : ℝ) : Prop := y = 2 * x - 1

/-- The point (2, 3) lies on the graph of y = 2x - 1 -/
theorem point_on_graph : lies_on_graph 2 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_graph_l3314_331484


namespace NUMINAMATH_CALUDE_flax_acreage_is_80_l3314_331469

/-- Represents the acreage of a farm with sunflowers and flax -/
structure FarmAcreage where
  total : ℕ
  flax : ℕ
  sunflowers : ℕ
  total_eq : total = flax + sunflowers
  sunflower_excess : sunflowers = flax + 80

/-- The theorem stating that for a 240-acre farm with the given conditions, 
    the flax acreage is 80 acres -/
theorem flax_acreage_is_80 (farm : FarmAcreage) 
    (h : farm.total = 240) : farm.flax = 80 := by
  sorry

end NUMINAMATH_CALUDE_flax_acreage_is_80_l3314_331469


namespace NUMINAMATH_CALUDE_intersection_values_l3314_331400

-- Define the complex plane
variable (z : ℂ)

-- Define the equation |z - 4| = 3|z + 4|
def equation (z : ℂ) : Prop := Complex.abs (z - 4) = 3 * Complex.abs (z + 4)

-- Define the intersection condition
def intersects_once (k : ℝ) : Prop :=
  ∃! z, equation z ∧ Complex.abs z = k

-- Theorem statement
theorem intersection_values :
  ∀ k, intersects_once k → k = 2 ∨ k = 14 :=
sorry

end NUMINAMATH_CALUDE_intersection_values_l3314_331400


namespace NUMINAMATH_CALUDE_existence_of_special_fractions_l3314_331419

theorem existence_of_special_fractions : 
  ∃ (a b c d : ℕ), (a : ℚ) / b + (c : ℚ) / d = 1 ∧ (a : ℚ) / d + (d : ℚ) / b = 2008 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_fractions_l3314_331419


namespace NUMINAMATH_CALUDE_max_consecutive_sum_15_l3314_331430

/-- The sum of the first n positive integers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- A sequence of n consecutive positive integers starting from k -/
def consecutive_sum (n k : ℕ) : ℕ := n * k + triangular_number n

theorem max_consecutive_sum_15 :
  (∃ (n : ℕ), n > 0 ∧ consecutive_sum n 1 = 15) ∧
  (∀ (m : ℕ), m > 5 → consecutive_sum m 1 > 15) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_15_l3314_331430


namespace NUMINAMATH_CALUDE_davids_biology_mark_l3314_331426

def marks_english : ℕ := 45
def marks_mathematics : ℕ := 35
def marks_physics : ℕ := 52
def marks_chemistry : ℕ := 47
def average_marks : ℚ := 46.8

theorem davids_biology_mark (marks_biology : ℕ) :
  (marks_english + marks_mathematics + marks_physics + marks_chemistry + marks_biology : ℚ) / 5 = average_marks →
  marks_biology = 55 := by
sorry

end NUMINAMATH_CALUDE_davids_biology_mark_l3314_331426


namespace NUMINAMATH_CALUDE_square_plate_nails_l3314_331496

/-- Calculates the total number of unique nails on a square plate -/
def total_nails (nails_per_side : ℕ) : ℕ :=
  nails_per_side * 4 - 4

/-- Theorem stating that a square plate with 25 nails per side has 96 unique nails -/
theorem square_plate_nails :
  total_nails 25 = 96 := by
  sorry

#eval total_nails 25  -- This should output 96

end NUMINAMATH_CALUDE_square_plate_nails_l3314_331496


namespace NUMINAMATH_CALUDE_mixture_composition_l3314_331442

theorem mixture_composition (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hsum : x + y = 100) :
  0.4 * x + 0.5 * y = 47 → x = 30 :=
by sorry

end NUMINAMATH_CALUDE_mixture_composition_l3314_331442


namespace NUMINAMATH_CALUDE_stick_cutting_theorem_l3314_331465

/-- Represents a marked stick with cuts -/
structure MarkedStick :=
  (length : ℕ)
  (left_interval : ℕ)
  (right_interval : ℕ)

/-- Counts the number of segments of a given length in a marked stick -/
def count_segments (stick : MarkedStick) (segment_length : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that a 240 cm stick marked as described yields 12 pieces of 3 cm -/
theorem stick_cutting_theorem :
  let stick : MarkedStick := ⟨240, 7, 6⟩
  count_segments stick 3 = 12 := by sorry

end NUMINAMATH_CALUDE_stick_cutting_theorem_l3314_331465


namespace NUMINAMATH_CALUDE_trapezoid_properties_l3314_331494

structure Trapezoid where
  EF : ℝ
  GH : ℝ
  EG : ℝ
  FH : ℝ
  height : ℝ

def perimeter (t : Trapezoid) : ℝ := t.EF + t.GH + t.EG + t.FH

theorem trapezoid_properties (t : Trapezoid) 
  (h1 : t.EF = 60)
  (h2 : t.GH = 30)
  (h3 : t.EG = 40)
  (h4 : t.FH = 50)
  (h5 : t.height = 24) :
  perimeter t = 191 ∧ t.EG = 51 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_properties_l3314_331494


namespace NUMINAMATH_CALUDE_jimmy_win_probability_remainder_mod_1000_l3314_331466

/-- Probability of rolling an odd number on a single die -/
def prob_odd_single : ℚ := 3/4

/-- Probability of Jimmy winning a single game -/
def prob_jimmy_win : ℚ := 1 - prob_odd_single^2

/-- Probability of Jimmy winning exactly k out of n games -/
def prob_jimmy_win_k_of_n (k n : ℕ) : ℚ :=
  Nat.choose n k * prob_jimmy_win^k * (1 - prob_jimmy_win)^(n - k)

/-- Probability of Jimmy winning 3 games before Jacob wins 3 games -/
def prob_jimmy_wins_3_first : ℚ :=
  prob_jimmy_win_k_of_n 3 3 +
  prob_jimmy_win_k_of_n 3 4 +
  prob_jimmy_win_k_of_n 3 5

theorem jimmy_win_probability :
  prob_jimmy_wins_3_first = 201341 / 2^19 :=
sorry

theorem remainder_mod_1000 :
  (201341 : ℤ) + 19 ≡ 360 [ZMOD 1000] :=
sorry

end NUMINAMATH_CALUDE_jimmy_win_probability_remainder_mod_1000_l3314_331466


namespace NUMINAMATH_CALUDE_shelter_new_pets_l3314_331450

theorem shelter_new_pets (initial_dogs : ℕ) (initial_cats : ℕ) (initial_lizards : ℕ)
  (dog_adoption_rate : ℚ) (cat_adoption_rate : ℚ) (lizard_adoption_rate : ℚ)
  (pets_after_month : ℕ) :
  initial_dogs = 30 →
  initial_cats = 28 →
  initial_lizards = 20 →
  dog_adoption_rate = 1/2 →
  cat_adoption_rate = 1/4 →
  lizard_adoption_rate = 1/5 →
  pets_after_month = 65 →
  ∃ new_pets : ℕ,
    new_pets = 13 ∧
    pets_after_month = 
      (initial_dogs - initial_dogs * dog_adoption_rate).floor +
      (initial_cats - initial_cats * cat_adoption_rate).floor +
      (initial_lizards - initial_lizards * lizard_adoption_rate).floor +
      new_pets :=
by
  sorry

end NUMINAMATH_CALUDE_shelter_new_pets_l3314_331450


namespace NUMINAMATH_CALUDE_quadratic_root_existence_l3314_331420

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_root_existence (a b c : ℝ) (ha : a ≠ 0) :
  ∃ x : ℝ, x > 0.6 ∧ x < 0.7 ∧ quadratic_function a b c x = 0 :=
by
  have h1 : quadratic_function a b c 0.6 < 0 := by sorry
  have h2 : quadratic_function a b c 0.7 > 0 := by sorry
  sorry

#check quadratic_root_existence

end NUMINAMATH_CALUDE_quadratic_root_existence_l3314_331420


namespace NUMINAMATH_CALUDE_chrysler_has_23_floors_l3314_331437

/-- The number of floors in the Leeward Center -/
def leeward_floors : ℕ := sorry

/-- The number of floors in the Chrysler Building -/
def chrysler_floors : ℕ := sorry

/-- The Chrysler Building has 11 more floors than the Leeward Center -/
axiom chrysler_leeward_difference : chrysler_floors = leeward_floors + 11

/-- The total number of floors in both buildings is 35 -/
axiom total_floors : leeward_floors + chrysler_floors = 35

/-- Theorem: The Chrysler Building has 23 floors -/
theorem chrysler_has_23_floors : chrysler_floors = 23 := by sorry

end NUMINAMATH_CALUDE_chrysler_has_23_floors_l3314_331437


namespace NUMINAMATH_CALUDE_interval_intersection_l3314_331499

theorem interval_intersection (x : ℝ) : 
  (2 < 4*x ∧ 4*x < 3 ∧ 2 < 5*x ∧ 5*x < 3) ↔ (1/2 < x ∧ x < 3/5) :=
sorry

end NUMINAMATH_CALUDE_interval_intersection_l3314_331499


namespace NUMINAMATH_CALUDE_random_selection_more_representative_l3314_331432

/-- Represents a student in the school -/
structure Student where
  grade : ℕ
  gender : Bool

/-- Represents the entire student population of the school -/
def StudentPopulation := List Student

/-- Represents a sample of students -/
def StudentSample := List Student

/-- Function to check if a sample is representative of the population -/
def isRepresentative (population : StudentPopulation) (sample : StudentSample) : Prop :=
  -- Definition of what makes a sample representative
  sorry

/-- Function to randomly select students from various grades -/
def randomSelectFromGrades (population : StudentPopulation) (sampleSize : ℕ) : StudentSample :=
  -- Implementation of random selection from various grades
  sorry

/-- Function to select students from a single class -/
def selectFromSingleClass (population : StudentPopulation) (sampleSize : ℕ) : StudentSample :=
  -- Implementation of selection from a single class
  sorry

/-- Function to select students of a single gender -/
def selectSingleGender (population : StudentPopulation) (sampleSize : ℕ) : StudentSample :=
  -- Implementation of selection of a single gender
  sorry

/-- Theorem stating that random selection from various grades is more representative -/
theorem random_selection_more_representative 
  (population : StudentPopulation) (sampleSize : ℕ) : 
  isRepresentative population (randomSelectFromGrades population sampleSize) ∧
  ¬isRepresentative population (selectFromSingleClass population sampleSize) ∧
  ¬isRepresentative population (selectSingleGender population sampleSize) :=
by
  sorry


end NUMINAMATH_CALUDE_random_selection_more_representative_l3314_331432


namespace NUMINAMATH_CALUDE_function_value_proof_l3314_331414

theorem function_value_proof : 
  ∀ f : ℝ → ℝ, 
  (∀ x, f x = (x - 3) * (x + 4)) → 
  f 29 = 170 → 
  ∃ x, f x = 170 ∧ x = 13 := by
sorry

end NUMINAMATH_CALUDE_function_value_proof_l3314_331414


namespace NUMINAMATH_CALUDE_complex_number_magnitude_l3314_331413

theorem complex_number_magnitude (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 15)
  (h2 : Complex.abs (z + 3 * w) = 3)
  (h3 : Complex.abs (z - w) = 6) :
  Complex.abs w ^ 2 = 3.375 := by sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_l3314_331413


namespace NUMINAMATH_CALUDE_fraction_simplification_l3314_331486

theorem fraction_simplification : (252 : ℚ) / 8820 * 21 = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3314_331486


namespace NUMINAMATH_CALUDE_transformed_curve_equation_l3314_331441

/-- Given a curve and a scaling transformation, prove the equation of the transformed curve -/
theorem transformed_curve_equation (x y x' y' : ℝ) :
  (x^2 / 4 - y^2 = 1) →
  (x' = x / 2) →
  (y' = 2 * y) →
  (x'^2 - y'^2 / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_transformed_curve_equation_l3314_331441


namespace NUMINAMATH_CALUDE_a_can_be_any_real_l3314_331463

theorem a_can_be_any_real (a b c d : ℝ) 
  (h1 : (a / b) ^ 2 < (c / d) ^ 2)
  (h2 : b ≠ 0)
  (h3 : d ≠ 0)
  (h4 : c = -d) :
  ∃ (x : ℝ), x = a ∧ (x < 0 ∨ x = 0 ∨ x > 0) :=
by sorry

end NUMINAMATH_CALUDE_a_can_be_any_real_l3314_331463


namespace NUMINAMATH_CALUDE_train_journey_equation_l3314_331468

/-- Represents the equation for a train journey where:
    - x is the distance in km
    - The speed increases from 160 km/h to 200 km/h
    - The travel time reduces by 2.5 hours
-/
theorem train_journey_equation (x : ℝ) : x / 160 - x / 200 = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_equation_l3314_331468


namespace NUMINAMATH_CALUDE_multiples_of_4_and_5_between_100_and_350_l3314_331481

theorem multiples_of_4_and_5_between_100_and_350 : 
  (Finset.filter (fun n => n % 4 = 0 ∧ n % 5 = 0 ∧ 100 < n ∧ n < 350) (Finset.range 350)).card = 12 :=
by sorry

end NUMINAMATH_CALUDE_multiples_of_4_and_5_between_100_and_350_l3314_331481


namespace NUMINAMATH_CALUDE_no_solution_system_l3314_331436

/-- Proves that the system of equations 3x - 4y = 5 and 6x - 8y = 7 has no solution -/
theorem no_solution_system :
  ¬ ∃ (x y : ℝ), (3 * x - 4 * y = 5) ∧ (6 * x - 8 * y = 7) := by
sorry

end NUMINAMATH_CALUDE_no_solution_system_l3314_331436


namespace NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l3314_331454

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 - 4*x + 3 = 0
def equation2 (y : ℝ) : Prop := 4*(2*y - 5)^2 = (3*y - 1)^2

-- Theorem for the first equation
theorem solutions_equation1 : 
  (∃ x : ℝ, equation1 x) ↔ (equation1 1 ∧ equation1 3) :=
sorry

-- Theorem for the second equation
theorem solutions_equation2 : 
  (∃ y : ℝ, equation2 y) ↔ (equation2 9 ∧ equation2 (11/7)) :=
sorry

end NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l3314_331454


namespace NUMINAMATH_CALUDE_tangent_and_zeros_theorem_l3314_331434

noncomputable section

-- Define the functions
def f (x : ℝ) := 2 * x^3 - 3 * x^2 + 1
def g (k x : ℝ) := k * x + 1 - Real.log x
def h (k x : ℝ) := min (f x) (g k x)

-- Define the theorem
theorem tangent_and_zeros_theorem :
  -- Part 1: Tangent condition
  (∀ a : ℝ, (∃! t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
    (f t₁ - (-4) = (6 * t₁^2 - 6 * t₁) * (t₁ - a)) ∧
    (f t₂ - (-4) = (6 * t₂^2 - 6 * t₂) * (t₂ - a)))
   ↔ (a = -1 ∨ a = 7/2)) ∧
  -- Part 2: Zeros condition
  (∀ k : ℝ, (∃! x₁ x₂ x₃ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ 0 < x₃ ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    h k x₁ = 0 ∧ h k x₂ = 0 ∧ h k x₃ = 0)
   ↔ (0 < k ∧ k < Real.exp (-2))) := by
  sorry


end NUMINAMATH_CALUDE_tangent_and_zeros_theorem_l3314_331434


namespace NUMINAMATH_CALUDE_correct_proportions_l3314_331498

/-- Represents the count of shirts for each color --/
structure ShirtCounts where
  yellow : ℕ
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Represents the proportion of shirts for each color --/
structure ShirtProportions where
  yellow : ℚ
  red : ℚ
  blue : ℚ
  green : ℚ

/-- Given shirt counts, calculates the correct proportions --/
def calculateProportions (counts : ShirtCounts) : ShirtProportions :=
  let total := counts.yellow + counts.red + counts.blue + counts.green
  { yellow := counts.yellow / total
  , red := counts.red / total
  , blue := counts.blue / total
  , green := counts.green / total }

/-- Theorem stating that given the specific shirt counts, the calculated proportions are correct --/
theorem correct_proportions (counts : ShirtCounts)
  (h1 : counts.yellow = 8)
  (h2 : counts.red = 4)
  (h3 : counts.blue = 2)
  (h4 : counts.green = 2) :
  let props := calculateProportions counts
  props.yellow = 1/2 ∧ props.red = 1/4 ∧ props.blue = 1/8 ∧ props.green = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_correct_proportions_l3314_331498


namespace NUMINAMATH_CALUDE_smallest_n_with_eight_and_terminating_l3314_331447

/-- A function that checks if a positive integer contains the digit 8 -/
def containsEight (n : ℕ+) : Prop := sorry

/-- A function that checks if the reciprocal of a positive integer is a terminating decimal -/
def isTerminatingDecimal (n : ℕ+) : Prop := ∃ (a b : ℕ), n = 2^a * 5^b

theorem smallest_n_with_eight_and_terminating : 
  (∀ m : ℕ+, m < 8 → ¬(containsEight m ∧ isTerminatingDecimal m)) ∧ 
  (containsEight 8 ∧ isTerminatingDecimal 8) := by sorry

end NUMINAMATH_CALUDE_smallest_n_with_eight_and_terminating_l3314_331447


namespace NUMINAMATH_CALUDE_red_area_after_four_changes_l3314_331467

/-- Represents the fraction of red area remaining after one execution of the process -/
def remaining_fraction : ℚ := 8 / 9

/-- The number of times the process is executed -/
def process_iterations : ℕ := 4

/-- Calculates the fraction of the original area that remains red after n iterations -/
def red_area_fraction (n : ℕ) : ℚ := remaining_fraction ^ n

theorem red_area_after_four_changes :
  red_area_fraction process_iterations = 4096 / 6561 := by
  sorry

end NUMINAMATH_CALUDE_red_area_after_four_changes_l3314_331467


namespace NUMINAMATH_CALUDE_custard_pies_sold_is_five_l3314_331478

/-- Represents the bakery sales problem --/
structure BakerySales where
  pumpkin_slices_per_pie : ℕ
  custard_slices_per_pie : ℕ
  pumpkin_price_per_slice : ℕ
  custard_price_per_slice : ℕ
  pumpkin_pies_sold : ℕ
  total_revenue : ℕ

/-- Calculates the number of custard pies sold --/
def custard_pies_sold (bs : BakerySales) : ℕ :=
  sorry

/-- Theorem stating that the number of custard pies sold is 5 --/
theorem custard_pies_sold_is_five (bs : BakerySales)
  (h1 : bs.pumpkin_slices_per_pie = 8)
  (h2 : bs.custard_slices_per_pie = 6)
  (h3 : bs.pumpkin_price_per_slice = 5)
  (h4 : bs.custard_price_per_slice = 6)
  (h5 : bs.pumpkin_pies_sold = 4)
  (h6 : bs.total_revenue = 340) :
  custard_pies_sold bs = 5 :=
sorry

end NUMINAMATH_CALUDE_custard_pies_sold_is_five_l3314_331478


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l3314_331444

/-- Given a rectangle with an inscribed circle of radius 7 and a length-to-width ratio of 3:1, 
    prove that its area is 588. -/
theorem rectangle_area_with_inscribed_circle (r w l : ℝ) : 
  r = 7 ∧ w = 2 * r ∧ l = 3 * w → l * w = 588 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l3314_331444


namespace NUMINAMATH_CALUDE_cosA_value_triangle_area_l3314_331401

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Part I
theorem cosA_value (t : Triangle) 
  (h1 : t.a^2 = 3 * t.b * t.c) 
  (h2 : Real.sin t.A = Real.sin t.C) : 
  Real.cos t.A = 1/6 := by
sorry

-- Part II
theorem triangle_area (t : Triangle) 
  (h1 : t.a^2 = 3 * t.b * t.c) 
  (h2 : t.A = π/4) 
  (h3 : t.a = 3) : 
  (1/2) * t.b * t.c * Real.sin t.A = (3/4) * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_cosA_value_triangle_area_l3314_331401


namespace NUMINAMATH_CALUDE_shift_increasing_interval_l3314_331424

-- Define a function f
variable (f : ℝ → ℝ)

-- Define what it means for f to be increasing on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem shift_increasing_interval
  (h : IncreasingOn f (-2) 3) :
  IncreasingOn (fun x ↦ f (x + 5)) (-7) (-2) :=
sorry

end NUMINAMATH_CALUDE_shift_increasing_interval_l3314_331424


namespace NUMINAMATH_CALUDE_closest_fraction_l3314_331472

def medals_won : ℚ := 24 / 150

def fractions : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]

theorem closest_fraction :
  ∃ (f : ℚ), f ∈ fractions ∧
  ∀ (g : ℚ), g ∈ fractions → |medals_won - f| ≤ |medals_won - g| ∧
  f = 1/6 :=
sorry

end NUMINAMATH_CALUDE_closest_fraction_l3314_331472


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l3314_331476

theorem sum_of_a_and_b (a b : ℝ) (h : Real.sqrt (a + 2) + (b - 3)^2 = 0) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l3314_331476


namespace NUMINAMATH_CALUDE_tom_found_four_seashells_today_l3314_331431

/-- The number of seashells Tom found yesterday -/
def yesterdays_seashells : ℕ := 7

/-- The total number of seashells Tom found -/
def total_seashells : ℕ := 11

/-- The number of seashells Tom found today -/
def todays_seashells : ℕ := total_seashells - yesterdays_seashells

theorem tom_found_four_seashells_today : todays_seashells = 4 := by
  sorry

end NUMINAMATH_CALUDE_tom_found_four_seashells_today_l3314_331431


namespace NUMINAMATH_CALUDE_total_sales_amount_theorem_l3314_331409

def weight_deviations : List ℤ := [-4, -1, -2, 2, 3, 4, 7, 1]
def qualification_criterion : ℤ := 4
def price_per_bag : ℚ := 86/10

def is_qualified (deviation : ℤ) : Bool :=
  deviation.natAbs ≤ qualification_criterion

theorem total_sales_amount_theorem :
  (weight_deviations.filter is_qualified).length * price_per_bag = 602/10 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_amount_theorem_l3314_331409


namespace NUMINAMATH_CALUDE_tan_alpha_values_l3314_331427

theorem tan_alpha_values (α : ℝ) :
  5 * Real.sin (2 * α) + 5 * Real.cos (2 * α) + 1 = 0 →
  Real.tan α = 3 ∨ Real.tan α = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_values_l3314_331427


namespace NUMINAMATH_CALUDE_longest_chord_in_circle_l3314_331495

theorem longest_chord_in_circle (r : ℝ) (h : r = 3) : 
  ∃ (c : ℝ), c = 6 ∧ ∀ (chord : ℝ), chord ≤ c :=
sorry

end NUMINAMATH_CALUDE_longest_chord_in_circle_l3314_331495


namespace NUMINAMATH_CALUDE_magician_trick_min_digits_l3314_331405

/-- The minimum number of digits required for the magician's trick -/
def min_digits : ℕ := 101

/-- The number of possible two-digit combinations -/
def two_digit_combinations (n : ℕ) : ℕ := (n - 1) * (10^(n - 2))

/-- The total number of possible arrangements -/
def total_arrangements (n : ℕ) : ℕ := 10^n

/-- Theorem stating that 101 is the minimum number of digits required for the magician's trick -/
theorem magician_trick_min_digits :
  (∀ n : ℕ, n ≥ min_digits → two_digit_combinations n ≥ total_arrangements n) ∧
  (∀ n : ℕ, n < min_digits → two_digit_combinations n < total_arrangements n) :=
sorry

end NUMINAMATH_CALUDE_magician_trick_min_digits_l3314_331405


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3314_331410

theorem x_squared_minus_y_squared (x y : ℝ) 
  (eq1 : x + y = 4) 
  (eq2 : 2 * x - 2 * y = 1) : 
  x^2 - y^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3314_331410


namespace NUMINAMATH_CALUDE_line_with_definite_slope_line_equation_through_two_points_l3314_331435

-- Statement B
theorem line_with_definite_slope (m : ℝ) :
  ∃ (k : ℝ), ∀ (x y : ℝ), m * x + y - 2 = 0 → y = k * x + (2 : ℝ) :=
sorry

-- Statement D
theorem line_equation_through_two_points (x₁ y₁ x₂ y₂ : ℝ) (h : x₁ ≠ x₂) :
  ∀ (x y : ℝ), y - y₁ = ((y₂ - y₁) / (x₂ - x₁)) * (x - x₁) →
  ∃ (m b : ℝ), y = m * x + b :=
sorry

end NUMINAMATH_CALUDE_line_with_definite_slope_line_equation_through_two_points_l3314_331435


namespace NUMINAMATH_CALUDE_four_integers_with_average_five_l3314_331425

theorem four_integers_with_average_five (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (a + b + c + d : ℚ) / 4 = 5 →
  (max a (max b (max c d)) - min a (min b (min c d)) : ℕ) = 
    max a (max b (max c d)) - min a (min b (min c d)) →
  ((a + b + c + d) - max a (max b (max c d)) - min a (min b (min c d)) : ℚ) / 2 = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_four_integers_with_average_five_l3314_331425


namespace NUMINAMATH_CALUDE_bella_items_after_purchase_l3314_331479

def total_items (marbles frisbees deck_cards action_figures : ℕ) : ℕ :=
  marbles + frisbees + deck_cards + action_figures

theorem bella_items_after_purchase : 
  ∀ (marbles frisbees deck_cards action_figures : ℕ),
    marbles = 60 →
    marbles = 2 * frisbees →
    frisbees = deck_cards + 20 →
    marbles = 5 * action_figures →
    total_items (marbles + (2 * marbles) / 5)
                (frisbees + (2 * frisbees) / 5)
                (deck_cards + (2 * deck_cards) / 5)
                (action_figures + action_figures / 3) = 156 := by
  sorry

#check bella_items_after_purchase

end NUMINAMATH_CALUDE_bella_items_after_purchase_l3314_331479


namespace NUMINAMATH_CALUDE_pencil_packaging_remainder_l3314_331490

theorem pencil_packaging_remainder : 48305312 % 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencil_packaging_remainder_l3314_331490


namespace NUMINAMATH_CALUDE_divide_8900_by_6_and_4_l3314_331492

theorem divide_8900_by_6_and_4 : (8900 / 6) / 4 = 370.8333333333333 := by
  sorry

end NUMINAMATH_CALUDE_divide_8900_by_6_and_4_l3314_331492


namespace NUMINAMATH_CALUDE_combined_return_percentage_l3314_331421

def investment1 : ℝ := 500
def investment2 : ℝ := 1500
def return1 : ℝ := 0.07
def return2 : ℝ := 0.23

def total_investment : ℝ := investment1 + investment2
def total_return : ℝ := investment1 * return1 + investment2 * return2

theorem combined_return_percentage :
  (total_return / total_investment) * 100 = 19 := by sorry

end NUMINAMATH_CALUDE_combined_return_percentage_l3314_331421


namespace NUMINAMATH_CALUDE_division_theorem_l3314_331448

theorem division_theorem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = divisor * quotient + remainder →
  dividend = 162 →
  divisor = 17 →
  remainder = 9 →
  quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_theorem_l3314_331448


namespace NUMINAMATH_CALUDE_line_circle_intersection_k_range_l3314_331488

theorem line_circle_intersection_k_range 
  (k : ℝ) 
  (line : ℝ → ℝ → Prop)
  (circle : ℝ → ℝ → Prop)
  (M N : ℝ × ℝ) :
  (∀ x y, line x y ↔ y = k * x + 3) →
  (∀ x y, circle x y ↔ (x - 3)^2 + (y - 2)^2 = 4) →
  (line M.1 M.2 ∧ circle M.1 M.2) →
  (line N.1 N.2 ∧ circle N.1 N.2) →
  ((M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 12) →
  -3/4 ≤ k ∧ k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_k_range_l3314_331488


namespace NUMINAMATH_CALUDE_line_intersection_l3314_331416

theorem line_intersection :
  ∀ (x y : ℚ),
  (12 * x - 3 * y = 33) →
  (8 * x + 2 * y = 18) →
  (x = 29/12 ∧ y = -2/3) :=
by
  sorry

end NUMINAMATH_CALUDE_line_intersection_l3314_331416


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3314_331418

-- Problem 1
theorem problem_1 : (-10) - (-4) + 5 = -1 := by sorry

-- Problem 2
theorem problem_2 : (-72) * (2/3 - 1/4 - 5/6) = 30 := by sorry

-- Problem 3
theorem problem_3 : -3^2 - (-2)^3 * (-1)^4 + Real.rpow 27 (1/3) = 2 := by sorry

-- Problem 4
theorem problem_4 : 5 + 4 * (Real.sqrt 6 - 2) - 4 * (Real.sqrt 6 - 1) = 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3314_331418


namespace NUMINAMATH_CALUDE_divisibility_of_n_squared_plus_n_plus_two_l3314_331449

theorem divisibility_of_n_squared_plus_n_plus_two (n : ℕ) :
  (∃ k : ℕ, n^2 + n + 2 = 2 * k) ∧ (∃ m : ℕ, ¬(∃ l : ℕ, m^2 + m + 2 = 5 * l)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_n_squared_plus_n_plus_two_l3314_331449


namespace NUMINAMATH_CALUDE_percentage_difference_l3314_331464

theorem percentage_difference : 
  (60 * (50 / 100) * (40 / 100)) - (70 * (60 / 100) * (50 / 100)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3314_331464


namespace NUMINAMATH_CALUDE_f_is_linear_l3314_331487

/-- A function representing the total price of masks based on quantity -/
def f (x : ℝ) : ℝ := 0.9 * x

/-- The unit price of a mask in yuan -/
def unit_price : ℝ := 0.9

/-- Theorem stating that f is a linear function -/
theorem f_is_linear : 
  ∃ (m b : ℝ), ∀ x, f x = m * x + b :=
sorry

end NUMINAMATH_CALUDE_f_is_linear_l3314_331487


namespace NUMINAMATH_CALUDE_rowing_coach_votes_l3314_331403

theorem rowing_coach_votes (num_coaches : ℕ) (votes_per_coach : ℕ) (coaches_per_voter : ℕ) : 
  num_coaches = 36 → 
  votes_per_coach = 5 → 
  coaches_per_voter = 3 → 
  (num_coaches * votes_per_coach) / coaches_per_voter = 60 := by
  sorry

end NUMINAMATH_CALUDE_rowing_coach_votes_l3314_331403


namespace NUMINAMATH_CALUDE_square_diff_simplification_l3314_331489

theorem square_diff_simplification (a : ℝ) : (a + 1)^2 - a^2 = 2*a + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_simplification_l3314_331489


namespace NUMINAMATH_CALUDE_problem_statement_l3314_331408

-- Define the function f and its derivative g
variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)

-- Define the conditions
axiom f_diff : ∀ x, HasDerivAt f (g x) x
axiom f_even : ∀ x, f (3/2 - 2*x) = f (3/2 + 2*x)
axiom g_even : ∀ x, g (2 + x) = g (2 - x)

-- State the theorem to be proved
theorem problem_statement :
  (f (-1) = f 4) ∧ (g (-1/2) = 0) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3314_331408


namespace NUMINAMATH_CALUDE_average_of_z_multiples_l3314_331473

/-- The average of z, 4z, 10z, 22z, and 46z is 16.6z -/
theorem average_of_z_multiples (z : ℝ) : 
  (z + 4*z + 10*z + 22*z + 46*z) / 5 = 16.6 * z := by
  sorry

end NUMINAMATH_CALUDE_average_of_z_multiples_l3314_331473


namespace NUMINAMATH_CALUDE_min_triangle_area_l3314_331461

/-- Given a triangle ABC with side AB = 2 and 2/sin(A) + 1/tan(B) = 2√3, 
    its area is greater than or equal to 2√3/3 -/
theorem min_triangle_area (A B C : ℝ) (h1 : 0 < A ∧ A < π) (h2 : 0 < B ∧ B < π) 
  (h3 : 0 < C ∧ C < π) (h4 : A + B + C = π) (h5 : Real.sin A ≠ 0) (h6 : Real.tan B ≠ 0)
  (h7 : 2 / Real.sin A + 1 / Real.tan B = 2 * Real.sqrt 3) :
  1 / 2 * 2 * Real.sin C ≥ 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_min_triangle_area_l3314_331461


namespace NUMINAMATH_CALUDE_freight_train_speed_l3314_331411

/-- Proves that the speed of the freight train is 50 km/hr given the problem conditions --/
theorem freight_train_speed 
  (distance : ℝ) 
  (speed_difference : ℝ) 
  (express_speed : ℝ) 
  (time : ℝ) 
  (h1 : distance = 390) 
  (h2 : speed_difference = 30) 
  (h3 : express_speed = 80) 
  (h4 : time = 3) 
  (h5 : distance = (express_speed * time) + ((express_speed - speed_difference) * time)) : 
  express_speed - speed_difference = 50 := by
  sorry

end NUMINAMATH_CALUDE_freight_train_speed_l3314_331411


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3314_331440

/-- An arithmetic sequence with its partial sums -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- S_n is the sum of the first n terms

/-- Given an arithmetic sequence with S_10 = 12 and S_20 = 17, prove S_30 = 15 -/
theorem arithmetic_sequence_sum (a : ArithmeticSequence)
    (h1 : a.S 10 = 12)
    (h2 : a.S 20 = 17) :
  a.S 30 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3314_331440


namespace NUMINAMATH_CALUDE_dana_weekend_earnings_l3314_331445

def dana_earnings (hourly_rate : ℝ) (commission_rate : ℝ) 
  (friday_hours : ℝ) (friday_sales : ℝ)
  (saturday_hours : ℝ) (saturday_sales : ℝ)
  (sunday_hours : ℝ) (sunday_sales : ℝ) : ℝ :=
  let total_hours := friday_hours + saturday_hours + sunday_hours
  let total_sales := friday_sales + saturday_sales + sunday_sales
  let hourly_earnings := hourly_rate * total_hours
  let commission_earnings := commission_rate * total_sales
  hourly_earnings + commission_earnings

theorem dana_weekend_earnings :
  dana_earnings 13 0.05 9 800 10 1000 3 300 = 391 := by
  sorry

end NUMINAMATH_CALUDE_dana_weekend_earnings_l3314_331445


namespace NUMINAMATH_CALUDE_scooter_gain_percent_l3314_331460

/-- Calculate the gain percent from a scooter sale -/
theorem scooter_gain_percent 
  (purchase_price : ℝ) 
  (repair_cost : ℝ) 
  (selling_price : ℝ) 
  (h1 : purchase_price = 800)
  (h2 : repair_cost = 200)
  (h3 : selling_price = 1200) : 
  (selling_price - (purchase_price + repair_cost)) / (purchase_price + repair_cost) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_scooter_gain_percent_l3314_331460


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l3314_331429

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^48 + i^96 + i^144 = 3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l3314_331429


namespace NUMINAMATH_CALUDE_age_difference_l3314_331404

theorem age_difference (A B C : ℕ) : A + B = B + C + 14 → A = C + 14 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3314_331404


namespace NUMINAMATH_CALUDE_street_lamp_combinations_l3314_331456

/-- The number of lamps in the row -/
def total_lamps : ℕ := 12

/-- The number of lamps that can be turned off -/
def lamps_to_turn_off : ℕ := 3

/-- The number of valid positions to insert turned-off lamps -/
def valid_positions : ℕ := total_lamps - lamps_to_turn_off - 1

theorem street_lamp_combinations : 
  (valid_positions.choose lamps_to_turn_off) = 56 := by
  sorry

#eval valid_positions.choose lamps_to_turn_off

end NUMINAMATH_CALUDE_street_lamp_combinations_l3314_331456


namespace NUMINAMATH_CALUDE_intersection_area_theorem_l3314_331485

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle defined by its vertices -/
structure Rectangle where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Represents a circle defined by its center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Calculates the area of intersection between a rectangle and a circle -/
def intersectionArea (r : Rectangle) (c : Circle) : ℝ := sorry

/-- The main theorem stating the area of intersection -/
theorem intersection_area_theorem (r : Rectangle) (c : Circle) : 
  r.v1 = ⟨3, 9⟩ → 
  r.v2 = ⟨20, 9⟩ → 
  r.v3 = ⟨20, -6⟩ → 
  r.v4 = ⟨3, -6⟩ → 
  c.center = ⟨3, -6⟩ → 
  c.radius = 5 → 
  intersectionArea r c = 25 * Real.pi / 4 := by sorry

end NUMINAMATH_CALUDE_intersection_area_theorem_l3314_331485


namespace NUMINAMATH_CALUDE_horner_method_v3_l3314_331497

def f (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def horner_v3 (a₆ a₅ a₄ a₃ : ℝ) (x : ℝ) : ℝ :=
  ((a₆ * x + a₅) * x + a₄) * x + a₃

theorem horner_method_v3 :
  horner_v3 3 5 6 79 (-4) = -57 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l3314_331497


namespace NUMINAMATH_CALUDE_isabelle_concert_savings_l3314_331439

/-- Calculates the number of weeks Isabelle must work to afford concert tickets for herself and her brothers. -/
theorem isabelle_concert_savings (isabelle_ticket : ℕ) (brother_ticket : ℕ) (isabelle_savings : ℕ) (brothers_savings : ℕ) (weekly_earnings : ℕ) : 
  isabelle_ticket = 20 →
  brother_ticket = 10 →
  isabelle_savings = 5 →
  brothers_savings = 5 →
  weekly_earnings = 3 →
  (isabelle_ticket + 2 * brother_ticket - isabelle_savings - brothers_savings) / weekly_earnings = 10 := by
sorry

end NUMINAMATH_CALUDE_isabelle_concert_savings_l3314_331439


namespace NUMINAMATH_CALUDE_incorrect_step_is_count_bacteria_l3314_331451

/-- Represents a step in the bacterial counting experiment -/
inductive ExperimentStep
  | PrepMedium
  | SpreadSamples
  | Incubate
  | CountBacteria

/-- Represents a range of bacterial counts -/
structure CountRange where
  lower : ℕ
  upper : ℕ

/-- Defines the correct count range for bacterial counting -/
def correct_count_range : CountRange := { lower := 30, upper := 300 }

/-- Defines whether a step is correct in the experiment -/
def is_correct_step (step : ExperimentStep) : Prop :=
  match step with
  | ExperimentStep.PrepMedium => True
  | ExperimentStep.SpreadSamples => True
  | ExperimentStep.Incubate => True
  | ExperimentStep.CountBacteria => False

/-- Theorem stating that the CountBacteria step is the incorrect one -/
theorem incorrect_step_is_count_bacteria :
  ∃ (step : ExperimentStep), ¬(is_correct_step step) ↔ step = ExperimentStep.CountBacteria :=
sorry

end NUMINAMATH_CALUDE_incorrect_step_is_count_bacteria_l3314_331451


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3314_331475

theorem rectangle_dimensions (perimeter : ℝ) (area : ℝ) 
  (h_perimeter : perimeter = 26) (h_area : area = 42) :
  ∃ (length width : ℝ),
    length + width = perimeter / 2 ∧
    length * width = area ∧
    length = 7 ∧
    width = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3314_331475


namespace NUMINAMATH_CALUDE_percentage_of_students_with_birds_l3314_331482

/-- Given a school with 500 students where 75 students own birds,
    prove that 15% of the students own birds. -/
theorem percentage_of_students_with_birds :
  ∀ (total_students : ℕ) (students_with_birds : ℕ),
    total_students = 500 →
    students_with_birds = 75 →
    (students_with_birds : ℚ) / total_students * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_students_with_birds_l3314_331482


namespace NUMINAMATH_CALUDE_transform_equivalence_l3314_331483

-- Define the original function
def f : ℝ → ℝ := sorry

-- Define the transformation
def transform (f : ℝ → ℝ) (x : ℝ) : ℝ := f (2*x - 2) + 1

-- Define the horizontal shift
def shift_right (f : ℝ → ℝ) (x : ℝ) : ℝ := f (2*(x - 1))

-- Define the vertical shift
def shift_up (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 1

-- Theorem statement
theorem transform_equivalence (x : ℝ) : 
  transform f x = shift_up (shift_right (f ∘ (fun x => 2*x))) x := by sorry

end NUMINAMATH_CALUDE_transform_equivalence_l3314_331483


namespace NUMINAMATH_CALUDE_total_length_is_16cm_l3314_331407

/-- Represents the dimensions of a rectangle in centimeters -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents the segments removed from the rectangle -/
structure RemovedSegments where
  long_side : ℝ
  short_side_ends : ℝ

/-- Represents the split in the remaining short side -/
structure SplitSegment where
  distance_from_middle : ℝ

/-- Calculates the total length of segments after modifications -/
def total_length_after_modifications (rect : Rectangle) (removed : RemovedSegments) (split : SplitSegment) : ℝ :=
  let remaining_long_side := rect.length - removed.long_side
  let remaining_short_side := rect.width - 2 * removed.short_side_ends
  let split_segment := min split.distance_from_middle (remaining_short_side / 2)
  remaining_long_side + remaining_short_side + 2 * removed.short_side_ends

/-- Theorem stating that the total length of segments after modifications is 16 cm -/
theorem total_length_is_16cm (rect : Rectangle) (removed : RemovedSegments) (split : SplitSegment)
    (h1 : rect.length = 10)
    (h2 : rect.width = 5)
    (h3 : removed.long_side = 3)
    (h4 : removed.short_side_ends = 2)
    (h5 : split.distance_from_middle = 1) :
  total_length_after_modifications rect removed split = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_length_is_16cm_l3314_331407


namespace NUMINAMATH_CALUDE_collinear_opposite_vectors_l3314_331462

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b

/-- Two vectors have opposite directions if their scalar multiple is negative -/
def opposite_directions (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k < 0 ∧ a = k • b

theorem collinear_opposite_vectors (m : ℝ) :
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (1, m)
  collinear a b ∧ opposite_directions a b → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_collinear_opposite_vectors_l3314_331462


namespace NUMINAMATH_CALUDE_no_real_roots_implies_nonzero_sum_l3314_331428

theorem no_real_roots_implies_nonzero_sum (a b c : ℝ) : 
  a ≠ 0 → 
  (∀ x : ℝ, a * x^2 + b * x + c ≠ 0) → 
  a^3 + a * b + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_implies_nonzero_sum_l3314_331428


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3314_331453

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | (x - 1) * (x - 2*a + 1) < 0}
  (a = 1 → S = ∅) ∧
  (a > 1 → S = {x : ℝ | 1 < x ∧ x < 2*a - 1}) ∧
  (a < 1 → S = {x : ℝ | 2*a - 1 < x ∧ x < 1}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3314_331453


namespace NUMINAMATH_CALUDE_probability_square_or_circle_l3314_331438

theorem probability_square_or_circle (total : ℕ) (triangles squares circles : ℕ) : 
  total = triangles + squares + circles →
  triangles = 4 →
  squares = 3 →
  circles = 5 →
  (squares + circles : ℚ) / total = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_square_or_circle_l3314_331438


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_l3314_331433

/-- If the terminal side of angle α passes through the point (-1, 2) in the Cartesian coordinate system, then sin α = (2√5) / 5 -/
theorem sin_alpha_for_point (α : Real) : 
  (∃ (x y : Real), x = -1 ∧ y = 2 ∧ x = Real.cos α ∧ y = Real.sin α) →
  Real.sin α = (2 * Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_for_point_l3314_331433


namespace NUMINAMATH_CALUDE_houses_on_block_l3314_331415

/-- Given a block of houses where:
  * The total number of pieces of junk mail for the block is 24
  * Each house receives 4 pieces of junk mail
  This theorem proves that there are 6 houses on the block. -/
theorem houses_on_block (total_mail : ℕ) (mail_per_house : ℕ) 
  (h1 : total_mail = 24) 
  (h2 : mail_per_house = 4) : 
  total_mail / mail_per_house = 6 := by
  sorry

end NUMINAMATH_CALUDE_houses_on_block_l3314_331415


namespace NUMINAMATH_CALUDE_mabels_daisy_problem_l3314_331493

/-- Given a number of daisies and petals per daisy, calculate the total number of petals --/
def total_petals (num_daisies : ℕ) (petals_per_daisy : ℕ) : ℕ :=
  num_daisies * petals_per_daisy

/-- Given an initial number of daisies and the number of daisies given away,
    calculate the remaining number of daisies --/
def remaining_daisies (initial_daisies : ℕ) (daisies_given : ℕ) : ℕ :=
  initial_daisies - daisies_given

theorem mabels_daisy_problem (initial_daisies : ℕ) (petals_per_daisy : ℕ) (daisies_given : ℕ)
    (h1 : initial_daisies = 5)
    (h2 : petals_per_daisy = 8)
    (h3 : daisies_given = 2) :
  total_petals (remaining_daisies initial_daisies daisies_given) petals_per_daisy = 24 := by
  sorry


end NUMINAMATH_CALUDE_mabels_daisy_problem_l3314_331493


namespace NUMINAMATH_CALUDE_oscar_wins_three_l3314_331491

/-- Represents a player in the chess tournament -/
inductive Player : Type
| Lucy : Player
| Maya : Player
| Oscar : Player

/-- The number of games won by a player -/
def games_won (p : Player) : ℕ :=
  match p with
  | Player.Lucy => 5
  | Player.Maya => 2
  | Player.Oscar => 3  -- This is what we want to prove

/-- The number of games lost by a player -/
def games_lost (p : Player) : ℕ :=
  match p with
  | Player.Lucy => 4
  | Player.Maya => 2
  | Player.Oscar => 4

/-- The total number of games played in the tournament -/
def total_games : ℕ := (games_won Player.Lucy + games_lost Player.Lucy +
                        games_won Player.Maya + games_lost Player.Maya +
                        games_won Player.Oscar + games_lost Player.Oscar) / 2

theorem oscar_wins_three :
  (∀ p : Player, games_won p + games_lost p = total_games) ∧
  (games_won Player.Lucy + games_won Player.Maya + games_won Player.Oscar =
   games_lost Player.Lucy + games_lost Player.Maya + games_lost Player.Oscar) →
  games_won Player.Oscar = 3 := by
  sorry

end NUMINAMATH_CALUDE_oscar_wins_three_l3314_331491
