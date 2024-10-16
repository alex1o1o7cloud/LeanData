import Mathlib

namespace NUMINAMATH_CALUDE_distinct_integers_with_swapped_digits_l4172_417255

def has_2n_digits (x : ℕ) (n : ℕ) : Prop :=
  10^(2*n - 1) ≤ x ∧ x < 10^(2*n)

def first_n_digits (x : ℕ) (n : ℕ) : ℕ :=
  x / 10^n

def last_n_digits (x : ℕ) (n : ℕ) : ℕ :=
  x % 10^n

theorem distinct_integers_with_swapped_digits (n : ℕ) (a b : ℕ) :
  n > 0 →
  a ≠ b →
  a > 0 ∧ b > 0 →
  has_2n_digits a n →
  has_2n_digits b n →
  a ∣ b →
  first_n_digits a n = last_n_digits b n →
  last_n_digits a n = first_n_digits b n →
  ((a = 2442 ∧ b = 4224) ∨ (a = 3993 ∧ b = 9339)) :=
by sorry

end NUMINAMATH_CALUDE_distinct_integers_with_swapped_digits_l4172_417255


namespace NUMINAMATH_CALUDE_sum_of_digits_problem_l4172_417261

/-- S(n) is the sum of digits of a natural number n -/
def S (n : ℕ) : ℕ := sorry

/-- If n is a natural number such that n + S(n) = 2009, then n = 1990 -/
theorem sum_of_digits_problem (n : ℕ) (h : n + S n = 2009) : n = 1990 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_problem_l4172_417261


namespace NUMINAMATH_CALUDE_sculpture_cost_in_inr_l4172_417266

/-- Exchange rate from British pounds to Indian rupees -/
def gbp_to_inr : ℚ := 20

/-- Exchange rate from British pounds to Namibian dollars -/
def gbp_to_nad : ℚ := 18

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℚ := 360

/-- Theorem stating the equivalent cost of the sculpture in Indian rupees -/
theorem sculpture_cost_in_inr :
  (sculpture_cost_nad / gbp_to_nad) * gbp_to_inr = 400 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_in_inr_l4172_417266


namespace NUMINAMATH_CALUDE_sector_area_l4172_417207

/-- The area of a circular sector with central angle 72° and radius 5 is 5π. -/
theorem sector_area (S : ℝ) : S = 5 * Real.pi := by
  -- Given:
  -- Central angle is 72°
  -- Radius is 5
  sorry

end NUMINAMATH_CALUDE_sector_area_l4172_417207


namespace NUMINAMATH_CALUDE_truck_load_problem_l4172_417298

/-- Proves that the number of crates loaded yesterday is 10 --/
theorem truck_load_problem :
  let truck_capacity : ℕ := 13500
  let box_weight : ℕ := 100
  let box_count : ℕ := 100
  let crate_weight : ℕ := 60
  let sack_weight : ℕ := 50
  let sack_count : ℕ := 50
  let bag_weight : ℕ := 40
  let bag_count : ℕ := 10

  let total_box_weight := box_weight * box_count
  let total_sack_weight := sack_weight * sack_count
  let total_bag_weight := bag_weight * bag_count

  let remaining_weight := truck_capacity - (total_box_weight + total_sack_weight + total_bag_weight)

  ∃ crate_count : ℕ, crate_count * crate_weight = remaining_weight ∧ crate_count = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_truck_load_problem_l4172_417298


namespace NUMINAMATH_CALUDE_marble_fraction_after_tripling_l4172_417285

theorem marble_fraction_after_tripling (total : ℚ) (h1 : total > 0) : 
  let blue := (2/3) * total
  let red := total - blue
  let new_red := 3 * red
  let new_total := blue + new_red
  new_red / new_total = 3/5 := by
sorry

end NUMINAMATH_CALUDE_marble_fraction_after_tripling_l4172_417285


namespace NUMINAMATH_CALUDE_jackson_pbj_sandwiches_l4172_417232

/-- The number of weeks in the school year -/
def school_weeks : ℕ := 36

/-- The number of days per week Jackson could eat peanut butter and jelly sandwiches -/
def pbj_days_per_week : ℕ := 2

/-- The number of Wednesdays Jackson missed -/
def missed_wednesdays : ℕ := 1

/-- The number of Fridays Jackson missed -/
def missed_fridays : ℕ := 2

/-- The total number of peanut butter and jelly sandwiches Jackson ate -/
def total_pbj_sandwiches : ℕ := school_weeks * pbj_days_per_week - (missed_wednesdays + missed_fridays)

theorem jackson_pbj_sandwiches :
  total_pbj_sandwiches = 69 := by
  sorry

end NUMINAMATH_CALUDE_jackson_pbj_sandwiches_l4172_417232


namespace NUMINAMATH_CALUDE_steps_per_floor_l4172_417238

/-- Proves that the number of steps across each floor is 30 --/
theorem steps_per_floor (
  num_floors : ℕ) 
  (steps_per_second : ℕ)
  (total_time : ℕ)
  (h1 : num_floors = 9)
  (h2 : steps_per_second = 3)
  (h3 : total_time = 90)
  : (steps_per_second * total_time) / num_floors = 30 := by
  sorry

end NUMINAMATH_CALUDE_steps_per_floor_l4172_417238


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_60_429_l4172_417201

theorem gcd_lcm_sum_60_429 : Nat.gcd 60 429 + Nat.lcm 60 429 = 8583 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_60_429_l4172_417201


namespace NUMINAMATH_CALUDE_variable_prime_count_l4172_417242

/-- The number of primes between n^2 + 1 and n^2 + n is not constant for n > 1 -/
theorem variable_prime_count (n : ℕ) (h : n > 1) :
  ∃ m : ℕ, m > n ∧ 
  (Finset.filter (Nat.Prime) (Finset.range (n^2 + n - (n^2 + 2) + 1))).card ≠
  (Finset.filter (Nat.Prime) (Finset.range (m^2 + m - (m^2 + 2) + 1))).card :=
by sorry

end NUMINAMATH_CALUDE_variable_prime_count_l4172_417242


namespace NUMINAMATH_CALUDE_initial_working_hours_l4172_417245

/-- Given the following conditions:
    - 63 men initially work h hours per day to dig 30 m deep
    - To dig 50 m deep in 6 hours per day, 77 extra men are needed
    Prove that the initial working hours (h) is 8 hours per day -/
theorem initial_working_hours (initial_men : ℕ) (initial_depth : ℕ) 
  (target_depth : ℕ) (new_hours : ℕ) (extra_men : ℕ) :
  initial_men = 63 →
  initial_depth = 30 →
  target_depth = 50 →
  new_hours = 6 →
  extra_men = 77 →
  ∃ h : ℕ, 
    h * initial_men * target_depth = new_hours * (initial_men + extra_men) * initial_depth ∧
    h = 8 := by
  sorry


end NUMINAMATH_CALUDE_initial_working_hours_l4172_417245


namespace NUMINAMATH_CALUDE_bob_candy_count_l4172_417221

/-- Bob's share of items -/
structure BobsShare where
  chewing_gums : ℕ
  chocolate_bars : ℕ
  assorted_candies : ℕ

/-- Definition of Bob's relationship and actions -/
structure BobInfo where
  is_sams_neighbor : Prop
  accompanies_sam : Prop
  share : BobsShare

/-- Theorem stating the number of candies Bob got -/
theorem bob_candy_count (bob : BobInfo) 
  (h1 : bob.is_sams_neighbor)
  (h2 : bob.accompanies_sam)
  (h3 : bob.share.chewing_gums = 15)
  (h4 : bob.share.chocolate_bars = 20)
  (h5 : bob.share.assorted_candies = 15) : 
  bob.share.assorted_candies = 15 := by
  sorry

end NUMINAMATH_CALUDE_bob_candy_count_l4172_417221


namespace NUMINAMATH_CALUDE_chord_length_is_2_sqrt_2_l4172_417254

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The chord length of a circle intercepted by a line -/
def chordLength (c : Circle) (l : Line) : ℝ := sorry

/-- The given circle x^2 + y^2 - 4y = 0 -/
def givenCircle : Circle :=
  { center := (0, 2),
    radius := 2 }

/-- The line passing through the origin with slope 1 -/
def givenLine : Line :=
  { slope := 1,
    yIntercept := 0 }

/-- Theorem: The chord length of the given circle intercepted by the given line is 2√2 -/
theorem chord_length_is_2_sqrt_2 :
  chordLength givenCircle givenLine = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_chord_length_is_2_sqrt_2_l4172_417254


namespace NUMINAMATH_CALUDE_steps_on_sunday_l4172_417227

def target_average : ℕ := 9000
def days_in_week : ℕ := 7
def known_days : ℕ := 4
def friday_saturday_average : ℕ := 9050

def steps_known_days : List ℕ := [9100, 8300, 9200, 8900]

theorem steps_on_sunday (
  target_total : target_average * days_in_week = 63000)
  (known_total : steps_known_days.sum = 35500)
  (friday_saturday_total : friday_saturday_average * 2 = 18100)
  : 63000 - 35500 - 18100 = 9400 := by
  sorry

end NUMINAMATH_CALUDE_steps_on_sunday_l4172_417227


namespace NUMINAMATH_CALUDE_inequality_proof_l4172_417288

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  Real.sqrt (a * b) < (a - b) / (Real.log a - Real.log b) ∧ 
  (a - b) / (Real.log a - Real.log b) < (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4172_417288


namespace NUMINAMATH_CALUDE_max_quad_area_l4172_417246

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the lines
def Line1 (m x y : ℝ) : Prop := m*x - y + 1 = 0
def Line2 (m x y : ℝ) : Prop := x + m*y - m = 0

-- Define the quadrilateral area
def QuadArea (A B C D : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem max_quad_area :
  ∀ (m : ℝ) (A B C D : ℝ × ℝ),
    Circle A.1 A.2 ∧ Circle B.1 B.2 ∧ Circle C.1 C.2 ∧ Circle D.1 D.2 →
    Line1 m A.1 A.2 ∧ Line1 m C.1 C.2 →
    Line2 m B.1 B.2 ∧ Line2 m D.1 D.2 →
    QuadArea A B C D ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_quad_area_l4172_417246


namespace NUMINAMATH_CALUDE_dog_walking_homework_diff_l4172_417279

/-- Represents the time in minutes for various activities -/
structure ActivityTimes where
  total : ℕ
  homework : ℕ
  cleaning : ℕ
  trash : ℕ
  remaining : ℕ

/-- Calculates the time spent walking the dog -/
def walkingTime (t : ActivityTimes) : ℕ :=
  t.total - t.remaining - (t.homework + t.cleaning + t.trash)

/-- Theorem stating the difference between dog walking and homework time -/
theorem dog_walking_homework_diff (t : ActivityTimes) : 
  t.total = 120 ∧ 
  t.homework = 30 ∧ 
  t.cleaning = t.homework / 2 ∧ 
  t.trash = t.homework / 6 ∧ 
  t.remaining = 35 → 
  walkingTime t - t.homework = 5 := by
  sorry


end NUMINAMATH_CALUDE_dog_walking_homework_diff_l4172_417279


namespace NUMINAMATH_CALUDE_figure_b_cannot_be_assembled_l4172_417253

-- Define the basic rhombus
structure Rhombus :=
  (color1 : String)
  (color2 : String)

-- Define the operation of rotation
def rotate (r : Rhombus) : Rhombus := r

-- Define the larger figures
inductive LargerFigure
  | A
  | B
  | C
  | D

-- Define a function to check if a larger figure can be assembled
def can_assemble (figure : LargerFigure) (r : Rhombus) : Prop :=
  match figure with
  | LargerFigure.A => True
  | LargerFigure.B => False
  | LargerFigure.C => True
  | LargerFigure.D => True

-- Theorem statement
theorem figure_b_cannot_be_assembled (r : Rhombus) :
  ¬(can_assemble LargerFigure.B r) ∧
  (can_assemble LargerFigure.A r) ∧
  (can_assemble LargerFigure.C r) ∧
  (can_assemble LargerFigure.D r) :=
sorry

end NUMINAMATH_CALUDE_figure_b_cannot_be_assembled_l4172_417253


namespace NUMINAMATH_CALUDE_correct_subtraction_result_l4172_417273

theorem correct_subtraction_result (original : ℕ) (incorrect : ℕ) (h1 : original = 514) (h2 : incorrect = 913) :
  original - (incorrect - original) = 115 := by
  sorry

end NUMINAMATH_CALUDE_correct_subtraction_result_l4172_417273


namespace NUMINAMATH_CALUDE_player_a_not_losing_probability_l4172_417239

theorem player_a_not_losing_probability
  (p_win : ℝ)
  (p_draw : ℝ)
  (h_win : p_win = 0.3)
  (h_draw : p_draw = 0.5) :
  p_win + p_draw = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_player_a_not_losing_probability_l4172_417239


namespace NUMINAMATH_CALUDE_min_distance_to_circle_l4172_417224

/-- Line l in polar form -/
def line_l (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.cos θ + ρ * Real.sin θ + 4 = 0

/-- Circle C in Cartesian form -/
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*y = 0

/-- Distance between a point (ρ, θ) and its tangent to circle C -/
noncomputable def distance_to_tangent (ρ θ : ℝ) : ℝ :=
  sorry

/-- Theorem stating the minimum distance and its occurrence -/
theorem min_distance_to_circle (ρ θ : ℝ) :
  line_l ρ θ →
  distance_to_tangent ρ θ ≥ 2 ∧
  (distance_to_tangent ρ θ = 2 ↔ ρ = 2 ∧ θ = Real.pi) :=
  sorry

end NUMINAMATH_CALUDE_min_distance_to_circle_l4172_417224


namespace NUMINAMATH_CALUDE_not_divisible_by_4p_l4172_417206

theorem not_divisible_by_4p (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ¬ (4 * p ∣ (2 * p - 1)^(p - 1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_4p_l4172_417206


namespace NUMINAMATH_CALUDE_nora_muffin_sales_l4172_417296

/-- The number of cases of muffins Nora needs to sell to raise $120 -/
def cases_needed (packs_per_case : ℕ) (muffins_per_pack : ℕ) (price_per_muffin : ℕ) (target_amount : ℕ) : ℕ :=
  target_amount / (packs_per_case * muffins_per_pack * price_per_muffin)

/-- Proof that Nora needs to sell 5 cases of muffins to raise $120 -/
theorem nora_muffin_sales :
  cases_needed 3 4 2 120 = 5 := by
  sorry

end NUMINAMATH_CALUDE_nora_muffin_sales_l4172_417296


namespace NUMINAMATH_CALUDE_students_with_all_pets_l4172_417271

theorem students_with_all_pets (total_students : ℕ) 
  (dog_fraction : ℚ) (cat_fraction : ℚ)
  (other_pet_count : ℕ) (no_pet_count : ℕ)
  (only_dog_count : ℕ) (only_other_count : ℕ)
  (cat_and_other_count : ℕ) :
  total_students = 40 →
  dog_fraction = 5 / 8 →
  cat_fraction = 1 / 4 →
  other_pet_count = 8 →
  no_pet_count = 6 →
  only_dog_count = 12 →
  only_other_count = 3 →
  cat_and_other_count = 10 →
  (∃ (all_pets_count : ℕ),
    all_pets_count = 0 ∧
    total_students * dog_fraction = only_dog_count + all_pets_count + cat_and_other_count ∧
    total_students * cat_fraction = cat_and_other_count + all_pets_count ∧
    other_pet_count = only_other_count + all_pets_count + cat_and_other_count ∧
    total_students - no_pet_count = only_dog_count + only_other_count + all_pets_count + cat_and_other_count) :=
by
  sorry

end NUMINAMATH_CALUDE_students_with_all_pets_l4172_417271


namespace NUMINAMATH_CALUDE_boys_camp_total_l4172_417240

theorem boys_camp_total (total : ℕ) 
  (h1 : (total : ℚ) * (1/5) = (total : ℚ) * (20/100))  -- 20% of boys are from school A
  (h2 : (total : ℚ) * (1/5) * (3/10) = (total : ℚ) * (1/5) * (30/100))  -- 30% of boys from school A study science
  (h3 : (total : ℚ) * (1/5) * (7/10) = 49)  -- 49 boys are from school A but do not study science
  : total = 350 := by
sorry


end NUMINAMATH_CALUDE_boys_camp_total_l4172_417240


namespace NUMINAMATH_CALUDE_sphere_surface_area_l4172_417256

theorem sphere_surface_area (volume : ℝ) (h : volume = 72 * Real.pi) :
  let r := (3 * volume / (4 * Real.pi)) ^ (1/3)
  4 * Real.pi * r^2 = 36 * 2^(2/3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l4172_417256


namespace NUMINAMATH_CALUDE_f_2013_plus_f_neg_2014_l4172_417237

open Real

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic_2 (f : ℝ → ℝ) : Prop := ∀ x ≥ 0, f (x + 2) = f x

def matches_exp_minus_one_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 2 → f x = exp x - 1

theorem f_2013_plus_f_neg_2014 (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_periodic : is_periodic_2 f)
  (h_match : matches_exp_minus_one_on_interval f) :
  f 2013 + f (-2014) = exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_2013_plus_f_neg_2014_l4172_417237


namespace NUMINAMATH_CALUDE_purchase_equation_l4172_417223

/-- 
Given a group of people jointly purchasing an item, where:
- Contributing 8 units per person results in an excess of 3 units
- Contributing 7 units per person results in a shortage of 4 units
Prove that the number of people satisfies the equation 8x - 3 = 7x + 4
-/
theorem purchase_equation (x : ℕ) 
  (h1 : 8 * x - 3 = (8 * x - 3)) 
  (h2 : 7 * x + 4 = (7 * x + 4)) : 
  8 * x - 3 = 7 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_purchase_equation_l4172_417223


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l4172_417205

theorem initial_mean_calculation (n : ℕ) (wrong_value correct_value : ℝ) (new_mean : ℝ) :
  n = 50 ∧ 
  wrong_value = 23 ∧ 
  correct_value = 48 ∧ 
  new_mean = 41.5 →
  ∃ (initial_mean : ℝ),
    initial_mean * n + (correct_value - wrong_value) = new_mean * n ∧
    initial_mean = 41 :=
by sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l4172_417205


namespace NUMINAMATH_CALUDE_max_triangle_area_in_rectangle_l4172_417289

/-- The maximum area of a right triangle with a 30° angle inside a 12x5 rectangle -/
theorem max_triangle_area_in_rectangle :
  let rectangle_length : ℝ := 12
  let rectangle_width : ℝ := 5
  let angle : ℝ := 30 * π / 180  -- 30° in radians
  ∃ (triangle_area : ℝ),
    triangle_area = 25 * Real.sqrt 3 / 4 ∧
    ∀ (a : ℝ), a ≤ rectangle_width →
      a * (2 * a) / 2 ≤ triangle_area :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_area_in_rectangle_l4172_417289


namespace NUMINAMATH_CALUDE_stating_correct_equation_representation_l4172_417294

/-- Represents the distribution of people in a campus beautification activity -/
def campus_beautification (initial_weeding : ℕ) (initial_planting : ℕ) (total_support : ℕ) 
  (support_weeding : ℕ) : Prop :=
  let final_weeding := initial_weeding + support_weeding
  let final_planting := initial_planting + (total_support - support_weeding)
  final_weeding = 2 * final_planting

/-- 
Theorem stating that the equation correctly represents the final distribution
of people in the campus beautification activity.
-/
theorem correct_equation_representation 
  (initial_weeding : ℕ) (initial_planting : ℕ) (total_support : ℕ) (support_weeding : ℕ) :
  campus_beautification initial_weeding initial_planting total_support support_weeding →
  initial_weeding + support_weeding = 2 * (initial_planting + (total_support - support_weeding)) :=
by
  sorry

end NUMINAMATH_CALUDE_stating_correct_equation_representation_l4172_417294


namespace NUMINAMATH_CALUDE_pentagon_area_ratio_l4172_417267

-- Define the pentagon
structure Pentagon where
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  I : ℝ × ℝ
  J : ℝ × ℝ

def is_convex (p : Pentagon) : Prop := sorry

-- Define parallel lines
def parallel (a b c d : ℝ × ℝ) : Prop := sorry

-- Define angle measurement
def angle (a b c : ℝ × ℝ) : ℝ := sorry

-- Define distance between points
def distance (a b : ℝ × ℝ) : ℝ := sorry

-- Define area of a triangle
def triangle_area (a b c : ℝ × ℝ) : ℝ := sorry

theorem pentagon_area_ratio (p : Pentagon) :
  is_convex p →
  parallel p.F p.G p.I p.J →
  parallel p.G p.H p.F p.I →
  parallel p.G p.I p.H p.J →
  angle p.F p.G p.H = 120 * π / 180 →
  distance p.F p.G = 4 →
  distance p.G p.H = 6 →
  distance p.H p.J = 18 →
  (triangle_area p.F p.G p.H) / (triangle_area p.G p.H p.I) = 16 / 171 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_ratio_l4172_417267


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l4172_417249

theorem polynomial_division_theorem (x : ℝ) :
  x^4 + 3*x^3 - 17*x^2 + 8*x - 12 = (x - 3) * (x^3 + 6*x^2 + x + 11) + 21 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l4172_417249


namespace NUMINAMATH_CALUDE_sequence_on_line_is_arithmetic_l4172_417283

/-- Given a sequence {a_n} where (n, a_n) lies on the line y = 2x,
    prove that it is an arithmetic sequence with common difference 2 -/
theorem sequence_on_line_is_arithmetic (a : ℕ → ℝ) :
  (∀ n : ℕ, a n = 2 * n) →
  (∀ n : ℕ, a (n + 1) - a n = 2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_on_line_is_arithmetic_l4172_417283


namespace NUMINAMATH_CALUDE_division_problem_l4172_417241

theorem division_problem (remainder quotient divisor dividend : ℕ) : 
  remainder = 8 →
  divisor = 3 * quotient →
  divisor = 3 * remainder + 3 →
  dividend = divisor * quotient + remainder →
  dividend = 251 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l4172_417241


namespace NUMINAMATH_CALUDE_max_a_for_monotonic_increasing_l4172_417213

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

-- State the theorem
theorem max_a_for_monotonic_increasing (a : ℝ) :
  a > 0 →
  (∀ x ≥ 1, Monotone (fun x => f a x)) →
  a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_a_for_monotonic_increasing_l4172_417213


namespace NUMINAMATH_CALUDE_g_zero_at_seven_fifths_l4172_417204

def g (x : ℝ) : ℝ := 5 * x - 7

theorem g_zero_at_seven_fifths : g (7 / 5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_zero_at_seven_fifths_l4172_417204


namespace NUMINAMATH_CALUDE_johns_final_push_l4172_417292

/-- John's final push in a speed walking race -/
theorem johns_final_push (john_pace : ℝ) : 
  john_pace * 34 = 3.7 * 34 + (15 + 2) → john_pace = 4.2 := by
  sorry

end NUMINAMATH_CALUDE_johns_final_push_l4172_417292


namespace NUMINAMATH_CALUDE_circle_area_ratio_l4172_417228

-- Define the circles C and D
variables (C D : ℝ → Prop)

-- Define the radii of circles C and D
variables (r_C r_D : ℝ)

-- Define the common arc length
variable (L : ℝ)

-- State the theorem
theorem circle_area_ratio 
  (h1 : L = (60 / 360) * (2 * Real.pi * r_C)) 
  (h2 : L = (45 / 360) * (2 * Real.pi * r_D)) 
  (h3 : 2 * Real.pi * r_D = 2 * (2 * Real.pi * r_C)) :
  (Real.pi * r_D^2) / (Real.pi * r_C^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l4172_417228


namespace NUMINAMATH_CALUDE_price_difference_theorem_l4172_417218

-- Define the discounted price
def discounted_price : ℝ := 71.4

-- Define the discount rate
def discount_rate : ℝ := 0.15

-- Define the price increase rate
def increase_rate : ℝ := 0.25

-- Theorem statement
theorem price_difference_theorem :
  let original_price := discounted_price / (1 - discount_rate)
  let final_price := discounted_price * (1 + increase_rate)
  final_price - original_price = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_theorem_l4172_417218


namespace NUMINAMATH_CALUDE_thomas_can_be_faster_l4172_417236

/-- Represents a walker with a constant speed --/
structure Walker where
  speed : ℝ
  markers_passed : ℕ

/-- Thomas passes 5 markers in an hour --/
def thomas : Walker :=
  { speed := 5000, markers_passed := 5 }

/-- Jeremiah passes 6 markers in an hour --/
def jeremiah : Walker :=
  { speed := 6000, markers_passed := 6 }

/-- The distance between two consecutive markers in meters --/
def marker_distance : ℝ := 1000

/-- Theorem stating that it's possible for Thomas's speed to be greater than Jeremiah's --/
theorem thomas_can_be_faster (d : ℝ) (h : d > 0) : ∃ (t j : ℝ), 
  t > j ∧ 
  t ≤ thomas.speed + 2 * d ∧ 
  j ≥ jeremiah.speed - 2 * d :=
sorry

end NUMINAMATH_CALUDE_thomas_can_be_faster_l4172_417236


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_union_A_B_l4172_417215

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -4 < x ∧ x < 4}
def B : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}

-- Theorem for A ∩ B
theorem intersection_A_B :
  A ∩ B = {x : ℝ | (-4 < x ∧ x ≤ 1) ∨ (3 ≤ x ∧ x < 4)} := by sorry

-- Theorem for ∁_U (A ∪ B)
theorem complement_union_A_B :
  (A ∪ B)ᶜ = ∅ := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_union_A_B_l4172_417215


namespace NUMINAMATH_CALUDE_second_train_speed_l4172_417216

/-- Proves that the speed of the second train is 60 km/h given the conditions of the problem -/
theorem second_train_speed
  (first_train_speed : ℝ)
  (time_difference : ℝ)
  (meeting_distance : ℝ)
  (h1 : first_train_speed = 40)
  (h2 : time_difference = 1)
  (h3 : meeting_distance = 120) :
  let second_train_speed := meeting_distance / (meeting_distance / first_train_speed - time_difference)
  second_train_speed = 60 := by
sorry

end NUMINAMATH_CALUDE_second_train_speed_l4172_417216


namespace NUMINAMATH_CALUDE_square_vertex_locus_l4172_417235

/-- Represents a line in 2D plane with equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square with vertices A, B, C, D and center O -/
structure Square where
  A : Point
  B : Point
  C : Point
  D : Point
  O : Point

def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem square_vertex_locus 
  (a c o : Line) 
  (h_not_parallel : a.a * c.b ≠ a.b * c.a) :
  ∃ (F G H : ℝ),
    ∀ (ABCD : Square),
      on_line ABCD.A a → 
      on_line ABCD.C c → 
      on_line ABCD.O o → 
      (on_line ABCD.B ⟨F, G, H⟩ ∧ on_line ABCD.D ⟨F, G, H⟩) :=
sorry

end NUMINAMATH_CALUDE_square_vertex_locus_l4172_417235


namespace NUMINAMATH_CALUDE_segment_length_product_l4172_417217

theorem segment_length_product (b : ℝ) : 
  (∃ b₁ b₂ : ℝ, 
    (∀ b : ℝ, (((3*b - 7)^2 + (2*b + 1)^2 : ℝ) = 50) ↔ (b = b₁ ∨ b = b₂)) ∧ 
    (b₁ * b₂ = 0)) := by
  sorry

end NUMINAMATH_CALUDE_segment_length_product_l4172_417217


namespace NUMINAMATH_CALUDE_seventeen_in_both_competitions_l4172_417278

/-- The number of students who participated in both math and physics competitions -/
def students_in_both_competitions (total : ℕ) (math : ℕ) (physics : ℕ) (none : ℕ) : ℕ :=
  math + physics + none - total

/-- Theorem stating that 17 students participated in both competitions -/
theorem seventeen_in_both_competitions :
  students_in_both_competitions 37 30 20 4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_seventeen_in_both_competitions_l4172_417278


namespace NUMINAMATH_CALUDE_triangle_ABC_c_value_l4172_417226

/-- Triangle ABC with vertices A(0, 4), B(3, 0), and C(c, 6) has area 7 and 0 < c < 3 -/
def triangle_ABC (c : ℝ) : Prop :=
  let A : ℝ × ℝ := (0, 4)
  let B : ℝ × ℝ := (3, 0)
  let C : ℝ × ℝ := (c, 6)
  let area := (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))
  (area = 7) ∧ (0 < c) ∧ (c < 3)

/-- If triangle ABC satisfies the given conditions, then c = 2 -/
theorem triangle_ABC_c_value :
  ∀ c : ℝ, triangle_ABC c → c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_c_value_l4172_417226


namespace NUMINAMATH_CALUDE_specific_pyramid_volume_l4172_417270

/-- A pyramid with a parallelogram base and specific dimensions --/
structure Pyramid where
  base_side1 : ℝ
  base_side2 : ℝ
  base_diagonal : ℝ
  lateral_edge : ℝ

/-- The volume of the pyramid --/
def pyramid_volume (p : Pyramid) : ℝ := sorry

/-- Theorem stating the volume of the specific pyramid is 200 --/
theorem specific_pyramid_volume :
  let p : Pyramid := {
    base_side1 := 9,
    base_side2 := 10,
    base_diagonal := 11,
    lateral_edge := Real.sqrt 10
  }
  pyramid_volume p = 200 := by sorry

end NUMINAMATH_CALUDE_specific_pyramid_volume_l4172_417270


namespace NUMINAMATH_CALUDE_probability_one_success_out_of_three_l4172_417257

/-- The probability of passing a single computer test -/
def p : ℚ := 1 / 3

/-- The number of tests taken -/
def n : ℕ := 3

/-- The number of successful attempts -/
def k : ℕ := 1

/-- Binomial coefficient function -/
def binomial_coeff (n k : ℕ) : ℚ := (n.choose k : ℚ)

/-- The probability of passing exactly k tests out of n attempts -/
def probability_k_successes (n k : ℕ) (p : ℚ) : ℚ :=
  binomial_coeff n k * p^k * (1 - p)^(n - k)

theorem probability_one_success_out_of_three :
  probability_k_successes n k p = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_probability_one_success_out_of_three_l4172_417257


namespace NUMINAMATH_CALUDE_number_problem_l4172_417287

theorem number_problem (x : ℤ) : x + 14 = 56 → 3 * x = 126 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l4172_417287


namespace NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l4172_417293

theorem quadratic_roots_to_coefficients :
  ∀ (b c : ℝ), 
    (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 1 ∨ x = -2) →
    b = 1 ∧ c = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l4172_417293


namespace NUMINAMATH_CALUDE_parallelogram_area_10_20_l4172_417202

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 10 cm and height 20 cm is 200 square centimeters -/
theorem parallelogram_area_10_20 :
  parallelogram_area 10 20 = 200 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_10_20_l4172_417202


namespace NUMINAMATH_CALUDE_total_shirts_is_ten_l4172_417209

/-- Represents the total number of shirts sold by the retailer -/
def total_shirts : ℕ := 10

/-- Represents the number of initially sold shirts -/
def initial_shirts : ℕ := 3

/-- Represents the prices of the initially sold shirts -/
def initial_prices : List ℝ := [20, 22, 25]

/-- Represents the desired overall average price -/
def desired_average : ℝ := 20

/-- Represents the minimum average price of the remaining shirts -/
def min_remaining_average : ℝ := 19

/-- Theorem stating that the total number of shirts is 10 given the conditions -/
theorem total_shirts_is_ten :
  total_shirts = initial_shirts + (total_shirts - initial_shirts) ∧
  (List.sum initial_prices + min_remaining_average * (total_shirts - initial_shirts)) / total_shirts > desired_average :=
by sorry

end NUMINAMATH_CALUDE_total_shirts_is_ten_l4172_417209


namespace NUMINAMATH_CALUDE_ellipse_properties_l4172_417214

/-- An ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  eccentricity : ℝ
  right_focus_to_vertex : ℝ

/-- The standard form of an ellipse equation -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- A line in slope-intercept form -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem ellipse_properties (C : Ellipse) 
  (h1 : C.center = (0, 0))
  (h2 : C.foci_on_x_axis = true)
  (h3 : C.eccentricity = 1/2)
  (h4 : C.right_focus_to_vertex = 1) :
  (∃ (x y : ℝ), standard_equation 4 3 x y) ∧
  (∃ (l : Line) (A B : ℝ × ℝ), 
    (standard_equation 4 3 A.1 A.2) ∧
    (standard_equation 4 3 B.1 B.2) ∧
    (A.2 = l.slope * A.1 + l.intercept) ∧
    (B.2 = l.slope * B.1 + l.intercept) ∧
    (dot_product A B = 0)) ∧
  (∀ (m : ℝ), (∃ (k : ℝ), 
    ∃ (A B : ℝ × ℝ),
      (standard_equation 4 3 A.1 A.2) ∧
      (standard_equation 4 3 B.1 B.2) ∧
      (A.2 = k * A.1 + m) ∧
      (B.2 = k * B.1 + m) ∧
      (dot_product A B = 0)) ↔ 
    (m ≤ -2 * Real.sqrt 21 / 7 ∨ m ≥ 2 * Real.sqrt 21 / 7)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l4172_417214


namespace NUMINAMATH_CALUDE_cube_root_simplification_l4172_417222

theorem cube_root_simplification :
  Real.rpow (20^3 + 30^3 + 40^3) (1/3) = 10 * Real.rpow 99 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l4172_417222


namespace NUMINAMATH_CALUDE_valid_seven_digit_integers_l4172_417231

-- Define the recurrence relation
def a : ℕ → ℕ
  | 0 => 0  -- Base case (not used)
  | 1 => 4  -- a₁ = 4
  | 2 => 17 -- a₂ = 17
  | n + 3 => 4 * a (n + 2) + 2 * a (n + 1)

-- Theorem statement
theorem valid_seven_digit_integers : a 7 = 29776 := by
  sorry

end NUMINAMATH_CALUDE_valid_seven_digit_integers_l4172_417231


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_equals_27_l4172_417233

theorem x_squared_minus_y_squared_equals_27
  (x y : ℝ)
  (h1 : y + 6 = (x - 3)^2)
  (h2 : x + 6 = (y - 3)^2)
  (h3 : x ≠ y) :
  x^2 - y^2 = 27 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_equals_27_l4172_417233


namespace NUMINAMATH_CALUDE_root_in_interval_l4172_417282

-- Define the function
def f (x : ℝ) := x^3 - 2*x - 5

-- Theorem statement
theorem root_in_interval :
  (∃ x ∈ Set.Icc 2 3, f x = 0) →  -- root exists in [2,3]
  f 2.5 > 0 →                    -- f(2.5) > 0
  (∃ x ∈ Set.Ioo 2 2.5, f x = 0) -- root exists in (2,2.5)
  := by sorry

end NUMINAMATH_CALUDE_root_in_interval_l4172_417282


namespace NUMINAMATH_CALUDE_average_speed_two_hours_l4172_417262

/-- Calculates the average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) : 
  speed1 = 145 → speed2 = 60 → (speed1 + speed2) / 2 = 102.5 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_l4172_417262


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l4172_417250

theorem integer_roots_of_polynomial (a : ℤ) : 
  a = -4 →
  (∀ x : ℤ, x^4 - 16*x^3 + (81-2*a)*x^2 + (16*a-142)*x + a^2 - 21*a + 68 = 0 ↔ 
    x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 7) := by
  sorry

#check integer_roots_of_polynomial

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l4172_417250


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l4172_417299

theorem triangle_abc_properties (a b c A B C m : ℝ) :
  0 < A → A ≤ 2 * Real.pi / 3 →
  a > 0 → b > 0 → c > 0 →
  A + B + C = Real.pi →
  a^2 + b^2 - c^2 = Real.sqrt 3 * a * b →
  m = 2 * (Real.cos (A / 2))^2 - Real.sin B - 1 →
  (C = Real.pi / 6 ∧ -1 ≤ m ∧ m < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l4172_417299


namespace NUMINAMATH_CALUDE_samuel_initial_skittles_l4172_417276

/-- The number of friends Samuel gave Skittles to -/
def num_friends : ℕ := 4

/-- The number of Skittles each person (including Samuel) ate -/
def skittles_per_person : ℕ := 3

/-- The initial number of Skittles Samuel had -/
def initial_skittles : ℕ := num_friends * skittles_per_person + skittles_per_person

/-- Theorem stating that Samuel initially had 15 Skittles -/
theorem samuel_initial_skittles : initial_skittles = 15 := by
  sorry

end NUMINAMATH_CALUDE_samuel_initial_skittles_l4172_417276


namespace NUMINAMATH_CALUDE_ellen_chairs_count_l4172_417225

/-- The number of chairs Ellen bought at a garage sale -/
def num_chairs : ℕ := 180 / 15

/-- The cost of each chair in dollars -/
def chair_cost : ℕ := 15

/-- The total amount Ellen spent in dollars -/
def total_spent : ℕ := 180

theorem ellen_chairs_count :
  num_chairs = 12 ∧ chair_cost * num_chairs = total_spent :=
sorry

end NUMINAMATH_CALUDE_ellen_chairs_count_l4172_417225


namespace NUMINAMATH_CALUDE_g_comp_three_roots_l4172_417281

/-- The function g(x) = x^2 + 8x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 8*x + d

/-- The composition of g with itself -/
def g_comp (d : ℝ) (x : ℝ) : ℝ := g d (g d x)

/-- The statement that g(g(x)) has exactly 3 distinct real roots -/
def has_exactly_three_roots (d : ℝ) : Prop :=
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x : ℝ, g_comp d x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧
                    r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃

theorem g_comp_three_roots :
  ∀ d : ℝ, has_exactly_three_roots d ↔ d = -20 + 4 * Real.sqrt 14 ∨ d = -20 - 4 * Real.sqrt 14 :=
sorry

end NUMINAMATH_CALUDE_g_comp_three_roots_l4172_417281


namespace NUMINAMATH_CALUDE_fraction_value_l4172_417211

theorem fraction_value (a b c d : ℚ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 4 * d) :
  a * c / (b * d) = 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l4172_417211


namespace NUMINAMATH_CALUDE_inequalities_always_true_l4172_417229

theorem inequalities_always_true 
  (x y a b : ℝ) 
  (hx : x > 0) (hy : y > 0) (ha : a > 0) (hb : b > 0)
  (hxa : x ≤ a) (hyb : y ≤ b) : 
  (x + y ≤ a + b) ∧ 
  (x - y ≤ a - b) ∧ 
  (x * y ≤ a * b) ∧ 
  (x / y ≤ a / b) := by
sorry

end NUMINAMATH_CALUDE_inequalities_always_true_l4172_417229


namespace NUMINAMATH_CALUDE_sum_of_evens_between_1_and_31_l4172_417291

def sumOfEvens : ℕ → ℕ
  | 0 => 0
  | n + 1 => if (n + 1) % 2 = 0 ∧ n + 1 < 31 then n + 1 + sumOfEvens n else sumOfEvens n

theorem sum_of_evens_between_1_and_31 : sumOfEvens 30 = 240 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_evens_between_1_and_31_l4172_417291


namespace NUMINAMATH_CALUDE_no_perfect_squares_l4172_417268

theorem no_perfect_squares (n : ℕ+) : 
  ¬(∃ (a b c : ℕ), (2 * n^2 + 1 = a^2) ∧ (3 * n^2 + 1 = b^2) ∧ (6 * n^2 + 1 = c^2)) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_l4172_417268


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l4172_417212

theorem stewart_farm_sheep_count :
  ∀ (num_sheep num_horses : ℕ),
    (num_sheep : ℚ) / num_horses = 4 / 7 →
    num_horses * 230 = 12880 →
    num_sheep = 32 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l4172_417212


namespace NUMINAMATH_CALUDE_largest_n_binomial_sum_existence_n_6_largest_n_is_6_l4172_417200

theorem largest_n_binomial_sum (n : ℕ) : 
  (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) → n ≤ 6 :=
by sorry

theorem existence_n_6 : 
  Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 6 :=
by sorry

theorem largest_n_is_6 : 
  ∃ (n : ℕ), (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) ∧ 
  (∀ (m : ℕ), (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m) → m ≤ n) ∧
  n = 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_sum_existence_n_6_largest_n_is_6_l4172_417200


namespace NUMINAMATH_CALUDE_not_perfect_square_l4172_417286

theorem not_perfect_square : 
  ¬ ∃ n : ℕ, 5^2023 = n^2 ∧ 
  ∃ a : ℕ, 3^2021 = a^2 ∧
  ∃ b : ℕ, 7^2024 = b^2 ∧
  ∃ c : ℕ, 6^2025 = c^2 ∧
  ∃ d : ℕ, 8^2026 = d^2 :=
by sorry

end NUMINAMATH_CALUDE_not_perfect_square_l4172_417286


namespace NUMINAMATH_CALUDE_diameter_length_l4172_417280

/-- Represents a circle with diameter AB and perpendicular chord CD -/
structure Circle where
  AB : ℕ
  CD : ℕ
  is_two_digit : 10 ≤ AB ∧ AB < 100
  is_reversed : CD = (AB % 10) * 10 + (AB / 10)

/-- The distance OH is rational -/
def rational_OH (c : Circle) : Prop :=
  ∃ (q : ℚ), q > 0 ∧ q^2 * 4 = 99 * (c.AB / 10 - c.AB % 10) * (c.AB / 10 + c.AB % 10)

theorem diameter_length (c : Circle) (h : rational_OH c) : c.AB = 65 :=
sorry

end NUMINAMATH_CALUDE_diameter_length_l4172_417280


namespace NUMINAMATH_CALUDE_m_geq_two_l4172_417220

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Assume f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Given condition: f'(x) < x for all x ∈ ℝ
axiom f'_less_than_x : ∀ x, f' x < x

-- Define m as a real number
variable (m : ℝ)

-- Given inequality involving f
axiom f_inequality : f (4 - m) - f m ≥ 8 - 4 * m

-- Theorem to prove
theorem m_geq_two : m ≥ 2 := by sorry

end NUMINAMATH_CALUDE_m_geq_two_l4172_417220


namespace NUMINAMATH_CALUDE_football_match_problem_l4172_417295

/-- Represents a football team's match statistics -/
structure TeamStats :=
  (wins : ℕ)
  (draws : ℕ)
  (losses : ℕ)

/-- Calculate the total matches played by a team -/
def total_matches (team : TeamStats) : ℕ :=
  team.wins + team.draws + team.losses

/-- The football match problem -/
theorem football_match_problem 
  (home : TeamStats)
  (rival : TeamStats)
  (h1 : home.wins = 3)
  (h2 : home.draws = 4)
  (h3 : home.losses = 0)
  (h4 : rival.wins = 2 * home.wins)
  (h5 : rival.draws = 4)
  (h6 : rival.losses = 0) :
  total_matches home + total_matches rival = 17 :=
sorry

end NUMINAMATH_CALUDE_football_match_problem_l4172_417295


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l4172_417269

-- Problem 1
theorem problem_1 : (-1)^3 + Real.sqrt 4 - (2 - Real.sqrt 2)^0 = 0 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) : (a + 3) * (a - 3) - a * (a - 2) = 2 * a - 9 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l4172_417269


namespace NUMINAMATH_CALUDE_green_block_weight_l4172_417244

theorem green_block_weight (yellow_weight green_weight : ℝ) 
  (h1 : yellow_weight = 0.6)
  (h2 : yellow_weight = green_weight + 0.2) : 
  green_weight = 0.4 := by
sorry

end NUMINAMATH_CALUDE_green_block_weight_l4172_417244


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4172_417284

theorem arithmetic_sequence_problem (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 1 = 2 →                                            -- given: a_1 = 2
  a 3 + a 5 = 8 →                                      -- given: a_3 + a_5 = 8
  a 7 = 6 :=                                           -- to prove: a_7 = 6
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4172_417284


namespace NUMINAMATH_CALUDE_triangle_division_into_congruent_parts_l4172_417234

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (h₁ : a > 0)
  (h₂ : b > 0)
  (h₃ : c > 0)
  (h₄ : a + b > c)
  (h₅ : b + c > a)
  (h₆ : c + a > b)

-- Define congruence for triangles
def CongruentTriangles (t₁ t₂ : Triangle) : Prop :=
  t₁.a = t₂.a ∧ t₁.b = t₂.b ∧ t₁.c = t₂.c

-- Define a division of a triangle into five smaller triangles
structure TriangleDivision (t : Triangle) :=
  (t₁ t₂ t₃ t₄ t₅ : Triangle)

-- State the theorem
theorem triangle_division_into_congruent_parts (t : Triangle) :
  ∃ (d : TriangleDivision t), 
    CongruentTriangles d.t₁ d.t₂ ∧
    CongruentTriangles d.t₁ d.t₃ ∧
    CongruentTriangles d.t₁ d.t₄ ∧
    CongruentTriangles d.t₁ d.t₅ :=
sorry

end NUMINAMATH_CALUDE_triangle_division_into_congruent_parts_l4172_417234


namespace NUMINAMATH_CALUDE_divisible_by_ten_l4172_417297

theorem divisible_by_ten (n : ℕ) : ∃ k : ℤ, 3^(n+2) - 2^(n+2) + 3^n - 2^n = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_ten_l4172_417297


namespace NUMINAMATH_CALUDE_string_cheese_calculation_l4172_417251

/-- The number of string cheeses in each package for Kelly's kids' lunches. -/
def string_cheeses_per_package : ℕ := by sorry

theorem string_cheese_calculation (days_per_week : ℕ) (oldest_daily : ℕ) (youngest_daily : ℕ) 
  (weeks : ℕ) (packages : ℕ) (h1 : days_per_week = 5) (h2 : oldest_daily = 2) 
  (h3 : youngest_daily = 1) (h4 : weeks = 4) (h5 : packages = 2) : 
  string_cheeses_per_package = 30 := by sorry

end NUMINAMATH_CALUDE_string_cheese_calculation_l4172_417251


namespace NUMINAMATH_CALUDE_hexagon_trapezoid_height_l4172_417208

/-- Given a 9 × 16 rectangle cut into two congruent hexagons that can form a larger rectangle
    with width 12, prove that the height of the internal trapezoid in one hexagon is 12. -/
theorem hexagon_trapezoid_height (original_width : ℝ) (original_height : ℝ)
  (resultant_width : ℝ) (y : ℝ) :
  original_width = 16 ∧ original_height = 9 ∧ resultant_width = 12 →
  y = 12 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_trapezoid_height_l4172_417208


namespace NUMINAMATH_CALUDE_power_outage_duration_is_three_l4172_417265

/-- The duration of the power outage in hours -/
def power_outage_duration : ℝ := 3

/-- The temperature rise rate during the power outage in degrees per hour -/
def temperature_rise_rate : ℝ := 8

/-- The temperature decrease rate when the air conditioner is on in degrees per hour -/
def temperature_decrease_rate : ℝ := 4

/-- The time taken by the air conditioner to restore the temperature in hours -/
def air_conditioner_duration : ℝ := 6

/-- Theorem stating that the power outage duration is 3 hours -/
theorem power_outage_duration_is_three :
  power_outage_duration = temperature_rise_rate⁻¹ * temperature_decrease_rate * air_conditioner_duration :=
by sorry

end NUMINAMATH_CALUDE_power_outage_duration_is_three_l4172_417265


namespace NUMINAMATH_CALUDE_cary_shoe_savings_l4172_417274

def cost_of_shoes : ℕ := 120
def amount_saved : ℕ := 30
def earnings_per_lawn : ℕ := 5
def lawns_per_weekend : ℕ := 3

def weekends_needed : ℕ :=
  (cost_of_shoes - amount_saved) / (earnings_per_lawn * lawns_per_weekend)

theorem cary_shoe_savings : weekends_needed = 6 := by
  sorry

end NUMINAMATH_CALUDE_cary_shoe_savings_l4172_417274


namespace NUMINAMATH_CALUDE_multiple_of_p_l4172_417247

theorem multiple_of_p (p q : ℚ) (k : ℚ) : 
  p / q = 3 / 5 → kp + q = 11 → k = 2 := by sorry

end NUMINAMATH_CALUDE_multiple_of_p_l4172_417247


namespace NUMINAMATH_CALUDE_system_solution_l4172_417290

theorem system_solution :
  ∃! (x y : ℚ), 2 * x - 3 * y = 5 ∧ 4 * x - 6 * y = 10 ∧ x + 2 * y = 7 ∧ x = 31/7 ∧ y = 9/7 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4172_417290


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_eight_satisfies_inequality_exists_no_greater_value_l4172_417252

theorem greatest_value_quadratic_inequality :
  ∀ x : ℝ, x^2 - 12*x + 32 ≤ 0 → x ≤ 8 :=
by
  sorry

theorem eight_satisfies_inequality :
  8^2 - 12*8 + 32 = 0 :=
by
  sorry

theorem exists_no_greater_value :
  ¬∃ y : ℝ, y > 8 ∧ y^2 - 12*y + 32 ≤ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_eight_satisfies_inequality_exists_no_greater_value_l4172_417252


namespace NUMINAMATH_CALUDE_min_beta_delta_sum_l4172_417263

open Complex

/-- The complex number i -/
def i : ℂ := Complex.I

/-- The function g as defined in the problem -/
def g (β δ : ℂ) (z : ℂ) : ℂ := (3 + 2*i)*z^2 + β*z + δ

/-- The theorem statement -/
theorem min_beta_delta_sum :
  ∀ β δ : ℂ, (g β δ 1).im = 0 → (g β δ (-i)).im = 0 → 
  ∃ (min : ℝ), min = 2 * Real.sqrt 2 ∧ 
  ∀ β' δ' : ℂ, (g β' δ' 1).im = 0 → (g β' δ' (-i)).im = 0 → 
  Complex.abs β' + Complex.abs δ' ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_beta_delta_sum_l4172_417263


namespace NUMINAMATH_CALUDE_offer_price_per_year_is_half_l4172_417277

/-- Represents a magazine subscription offer -/
structure MagazineOffer where
  regularYearlyFee : ℕ
  offerYears : ℕ
  offerPrice : ℕ
  issuesPerYear : ℕ

/-- The Parents magazine offer -/
def parentsOffer : MagazineOffer :=
  { regularYearlyFee := 12
  , offerYears := 2
  , offerPrice := 12
  , issuesPerYear := 12
  }

/-- Theorem stating that the offer price per year is half of the regular price per year -/
theorem offer_price_per_year_is_half (o : MagazineOffer) 
    (h1 : o.offerYears = 2)
    (h2 : o.offerPrice = o.regularYearlyFee) :
    o.offerPrice / o.offerYears = o.regularYearlyFee / 2 := by
  sorry

#check offer_price_per_year_is_half parentsOffer

end NUMINAMATH_CALUDE_offer_price_per_year_is_half_l4172_417277


namespace NUMINAMATH_CALUDE_complex_equation_sum_l4172_417275

theorem complex_equation_sum (a b : ℝ) : 
  (Complex.mk a 3 + Complex.mk 2 (-1) = Complex.mk 5 b) → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l4172_417275


namespace NUMINAMATH_CALUDE_equation_equivalence_l4172_417272

theorem equation_equivalence (a b c : ℝ) :
  2 * b^2 = a^2 + c^2 ↔ 1 / (a + b) + 1 / (b + c) = 2 / (c + a) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l4172_417272


namespace NUMINAMATH_CALUDE_quadratic_inequality_result_l4172_417259

theorem quadratic_inequality_result (x : ℝ) :
  x^2 - 5*x + 6 < 0 → x^2 - 5*x + 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_result_l4172_417259


namespace NUMINAMATH_CALUDE_min_balls_for_twenty_of_one_color_l4172_417258

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls needed to guarantee at least n balls of a single color -/
def minBallsForColor (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The specific ball counts in our problem -/
def ourBox : BallCounts :=
  { red := 30, green := 22, yellow := 18, blue := 15, white := 10, black := 6 }

/-- The theorem to be proved -/
theorem min_balls_for_twenty_of_one_color :
  minBallsForColor ourBox 20 = 88 := by
  sorry

end NUMINAMATH_CALUDE_min_balls_for_twenty_of_one_color_l4172_417258


namespace NUMINAMATH_CALUDE_solution_set_inequalities_l4172_417203

theorem solution_set_inequalities (a b : ℝ) 
  (h : ∃ x, x > a ∧ x < b) : 
  {x : ℝ | x < 1 - a ∧ x < 1 - b} = {x : ℝ | x < 1 - b} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequalities_l4172_417203


namespace NUMINAMATH_CALUDE_distance_between_docks_l4172_417230

/-- The distance between docks A and B in kilometers. -/
def distance : ℝ := 105

/-- The speed of the water flow in kilometers per hour. -/
def water_speed : ℝ := 3

/-- The time taken to travel downstream in hours. -/
def downstream_time : ℝ := 5

/-- The time taken to travel upstream in hours. -/
def upstream_time : ℝ := 7

/-- Theorem stating that the distance between docks A and B is 105 kilometers. -/
theorem distance_between_docks :
  distance = 105 ∧
  water_speed = 3 ∧
  downstream_time = 5 ∧
  upstream_time = 7 ∧
  (distance / downstream_time - water_speed = distance / upstream_time + water_speed) :=
by sorry

end NUMINAMATH_CALUDE_distance_between_docks_l4172_417230


namespace NUMINAMATH_CALUDE_writer_productivity_l4172_417219

/-- Given a writer's manuscript details, calculate their writing productivity. -/
theorem writer_productivity (total_words : ℕ) (total_hours : ℕ) (break_hours : ℕ) :
  total_words = 60000 →
  total_hours = 120 →
  break_hours = 20 →
  (total_words : ℝ) / (total_hours - break_hours : ℝ) = 600 := by
  sorry

end NUMINAMATH_CALUDE_writer_productivity_l4172_417219


namespace NUMINAMATH_CALUDE_gcd_2023_2048_l4172_417264

theorem gcd_2023_2048 : Nat.gcd 2023 2048 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2023_2048_l4172_417264


namespace NUMINAMATH_CALUDE_max_value_on_interval_max_value_is_11_l4172_417243

def f (x : ℝ) : ℝ := x^4 - 8*x^2 + 2

theorem max_value_on_interval (a b : ℝ) (h : a ≤ b) :
  ∃ c ∈ Set.Icc a b, ∀ x ∈ Set.Icc a b, f x ≤ f c :=
sorry

theorem max_value_is_11 :
  ∃ c ∈ Set.Icc (-1) 3, f c = 11 ∧ ∀ x ∈ Set.Icc (-1) 3, f x ≤ f c :=
sorry

end NUMINAMATH_CALUDE_max_value_on_interval_max_value_is_11_l4172_417243


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4172_417210

theorem complex_equation_solution (m A B : ℝ) :
  (((2 : ℂ) - m * I) / ((1 : ℂ) + 2 * I) = A + B * I) →
  A + B = 0 →
  m = -2/3 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4172_417210


namespace NUMINAMATH_CALUDE_expression_evaluation_l4172_417248

theorem expression_evaluation :
  (-1)^2008 + (-1)^2009 + 2^2006 * (-1)^2007 + 1^2010 = -2^2006 + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4172_417248


namespace NUMINAMATH_CALUDE_expression_equality_l4172_417260

theorem expression_equality : (3^2015 + 3^2013) / (3^2015 - 3^2013) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l4172_417260
