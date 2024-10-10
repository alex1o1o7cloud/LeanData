import Mathlib

namespace power_function_through_point_l382_38200

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop := ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the theorem
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2) : 
  f 4 = 2 := by
sorry

end power_function_through_point_l382_38200


namespace min_value_theorem_l382_38208

theorem min_value_theorem (x : ℝ) (h : x > 1) :
  3 * x + 1 / (x - 1) ≥ 2 * Real.sqrt 3 + 3 := by
  sorry

end min_value_theorem_l382_38208


namespace gas_refill_amount_l382_38218

def gas_problem (initial_gas tank_capacity gas_to_store gas_to_doctor : ℕ) : ℕ :=
  tank_capacity - (initial_gas - gas_to_store - gas_to_doctor)

theorem gas_refill_amount :
  gas_problem 10 12 6 2 = 10 := by sorry

end gas_refill_amount_l382_38218


namespace inspection_sample_size_l382_38242

/-- Represents a batch of leather shoes -/
structure ShoeBatch where
  total : ℕ

/-- Represents a quality inspection of shoes -/
structure QualityInspection where
  batch : ShoeBatch
  drawn : ℕ

/-- Definition of sample size for a quality inspection -/
def sampleSize (inspection : QualityInspection) : ℕ :=
  inspection.drawn

theorem inspection_sample_size (batch : ShoeBatch) :
  let inspection := QualityInspection.mk batch 50
  sampleSize inspection = 50 := by
  sorry

end inspection_sample_size_l382_38242


namespace integral_split_l382_38268

-- Define f as a real-valued function on the real line
variable (f : ℝ → ℝ)

-- State the theorem
theorem integral_split (h : ∫ x in (1:ℝ)..(3:ℝ), f x = 56) :
  ∫ x in (1:ℝ)..(2:ℝ), f x + ∫ x in (2:ℝ)..(3:ℝ), f x = 56 := by
  sorry

end integral_split_l382_38268


namespace same_color_plate_probability_l382_38277

/-- The probability of selecting two plates of the same color -/
theorem same_color_plate_probability 
  (total_plates : ℕ) 
  (red_plates : ℕ) 
  (blue_plates : ℕ) 
  (h1 : total_plates = red_plates + blue_plates) 
  (h2 : total_plates = 13) 
  (h3 : red_plates = 7) 
  (h4 : blue_plates = 6) : 
  (red_plates.choose 2 + blue_plates.choose 2 : ℚ) / total_plates.choose 2 = 4/9 := by
  sorry

end same_color_plate_probability_l382_38277


namespace range_of_2x_plus_y_range_of_a_l382_38299

-- Define the circle
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 2*y

-- Theorem for the range of 2x + y
theorem range_of_2x_plus_y :
  ∀ x y : ℝ, on_circle x y → -Real.sqrt 5 + 1 ≤ 2*x + y ∧ 2*x + y ≤ Real.sqrt 5 + 1 := by
  sorry

-- Theorem for the range of a
theorem range_of_a :
  (∀ x y : ℝ, on_circle x y → ∀ a : ℝ, x + y + a ≥ 0) →
  ∀ a : ℝ, a ≥ Real.sqrt 2 - 1 := by
  sorry

end range_of_2x_plus_y_range_of_a_l382_38299


namespace quilt_shaded_fraction_l382_38245

/-- Represents a square quilt block -/
structure QuiltBlock where
  size : Nat
  full_shaded : Nat
  half_shaded : Nat

/-- Calculates the fraction of shaded area in a quilt block -/
def shaded_fraction (q : QuiltBlock) : Rat :=
  (q.full_shaded + q.half_shaded / 2 : Rat) / (q.size * q.size)

/-- Theorem stating that a 4x4 quilt block with 2 fully shaded squares and 4 half-shaded squares has 1/4 of its area shaded -/
theorem quilt_shaded_fraction :
  let q : QuiltBlock := { size := 4, full_shaded := 2, half_shaded := 4 }
  shaded_fraction q = 1/4 := by
  sorry

end quilt_shaded_fraction_l382_38245


namespace no_arithmetic_mean_among_fractions_l382_38201

theorem no_arithmetic_mean_among_fractions : 
  let a := 8 / 13
  let b := 11 / 17
  let c := 5 / 8
  ¬(a = (b + c) / 2 ∨ b = (a + c) / 2 ∨ c = (a + b) / 2) := by
sorry

end no_arithmetic_mean_among_fractions_l382_38201


namespace ruths_track_length_l382_38204

theorem ruths_track_length (sean_piece_length ruth_piece_length total_length : ℝ) :
  sean_piece_length = 8 →
  total_length = 72 →
  (total_length / sean_piece_length) * sean_piece_length = (total_length / ruth_piece_length) * ruth_piece_length →
  ruth_piece_length = 8 :=
by sorry

end ruths_track_length_l382_38204


namespace hundredth_stationary_is_hundred_l382_38284

/-- A function representing the sorting algorithm that swaps adjacent numbers if the larger number is on the left -/
def sortPass (s : List ℕ) : List ℕ := sorry

/-- A predicate that checks if a number at a given index remains stationary during both passes -/
def isStationary (s : List ℕ) (index : ℕ) : Prop := sorry

theorem hundredth_stationary_is_hundred {s : List ℕ} (h1 : s.length = 1982) 
  (h2 : ∀ n, n ∈ s → 1 ≤ n ∧ n ≤ 1982) 
  (h3 : isStationary s 100) : 
  s[99] = 100 := by sorry

end hundredth_stationary_is_hundred_l382_38284


namespace intersection_y_coordinate_l382_38246

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 2*y

-- Define points P and Q on the parabola
def P : ℝ × ℝ := (4, 8)
def Q : ℝ × ℝ := (-2, 2)

-- Define the tangent lines at P and Q
def tangent_P (x y : ℝ) : Prop := y = 4*x - 8
def tangent_Q (x y : ℝ) : Prop := y = -2*x - 2

-- Define the intersection point A
def A : ℝ × ℝ := (1, -4)

-- Theorem statement
theorem intersection_y_coordinate :
  parabola P.1 P.2 ∧ 
  parabola Q.1 Q.2 ∧ 
  tangent_P A.1 A.2 ∧ 
  tangent_Q A.1 A.2 →
  A.2 = -4 :=
by sorry

end intersection_y_coordinate_l382_38246


namespace square_root_of_sixteen_l382_38219

theorem square_root_of_sixteen : ∃ (x : ℝ), x^2 = 16 ↔ x = 4 ∨ x = -4 := by
  sorry

end square_root_of_sixteen_l382_38219


namespace negative_abs_negative_three_l382_38216

theorem negative_abs_negative_three : -|-3| = -3 := by sorry

end negative_abs_negative_three_l382_38216


namespace transfer_ratio_l382_38212

def initial_balance : ℕ := 190
def mom_transfer : ℕ := 60
def final_balance : ℕ := 100

def sister_transfer : ℕ := initial_balance - mom_transfer - final_balance

theorem transfer_ratio : 
  sister_transfer * 2 = mom_transfer := by sorry

end transfer_ratio_l382_38212


namespace shortest_chord_parallel_and_separate_l382_38287

/-- Circle with center at origin and radius r -/
def Circle (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

/-- Point P inside the circle -/
structure PointInCircle (r : ℝ) where
  a : ℝ
  b : ℝ
  h1 : a ≠ 0
  h2 : b ≠ 0
  h3 : a^2 + b^2 < r^2

/-- Line l1 containing the shortest chord through P -/
def ShortestChordLine (r : ℝ) (p : PointInCircle r) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | p.a * q.1 + p.b * q.2 = p.a^2 + p.b^2}

/-- Line l2 -/
def Line_l2 (r : ℝ) (p : PointInCircle r) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | p.b * q.1 - p.a * q.2 + r^2 = 0}

/-- Two lines are parallel -/
def Parallel (l1 l2 : Set (ℝ × ℝ)) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ l1 ↔ (x, y) ∈ l2 ∨ (x + k, y) ∈ l2

/-- A line is separate from a circle -/
def Separate (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop :=
  ∀ (p : ℝ × ℝ), p ∈ l → p ∉ c

theorem shortest_chord_parallel_and_separate (r : ℝ) (p : PointInCircle r) :
  Parallel (ShortestChordLine r p) (Line_l2 r p) ∧
  Separate (Line_l2 r p) (Circle r) := by
  sorry

end shortest_chord_parallel_and_separate_l382_38287


namespace consecutive_integers_product_l382_38258

theorem consecutive_integers_product (a b c d : ℤ) : 
  (b = a + 1) → (c = b + 1) → (d = c + 1) → (a + d = 109) → (b * c = 2970) := by
  sorry

end consecutive_integers_product_l382_38258


namespace internal_tangent_locus_l382_38233

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are internally tangent -/
def are_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius - c2.radius)^2

/-- The locus of points -/
def locus (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

theorem internal_tangent_locus (O1 O2 : Circle) 
  (h1 : O1.radius = 7)
  (h2 : O2.radius = 4)
  (h3 : are_internally_tangent O1 O2) :
  locus { center := O1.center, radius := 3 } O2.center :=
sorry

end internal_tangent_locus_l382_38233


namespace brother_lower_limit_l382_38202

-- Define Arun's weight
def W : ℝ := sorry

-- Define brother's lower limit
def B : ℝ := sorry

-- Arun's opinion
axiom arun_opinion : 64 < W ∧ W < 72

-- Brother's opinion
axiom brother_opinion : B < W ∧ W < 70

-- Mother's opinion
axiom mother_opinion : W ≤ 67

-- Average weight
axiom average_weight : (W + 67) / 2 = 66

-- Theorem to prove
theorem brother_lower_limit : B > 64 := by sorry

end brother_lower_limit_l382_38202


namespace extreme_value_probability_l382_38269

-- Define the die outcomes
def DieOutcome := Fin 6

-- Define the probability space
def Ω := DieOutcome × DieOutcome

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define the condition for extreme value
def hasExtremeValue (a b : ℕ) : Prop := a^2 > 4*b

-- State the theorem
theorem extreme_value_probability : 
  P {ω : Ω | hasExtremeValue ω.1.val.succ ω.2.val.succ} = 17/36 := by sorry

end extreme_value_probability_l382_38269


namespace coin_toss_probability_l382_38238

theorem coin_toss_probability : 
  let p_head : ℝ := 1/2  -- Probability of getting heads on a single toss
  let n : ℕ := 3  -- Number of tosses
  let p_all_tails : ℝ := (1 - p_head)^n  -- Probability of getting all tails
  1 - p_all_tails = 7/8 := by sorry

end coin_toss_probability_l382_38238


namespace a_range_characterization_l382_38226

/-- Proposition p: The domain of the logarithm function is all real numbers -/
def prop_p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*a*x + 7*a - 6 > 0

/-- Proposition q: There exists a real x satisfying the quadratic inequality -/
def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - a*x + 4 < 0

/-- The set of real numbers a where either p is true and q is false, or p is false and q is true -/
def a_range : Set ℝ := {a | (prop_p a ∧ ¬prop_q a) ∨ (¬prop_p a ∧ prop_q a)}

theorem a_range_characterization : 
  a_range = {a | a < -4 ∨ (1 < a ∧ a ≤ 4) ∨ 6 ≤ a} :=
sorry

end a_range_characterization_l382_38226


namespace fraction_multiplication_l382_38293

theorem fraction_multiplication : (1 : ℚ) / 3 * 4 / 7 * 9 / 13 * 2 / 5 = 72 / 1365 := by
  sorry

end fraction_multiplication_l382_38293


namespace complement_of_A_union_B_l382_38239

open Set

def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | x ≥ 0}

theorem complement_of_A_union_B :
  (A ∪ B)ᶜ = {x : ℝ | x ≤ -1} :=
by sorry

end complement_of_A_union_B_l382_38239


namespace lcm_problem_l382_38282

theorem lcm_problem (a b : ℕ) (h : Nat.gcd a b = 47) (ha : a = 210) (hb : b = 517) :
  Nat.lcm a b = 2310 := by
  sorry

end lcm_problem_l382_38282


namespace intersection_distance_l382_38290

/-- The distance between intersection points of a line and circle -/
theorem intersection_distance (x y : ℝ) : 
  -- Line equation
  (y = Real.sqrt 3 * x + Real.sqrt 2 / 2) →
  -- Circle equation
  ((x - Real.sqrt 2 / 2)^2 + (y - Real.sqrt 2 / 2)^2 = 1) →
  -- Distance between intersection points
  ∃ (a b : ℝ × ℝ), 
    (a.1 - b.1)^2 + (a.2 - b.2)^2 = Real.sqrt 10 / 2 := by
  sorry

end intersection_distance_l382_38290


namespace integer_solutions_equation_l382_38251

theorem integer_solutions_equation (n m : ℤ) : 
  n^6 + 3*n^5 + 3*n^4 + 2*n^3 + 3*n^2 + 3*n + 1 = m^3 ↔ (n = 0 ∧ m = 1) ∨ (n = -1 ∧ m = 0) :=
sorry

end integer_solutions_equation_l382_38251


namespace child_ticket_cost_l382_38264

/-- Proves that the cost of a child ticket is 25 cents given the specified conditions. -/
theorem child_ticket_cost
  (adult_price : ℕ)
  (total_attendees : ℕ)
  (total_revenue : ℕ)
  (num_children : ℕ)
  (h1 : adult_price = 60)
  (h2 : total_attendees = 280)
  (h3 : total_revenue = 14000)  -- in cents
  (h4 : num_children = 80) :
  ∃ (child_price : ℕ),
    child_price * num_children + adult_price * (total_attendees - num_children) = total_revenue ∧
    child_price = 25 :=
by sorry

end child_ticket_cost_l382_38264


namespace symmetric_points_values_l382_38248

/-- Two points are symmetric about the y-axis if their x-coordinates are opposite and their y-coordinates are equal -/
def symmetric_about_y_axis (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 = -x2 ∧ y1 = y2

theorem symmetric_points_values :
  ∀ m n : ℝ,
  symmetric_about_y_axis (-3) (2*m - 1) (n + 1) 4 →
  m = 2.5 ∧ n = 2 :=
by sorry

end symmetric_points_values_l382_38248


namespace price_reduction_equation_l382_38254

theorem price_reduction_equation (initial_price final_price : ℝ) 
  (h1 : initial_price = 188) 
  (h2 : final_price = 108) 
  (x : ℝ) -- x represents the percentage of each reduction
  (h3 : x ≥ 0 ∧ x < 1) -- ensure x is a valid percentage
  (h4 : final_price = initial_price * (1 - x)^2) -- two equal reductions
  : initial_price * (1 - x)^2 = final_price := by
  sorry

end price_reduction_equation_l382_38254


namespace volleyball_preference_percentage_l382_38252

theorem volleyball_preference_percentage
  (north_students : ℕ)
  (south_students : ℕ)
  (north_volleyball_percentage : ℚ)
  (south_volleyball_percentage : ℚ)
  (h1 : north_students = 1800)
  (h2 : south_students = 2700)
  (h3 : north_volleyball_percentage = 25 / 100)
  (h4 : south_volleyball_percentage = 35 / 100)
  : (north_students * north_volleyball_percentage + south_students * south_volleyball_percentage) /
    (north_students + south_students) = 31 / 100 := by
  sorry


end volleyball_preference_percentage_l382_38252


namespace abs_neg_three_l382_38295

theorem abs_neg_three : |(-3 : ℤ)| = 3 := by sorry

end abs_neg_three_l382_38295


namespace print_360_pages_in_15_minutes_l382_38296

/-- Calculates the time needed to print a given number of pages at a specific rate. -/
def print_time (pages : ℕ) (rate : ℕ) : ℚ :=
  pages / rate

/-- Theorem stating that printing 360 pages at a rate of 24 pages per minute takes 15 minutes. -/
theorem print_360_pages_in_15_minutes :
  print_time 360 24 = 15 := by
  sorry

end print_360_pages_in_15_minutes_l382_38296


namespace imaginary_power_2019_l382_38228

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_2019 : i^2019 = -i := by sorry

end imaginary_power_2019_l382_38228


namespace total_books_count_l382_38250

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 6

/-- The number of shelves with mystery books -/
def mystery_shelves : ℕ := 5

/-- The number of shelves with picture books -/
def picture_shelves : ℕ := 4

/-- The total number of books -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem total_books_count : total_books = 54 := by sorry

end total_books_count_l382_38250


namespace infinite_non_prime_generating_numbers_l382_38274

theorem infinite_non_prime_generating_numbers :
  ∃ f : ℕ → ℕ, ∀ m : ℕ, m > 1 → ∀ n : ℕ, ¬ Nat.Prime (n^4 + f m) := by
  sorry

end infinite_non_prime_generating_numbers_l382_38274


namespace function_properties_l382_38230

/-- Given a function f(x) = x - a*exp(x) + b, where a > 0 and b is real,
    this theorem proves properties about its maximum value and zero points. -/
theorem function_properties (a b : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ x - a * Real.exp x + b
  -- The maximum value of f occurs at ln(1/a) and equals ln(1/a) - 1 + b
  ∃ (x_max : ℝ), x_max = Real.log (1/a) ∧
    ∀ x, f x ≤ f x_max ∧ f x_max = Real.log (1/a) - 1 + b ∧
  -- If f has two distinct zero points, their sum is less than -2*ln(a)
  ∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → f x₁ = 0 → f x₂ = 0 → x₁ + x₂ < -2 * Real.log a :=
by
  sorry

end function_properties_l382_38230


namespace air_quality_probability_l382_38235

theorem air_quality_probability (p_good : ℝ) (p_consecutive : ℝ) 
  (h1 : p_good = 0.75) (h2 : p_consecutive = 0.6) : 
  p_consecutive / p_good = 0.8 := by
  sorry

end air_quality_probability_l382_38235


namespace point_position_l382_38288

theorem point_position (x : ℝ) (h1 : x < -2) (h2 : |x - (-2)| = 5) : x = -7 := by
  sorry

end point_position_l382_38288


namespace sock_drawer_problem_l382_38267

/-- The number of ways to choose k items from n distinguishable items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose a pair of socks of the same color -/
def sameColorPairs (white brown blue red : ℕ) : ℕ :=
  choose white 2 + choose brown 2 + choose blue 2 + choose red 2

theorem sock_drawer_problem :
  sameColorPairs 5 5 3 2 = 24 := by sorry

end sock_drawer_problem_l382_38267


namespace symmetry_wrt_y_axis_l382_38227

/-- Given two real numbers a and b such that lg a + lg b = 0, a ≠ 1, and b ≠ 1,
    the functions f(x) = a^x and g(x) = b^x are symmetric with respect to the y-axis. -/
theorem symmetry_wrt_y_axis (a b : ℝ) (ha : a ≠ 1) (hb : b ≠ 1) 
    (h : Real.log a + Real.log b = 0) :
  ∀ x : ℝ, a^(-x) = b^x := by
  sorry

end symmetry_wrt_y_axis_l382_38227


namespace computers_waiting_for_parts_l382_38276

theorem computers_waiting_for_parts (total : ℕ) (unfixable_percent : ℚ) (fixed_immediately : ℕ) : 
  total = 20 →
  unfixable_percent = 1/5 →
  fixed_immediately = 8 →
  (total - (unfixable_percent * total).num - fixed_immediately : ℚ) / total = 2/5 := by
  sorry

end computers_waiting_for_parts_l382_38276


namespace annas_money_l382_38244

theorem annas_money (original : ℝ) : 
  (original - original * (1/4) = 24) → original = 32 :=
by
  sorry

end annas_money_l382_38244


namespace diophantine_equation_solution_l382_38234

theorem diophantine_equation_solution : ∃ (x y : ℕ), 1984 * x - 1983 * y = 1985 ∧ x = 27764 ∧ y = 27777 := by
  sorry

end diophantine_equation_solution_l382_38234


namespace simplify_expression_l382_38232

theorem simplify_expression (a b : ℝ) : (2*a^2 - 3*a*b + 8) - (-a*b - a^2 + 8) = 3*a^2 - 2*a*b := by
  sorry

end simplify_expression_l382_38232


namespace equation_solution_l382_38213

theorem equation_solution (x y : ℝ) :
  (Real.sqrt (8 * x) / Real.sqrt (4 * (y - 2)) = 3) →
  (x = (9 * y - 18) / 2) := by
sorry

end equation_solution_l382_38213


namespace platform_length_l382_38271

/-- Given a train of length 300 meters that takes 39 seconds to cross a platform
    and 26 seconds to cross a signal pole, prove that the length of the platform is 150 meters. -/
theorem platform_length
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_cross_platform = 39)
  (h3 : time_cross_pole = 26) :
  let train_speed := train_length / time_cross_pole
  let platform_length := train_speed * time_cross_platform - train_length
  platform_length = 150 := by
sorry

end platform_length_l382_38271


namespace existence_of_distinct_integers_l382_38224

theorem existence_of_distinct_integers (n : ℤ) (h : n > 1) :
  ∃ (a b c : ℤ),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    n^2 < a ∧ a < (n+1)^2 ∧
    n^2 < b ∧ b < (n+1)^2 ∧
    n^2 < c ∧ c < (n+1)^2 ∧
    (c ∣ a^2 + b^2) := by
  sorry

end existence_of_distinct_integers_l382_38224


namespace four_painters_work_days_l382_38231

/-- The number of work-days required for a given number of painters to complete a job -/
def work_days (num_painters : ℕ) (total_work : ℚ) : ℚ :=
  total_work / num_painters

theorem four_painters_work_days :
  let total_work : ℚ := 6 * (3/2)  -- 6 painters * 1.5 days
  (work_days 4 total_work) = 2 + (1/4) := by sorry

end four_painters_work_days_l382_38231


namespace sin_cos_identity_l382_38241

theorem sin_cos_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_identity_l382_38241


namespace largest_x_value_l382_38266

theorem largest_x_value (x : ℝ) : 
  x ≠ 7 → 
  ((x^2 - 5*x - 84) / (x - 7) = 2 / (x + 6)) → 
  x ≤ -5 :=
by
  sorry

end largest_x_value_l382_38266


namespace function_odd_iff_sum_squares_zero_l382_38292

/-- The function f(x) = x|x-a| + b is odd if and only if a^2 + b^2 = 0 -/
theorem function_odd_iff_sum_squares_zero (a b : ℝ) :
  (∀ x : ℝ, x * |x - a| + b = -((-x) * |(-x) - a| + b)) ↔ a^2 + b^2 = 0 := by
  sorry

end function_odd_iff_sum_squares_zero_l382_38292


namespace johns_goals_l382_38206

theorem johns_goals (total_goals : ℝ) (teammate_count : ℕ) (avg_teammate_goals : ℝ) :
  total_goals = 65 ∧
  teammate_count = 9 ∧
  avg_teammate_goals = 4.5 →
  total_goals - (teammate_count : ℝ) * avg_teammate_goals = 24.5 :=
by sorry

end johns_goals_l382_38206


namespace rectangle_area_l382_38285

/-- Proves that a rectangle with length thrice its breadth and perimeter 64 has area 192 -/
theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let perimeter := 2 * (l + b)
  perimeter = 64 → l * b = 192 := by
  sorry

end rectangle_area_l382_38285


namespace train_departure_time_l382_38256

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDiff (t1 t2 : Time) : Nat :=
  (t1.hours * 60 + t1.minutes) - (t2.hours * 60 + t2.minutes)

theorem train_departure_time 
  (arrival : Time)
  (journey_duration : Nat)
  (h_arrival : arrival.hours = 10 ∧ arrival.minutes = 0)
  (h_duration : journey_duration = 15) :
  ∃ (departure : Time), 
    timeDiff arrival departure = journey_duration ∧ 
    departure.hours = 9 ∧ 
    departure.minutes = 45 := by
  sorry

end train_departure_time_l382_38256


namespace compute_expression_simplify_expression_l382_38260

-- Part 1
theorem compute_expression : (1/2)⁻¹ - Real.sqrt 3 * Real.cos (30 * π / 180) + (2014 - Real.pi)^0 = 3/2 := by
  sorry

-- Part 2
theorem simplify_expression (a : ℝ) : a * (a + 1) - (a + 1) * (a - 1) = a + 1 := by
  sorry

end compute_expression_simplify_expression_l382_38260


namespace g_of_5_eq_neg_7_l382_38272

/-- The polynomial function g(x) -/
def g (x : ℝ) : ℝ := 2 * x^4 - 15 * x^3 + 24 * x^2 - 18 * x - 72

/-- Theorem: g(5) equals -7 -/
theorem g_of_5_eq_neg_7 : g 5 = -7 := by
  sorry

end g_of_5_eq_neg_7_l382_38272


namespace distance_between_homes_l382_38281

/-- The distance between Xiaohong's and Xiaoli's homes given their walking speeds and arrival times -/
theorem distance_between_homes 
  (x_speed : ℝ) 
  (l_speed_to_cinema : ℝ) 
  (l_speed_from_cinema : ℝ) 
  (delay : ℝ) 
  (h_x_speed : x_speed = 52) 
  (h_l_speed_to_cinema : l_speed_to_cinema = 70) 
  (h_l_speed_from_cinema : l_speed_from_cinema = 90) 
  (h_delay : delay = 4) : 
  ∃ (t : ℝ), x_speed * t + l_speed_to_cinema * t = 2196 ∧ 
  x_speed * (t + delay + (x_speed * t / x_speed)) = l_speed_from_cinema * ((x_speed * t / x_speed) - delay) :=
sorry

end distance_between_homes_l382_38281


namespace profit_percentage_l382_38253

theorem profit_percentage (selling_price cost_price : ℝ) (h : cost_price = 0.95 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100 / 95 - 1) * 100 := by
  sorry

end profit_percentage_l382_38253


namespace sun_division_l382_38217

theorem sun_division (x y z total : ℚ) : 
  (y = (45/100) * x) →  -- y gets 45 paisa for each rupee x gets
  (z = (30/100) * x) →  -- z gets 30 paisa for each rupee x gets
  (y = 63) →            -- y's share is Rs. 63
  (total = x + y + z) → -- total is the sum of all shares
  (total = 245) :=      -- prove that the total is Rs. 245
by
  sorry

end sun_division_l382_38217


namespace vip_seat_cost_l382_38210

theorem vip_seat_cost (total_tickets : ℕ) (total_revenue : ℕ) 
  (general_price : ℕ) (vip_difference : ℕ) :
  total_tickets = 320 →
  total_revenue = 7500 →
  general_price = 15 →
  vip_difference = 212 →
  ∃ (vip_price : ℕ), 
    vip_price = 65 ∧
    (total_tickets - vip_difference) * general_price + 
    vip_difference * vip_price = total_revenue :=
by
  sorry

end vip_seat_cost_l382_38210


namespace median_squares_sum_l382_38279

/-- Given a triangle with side lengths 13, 14, and 15, the sum of the squares of its median lengths is 442.5 -/
theorem median_squares_sum (a b c : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) :
  let median_sum_squares := 3/4 * (a^2 + b^2 + c^2)
  median_sum_squares = 442.5 := by
  sorry

end median_squares_sum_l382_38279


namespace isosceles_triangle_part1_isosceles_triangle_part2_l382_38209

/-- Represents an isosceles triangle with given side lengths -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  isIsosceles : leg ≥ base
  perimeter : ℝ
  sumOfSides : base + 2 * leg = perimeter

/-- Theorem for part 1 of the problem -/
theorem isosceles_triangle_part1 :
  ∃ (t : IsoscelesTriangle),
    t.perimeter = 20 ∧ t.leg = 2 * t.base ∧ t.base = 4 ∧ t.leg = 8 := by
  sorry

/-- Theorem for part 2 of the problem -/
theorem isosceles_triangle_part2 :
  ∃ (t : IsoscelesTriangle),
    t.perimeter = 20 ∧ t.base = 5 ∧ t.leg = 7.5 := by
  sorry

end isosceles_triangle_part1_isosceles_triangle_part2_l382_38209


namespace book_sales_proof_l382_38214

/-- Calculates the number of copies sold given the revenue per book, agent's commission percentage, and total amount kept by the author. -/
def calculate_copies_sold (revenue_per_book : ℚ) (agent_commission_percent : ℚ) (total_kept : ℚ) : ℚ :=
  total_kept / (revenue_per_book * (1 - agent_commission_percent / 100))

/-- Proves that given the specific conditions, the number of copies sold is 900,000. -/
theorem book_sales_proof (revenue_per_book : ℚ) (agent_commission_percent : ℚ) (total_kept : ℚ) 
    (h1 : revenue_per_book = 2)
    (h2 : agent_commission_percent = 10)
    (h3 : total_kept = 1620000) :
  calculate_copies_sold revenue_per_book agent_commission_percent total_kept = 900000 := by
  sorry

end book_sales_proof_l382_38214


namespace simplify_fraction_l382_38273

theorem simplify_fraction (b : ℝ) (h : b = 5) : 15 * b^4 / (75 * b^3) = 1 := by
  sorry

end simplify_fraction_l382_38273


namespace student_tickets_sold_l382_38255

theorem student_tickets_sold (total_tickets : ℕ) (student_price non_student_price total_money : ℚ)
  (h1 : total_tickets = 193)
  (h2 : student_price = 1/2)
  (h3 : non_student_price = 3/2)
  (h4 : total_money = 206.5)
  (h5 : ∃ (student_tickets non_student_tickets : ℕ),
    student_tickets + non_student_tickets = total_tickets ∧
    student_tickets * student_price + non_student_tickets * non_student_price = total_money) :
  ∃ (student_tickets : ℕ), student_tickets = 83 :=
by sorry

end student_tickets_sold_l382_38255


namespace spending_difference_l382_38243

/-- The cost of the computer table in dollars -/
def table_cost : ℚ := 140

/-- The cost of the computer chair in dollars -/
def chair_cost : ℚ := 100

/-- The cost of the joystick in dollars -/
def joystick_cost : ℚ := 20

/-- Frank's share of the joystick cost -/
def frank_joystick_share : ℚ := 1/4

/-- Eman's share of the joystick cost -/
def eman_joystick_share : ℚ := 1 - frank_joystick_share

/-- Frank's total spending -/
def frank_total : ℚ := table_cost + frank_joystick_share * joystick_cost

/-- Eman's total spending -/
def eman_total : ℚ := chair_cost + eman_joystick_share * joystick_cost

theorem spending_difference : frank_total - eman_total = 30 := by
  sorry

end spending_difference_l382_38243


namespace total_distance_covered_l382_38283

/-- The total distance covered by a fox, rabbit, and deer given their speeds and running times -/
theorem total_distance_covered 
  (fox_speed : ℝ) 
  (rabbit_speed : ℝ) 
  (deer_speed : ℝ) 
  (fox_time : ℝ) 
  (rabbit_time : ℝ) 
  (deer_time : ℝ) 
  (h1 : fox_speed = 50) 
  (h2 : rabbit_speed = 60) 
  (h3 : deer_speed = 80) 
  (h4 : fox_time = 2) 
  (h5 : rabbit_time = 5/3) 
  (h6 : deer_time = 3/2) : 
  fox_speed * fox_time + rabbit_speed * rabbit_time + deer_speed * deer_time = 320 := by
  sorry


end total_distance_covered_l382_38283


namespace triangle_ratio_theorem_l382_38286

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_ratio_theorem (t : Triangle) 
  (h1 : Real.sqrt 3 * t.a * Real.cos t.B = t.b * Real.sin t.A)
  (h2 : (Real.sqrt 3 / 4) * t.b^2 = (1/2) * t.a * t.c * Real.sin t.B) :
  t.a / t.c = 1 := by
  sorry

end triangle_ratio_theorem_l382_38286


namespace solution_comparison_l382_38259

theorem solution_comparison (a a' b b' : ℝ) (ha : a ≠ 0) (ha' : a' ≠ 0) :
  (-b / a < -b' / a') ↔ (b' / a' < b / a) := by
  sorry

end solution_comparison_l382_38259


namespace complex_equation_solution_l382_38215

theorem complex_equation_solution (b : ℝ) : 
  (2 - Complex.I) * (4 * Complex.I) = 4 + b * Complex.I → b = 8 := by
sorry

end complex_equation_solution_l382_38215


namespace short_story_booklets_l382_38275

/-- The number of booklets in Jack's short story section -/
def num_booklets : ℕ := 441 / 9

/-- The number of pages in each booklet -/
def pages_per_booklet : ℕ := 9

/-- The total number of pages Jack needs to read -/
def total_pages : ℕ := 441

theorem short_story_booklets :
  num_booklets = 49 ∧
  pages_per_booklet * num_booklets = total_pages :=
sorry

end short_story_booklets_l382_38275


namespace base9_repeating_fraction_l382_38294

/-- Represents a digit in base-9 system -/
def Base9Digit := Fin 9

/-- Converts a base-10 number to its base-9 representation -/
def toBase9 (n : ℚ) : List Base9Digit :=
  sorry

/-- Checks if a list of digits is repeating -/
def isRepeating (l : List Base9Digit) : Prop :=
  sorry

/-- The main theorem -/
theorem base9_repeating_fraction :
  ∃ (n d : ℕ) (l : List Base9Digit),
    n ≠ 0 ∧ d ≠ 0 ∧
    (n : ℚ) / d < 1 / 2 ∧
    isRepeating (toBase9 ((n : ℚ) / d)) ∧
    n = 13 ∧ d = 37 :=
  sorry

end base9_repeating_fraction_l382_38294


namespace committee_formation_count_l382_38236

/-- The number of ways to form a committee with specified conditions -/
def committee_formations (total_members : ℕ) (committee_size : ℕ) (required_members : ℕ) : ℕ :=
  Nat.choose (total_members - required_members) (committee_size - required_members)

/-- Theorem: The number of ways to form a 5-person committee from a 12-member club,
    where two specific members must always be included, is equal to 120. -/
theorem committee_formation_count :
  committee_formations 12 5 2 = 120 := by
  sorry

end committee_formation_count_l382_38236


namespace expression_simplification_and_evaluation_l382_38229

theorem expression_simplification_and_evaluation :
  let x : ℚ := -1
  let y : ℚ := -1/2
  let original_expression := 4*x*y + (2*x^2 + 5*x*y - y^2) - 2*(x^2 + 3*x*y)
  let simplified_expression := 3*x*y - y^2
  original_expression = simplified_expression ∧ simplified_expression = 5/4 :=
by sorry

end expression_simplification_and_evaluation_l382_38229


namespace curve_equation_relationship_l382_38207

-- Define a type for points in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a type for curves in 2D space
structure Curve where
  points : Set Point2D

-- Define a function type for equations in 2D space
def Equation2D := Point2D → Prop

-- Define the given condition
def satisfiesEquation (C : Curve) (f : Equation2D) : Prop :=
  ∀ p ∈ C.points, f p

-- Theorem statement
theorem curve_equation_relationship (C : Curve) (f : Equation2D) :
  satisfiesEquation C f →
  ¬ (∀ p : Point2D, f p ↔ p ∈ C.points) :=
by sorry

end curve_equation_relationship_l382_38207


namespace min_value_of_f_l382_38223

/-- The quadratic function f(x) = (x-2)^2 - 3 -/
def f (x : ℝ) : ℝ := (x - 2)^2 - 3

/-- The minimum value of f(x) is -3 -/
theorem min_value_of_f :
  ∃ (m : ℝ), m = -3 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

end min_value_of_f_l382_38223


namespace lastTwoDigitsOf7To2020_l382_38247

-- Define the function that gives the last two digits of 7^n
def lastTwoDigits (n : ℕ) : ℕ :=
  (7^n) % 100

-- State the periodicity of the last two digits
axiom lastTwoDigitsPeriodicity (n : ℕ) (h : n ≥ 2) : 
  lastTwoDigits n = lastTwoDigits (n % 4 + 4)

-- Define the theorem
theorem lastTwoDigitsOf7To2020 : lastTwoDigits 2020 = 01 := by
  sorry

end lastTwoDigitsOf7To2020_l382_38247


namespace field_trip_attendance_is_76_l382_38280

/-- The number of people on a field trip given the number of vans and buses,
    and the number of people in each vehicle type. -/
def field_trip_attendance (num_vans num_buses : ℕ) (people_per_van people_per_bus : ℕ) : ℕ :=
  num_vans * people_per_van + num_buses * people_per_bus

/-- Theorem stating that the total number of people on the field trip is 76. -/
theorem field_trip_attendance_is_76 :
  field_trip_attendance 2 3 8 20 = 76 := by
  sorry

#eval field_trip_attendance 2 3 8 20

end field_trip_attendance_is_76_l382_38280


namespace product_of_divisors_of_product_of_divisors_of_2005_l382_38240

def divisors (n : ℕ) : Finset ℕ := sorry

def divisor_product (n : ℕ) : ℕ := (divisors n).prod id

theorem product_of_divisors_of_product_of_divisors_of_2005 :
  divisor_product (divisor_product 2005) = 2005^9 := by sorry

end product_of_divisors_of_product_of_divisors_of_2005_l382_38240


namespace units_digit_of_expression_l382_38278

theorem units_digit_of_expression : ∃ n : ℕ, (12 + Real.sqrt 36)^17 + (12 - Real.sqrt 36)^17 = 10 * n + 4 := by
  sorry

end units_digit_of_expression_l382_38278


namespace line_plane_parallelism_l382_38225

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel_line_line : Line → Line → Prop)

-- Define the subset relation between a line and a plane
variable (subset_line_plane : Line → Plane → Prop)

-- Define the intersection operation between two planes
variable (intersection_plane_plane : Plane → Plane → Line)

-- State the theorem
theorem line_plane_parallelism 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) 
  (h_m_parallel_α : parallel_line_plane m α) 
  (h_m_subset_β : subset_line_plane m β) 
  (h_intersection : intersection_plane_plane α β = n) : 
  parallel_line_line m n :=
sorry

end line_plane_parallelism_l382_38225


namespace smallest_palindrome_div_by_7_l382_38203

/-- A function that checks if a number is a four-digit palindrome -/
def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (n / 1000 = n % 10) ∧ ((n / 100) % 10 = (n / 10) % 10)

/-- A function that checks if a number has an odd first digit -/
def has_odd_first_digit (n : ℕ) : Prop :=
  (n / 1000) % 2 = 1

/-- The theorem stating that 1661 is the smallest four-digit palindrome divisible by 7 with an odd first digit -/
theorem smallest_palindrome_div_by_7 :
  (∀ n : ℕ, is_four_digit_palindrome n ∧ has_odd_first_digit n ∧ n % 7 = 0 → n ≥ 1661) ∧
  is_four_digit_palindrome 1661 ∧ has_odd_first_digit 1661 ∧ 1661 % 7 = 0 := by
  sorry

end smallest_palindrome_div_by_7_l382_38203


namespace total_triangles_is_53_l382_38289

/-- Represents a rectangular figure with internal divisions -/
structure RectangularFigure where
  /-- The number of smallest right triangles -/
  small_right_triangles : ℕ
  /-- The number of isosceles triangles with base equal to the width -/
  width_isosceles_triangles : ℕ
  /-- The number of isosceles triangles with base equal to half the length -/
  half_length_isosceles_triangles : ℕ
  /-- The number of large right triangles -/
  large_right_triangles : ℕ
  /-- The number of large isosceles triangles with base equal to the full width -/
  large_isosceles_triangles : ℕ

/-- Calculates the total number of triangles in the figure -/
def total_triangles (figure : RectangularFigure) : ℕ :=
  figure.small_right_triangles +
  figure.width_isosceles_triangles +
  figure.half_length_isosceles_triangles +
  figure.large_right_triangles +
  figure.large_isosceles_triangles

/-- The specific rectangular figure described in the problem -/
def problem_figure : RectangularFigure :=
  { small_right_triangles := 24
  , width_isosceles_triangles := 6
  , half_length_isosceles_triangles := 8
  , large_right_triangles := 12
  , large_isosceles_triangles := 3
  }

/-- Theorem stating that the total number of triangles in the problem figure is 53 -/
theorem total_triangles_is_53 : total_triangles problem_figure = 53 := by
  sorry

end total_triangles_is_53_l382_38289


namespace line_equation_theorem_l382_38220

/-- Represents a line in the 2D plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if the given equation represents the line -/
def isEquationOfLine (a b c : ℝ) (l : Line) : Prop :=
  a ≠ 0 ∧ 
  l.slope = -a / b ∧
  l.yIntercept = -c / b

/-- The main theorem: the equation 3x - y + 4 = 0 represents a line with slope 3 and y-intercept 4 -/
theorem line_equation_theorem : 
  let l : Line := { slope := 3, yIntercept := 4 }
  isEquationOfLine 3 (-1) 4 l := by
sorry

end line_equation_theorem_l382_38220


namespace union_of_sets_l382_38298

theorem union_of_sets : 
  let S : Set ℕ := {3, 4, 5}
  let T : Set ℕ := {4, 7, 8}
  S ∪ T = {3, 4, 5, 7, 8} := by
  sorry

end union_of_sets_l382_38298


namespace product_101_101_l382_38257

theorem product_101_101 : 101 * 101 = 10201 := by
  sorry

end product_101_101_l382_38257


namespace circus_ticket_sales_l382_38205

theorem circus_ticket_sales (lower_price upper_price : ℕ) 
  (total_tickets total_revenue : ℕ) : 
  lower_price = 30 → 
  upper_price = 20 → 
  total_tickets = 80 → 
  total_revenue = 2100 → 
  ∃ (lower_seats upper_seats : ℕ), 
    lower_seats + upper_seats = total_tickets ∧ 
    lower_price * lower_seats + upper_price * upper_seats = total_revenue ∧ 
    lower_seats = 50 := by
  sorry

end circus_ticket_sales_l382_38205


namespace cricket_average_proof_l382_38297

def cricket_average (total_matches : ℕ) (first_set : ℕ) (second_set : ℕ) 
  (first_avg : ℚ) (second_avg : ℚ) : ℚ :=
  let total_first := first_avg * first_set
  let total_second := second_avg * second_set
  (total_first + total_second) / total_matches

theorem cricket_average_proof :
  cricket_average 10 6 4 41 (35.75) = 38.9 := by
  sorry

#eval cricket_average 10 6 4 41 (35.75)

end cricket_average_proof_l382_38297


namespace solution_set_when_a_is_2_range_of_a_l382_38211

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x - a| + a
def g (x : ℝ) : ℝ := |2 * x - 1|

-- Part I
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

-- Part II
theorem range_of_a :
  ∀ x : ℝ, f a x + g x ≥ 3 → a ∈ Set.Ici 2 := by sorry

end solution_set_when_a_is_2_range_of_a_l382_38211


namespace equal_slope_implies_parallel_l382_38237

/-- Two lines in a plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Theorem: If two non-intersecting lines have equal slopes, then they are parallel -/
theorem equal_slope_implies_parallel (l1 l2 : Line) 
  (h1 : l1.slope = l2.slope) 
  (h2 : l1.yIntercept ≠ l2.yIntercept) : 
  parallel l1 l2 := by
  sorry

end equal_slope_implies_parallel_l382_38237


namespace arman_age_problem_l382_38262

/-- Given that Arman is six times older than his sister, his sister was 2 years old four years ago,
    prove that Arman will be 40 years old in 4 years. -/
theorem arman_age_problem (sister_age_4_years_ago : ℕ) (arman_age sister_age : ℕ) :
  sister_age_4_years_ago = 2 →
  sister_age = sister_age_4_years_ago + 4 →
  arman_age = 6 * sister_age →
  40 - arman_age = 4 := by
sorry

end arman_age_problem_l382_38262


namespace polynomial_factorization_l382_38261

theorem polynomial_factorization (x : ℝ) :
  x^12 + x^6 + 1 = (x^2 + x + 1) * (x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + 1) := by
  sorry

end polynomial_factorization_l382_38261


namespace function_inequalities_l382_38249

noncomputable section

variable (a : ℝ)
variable (x : ℝ)

def f (x : ℝ) : ℝ := a^(3*x + 1)
def g (x : ℝ) : ℝ := (1/a)^(5*x - 2)

theorem function_inequalities (h1 : a > 0) (h2 : a ≠ 1) :
  (0 < a ∧ a < 1 → (f a x < 1 ↔ x > -1/3)) ∧
  ((0 < a ∧ a < 1 → (f a x ≥ g a x ↔ x ≤ 1/8)) ∧
   (a > 1 → (f a x ≥ g a x ↔ x ≥ 1/8))) := by
  sorry

end

end function_inequalities_l382_38249


namespace gabby_makeup_set_l382_38221

/-- The amount of money Gabby's mom gave her -/
def moms_gift (cost savings needed_after : ℕ) : ℕ :=
  cost - savings - needed_after

/-- Proof that Gabby's mom gave her $20 -/
theorem gabby_makeup_set : moms_gift 65 35 10 = 20 := by
  sorry

end gabby_makeup_set_l382_38221


namespace find_x_value_l382_38263

theorem find_x_value (x : ℝ) :
  (Real.sqrt x / Real.sqrt 0.81 + Real.sqrt 1.44 / Real.sqrt 0.49 = 2.9365079365079367) →
  x = 1.21 := by
  sorry

end find_x_value_l382_38263


namespace pigeonhole_divisibility_l382_38270

theorem pigeonhole_divisibility (n : ℕ) (a : Fin (n + 1) → ℤ) :
  ∃ i j : Fin (n + 1), i ≠ j ∧ (a i - a j) % n = 0 := by
  sorry

end pigeonhole_divisibility_l382_38270


namespace mary_james_seating_probability_l382_38222

-- Define the number of chairs
def total_chairs : ℕ := 10

-- Define the number of available chairs (excluding first and last)
def available_chairs : ℕ := total_chairs - 2

-- Define the probability of not sitting next to each other
def prob_not_adjacent : ℚ := 3/4

theorem mary_james_seating_probability :
  (1 : ℚ) - (available_chairs - 1 : ℚ) / (available_chairs.choose 2 : ℚ) = prob_not_adjacent :=
by sorry

end mary_james_seating_probability_l382_38222


namespace tetrahedron_volume_lower_bound_l382_38291

/-- Theorem: The volume of a tetrahedron is at least one-third the product of its opposite edge distances. -/
theorem tetrahedron_volume_lower_bound (d₁ d₂ d₃ V : ℝ) (h₁ : d₁ > 0) (h₂ : d₂ > 0) (h₃ : d₃ > 0) (hV : V > 0) :
  V ≥ (1/3) * d₁ * d₂ * d₃ := by
  sorry

end tetrahedron_volume_lower_bound_l382_38291


namespace symmetric_line_is_symmetric_l382_38265

/-- The point of symmetry -/
def P : ℝ × ℝ := (2, -1)

/-- The equation of the original line: 3x - y - 4 = 0 -/
def original_line (x y : ℝ) : Prop := 3 * x - y - 4 = 0

/-- The equation of the symmetric line: 3x - y - 7 = 0 -/
def symmetric_line (x y : ℝ) : Prop := 3 * x - y - 7 = 0

/-- Definition of symmetry with respect to a point -/
def is_symmetric (line1 line2 : (ℝ → ℝ → Prop)) (p : ℝ × ℝ) : Prop :=
  ∀ (x1 y1 x2 y2 : ℝ),
    line1 x1 y1 → line2 x2 y2 →
    (x1 + x2) / 2 = p.1 ∧ (y1 + y2) / 2 = p.2

/-- The main theorem: the symmetric_line is symmetric to the original_line with respect to P -/
theorem symmetric_line_is_symmetric :
  is_symmetric original_line symmetric_line P :=
sorry

end symmetric_line_is_symmetric_l382_38265
