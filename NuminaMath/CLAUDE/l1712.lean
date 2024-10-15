import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_factorization_l1712_171219

theorem polynomial_factorization (x y : ℝ) : x * y^2 - 16 * x = x * (y + 4) * (y - 4) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1712_171219


namespace NUMINAMATH_CALUDE_picture_frame_area_l1712_171287

theorem picture_frame_area (x y : ℤ) 
  (x_gt_one : x > 1) 
  (y_gt_one : y > 1) 
  (frame_area : (2*x + 4)*(y + 2) - x*y = 45) : 
  x*y = 15 := by
sorry

end NUMINAMATH_CALUDE_picture_frame_area_l1712_171287


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1712_171237

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := 5 / (3 + 4 * I)
  Complex.im z = -(4 / 5) := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1712_171237


namespace NUMINAMATH_CALUDE_window_width_calculation_l1712_171289

def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12
def door_length : ℝ := 6
def door_width : ℝ := 3
def window_height : ℝ := 3
def num_windows : ℕ := 3
def cost_per_sqft : ℝ := 3
def total_cost : ℝ := 2718

theorem window_width_calculation (W : ℝ) :
  (2 * (room_length * room_height + room_width * room_height) -
   door_length * door_width - num_windows * W * window_height) * cost_per_sqft = total_cost →
  W = 4 := by sorry

end NUMINAMATH_CALUDE_window_width_calculation_l1712_171289


namespace NUMINAMATH_CALUDE_sphere_cylinder_volume_ratio_l1712_171267

theorem sphere_cylinder_volume_ratio :
  ∀ (r : ℝ), r > 0 →
  (4 / 3 * Real.pi * r^3) / (Real.pi * r^2 * (2 * r)) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_volume_ratio_l1712_171267


namespace NUMINAMATH_CALUDE_count_solutions_power_diff_l1712_171214

/-- The number of solutions to x^n - y^n = 2^100 where x, y, n are positive integers and n > 1 -/
theorem count_solutions_power_diff : 
  (Finset.filter 
    (fun t : ℕ × ℕ × ℕ => 
      let (x, y, n) := t
      x > 0 ∧ y > 0 ∧ n > 1 ∧ x^n - y^n = 2^100)
    (Finset.product (Finset.range (2^100 + 1)) 
      (Finset.product (Finset.range (2^100 + 1)) (Finset.range 101)))).card = 49 := by
  sorry

end NUMINAMATH_CALUDE_count_solutions_power_diff_l1712_171214


namespace NUMINAMATH_CALUDE_probability_two_boys_l1712_171245

theorem probability_two_boys (total : ℕ) (boys : ℕ) (girls : ℕ) :
  total = 15 →
  boys = 8 →
  girls = 7 →
  boys + girls = total →
  (Nat.choose boys 2 : ℚ) / (Nat.choose total 2 : ℚ) = 4 / 15 := by
sorry

end NUMINAMATH_CALUDE_probability_two_boys_l1712_171245


namespace NUMINAMATH_CALUDE_happy_boys_count_l1712_171283

theorem happy_boys_count (total_children : ℕ) (happy_children : ℕ) (sad_children : ℕ) 
  (neutral_children : ℕ) (total_boys : ℕ) (total_girls : ℕ) (sad_girls : ℕ) 
  (neutral_boys : ℕ) (happy_boys_exist : Prop) :
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neutral_children = 20 →
  total_boys = 19 →
  total_girls = 41 →
  sad_girls = 4 →
  neutral_boys = 7 →
  happy_boys_exist →
  ∃ (happy_boys : ℕ), happy_boys = 6 ∧ 
    happy_boys + (sad_children - sad_girls) + neutral_boys = total_boys :=
by sorry

end NUMINAMATH_CALUDE_happy_boys_count_l1712_171283


namespace NUMINAMATH_CALUDE_skittles_distribution_l1712_171207

theorem skittles_distribution (initial_skittles : ℕ) (additional_skittles : ℕ) (num_people : ℕ) :
  initial_skittles = 14 →
  additional_skittles = 22 →
  num_people = 7 →
  (initial_skittles + additional_skittles) / num_people = 5 :=
by sorry

end NUMINAMATH_CALUDE_skittles_distribution_l1712_171207


namespace NUMINAMATH_CALUDE_binomial_10_9_l1712_171298

theorem binomial_10_9 : Nat.choose 10 9 = 10 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_9_l1712_171298


namespace NUMINAMATH_CALUDE_math_team_probability_l1712_171296

theorem math_team_probability : 
  let team_sizes : List Nat := [6, 8, 9]
  let num_teams : Nat := 3
  let num_cocaptains : Nat := 3
  let prob_select_team : Rat := 1 / num_teams
  let prob_select_cocaptains (n : Nat) : Rat := 6 / (n * (n - 1) * (n - 2))
  (prob_select_team * (team_sizes.map prob_select_cocaptains).sum : Rat) = 1 / 70 := by
  sorry

end NUMINAMATH_CALUDE_math_team_probability_l1712_171296


namespace NUMINAMATH_CALUDE_complex_point_on_line_l1712_171240

theorem complex_point_on_line (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  (z.re - z.im = 1) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_point_on_line_l1712_171240


namespace NUMINAMATH_CALUDE_chipped_marbles_count_l1712_171226

/-- Represents the number of marbles in each bag -/
def bags : List Nat := [18, 19, 21, 23, 25, 34]

/-- The total number of marbles -/
def total_marbles : Nat := bags.sum

/-- Predicate to check if a number is divisible by 3 -/
def divisible_by_three (n : Nat) : Prop := n % 3 = 0

/-- The number of bags Jane takes -/
def jane_bags : Nat := 3

/-- The number of bags George takes -/
def george_bags : Nat := 2

/-- The number of bags that remain -/
def remaining_bags : Nat := bags.length - jane_bags - george_bags

/-- Theorem stating the number of chipped marbles -/
theorem chipped_marbles_count : 
  ∃ (chipped : Nat) (jane george : List Nat),
    chipped ∈ bags ∧
    jane.length = jane_bags ∧
    george.length = george_bags ∧
    (jane.sum = 2 * george.sum) ∧
    (∀ m ∈ jane ++ george, m ≠ chipped) ∧
    divisible_by_three (total_marbles - chipped) ∧
    chipped = 23 := by
  sorry

end NUMINAMATH_CALUDE_chipped_marbles_count_l1712_171226


namespace NUMINAMATH_CALUDE_right_triangle_ab_length_l1712_171256

/-- 
Given a right triangle ABC in the x-y plane where:
- ∠B = 90°
- The length of AC is 225
- The slope of line segment AC is 4/3
Prove that the length of AB is 180.
-/
theorem right_triangle_ab_length 
  (A B C : ℝ × ℝ) -- Points in the plane
  (right_angle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0) -- ∠B = 90°
  (ac_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 225) -- Length of AC is 225
  (ac_slope : (C.2 - A.2) / (C.1 - A.1) = 4 / 3) -- Slope of AC is 4/3
  : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 180 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ab_length_l1712_171256


namespace NUMINAMATH_CALUDE_shadow_length_change_l1712_171205

/-- Represents the length of a shadow -/
inductive ShadowLength
  | Long
  | Short

/-- Represents a time of day -/
inductive TimeOfDay
  | Morning
  | Noon
  | Afternoon

/-- Represents the direction of a shadow -/
inductive ShadowDirection
  | West
  | North
  | East

/-- Function to determine shadow length based on time of day -/
def shadowLengthAtTime (time : TimeOfDay) : ShadowLength :=
  match time with
  | TimeOfDay.Morning => ShadowLength.Long
  | TimeOfDay.Noon => ShadowLength.Short
  | TimeOfDay.Afternoon => ShadowLength.Long

/-- Function to determine shadow direction based on time of day -/
def shadowDirectionAtTime (time : TimeOfDay) : ShadowDirection :=
  match time with
  | TimeOfDay.Morning => ShadowDirection.West
  | TimeOfDay.Noon => ShadowDirection.North
  | TimeOfDay.Afternoon => ShadowDirection.East

/-- Theorem stating the change in shadow length throughout the day -/
theorem shadow_length_change :
  ∀ (t1 t2 t3 : TimeOfDay),
    t1 = TimeOfDay.Morning →
    t2 = TimeOfDay.Noon →
    t3 = TimeOfDay.Afternoon →
    (shadowLengthAtTime t1 = ShadowLength.Long ∧
     shadowLengthAtTime t2 = ShadowLength.Short ∧
     shadowLengthAtTime t3 = ShadowLength.Long) :=
by
  sorry

#check shadow_length_change

end NUMINAMATH_CALUDE_shadow_length_change_l1712_171205


namespace NUMINAMATH_CALUDE_carlo_friday_practice_time_l1712_171288

/-- Represents Carlo's practice times for each day of the week -/
structure PracticeTimes where
  M : ℕ  -- Monday
  T : ℕ  -- Tuesday
  W : ℕ  -- Wednesday
  Th : ℕ -- Thursday
  F : ℕ  -- Friday

/-- Conditions for Carlo's practice schedule -/
def valid_practice_schedule (pt : PracticeTimes) : Prop :=
  pt.M = 2 * pt.T ∧
  pt.T = pt.W - 10 ∧
  pt.W = pt.Th + 5 ∧
  pt.Th = 50 ∧
  pt.M + pt.T + pt.W + pt.Th + pt.F = 300

/-- Theorem stating that given the conditions, Carlo should practice 60 minutes on Friday -/
theorem carlo_friday_practice_time (pt : PracticeTimes) 
  (h : valid_practice_schedule pt) : pt.F = 60 := by
  sorry

end NUMINAMATH_CALUDE_carlo_friday_practice_time_l1712_171288


namespace NUMINAMATH_CALUDE_number_of_students_l1712_171206

theorem number_of_students (S : ℕ) (N : ℕ) : 
  (4 * S + 3 = N) → (5 * S = N + 6) → S = 9 := by
sorry

end NUMINAMATH_CALUDE_number_of_students_l1712_171206


namespace NUMINAMATH_CALUDE_distance_XY_proof_l1712_171257

/-- The distance between points X and Y -/
def distance_XY : ℝ := 52

/-- Yolanda's walking speed in miles per hour -/
def yolanda_speed : ℝ := 3

/-- Bob's walking speed in miles per hour -/
def bob_speed : ℝ := 4

/-- The time difference between Yolanda's and Bob's start in hours -/
def time_difference : ℝ := 1

/-- The distance Bob has walked when they meet -/
def bob_distance : ℝ := 28

theorem distance_XY_proof :
  distance_XY = yolanda_speed * (bob_distance / bob_speed + time_difference) + bob_distance :=
by sorry

end NUMINAMATH_CALUDE_distance_XY_proof_l1712_171257


namespace NUMINAMATH_CALUDE_rational_root_count_l1712_171272

def polynomial (a₁ : ℤ) (x : ℚ) : ℚ := 12 * x^3 - 4 * x^2 + a₁ * x + 18

def is_possible_root (x : ℚ) : Prop :=
  ∃ (p q : ℤ), x = p / q ∧ 
  (p ∣ 18 ∨ p = 0) ∧ 
  (q ∣ 12 ∧ q ≠ 0)

theorem rational_root_count :
  ∃! (roots : Finset ℚ), 
    (∀ x ∈ roots, is_possible_root x) ∧
    (∀ x, is_possible_root x → x ∈ roots) ∧
    roots.card = 20 :=
sorry

end NUMINAMATH_CALUDE_rational_root_count_l1712_171272


namespace NUMINAMATH_CALUDE_coeff_x4_is_negative_30_l1712_171269

/-- The coefficient of x^4 in the expansion of (4x^2-2x-5)(x^2+1)^5 -/
def coeff_x4 : ℤ :=
  4 * (Nat.choose 5 3) - 5 * (Nat.choose 5 1)

/-- Theorem stating that the coefficient of x^4 is -30 -/
theorem coeff_x4_is_negative_30 : coeff_x4 = -30 := by
  sorry

end NUMINAMATH_CALUDE_coeff_x4_is_negative_30_l1712_171269


namespace NUMINAMATH_CALUDE_rectangle_area_l1712_171255

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width^2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1712_171255


namespace NUMINAMATH_CALUDE_expansion_dissimilar_terms_l1712_171211

/-- The number of dissimilar terms in the expansion of (a + b + c + d)^10 -/
def dissimilarTerms : ℕ := 286

/-- The number of variables in the expansion -/
def numVariables : ℕ := 4

/-- The exponent in the expansion -/
def exponent : ℕ := 10

/-- Theorem: The number of dissimilar terms in (a + b + c + d)^10 is 286 -/
theorem expansion_dissimilar_terms :
  dissimilarTerms = (numVariables + exponent - 1).choose (numVariables - 1) :=
sorry

end NUMINAMATH_CALUDE_expansion_dissimilar_terms_l1712_171211


namespace NUMINAMATH_CALUDE_spurs_team_size_l1712_171258

theorem spurs_team_size :
  ∀ (num_players : ℕ) (basketballs_per_player : ℕ) (total_basketballs : ℕ),
    basketballs_per_player = 11 →
    total_basketballs = 242 →
    total_basketballs = num_players * basketballs_per_player →
    num_players = 22 := by
  sorry

end NUMINAMATH_CALUDE_spurs_team_size_l1712_171258


namespace NUMINAMATH_CALUDE_average_flux_1_to_999_l1712_171212

/-- The flux of a positive integer is the number of times the digits change from increasing to decreasing or vice versa, ignoring consecutive equal digits. -/
def flux (n : ℕ+) : ℕ := sorry

/-- The sum of fluxes for all positive integers from 1 to 999, inclusive. -/
def sum_of_fluxes : ℕ := sorry

theorem average_flux_1_to_999 :
  (sum_of_fluxes : ℚ) / 999 = 175 / 333 := by sorry

end NUMINAMATH_CALUDE_average_flux_1_to_999_l1712_171212


namespace NUMINAMATH_CALUDE_distribution_centers_count_l1712_171204

/-- The number of unique representations using either a single color or a pair of different colors -/
def uniqueRepresentations (n : ℕ) : ℕ := n + n.choose 2

/-- Theorem stating that with 5 colors, there are 15 unique representations -/
theorem distribution_centers_count : uniqueRepresentations 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_distribution_centers_count_l1712_171204


namespace NUMINAMATH_CALUDE_total_money_l1712_171241

/-- The total amount of money A, B, and C have together is 500, given the specified conditions. -/
theorem total_money (a b c : ℕ) : 
  a + c = 200 → b + c = 330 → c = 30 → a + b + c = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l1712_171241


namespace NUMINAMATH_CALUDE_perpendicular_lines_coefficient_l1712_171228

/-- Given two lines in the xy-plane, this theorem proves that if they are perpendicular,
    then the coefficient 'a' in the first line's equation must equal 2/3. -/
theorem perpendicular_lines_coefficient (a : ℝ) :
  (∀ x y : ℝ, x + a * y + 2 = 0 → 2 * x + 3 * y + 1 = 0 → 
   ((-1 : ℝ) / a) * (-2 / 3) = -1) →
  a = 2 / 3 := by
sorry


end NUMINAMATH_CALUDE_perpendicular_lines_coefficient_l1712_171228


namespace NUMINAMATH_CALUDE_gcd_properties_l1712_171213

theorem gcd_properties (a b : ℤ) (h : Nat.gcd a.natAbs b.natAbs = 1) :
  (Nat.gcd (a + b).natAbs (a * b).natAbs = 1 ∧
   Nat.gcd (a - b).natAbs (a * b).natAbs = 1) ∧
  (Nat.gcd (a + b).natAbs (a - b).natAbs = 1 ∨
   Nat.gcd (a + b).natAbs (a - b).natAbs = 2) := by
  sorry

end NUMINAMATH_CALUDE_gcd_properties_l1712_171213


namespace NUMINAMATH_CALUDE_alice_additional_spend_l1712_171224

/-- The amount Alice needs to spend for free delivery -/
def free_delivery_threshold : ℚ := 35

/-- The cost of chicken per pound -/
def chicken_price : ℚ := 6

/-- The amount of chicken in pounds -/
def chicken_amount : ℚ := 3/2

/-- The cost of lettuce -/
def lettuce_price : ℚ := 3

/-- The cost of cherry tomatoes -/
def tomatoes_price : ℚ := 5/2

/-- The cost of one sweet potato -/
def sweet_potato_price : ℚ := 3/4

/-- The number of sweet potatoes -/
def sweet_potato_count : ℕ := 4

/-- The cost of one head of broccoli -/
def broccoli_price : ℚ := 2

/-- The number of broccoli heads -/
def broccoli_count : ℕ := 2

/-- The cost of Brussel sprouts -/
def brussel_sprouts_price : ℚ := 5/2

/-- The total cost of items in Alice's cart -/
def cart_total : ℚ :=
  chicken_price * chicken_amount + lettuce_price + tomatoes_price +
  sweet_potato_price * sweet_potato_count + broccoli_price * broccoli_count +
  brussel_sprouts_price

/-- The additional amount Alice needs to spend for free delivery -/
def additional_spend : ℚ := free_delivery_threshold - cart_total

theorem alice_additional_spend :
  additional_spend = 11 := by sorry

end NUMINAMATH_CALUDE_alice_additional_spend_l1712_171224


namespace NUMINAMATH_CALUDE_valentines_count_l1712_171200

theorem valentines_count (boys girls : ℕ) : 
  boys * girls = boys + girls + 16 → boys * girls = 36 := by
  sorry

end NUMINAMATH_CALUDE_valentines_count_l1712_171200


namespace NUMINAMATH_CALUDE_weight_of_b_l1712_171227

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 41) :
  b = 27 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l1712_171227


namespace NUMINAMATH_CALUDE_triangle_division_l1712_171262

/-- A quadrilateral that is both inscribed in a circle and circumscribed about a circle -/
structure BicentricQuadrilateral where
  -- We don't need to define the structure completely, just its existence
  mk :: (dummy : Unit)

/-- Represents a division of a triangle into bicentric quadrilaterals -/
def TriangleDivision (n : ℕ) := 
  { division : List BicentricQuadrilateral // division.length = n }

/-- The main theorem: any triangle can be divided into n bicentric quadrilaterals for n ≥ 3 -/
theorem triangle_division (n : ℕ) (h : n ≥ 3) : 
  ∃ (division : TriangleDivision n), True :=
sorry

end NUMINAMATH_CALUDE_triangle_division_l1712_171262


namespace NUMINAMATH_CALUDE_yards_mowed_l1712_171220

/-- The problem of calculating how many yards Christian mowed --/
theorem yards_mowed (perfume_price : ℕ) (christian_savings sue_savings : ℕ)
  (yard_price : ℕ) (dogs_walked dog_price : ℕ) (remaining : ℕ) :
  perfume_price = 50 →
  christian_savings = 5 →
  sue_savings = 7 →
  yard_price = 5 →
  dogs_walked = 6 →
  dog_price = 2 →
  remaining = 6 →
  (perfume_price - (christian_savings + sue_savings + dogs_walked * dog_price + remaining)) / yard_price = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_yards_mowed_l1712_171220


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l1712_171285

theorem sqrt_sum_inequality : Real.sqrt 2 + Real.sqrt 11 < Real.sqrt 3 + Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l1712_171285


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l1712_171254

/-- 
Given a triangle with sides in the ratio 5 : 6 : 7 and a perimeter of 720 cm,
prove that the longest side has a length of 280 cm.
-/
theorem longest_side_of_triangle (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (ratio : a / 5 = b / 6 ∧ b / 6 = c / 7)
  (perimeter : a + b + c = 720) :
  c = 280 := by
  sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l1712_171254


namespace NUMINAMATH_CALUDE_friends_average_age_l1712_171276

def average_age (m : ℝ) : ℝ := 1.05 * m + 21.6

theorem friends_average_age (m : ℝ) :
  let john := 1.5 * m
  let mary := m
  let tonya := 60
  let sam := 0.8 * tonya
  let carol := 2.75 * m
  (john + mary + tonya + sam + carol) / 5 = average_age m := by
  sorry

end NUMINAMATH_CALUDE_friends_average_age_l1712_171276


namespace NUMINAMATH_CALUDE_min_value_expr_l1712_171270

theorem min_value_expr (x y : ℝ) (h1 : x > 0) (h2 : y > -1) (h3 : x + y = 1) :
  (x^2 + 3) / x + y^2 / (y + 1) ≥ 2 + Real.sqrt 3 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > -1 ∧ x₀ + y₀ = 1 ∧
    (x₀^2 + 3) / x₀ + y₀^2 / (y₀ + 1) = 2 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expr_l1712_171270


namespace NUMINAMATH_CALUDE_ellipse_sum_is_twelve_l1712_171273

/-- Represents an ellipse with center (h, k) and semi-axes a and b -/
structure Ellipse where
  h : ℝ  -- x-coordinate of center
  k : ℝ  -- y-coordinate of center
  a : ℝ  -- length of semi-major axis
  b : ℝ  -- length of semi-minor axis

/-- The sum of center coordinates and semi-axes lengths for a specific ellipse -/
def ellipse_sum (e : Ellipse) : ℝ :=
  e.h + e.k + e.a + e.b

/-- Theorem: The sum of center coordinates and semi-axes lengths for the given ellipse is 12 -/
theorem ellipse_sum_is_twelve : 
  let e : Ellipse := { h := 3, k := -2, a := 7, b := 4 }
  ellipse_sum e = 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_is_twelve_l1712_171273


namespace NUMINAMATH_CALUDE_mrs_heine_items_l1712_171229

/-- The number of items Mrs. Heine will buy for her dogs -/
def total_items (num_dogs : ℕ) (biscuits_per_dog : ℕ) (boots_per_set : ℕ) : ℕ :=
  num_dogs * (biscuits_per_dog + boots_per_set)

/-- Proof that Mrs. Heine will buy 18 items -/
theorem mrs_heine_items : total_items 2 5 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_mrs_heine_items_l1712_171229


namespace NUMINAMATH_CALUDE_fraction_addition_l1712_171294

theorem fraction_addition : (5 / (8/13)) + (4/7) = 487/56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1712_171294


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l1712_171248

theorem cheryl_material_usage 
  (material1 : ℚ) 
  (material2 : ℚ) 
  (leftover : ℚ) 
  (h1 : material1 = 2 / 9) 
  (h2 : material2 = 1 / 8) 
  (h3 : leftover = 4 / 18) : 
  material1 + material2 - leftover = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_material_usage_l1712_171248


namespace NUMINAMATH_CALUDE_students_in_line_l1712_171263

theorem students_in_line (total : ℕ) (behind : ℕ) (h1 : total = 25) (h2 : behind = 13) :
  total - (behind + 1) = 11 := by
  sorry

end NUMINAMATH_CALUDE_students_in_line_l1712_171263


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1712_171253

/-- Given a train of length 360 meters passing a bridge of length 240 meters in 4 minutes,
    prove that the speed of the train is 2.5 m/s. -/
theorem train_speed_calculation (train_length : ℝ) (bridge_length : ℝ) (time_minutes : ℝ) :
  train_length = 360 →
  bridge_length = 240 →
  time_minutes = 4 →
  (train_length + bridge_length) / (time_minutes * 60) = 2.5 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l1712_171253


namespace NUMINAMATH_CALUDE_lap_time_improvement_is_12_seconds_l1712_171290

-- Define the initial condition
def initial_laps : ℕ := 25
def initial_time : ℕ := 50

-- Define the later condition
def later_laps : ℕ := 30
def later_time : ℕ := 54

-- Define the function to calculate lap time in seconds
def lap_time_seconds (laps : ℕ) (time : ℕ) : ℚ :=
  (time * 60) / laps

-- Define the improvement in lap time
def lap_time_improvement : ℚ :=
  lap_time_seconds initial_laps initial_time - lap_time_seconds later_laps later_time

-- Theorem statement
theorem lap_time_improvement_is_12_seconds :
  lap_time_improvement = 12 := by sorry

end NUMINAMATH_CALUDE_lap_time_improvement_is_12_seconds_l1712_171290


namespace NUMINAMATH_CALUDE_baseball_cost_l1712_171297

/-- The cost of a baseball given the cost of a football, total payment, and change received. -/
theorem baseball_cost (football_cost change_received total_payment : ℚ) 
  (h1 : football_cost = 9.14)
  (h2 : change_received = 4.05)
  (h3 : total_payment = 20) : 
  total_payment - change_received - football_cost = 6.81 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cost_l1712_171297


namespace NUMINAMATH_CALUDE_filled_circles_in_2009_l1712_171280

/-- Represents the cumulative number of circles (both filled and empty) after n filled circles -/
def s (n : ℕ) : ℕ := (n^2 + n) / 2

/-- Represents the pattern where the nth filled circle is followed by n empty circles -/
def circle_pattern (n : ℕ) : ℕ := n + 1

theorem filled_circles_in_2009 : 
  ∃ k : ℕ, k = 63 ∧ s k ≤ 2009 ∧ s (k + 1) > 2009 :=
sorry

end NUMINAMATH_CALUDE_filled_circles_in_2009_l1712_171280


namespace NUMINAMATH_CALUDE_ship_supplies_l1712_171286

/-- Calculates the remaining supplies on a ship given initial amount and usage rates --/
theorem ship_supplies (initial_supply : ℚ) (first_day_usage : ℚ) (next_days_usage : ℚ) :
  initial_supply = 400 ∧ 
  first_day_usage = 2/5 ∧ 
  next_days_usage = 3/5 →
  initial_supply * (1 - first_day_usage) * (1 - next_days_usage) = 96 := by
  sorry

end NUMINAMATH_CALUDE_ship_supplies_l1712_171286


namespace NUMINAMATH_CALUDE_geometry_propositions_l1712_171277

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations between lines and planes
def subset (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def intersect (p1 p2 : Plane) (l : Line) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem geometry_propositions 
  (m n : Line) (α β : Plane) 
  (h_distinct_lines : m ≠ n) 
  (h_distinct_planes : α ≠ β) :
  (∀ (m : Line) (α β : Plane), 
    subset m α → perpendicular m β → perpendicular_planes α β) ∧
  (∃ (m : Line) (α β : Plane) (n : Line), 
    subset m α ∧ intersect α β n ∧ perpendicular_planes α β ∧ ¬(perpendicular m n)) ∧
  (∃ (m n : Line) (α β : Plane), 
    subset m α ∧ subset n β ∧ parallel_planes α β ∧ ¬(parallel_lines m n)) ∧
  (∀ (m n : Line) (α β : Plane), 
    parallel_line_plane m α → subset m β → intersect α β n → parallel_lines m n) := by
  sorry


end NUMINAMATH_CALUDE_geometry_propositions_l1712_171277


namespace NUMINAMATH_CALUDE_curve_transformation_l1712_171203

theorem curve_transformation (x : ℝ) : 2 * Real.cos (2 * (x - π/3)) = Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_curve_transformation_l1712_171203


namespace NUMINAMATH_CALUDE_function_multiple_preimages_l1712_171242

theorem function_multiple_preimages :
  ∃ (f : ℝ → ℝ) (y : ℝ), ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = y ∧ f x₂ = y := by
  sorry

end NUMINAMATH_CALUDE_function_multiple_preimages_l1712_171242


namespace NUMINAMATH_CALUDE_horner_operations_count_l1712_171234

/-- Represents a univariate polynomial --/
structure UnivariatePoly (α : Type*) where
  coeffs : List α

/-- Horner's method for polynomial evaluation --/
def hornerMethod (p : UnivariatePoly ℤ) : ℕ × ℕ :=
  (p.coeffs.length - 1, p.coeffs.length - 1)

/-- The given polynomial f(x) = 5x^5 + 4x^4 + 3x^3 + 2x^2 + x + 1 --/
def f : UnivariatePoly ℤ :=
  ⟨[1, 1, 2, 3, 4, 5]⟩

theorem horner_operations_count :
  hornerMethod f = (5, 5) := by sorry

end NUMINAMATH_CALUDE_horner_operations_count_l1712_171234


namespace NUMINAMATH_CALUDE_geometric_sum_first_eight_l1712_171243

def geometric_sequence (a r : ℚ) (n : ℕ) : ℚ := a * r ^ (n - 1)

def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_eight :
  let a : ℚ := 1/3
  let r : ℚ := 1/3
  let n : ℕ := 8
  geometric_sum a r n = 3280/6561 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_first_eight_l1712_171243


namespace NUMINAMATH_CALUDE_soda_cost_l1712_171246

theorem soda_cost (bill : ℕ) (change : ℕ) (num_sodas : ℕ) (h1 : bill = 20) (h2 : change = 14) (h3 : num_sodas = 3) :
  (bill - change) / num_sodas = 2 := by
sorry

end NUMINAMATH_CALUDE_soda_cost_l1712_171246


namespace NUMINAMATH_CALUDE_square_of_binomial_l1712_171271

theorem square_of_binomial (b : ℝ) : 
  (∃ c : ℝ, ∀ x : ℝ, 9 * x^2 + 24 * x + b = (3 * x + c)^2) → b = 16 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_l1712_171271


namespace NUMINAMATH_CALUDE_cubic_factorization_sum_l1712_171282

theorem cubic_factorization_sum (a b c d e : ℤ) : 
  (∀ x, 1728 * x^3 + 27 = (a*x + b) * (c*x^2 + d*x + e)) →
  a + b + c + d + e = 132 := by
sorry

end NUMINAMATH_CALUDE_cubic_factorization_sum_l1712_171282


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1712_171299

theorem quadratic_equation_solution (x : ℝ) : x^2 - 2*x - 8 = 0 → x = 4 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1712_171299


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1712_171295

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 + 5*x₁ - 3 = 0) → (x₂^2 + 5*x₂ - 3 = 0) → (x₁ + x₂ = -5) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1712_171295


namespace NUMINAMATH_CALUDE_cone_height_l1712_171202

-- Define the cone
structure Cone where
  surfaceArea : ℝ
  centralAngle : ℝ

-- Theorem statement
theorem cone_height (c : Cone) 
  (h1 : c.surfaceArea = π) 
  (h2 : c.centralAngle = 2 * π / 3) : 
  ∃ h : ℝ, h = Real.sqrt 2 ∧ h > 0 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_l1712_171202


namespace NUMINAMATH_CALUDE_apples_given_to_neighbor_l1712_171268

def initial_apples : ℕ := 127
def remaining_apples : ℕ := 39

theorem apples_given_to_neighbor :
  initial_apples - remaining_apples = 88 :=
by sorry

end NUMINAMATH_CALUDE_apples_given_to_neighbor_l1712_171268


namespace NUMINAMATH_CALUDE_weekly_distance_calculation_l1712_171265

/-- Calculates the weekly running distance given the number of days, hours per day, and speed. -/
def weekly_running_distance (days_per_week : ℕ) (hours_per_day : ℝ) (speed_mph : ℝ) : ℝ :=
  days_per_week * hours_per_day * speed_mph

/-- Proves that running 5 days a week, 1.5 hours each day, at 8 mph results in 60 miles per week. -/
theorem weekly_distance_calculation :
  weekly_running_distance 5 1.5 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_weekly_distance_calculation_l1712_171265


namespace NUMINAMATH_CALUDE_oldest_babysat_age_l1712_171236

-- Define constants
def jane_start_age : ℕ := 16
def jane_current_age : ℕ := 32
def years_since_stopped : ℕ := 10

-- Define the theorem
theorem oldest_babysat_age :
  ∀ (oldest_age : ℕ),
  (oldest_age = (jane_current_age - years_since_stopped) / 2 + years_since_stopped) →
  (oldest_age ≤ jane_current_age) →
  (∀ (jane_age : ℕ) (child_age : ℕ),
    jane_start_age ≤ jane_age →
    jane_age ≤ jane_current_age - years_since_stopped →
    child_age ≤ jane_age / 2 →
    child_age + (jane_current_age - jane_age) ≤ oldest_age) →
  oldest_age = 21 :=
by sorry

end NUMINAMATH_CALUDE_oldest_babysat_age_l1712_171236


namespace NUMINAMATH_CALUDE_solution_satisfies_conditions_l1712_171251

/-- Represents the number of teeth each person has -/
structure TeethCount where
  dima : ℕ
  yulia : ℕ
  kolya : ℕ
  vanya : ℕ

/-- Checks if the given teeth count satisfies all conditions of the problem -/
def satisfiesConditions (tc : TeethCount) : Prop :=
  tc.dima = tc.yulia + 2 ∧
  tc.kolya = tc.dima + tc.yulia ∧
  tc.vanya = 2 * tc.kolya ∧
  tc.dima + tc.yulia + tc.kolya + tc.vanya = 64

/-- The theorem stating that the solution satisfies all conditions -/
theorem solution_satisfies_conditions : 
  satisfiesConditions ⟨9, 7, 16, 32⟩ := by sorry

end NUMINAMATH_CALUDE_solution_satisfies_conditions_l1712_171251


namespace NUMINAMATH_CALUDE_tony_money_left_l1712_171281

/-- The amount of money Tony has left after purchases at a baseball game. -/
def money_left (initial_amount ticket_cost hot_dog_cost drink_cost cap_cost : ℕ) : ℕ :=
  initial_amount - ticket_cost - hot_dog_cost - drink_cost - cap_cost

/-- Theorem stating that Tony has $13 left after his purchases. -/
theorem tony_money_left : 
  money_left 50 16 5 4 12 = 13 := by
  sorry

end NUMINAMATH_CALUDE_tony_money_left_l1712_171281


namespace NUMINAMATH_CALUDE_kolya_is_wrong_l1712_171215

/-- Represents the statements made by each boy -/
structure Statements where
  vasya : ℕ → Prop
  kolya : ℕ → Prop
  petya : ℕ → ℕ → Prop
  misha : ℕ → ℕ → Prop

/-- The actual statements made by the boys -/
def boys_statements : Statements where
  vasya := λ b => b ≥ 4
  kolya := λ g => g ≥ 5
  petya := λ b g => b ≥ 3 ∧ g ≥ 4
  misha := λ b g => b ≥ 4 ∧ g ≥ 4

/-- Theorem stating that Kolya's statement is the only one that can be false -/
theorem kolya_is_wrong (s : Statements) (b g : ℕ) :
  s = boys_statements →
  (s.vasya b ∧ s.petya b g ∧ s.misha b g ∧ ¬s.kolya g) ↔
  (b ≥ 4 ∧ g = 4) :=
sorry

end NUMINAMATH_CALUDE_kolya_is_wrong_l1712_171215


namespace NUMINAMATH_CALUDE_gerald_remaining_money_l1712_171231

/-- Represents the cost of items and currency conversions --/
structure Costs where
  meat_pie : ℕ
  sausage_roll : ℕ
  farthings_per_pfennig : ℕ
  pfennigs_per_groat : ℕ
  groats_per_florin : ℕ

/-- Represents Gerald's initial money --/
structure GeraldMoney where
  farthings : ℕ
  groats : ℕ
  florins : ℕ

/-- Calculates the remaining pfennigs after purchase --/
def remaining_pfennigs (c : Costs) (m : GeraldMoney) : ℕ :=
  let total_pfennigs := 
    m.farthings / c.farthings_per_pfennig +
    m.groats * c.pfennigs_per_groat +
    m.florins * c.groats_per_florin * c.pfennigs_per_groat
  total_pfennigs - (c.meat_pie + c.sausage_roll)

/-- Theorem stating Gerald's remaining pfennigs --/
theorem gerald_remaining_money (c : Costs) (m : GeraldMoney) 
  (h1 : c.meat_pie = 120)
  (h2 : c.sausage_roll = 75)
  (h3 : m.farthings = 54)
  (h4 : m.groats = 8)
  (h5 : m.florins = 17)
  (h6 : c.farthings_per_pfennig = 6)
  (h7 : c.pfennigs_per_groat = 4)
  (h8 : c.groats_per_florin = 10) :
  remaining_pfennigs c m = 526 := by
  sorry

end NUMINAMATH_CALUDE_gerald_remaining_money_l1712_171231


namespace NUMINAMATH_CALUDE_larger_number_proof_l1712_171249

theorem larger_number_proof (x y : ℝ) (h1 : x + y = 45) (h2 : x - y = 5) (h3 : x ≥ y) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1712_171249


namespace NUMINAMATH_CALUDE_cone_volume_l1712_171279

theorem cone_volume (slant_height height : ℝ) (h1 : slant_height = 15) (h2 : height = 8) :
  (1 / 3 : ℝ) * π * (slant_height ^ 2 - height ^ 2) * height = 429 * (1 / 3 : ℝ) * π :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l1712_171279


namespace NUMINAMATH_CALUDE_venny_car_cost_l1712_171238

def original_price : ℝ := 37500

def discount_percentage : ℝ := 40

theorem venny_car_cost : ℝ := by
  -- Define the amount Venny spent as 40% of the original price
  let amount_spent := (discount_percentage / 100) * original_price
  
  -- Prove that this amount is equal to $15,000
  sorry

end NUMINAMATH_CALUDE_venny_car_cost_l1712_171238


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1712_171274

-- Define the quadratic expression
def f (a x : ℝ) : ℝ := a * x^2 + (a - 2) * x - 2

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then { x | x ≥ -1 }
  else if a > 0 then { x | -1 ≤ x ∧ x ≤ 2/a }
  else if -2 < a ∧ a < 0 then { x | x ≤ 2/a ∨ x ≥ -1 }
  else if a < -2 then { x | x ≤ -1 ∨ x ≥ 2/a }
  else Set.univ

-- State the theorem
theorem quadratic_inequality_solution (a : ℝ) :
  { x : ℝ | f a x ≤ 0 } = solution_set a :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1712_171274


namespace NUMINAMATH_CALUDE_tape_left_over_l1712_171222

/-- Calculates the amount of tape left over after wrapping a rectangular field once -/
theorem tape_left_over (total_tape : ℕ) (width : ℕ) (length : ℕ) : 
  total_tape = 250 → width = 20 → length = 60 → 
  total_tape - 2 * (width + length) = 90 := by
  sorry

end NUMINAMATH_CALUDE_tape_left_over_l1712_171222


namespace NUMINAMATH_CALUDE_apple_distribution_l1712_171264

theorem apple_distribution (martha_initial : ℕ) (jane_apples : ℕ) (martha_final : ℕ) (martha_remaining : ℕ) :
  martha_initial = 20 →
  jane_apples = 5 →
  martha_remaining = 4 →
  martha_final = martha_remaining + 4 →
  martha_initial - jane_apples - martha_final = jane_apples + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l1712_171264


namespace NUMINAMATH_CALUDE_x_plus_y_equals_two_l1712_171208

theorem x_plus_y_equals_two (x y : ℝ) 
  (hx : (x - 1)^3 + 1997*(x - 1) = -1)
  (hy : (y - 1)^3 + 1997*(y - 1) = 1) : 
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_two_l1712_171208


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1712_171291

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) → 
  a = 5 ∧ b = -6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1712_171291


namespace NUMINAMATH_CALUDE_sqrt_two_division_l1712_171250

theorem sqrt_two_division : 2 * Real.sqrt 2 / Real.sqrt 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_division_l1712_171250


namespace NUMINAMATH_CALUDE_quadratic_common_root_l1712_171278

-- Define the quadratic functions
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def g (a b c x : ℝ) : ℝ := c * x^2 + b * x + a

-- State the theorem
theorem quadratic_common_root (a b c : ℝ) :
  (∃! x, f a b c x + g a b c x = 0) →
  (∃ x, f a b c x = 0 ∧ g a b c x = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_common_root_l1712_171278


namespace NUMINAMATH_CALUDE_system_solution_l1712_171259

theorem system_solution (x y : ℝ) : 
  (6 * (1 - x)^2 = 1 / y ∧ 6 * (1 - y)^2 = 1 / x) ↔ 
  ((x = 3/2 ∧ y = 2/3) ∨ 
   (x = 2/3 ∧ y = 3/2) ∨ 
   (x = (1/6) * (4 + 2^(2/3) + 2^(4/3)) ∧ y = (1/6) * (4 + 2^(2/3) + 2^(4/3)))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1712_171259


namespace NUMINAMATH_CALUDE_candy_problem_l1712_171230

theorem candy_problem (initial_candies : ℕ) : 
  let day1_remaining := initial_candies - (initial_candies / 4) - 3
  let day2_remaining := day1_remaining - (day1_remaining / 4) - 5
  day2_remaining = 10 →
  initial_candies = 84 := by
sorry

end NUMINAMATH_CALUDE_candy_problem_l1712_171230


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1712_171221

/-- A line in the form kx + y + k = 0 passes through the point (-1, 0) for all real k. -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k * (-1) + 0 + k = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1712_171221


namespace NUMINAMATH_CALUDE_hash_five_two_l1712_171218

-- Define the # operation
def hash (a b : ℤ) : ℤ := (a + 2*b) * (a - 2*b)

-- Theorem statement
theorem hash_five_two : hash 5 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_hash_five_two_l1712_171218


namespace NUMINAMATH_CALUDE_special_polynomial_value_l1712_171209

theorem special_polynomial_value (x : ℝ) (h : x + 1/x = 3) : 
  x^10 - 5*x^6 + x^2 = 8436*x - 338 := by
sorry

end NUMINAMATH_CALUDE_special_polynomial_value_l1712_171209


namespace NUMINAMATH_CALUDE_children_toothpaste_sales_amount_l1712_171247

/-- Calculates the total sales amount for children's toothpaste. -/
def total_sales_amount (num_boxes : ℕ) (packs_per_box : ℕ) (price_per_pack : ℕ) : ℕ :=
  num_boxes * packs_per_box * price_per_pack

/-- Proves that the total sales amount for the given conditions is 1200 yuan. -/
theorem children_toothpaste_sales_amount :
  total_sales_amount 12 25 4 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_children_toothpaste_sales_amount_l1712_171247


namespace NUMINAMATH_CALUDE_range_of_a_l1712_171210

theorem range_of_a (a x : ℝ) : 
  (∀ x, (x^2 - 7*x + 10 ≤ 0 → a < x ∧ x < a + 1) ∧ 
        (a < x ∧ x < a + 1 → ¬(∀ y, a < y ∧ y < a + 1 → y^2 - 7*y + 10 ≤ 0))) →
  2 ≤ a ∧ a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1712_171210


namespace NUMINAMATH_CALUDE_half_vector_AB_l1712_171235

/-- Given two vectors OA and OB in ℝ², prove that half of vector AB equals (1/2, 5/2) -/
theorem half_vector_AB (OA OB : ℝ × ℝ) (h1 : OA = (3, 2)) (h2 : OB = (4, 7)) :
  (1 / 2 : ℝ) • (OB - OA) = (1/2, 5/2) := by sorry

end NUMINAMATH_CALUDE_half_vector_AB_l1712_171235


namespace NUMINAMATH_CALUDE_max_value_of_g_l1712_171260

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x - x^4

-- State the theorem
theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l1712_171260


namespace NUMINAMATH_CALUDE_specific_polygon_perimeter_l1712_171292

/-- The perimeter of a polygon consisting of a rectangle and a right triangle -/
def polygon_perimeter (rect_side1 rect_side2 triangle_hypotenuse : ℝ) : ℝ :=
  2 * (rect_side1 + rect_side2) - rect_side2 + triangle_hypotenuse

/-- Theorem: The perimeter of the specific polygon is 21 units -/
theorem specific_polygon_perimeter :
  polygon_perimeter 6 4 5 = 21 := by
  sorry

#eval polygon_perimeter 6 4 5

end NUMINAMATH_CALUDE_specific_polygon_perimeter_l1712_171292


namespace NUMINAMATH_CALUDE_population_growth_rate_exists_and_unique_l1712_171201

theorem population_growth_rate_exists_and_unique :
  ∃! r : ℝ, 0 < r ∧ r < 1 ∧ 20000 * (1 + r)^3 = 26620 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_rate_exists_and_unique_l1712_171201


namespace NUMINAMATH_CALUDE_jacob_needs_26_more_fish_l1712_171275

def fishing_tournament (jacob_initial : ℕ) (alex_multiplier : ℕ) (alex_loss : ℕ) : ℕ :=
  let alex_initial := jacob_initial * alex_multiplier
  let alex_final := alex_initial - alex_loss
  let jacob_target := alex_final + 1
  jacob_target - jacob_initial

theorem jacob_needs_26_more_fish :
  fishing_tournament 8 7 23 = 26 := by
  sorry

end NUMINAMATH_CALUDE_jacob_needs_26_more_fish_l1712_171275


namespace NUMINAMATH_CALUDE_sin_double_angle_minus_pi_half_l1712_171232

/-- Given an angle α in the Cartesian coordinate system with the specified properties,
    prove that sin(2α - π/2) = -1/2 -/
theorem sin_double_angle_minus_pi_half (α : ℝ) : 
  (∃ (x y : ℝ), x = Real.sqrt 3 ∧ y = -1 ∧ 
   x * Real.cos α = x ∧ x * Real.sin α = y) →
  Real.sin (2 * α - π / 2) = -1 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_double_angle_minus_pi_half_l1712_171232


namespace NUMINAMATH_CALUDE_sum_squares_50_rings_l1712_171266

/-- The number of squares in the nth ring of a square array -/
def squares_in_ring (n : ℕ) : ℕ := 8 * n

/-- The sum of squares from the 1st to the nth ring -/
def sum_squares (n : ℕ) : ℕ := 
  (List.range n).map squares_in_ring |>.sum

/-- Theorem stating that the sum of squares in the first 50 rings is 10200 -/
theorem sum_squares_50_rings : sum_squares 50 = 10200 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_50_rings_l1712_171266


namespace NUMINAMATH_CALUDE_least_multiple_of_25_greater_than_450_l1712_171233

theorem least_multiple_of_25_greater_than_450 :
  ∀ n : ℕ, n > 0 ∧ 25 ∣ n ∧ n > 450 → n ≥ 475 :=
by
  sorry

end NUMINAMATH_CALUDE_least_multiple_of_25_greater_than_450_l1712_171233


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l1712_171244

theorem right_triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (pythagorean : a^2 + b^2 = c^2) : (a + b) / (a * b / c) ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l1712_171244


namespace NUMINAMATH_CALUDE_summer_camp_boys_l1712_171293

theorem summer_camp_boys (total : ℕ) (teachers : ℕ) (boy_ratio girl_ratio : ℕ) :
  total = 65 →
  teachers = 5 →
  boy_ratio = 3 →
  girl_ratio = 4 →
  ∃ (boys girls : ℕ),
    boys + girls + teachers = total ∧
    boys * girl_ratio = girls * boy_ratio ∧
    boys = 26 :=
by sorry

end NUMINAMATH_CALUDE_summer_camp_boys_l1712_171293


namespace NUMINAMATH_CALUDE_cube_remainder_mod_nine_l1712_171225

theorem cube_remainder_mod_nine (n : ℤ) :
  (n % 9 = 2 ∨ n % 9 = 5 ∨ n % 9 = 8) → n^3 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_remainder_mod_nine_l1712_171225


namespace NUMINAMATH_CALUDE_calculation_proof_l1712_171239

theorem calculation_proof :
  let four_million : ℝ := 4 * 10^6
  let four_hundred_thousand : ℝ := 4 * 10^5
  let four_billion : ℝ := 4 * 10^9
  (four_million * four_hundred_thousand + four_billion) = 1.604 * 10^12 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1712_171239


namespace NUMINAMATH_CALUDE_hostel_cost_23_days_l1712_171284

/-- Calculate the cost of staying in a student youth hostel for a given number of days. -/
def hostelCost (days : ℕ) : ℚ :=
  let firstWeekRate : ℚ := 18
  let additionalWeekRate : ℚ := 11
  let firstWeekDays : ℕ := min days 7
  let additionalDays : ℕ := days - firstWeekDays
  firstWeekRate * firstWeekDays + additionalWeekRate * additionalDays

/-- The cost of staying for 23 days in the student youth hostel is $302.00. -/
theorem hostel_cost_23_days : hostelCost 23 = 302 := by
  sorry

#eval hostelCost 23

end NUMINAMATH_CALUDE_hostel_cost_23_days_l1712_171284


namespace NUMINAMATH_CALUDE_simplify_expressions_l1712_171223

theorem simplify_expressions (x y : ℝ) :
  (2 * (2 * x - y) - (x + y) = 3 * x - 3 * y) ∧
  (x^2 * y + (-3 * (2 * x * y - x^2 * y) - x * y) = 4 * x^2 * y - 7 * x * y) := by
sorry

end NUMINAMATH_CALUDE_simplify_expressions_l1712_171223


namespace NUMINAMATH_CALUDE_student_distribution_l1712_171217

theorem student_distribution (total : ℝ) (third_year : ℝ) (second_year : ℝ)
  (h1 : third_year = 0.5 * total)
  (h2 : second_year = 0.3 * total)
  (h3 : total > 0) :
  second_year / (total - third_year) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_student_distribution_l1712_171217


namespace NUMINAMATH_CALUDE_subset_M_l1712_171261

def M : Set ℝ := {x | x + 1 > 0}

theorem subset_M : {0} ⊆ M := by sorry

end NUMINAMATH_CALUDE_subset_M_l1712_171261


namespace NUMINAMATH_CALUDE_min_value_4a_plus_b_l1712_171216

theorem min_value_4a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 + a*b - 3 = 0) :
  4*a + b ≥ 6 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀^2 + a₀*b₀ - 3 = 0 ∧ 4*a₀ + b₀ = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_4a_plus_b_l1712_171216


namespace NUMINAMATH_CALUDE_tenth_term_is_18_l1712_171252

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 2 = 2 ∧ a 5 = 8 ∧ ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- The 10th term of the arithmetic sequence is 18 -/
theorem tenth_term_is_18 (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 10 = 18 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_18_l1712_171252
