import Mathlib

namespace point_translation_on_sine_curves_l3056_305695

theorem point_translation_on_sine_curves : ∃ (t s : ℝ),
  -- P(π/4, t) is on y = sin(x - π/12)
  t = Real.sin (π / 4 - π / 12) ∧
  -- s > 0
  s > 0 ∧
  -- P' is on y = sin(2x) after translation
  Real.sin (2 * (π / 4 - s)) = t ∧
  -- t = 1/2
  t = 1 / 2 ∧
  -- Minimum value of s = π/6
  s = π / 6 ∧
  -- s is the minimum positive value satisfying the conditions
  ∀ (s' : ℝ), s' > 0 → Real.sin (2 * (π / 4 - s')) = t → s ≤ s' := by
sorry

end point_translation_on_sine_curves_l3056_305695


namespace difference_of_squares_l3056_305637

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 10) :
  x^2 - y^2 = 200 := by
  sorry

end difference_of_squares_l3056_305637


namespace sally_peach_cost_l3056_305632

def total_spent : ℝ := 23.86
def cherry_cost : ℝ := 11.54
def peach_cost : ℝ := total_spent - cherry_cost

theorem sally_peach_cost : peach_cost = 12.32 := by
  sorry

end sally_peach_cost_l3056_305632


namespace desk_length_l3056_305605

/-- Given a rectangular desk with width 9 cm and perimeter 46 cm, prove its length is 14 cm. -/
theorem desk_length (width : ℝ) (perimeter : ℝ) (length : ℝ) : 
  width = 9 → perimeter = 46 → 2 * (length + width) = perimeter → length = 14 := by
  sorry

end desk_length_l3056_305605


namespace complex_quotient_pure_imaginary_l3056_305697

theorem complex_quotient_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := a + 2*Complex.I
  let z₂ : ℂ := 3 - 4*Complex.I
  (∃ b : ℝ, z₁ / z₂ = b*Complex.I ∧ b ≠ 0) → a = 8/3 := by
sorry

end complex_quotient_pure_imaginary_l3056_305697


namespace max_volume_rectangular_prism_l3056_305641

/-- Represents a right prism with a rectangular base -/
structure RectangularPrism where
  a : ℝ  -- length of the base
  b : ℝ  -- width of the base
  h : ℝ  -- height of the prism

/-- The sum of areas of two lateral faces and the base face is 32 -/
def area_constraint (p : RectangularPrism) : Prop :=
  p.a * p.h + p.b * p.h + p.a * p.b = 32

/-- The volume of the prism -/
def volume (p : RectangularPrism) : ℝ :=
  p.a * p.b * p.h

/-- Theorem stating the maximum volume of the prism -/
theorem max_volume_rectangular_prism :
  ∀ p : RectangularPrism, area_constraint p →
  volume p ≤ (128 * Real.sqrt 3) / 3 :=
by sorry

end max_volume_rectangular_prism_l3056_305641


namespace bounded_difference_exists_l3056_305644

/-- A monotonous function satisfying the given inequality condition. -/
structure MonotonousFunction (f : ℝ → ℝ) (c₁ c₂ : ℝ) : Prop :=
  (mono : Monotone f)
  (pos_const : c₁ > 0 ∧ c₂ > 0)
  (ineq : ∀ x y : ℝ, f x + f y - c₁ ≤ f (x + y) ∧ f (x + y) ≤ f x + f y + c₂)

/-- The main theorem stating the existence of k such that f(x) - kx is bounded. -/
theorem bounded_difference_exists (f : ℝ → ℝ) (c₁ c₂ : ℝ) 
  (hf : MonotonousFunction f c₁ c₂) : 
  ∃ k : ℝ, ∃ M : ℝ, ∀ x : ℝ, |f x - k * x| ≤ M :=
sorry

end bounded_difference_exists_l3056_305644


namespace g_of_negative_two_l3056_305623

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x + 2

-- Theorem statement
theorem g_of_negative_two : g (-2) = -8 := by
  sorry

end g_of_negative_two_l3056_305623


namespace set_operations_l3056_305636

def A : Set ℝ := {x | -4 ≤ x ∧ x ≤ -2}
def B : Set ℝ := {x | x + 3 ≥ 0}
def U : Set ℝ := {x | x ≤ -1}

theorem set_operations :
  (A ∩ B = {x | -3 ≤ x ∧ x ≤ -2}) ∧
  (A ∪ B = {x | x ≥ -4}) ∧
  (U \ (A ∩ B) = {x | x < -3 ∨ (-2 < x ∧ x ≤ -1)}) :=
by sorry

end set_operations_l3056_305636


namespace distance_between_points_l3056_305642

theorem distance_between_points : 
  let p1 : ℚ × ℚ := (-3/2, -1/2)
  let p2 : ℚ × ℚ := (9/2, 7/2)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 52 := by
  sorry

end distance_between_points_l3056_305642


namespace five_digit_divisibility_l3056_305619

theorem five_digit_divisibility (a b c d e : ℕ) (h1 : a ≥ 1) (h2 : a ≤ 9) 
  (h3 : b ≤ 9) (h4 : c ≤ 9) (h5 : d ≤ 9) (h6 : e ≤ 9) :
  let n := 10000 * a + 1000 * b + 100 * c + 10 * d + e
  let m := 1000 * a + 100 * b + 10 * d + e
  (∃ k : ℕ, n = k * m) →
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ 100 * c = (k - 1) * m :=
by sorry

end five_digit_divisibility_l3056_305619


namespace space_geometry_statements_l3056_305603

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (lineParallelPlane : Line → Plane → Prop)
variable (linePerpendicularPlane : Line → Plane → Prop)
variable (planeParallelPlane : Plane → Plane → Prop)
variable (planePerpendicularPlane : Plane → Plane → Prop)
variable (intersect : Line → Line → Prop)

-- Define the theorem
theorem space_geometry_statements 
  (m n : Line) (α β : Plane) (A : Point) :
  (∀ l₁ l₂ p, parallel l₁ l₂ → lineParallelPlane l₂ p → lineParallelPlane l₁ p) ∧
  (parallel m n → linePerpendicularPlane n β → lineParallelPlane m α → planePerpendicularPlane α β) ∧
  (intersect m n → lineParallelPlane m α → lineParallelPlane m β → 
   lineParallelPlane n α → lineParallelPlane n β → planeParallelPlane α β) :=
by sorry

end space_geometry_statements_l3056_305603


namespace intersection_A_B_intersection_C_R_A_B_l3056_305685

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

-- Define the complement of A in ℝ
def C_R_A : Set ℝ := {x : ℝ | x < 3 ∨ x ≥ 7}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | 3 ≤ x ∧ x < 7} := by sorry

-- Theorem for (C_R A) ∩ B
theorem intersection_C_R_A_B : (C_R_A ∩ B) = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

end intersection_A_B_intersection_C_R_A_B_l3056_305685


namespace triangle_problem_l3056_305660

-- Define the triangle ABC
def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Add conditions for a valid triangle if necessary
  True

-- Define the theorem
theorem triangle_problem (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_a : a = 8)
  (h_bc : b - c = 2)
  (h_cosA : Real.cos A = -1/4) :
  Real.sin B = (3 * Real.sqrt 15) / 16 ∧ 
  Real.cos (2 * A + π/6) = -(7 * Real.sqrt 3) / 16 - (Real.sqrt 15) / 16 :=
by
  sorry

end triangle_problem_l3056_305660


namespace min_value_reciprocal_sum_l3056_305630

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + 3*b = 2) :
  (1/a + 3/b) ≥ 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3*b₀ = 2 ∧ 1/a₀ + 3/b₀ = 8 :=
sorry

end min_value_reciprocal_sum_l3056_305630


namespace garcia_fourth_quarter_shots_l3056_305601

/-- Represents the number of shots taken and made in a basketball game --/
structure GameStats :=
  (shots_taken : ℕ)
  (shots_made : ℕ)

/-- Calculates the shooting accuracy as a rational number --/
def accuracy (stats : GameStats) : ℚ :=
  stats.shots_made / stats.shots_taken

theorem garcia_fourth_quarter_shots 
  (first_two_quarters : GameStats)
  (third_quarter : GameStats)
  (fourth_quarter : GameStats)
  (h1 : first_two_quarters.shots_taken = 20)
  (h2 : first_two_quarters.shots_made = 12)
  (h3 : third_quarter.shots_taken = 10)
  (h4 : accuracy third_quarter = (1/2) * accuracy first_two_quarters)
  (h5 : accuracy fourth_quarter = (4/3) * accuracy third_quarter)
  (h6 : accuracy (GameStats.mk 
    (first_two_quarters.shots_taken + third_quarter.shots_taken + fourth_quarter.shots_taken)
    (first_two_quarters.shots_made + third_quarter.shots_made + fourth_quarter.shots_made)) = 46/100)
  : fourth_quarter.shots_made = 8 := by
  sorry

end garcia_fourth_quarter_shots_l3056_305601


namespace sufficient_not_necessary_l3056_305694

theorem sufficient_not_necessary (y : ℝ) (h : y > 2) :
  (∀ x, x > 1 → x + y > 3) ∧ 
  (∃ x, x + y > 3 ∧ ¬(x > 1)) := by
  sorry

end sufficient_not_necessary_l3056_305694


namespace sum_of_cubes_l3056_305608

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) :
  x^3 + y^3 = 1008 := by sorry

end sum_of_cubes_l3056_305608


namespace sequence_sum_l3056_305621

def S (n : ℕ) : ℤ :=
  if n % 2 = 1 then (n + 1) / 2 else -(n / 2)

theorem sequence_sum : S 19 * S 31 + S 48 = 136 := by
  sorry

end sequence_sum_l3056_305621


namespace f_properties_l3056_305675

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp x - 1) / x - a * x + a

theorem f_properties (a : ℝ) :
  (∀ x y, 0 < x ∧ x < y → a ≤ 0 → f a x < f a y) ∧ 
  (∃ x₀, 0 < x₀ ∧ x₀ < 1 ∧ f a x₀ = Real.exp 1 - 1 → a < 1) ∧
  ¬(a < 1 → ∃ x₀, 0 < x₀ ∧ x₀ < 1 ∧ f a x₀ = Real.exp 1 - 1 ∧ 
    ∀ x, 0 < x ∧ x < 1 ∧ x ≠ x₀ → f a x ≠ Real.exp 1 - 1) :=
by sorry

end f_properties_l3056_305675


namespace marbles_ratio_l3056_305616

def total_marbles : ℕ := 63
def your_initial_marbles : ℕ := 16

def brother_marbles : ℕ → ℕ → ℕ
  | your_marbles, marbles_given => 
    (total_marbles - your_marbles - (3 * (your_marbles - marbles_given))) / 2 + marbles_given

def your_final_marbles : ℕ := your_initial_marbles - 2

theorem marbles_ratio : 
  ∃ (m : ℕ), m > 0 ∧ your_final_marbles = m * (brother_marbles your_initial_marbles 2) ∧
  (your_final_marbles : ℚ) / (brother_marbles your_initial_marbles 2) = 2 := by
sorry

end marbles_ratio_l3056_305616


namespace frequency_third_group_l3056_305674

theorem frequency_third_group (m : ℕ) (h1 : m ≥ 3) : 
  let total_frequency : ℝ := 1
  let third_rectangle_area : ℝ := (1 / 4) * (total_frequency - third_rectangle_area)
  let sample_size : ℕ := 100
  (third_rectangle_area * sample_size : ℝ) = 20 := by
  sorry

end frequency_third_group_l3056_305674


namespace journey_time_ratio_and_sum_l3056_305668

/-- Represents the ratio of road segments --/
def road_ratio : Fin 3 → ℚ
| 0 => 1
| 1 => 2
| 2 => 1

/-- Represents the ratio of speeds on different road types --/
def speed_ratio : Fin 3 → ℚ
| 0 => 3
| 1 => 2
| 2 => 4

/-- Calculates the time taken for a journey --/
def journey_time (r : Fin 3 → ℚ) (s : Fin 3 → ℚ) : ℚ :=
  (r 0 / s 0) + (r 1 / s 1) + (r 2 / s 2)

/-- Theorem stating the ratio of journey times and the sum of m and n --/
theorem journey_time_ratio_and_sum :
  let to_school := journey_time road_ratio speed_ratio
  let return_home := journey_time road_ratio (fun i => speed_ratio (2 - i))
  let ratio := to_school / return_home
  ∃ (m n : ℕ), m.Coprime n ∧ ratio = n / m ∧ m + n = 35 := by
  sorry


end journey_time_ratio_and_sum_l3056_305668


namespace cow_count_l3056_305663

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- The total number of legs in the group -/
def totalLegs (ac : AnimalCount) : ℕ :=
  2 * ac.ducks + 4 * ac.cows

/-- The total number of heads in the group -/
def totalHeads (ac : AnimalCount) : ℕ :=
  ac.ducks + ac.cows

/-- The problem statement -/
theorem cow_count (ac : AnimalCount) :
  totalLegs ac = 2 * totalHeads ac + 36 → ac.cows = 18 := by
  sorry

end cow_count_l3056_305663


namespace pave_hall_l3056_305699

/-- The number of stones required to pave a rectangular hall -/
def stones_required (hall_length hall_width stone_length stone_width : ℚ) : ℚ :=
  (hall_length * hall_width * 100) / (stone_length * stone_width)

/-- Theorem stating that 2700 stones are required to pave the given hall -/
theorem pave_hall : stones_required 36 15 4 5 = 2700 := by
  sorry

end pave_hall_l3056_305699


namespace ordering_of_logarithms_and_exponential_l3056_305684

theorem ordering_of_logarithms_and_exponential : 
  let a := Real.log 3 / Real.log 5
  let b := Real.log 8 / Real.log 13
  let c := Real.exp (-1/2)
  c < a ∧ a < b :=
by sorry

end ordering_of_logarithms_and_exponential_l3056_305684


namespace cosine_angle_POQ_l3056_305611

/-- Given points P, Q, and O, prove that the cosine of angle POQ is -√10/10 -/
theorem cosine_angle_POQ :
  let P : ℝ × ℝ := (1, 1)
  let Q : ℝ × ℝ := (-2, 1)
  let O : ℝ × ℝ := (0, 0)
  let OP : ℝ × ℝ := (P.1 - O.1, P.2 - O.2)
  let OQ : ℝ × ℝ := (Q.1 - O.1, Q.2 - O.2)
  let dot_product : ℝ := OP.1 * OQ.1 + OP.2 * OQ.2
  let magnitude_OP : ℝ := Real.sqrt (OP.1^2 + OP.2^2)
  let magnitude_OQ : ℝ := Real.sqrt (OQ.1^2 + OQ.2^2)
  dot_product / (magnitude_OP * magnitude_OQ) = -Real.sqrt 10 / 10 := by
  sorry

end cosine_angle_POQ_l3056_305611


namespace car_travel_theorem_l3056_305653

/-- Represents the distance-time relationship for a car traveling between two points --/
def distance_function (initial_distance : ℝ) (speed : ℝ) (time : ℝ) : ℝ :=
  initial_distance - speed * time

theorem car_travel_theorem (initial_distance speed time : ℝ) 
  (h1 : initial_distance = 120)
  (h2 : speed = 80)
  (h3 : 0 ≤ time)
  (h4 : time ≤ 1.5) :
  let y := distance_function initial_distance speed time
  ∀ x, x = time → y = 120 - 80 * x ∧ 
  (x = 0.8 → y = 56) := by
  sorry

#check car_travel_theorem

end car_travel_theorem_l3056_305653


namespace preference_change_difference_l3056_305651

theorem preference_change_difference (initial_online initial_traditional final_online final_traditional : ℚ) 
  (h_initial_sum : initial_online + initial_traditional = 1)
  (h_final_sum : final_online + final_traditional = 1)
  (h_initial_online : initial_online = 2/5)
  (h_initial_traditional : initial_traditional = 3/5)
  (h_final_online : final_online = 4/5)
  (h_final_traditional : final_traditional = 1/5) :
  let min_change := |final_online - initial_online|
  let max_change := min initial_traditional (1 - initial_online)
  max_change - min_change = 2/5 := by
sorry

#eval (2 : ℚ) / 5 -- This should evaluate to 0.4, which is 40%

end preference_change_difference_l3056_305651


namespace product_four_consecutive_even_divisible_by_96_largest_divisor_four_consecutive_even_l3056_305669

/-- The product of four consecutive even natural numbers is always divisible by 96 -/
theorem product_four_consecutive_even_divisible_by_96 :
  ∀ n : ℕ, 96 ∣ (2*n) * (2*n + 2) * (2*n + 4) * (2*n + 6) :=
by sorry

/-- 96 is the largest natural number that always divides the product of four consecutive even natural numbers -/
theorem largest_divisor_four_consecutive_even :
  ∀ m : ℕ, (∀ n : ℕ, m ∣ (2*n) * (2*n + 2) * (2*n + 4) * (2*n + 6)) → m ≤ 96 :=
by sorry

end product_four_consecutive_even_divisible_by_96_largest_divisor_four_consecutive_even_l3056_305669


namespace rotate_point_A_about_C_l3056_305654

/-- Rotates a point 180 degrees about a center point -/
def rotate180 (point center : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - point.1, 2 * center.2 - point.2)

theorem rotate_point_A_about_C : 
  let A : ℝ × ℝ := (-4, 1)
  let C : ℝ × ℝ := (-1, 1)
  rotate180 A C = (2, 1) := by sorry

end rotate_point_A_about_C_l3056_305654


namespace optimal_rental_plan_l3056_305666

/-- Represents the rental plan for cars -/
structure RentalPlan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a rental plan is valid according to the given conditions -/
def isValidPlan (plan : RentalPlan) : Prop :=
  plan.typeA + plan.typeB = 8 ∧
  35 * plan.typeA + 30 * plan.typeB ≥ 255 ∧
  400 * plan.typeA + 320 * plan.typeB ≤ 3000

/-- Calculates the total cost of a rental plan -/
def totalCost (plan : RentalPlan) : ℕ :=
  400 * plan.typeA + 320 * plan.typeB

/-- The optimal rental plan -/
def optimalPlan : RentalPlan :=
  { typeA := 3, typeB := 5 }

theorem optimal_rental_plan :
  isValidPlan optimalPlan ∧
  totalCost optimalPlan = 2800 ∧
  ∀ plan, isValidPlan plan → totalCost plan ≥ totalCost optimalPlan :=
by sorry

end optimal_rental_plan_l3056_305666


namespace complex_number_equality_l3056_305614

theorem complex_number_equality (z : ℂ) : 
  Complex.abs (z - 1) = Complex.abs (z - Complex.I) →
  ∃ (r : ℝ), r > 0 ∧ z - (z - 6) / (z - 1) = r →
  z = 2 + 2 * Complex.I :=
by sorry

end complex_number_equality_l3056_305614


namespace roots_cubic_sum_l3056_305638

theorem roots_cubic_sum (p q : ℝ) : 
  (p^2 - 5*p + 3 = 0) → (q^2 - 5*q + 3 = 0) → (p + q)^3 = 125 := by
  sorry

end roots_cubic_sum_l3056_305638


namespace crossed_out_number_is_21_l3056_305692

def first_n_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem crossed_out_number_is_21 :
  ∀ a : ℕ, 
    a > 0 ∧ a ≤ 20 →
    (∃ k : ℕ, k > 0 ∧ k ≤ 20 ∧ k ≠ a ∧ 
      k = (first_n_sum 20 - a) / 19 ∧ 
      (first_n_sum 20 - a) % 19 = 0) →
    a = 21 :=
by sorry

end crossed_out_number_is_21_l3056_305692


namespace squirrel_pine_cones_l3056_305688

/-- The number of pine cones the squirrel planned to eat per day -/
def planned_daily_cones : ℕ := 6

/-- The additional number of pine cones the squirrel actually ate per day -/
def additional_daily_cones : ℕ := 2

/-- The number of days earlier the pine cones were finished -/
def days_earlier : ℕ := 5

/-- The total number of pine cones stored by the squirrel -/
def total_cones : ℕ := 120

theorem squirrel_pine_cones :
  ∃ (planned_days : ℕ),
    planned_days * planned_daily_cones =
    (planned_days - days_earlier) * (planned_daily_cones + additional_daily_cones) ∧
    total_cones = planned_days * planned_daily_cones :=
by sorry

end squirrel_pine_cones_l3056_305688


namespace outer_circle_radius_l3056_305620

/-- Given a circular race track with an inner circumference of 440 meters and a width of 14 meters,
    the radius of the outer circle is equal to (440 / (2 * π)) + 14 meters. -/
theorem outer_circle_radius (inner_circumference : ℝ) (track_width : ℝ) 
    (h1 : inner_circumference = 440)
    (h2 : track_width = 14) : 
    (inner_circumference / (2 * Real.pi) + track_width) = (440 / (2 * Real.pi) + 14) := by
  sorry

#check outer_circle_radius

end outer_circle_radius_l3056_305620


namespace smallest_n_for_f_greater_than_15_l3056_305628

def digit_sum (x : ℚ) : ℕ :=
  sorry

def f (n : ℕ+) : ℕ :=
  digit_sum ((1 : ℚ) / (7 ^ (n : ℕ)))

theorem smallest_n_for_f_greater_than_15 :
  ∀ k : ℕ+, k < 7 → f k ≤ 15 ∧ f 7 > 15 := by
  sorry

end smallest_n_for_f_greater_than_15_l3056_305628


namespace function_minimum_value_l3056_305661

theorem function_minimum_value (a : ℝ) :
  (∃ x₀ : ℝ, (x₀ + a)^2 + (Real.exp x₀ + a / Real.exp 1)^2 ≤ 4 / (Real.exp 2 + 1)) →
  a = (Real.exp 2 - 1) / (Real.exp 2 + 1) := by
  sorry

end function_minimum_value_l3056_305661


namespace sum_of_roots_quadratic_equation_l3056_305610

theorem sum_of_roots_quadratic_equation :
  ∀ x₁ x₂ : ℝ, (x₁^2 - 3*x₁ + 2 = 0) ∧ (x₂^2 - 3*x₂ + 2 = 0) → x₁ + x₂ = 3 := by
  sorry

end sum_of_roots_quadratic_equation_l3056_305610


namespace bobbo_river_crossing_l3056_305680

/-- Bobbo's river crossing problem -/
theorem bobbo_river_crossing 
  (river_width : ℝ)
  (initial_speed : ℝ)
  (current_speed : ℝ)
  (waterfall_distance : ℝ)
  (h_river_width : river_width = 100)
  (h_initial_speed : initial_speed = 2)
  (h_current_speed : current_speed = 5)
  (h_waterfall_distance : waterfall_distance = 175) :
  let midway_distance := river_width / 2
  let time_to_midway := midway_distance / initial_speed
  let downstream_distance := current_speed * time_to_midway
  let remaining_distance := waterfall_distance - downstream_distance
  let time_left := remaining_distance / current_speed
  let required_speed := midway_distance / time_left
  required_speed - initial_speed = 3 := by
  sorry

end bobbo_river_crossing_l3056_305680


namespace sqrt_sum_powers_of_five_l3056_305606

theorem sqrt_sum_powers_of_five : 
  Real.sqrt (5^3 + 5^4 + 5^5) = 5 * Real.sqrt 155 := by
  sorry

end sqrt_sum_powers_of_five_l3056_305606


namespace lcm_12_15_18_l3056_305655

theorem lcm_12_15_18 : Nat.lcm (Nat.lcm 12 15) 18 = 180 := by
  sorry

end lcm_12_15_18_l3056_305655


namespace fourth_student_number_l3056_305679

def systematicSampling (totalStudents : Nat) (samplesToSelect : Nat) (selected : List Nat) : Nat :=
  sorry

theorem fourth_student_number
  (totalStudents : Nat)
  (samplesToSelect : Nat)
  (selected : List Nat)
  (h1 : totalStudents = 54)
  (h2 : samplesToSelect = 4)
  (h3 : selected = [3, 29, 42])
  : systematicSampling totalStudents samplesToSelect selected = 16 :=
by sorry

end fourth_student_number_l3056_305679


namespace range_of_f_l3056_305646

theorem range_of_f (x : ℝ) : 
  let f := fun (x : ℝ) => Real.sin x^4 - Real.sin x * Real.cos x + Real.cos x^4
  0 ≤ f x ∧ f x ≤ 9/8 ∧ 
  (∃ y : ℝ, f y = 0) ∧ 
  (∃ z : ℝ, f z = 9/8) :=
by sorry

end range_of_f_l3056_305646


namespace trip_equation_correct_l3056_305691

/-- Represents a car trip with a stop -/
structure CarTrip where
  totalDistance : ℝ
  totalTime : ℝ
  stopDuration : ℝ
  speedBefore : ℝ
  speedAfter : ℝ

/-- The equation for the trip is correct -/
theorem trip_equation_correct (trip : CarTrip) 
    (h1 : trip.totalDistance = 300)
    (h2 : trip.totalTime = 4)
    (h3 : trip.stopDuration = 0.5)
    (h4 : trip.speedBefore = 60)
    (h5 : trip.speedAfter = 90) :
  ∃ t : ℝ, 
    t ≥ 0 ∧ 
    t ≤ trip.totalTime - trip.stopDuration ∧
    trip.speedBefore * t + trip.speedAfter * (trip.totalTime - trip.stopDuration - t) = trip.totalDistance :=
by sorry

end trip_equation_correct_l3056_305691


namespace brown_ball_weight_l3056_305624

theorem brown_ball_weight (blue_weight : ℝ) (total_weight : ℝ) (brown_weight : ℝ) :
  blue_weight = 6 →
  total_weight = 9.12 →
  total_weight = blue_weight + brown_weight →
  brown_weight = 3.12 :=
by
  sorry

end brown_ball_weight_l3056_305624


namespace expression_simplification_l3056_305698

theorem expression_simplification (x y z : ℝ) 
  (h1 : x > y) (h2 : y > 1) (h3 : z > 0) : 
  (x^y * y^(x+z)) / (y^(y+z) * x^x) = (x/y)^(y-x) := by
  sorry

end expression_simplification_l3056_305698


namespace arc_length_sixty_degrees_l3056_305665

theorem arc_length_sixty_degrees (r : ℝ) (h : r = 1) :
  let angle : ℝ := π / 3
  let arc_length : ℝ := r * angle
  arc_length = π / 3 := by sorry

end arc_length_sixty_degrees_l3056_305665


namespace train_length_l3056_305635

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 50 → time = 9 → ∃ length : ℝ, 
  (length ≥ 124.5 ∧ length ≤ 125.5) ∧ length = speed * 1000 / 3600 * time :=
sorry

end train_length_l3056_305635


namespace rectangular_plot_breadth_l3056_305643

theorem rectangular_plot_breadth (b : ℝ) (l : ℝ) (A : ℝ) : 
  A = 23 * b →  -- Area is 23 times the breadth
  l = b + 10 →  -- Length is 10 meters more than breadth
  A = l * b →   -- Area formula for rectangle
  b = 13 := by
sorry

end rectangular_plot_breadth_l3056_305643


namespace integral_evaluation_l3056_305664

open Real

theorem integral_evaluation : 
  ∫ (x : ℝ) in Real.arccos (1 / Real.sqrt 10)..Real.arccos (1 / Real.sqrt 26), 
    12 / ((6 + 5 * tan x) * sin (2 * x)) = log (105 / 93) := by
  sorry

end integral_evaluation_l3056_305664


namespace cubic_equation_solutions_l3056_305640

theorem cubic_equation_solutions :
  ∀ m n : ℤ, m^3 - n^3 = 2*m*n + 8 ↔ (m = 0 ∧ n = -2) ∨ (m = 2 ∧ n = 0) :=
by sorry

end cubic_equation_solutions_l3056_305640


namespace hawks_score_l3056_305600

/-- Given the total score and winning margin in a basketball game, 
    calculate the score of the losing team. -/
theorem hawks_score (total_score winning_margin : ℕ) : 
  total_score = 58 → winning_margin = 12 → 
  (total_score - winning_margin) / 2 = 23 := by
  sorry

#check hawks_score

end hawks_score_l3056_305600


namespace triangle_max_area_l3056_305607

/-- Given a triangle ABC with area S and sides a, b, c, 
    if 4S = a² - (b - c)² and b + c = 4, 
    then the maximum value of S is 2 -/
theorem triangle_max_area (a b c S : ℝ) : 
  4 * S = a^2 - (b - c)^2 → b + c = 4 → S ≤ 2 ∧ ∃ b c, b + c = 4 ∧ S = 2 := by
  sorry

end triangle_max_area_l3056_305607


namespace theo_cookie_days_l3056_305626

/-- The number of days Theo eats cookies each month -/
def days_per_month (cookies_per_time cookies_per_day total_cookies months : ℕ) : ℕ :=
  (total_cookies / months) / (cookies_per_time * cookies_per_day)

/-- Theorem stating that Theo eats cookies for 20 days each month -/
theorem theo_cookie_days : days_per_month 13 3 2340 3 = 20 := by
  sorry

end theo_cookie_days_l3056_305626


namespace peanut_butter_servings_l3056_305689

/-- Represents the amount of peanut butter in tablespoons -/
def peanut_butter : ℚ := 37 + 2/3

/-- Represents the serving size in tablespoons -/
def serving_size : ℚ := 2 + 1/2

/-- Calculates the number of servings in the jar -/
def number_of_servings : ℚ := peanut_butter / serving_size

/-- Proves that the number of servings in the jar is equal to 15 1/15 -/
theorem peanut_butter_servings : number_of_servings = 15 + 1/15 := by
  sorry

end peanut_butter_servings_l3056_305689


namespace magnified_tissue_diameter_l3056_305634

/-- Calculates the diameter of a magnified image given the actual diameter and magnification factor. -/
def magnifiedDiameter (actualDiameter : ℝ) (magnificationFactor : ℝ) : ℝ :=
  actualDiameter * magnificationFactor

/-- Proves that for a tissue with actual diameter 0.0003 cm and a microscope with 1000x magnification,
    the magnified image diameter is 0.3 cm. -/
theorem magnified_tissue_diameter :
  magnifiedDiameter 0.0003 1000 = 0.3 := by
  sorry

end magnified_tissue_diameter_l3056_305634


namespace collinear_points_b_value_l3056_305672

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_b_value :
  ∀ b : ℝ,
  let A : Point := ⟨3, 1⟩
  let B : Point := ⟨-2, b⟩
  let C : Point := ⟨8, 11⟩
  collinear A B C → b = -9 := by
  sorry

end collinear_points_b_value_l3056_305672


namespace oranges_thrown_away_l3056_305629

theorem oranges_thrown_away (initial_oranges new_oranges final_oranges : ℕ) : 
  initial_oranges = 31 → new_oranges = 38 → final_oranges = 60 → 
  ∃ thrown_away : ℕ, initial_oranges - thrown_away + new_oranges = final_oranges ∧ thrown_away = 9 :=
by sorry

end oranges_thrown_away_l3056_305629


namespace f_properties_l3056_305662

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem f_properties :
  ∃ (period : ℝ),
    (∀ (x : ℝ), f (x + period) = f x) ∧
    (∀ (p : ℝ), (∀ (x : ℝ), f (x + p) = f x) → period ≤ p) ∧
    (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ 2) ∧
    (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ -1) ∧
    (∃ (x₁ : ℝ), x₁ ∈ Set.Icc 0 (Real.pi / 2) ∧ f x₁ = 2) ∧
    (∃ (x₂ : ℝ), x₂ ∈ Set.Icc 0 (Real.pi / 2) ∧ f x₂ = -1) ∧
    (∀ (x₀ : ℝ), x₀ ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) → 
      f x₀ = 6/5 → Real.cos (2 * x₀) = (3 - 4 * Real.sqrt 3) / 10) :=
by sorry

end f_properties_l3056_305662


namespace chessboard_division_impossible_l3056_305613

/-- Represents a chessboard --/
structure Chessboard :=
  (size : ℕ)

/-- Represents a line on the chessboard --/
structure Line

/-- Represents a division of the chessboard --/
def ChessboardDivision := List Line

/-- Function to check if a division is valid --/
def is_valid_division (board : Chessboard) (division : ChessboardDivision) : Prop :=
  sorry

/-- Theorem: It's impossible to divide an 8x8 chessboard with 13 lines
    such that each region contains at most one square center --/
theorem chessboard_division_impossible :
  ∀ (board : Chessboard) (division : ChessboardDivision),
    board.size = 8 →
    division.length = 13 →
    ¬(is_valid_division board division) :=
sorry

end chessboard_division_impossible_l3056_305613


namespace apples_for_juice_apples_for_juice_proof_l3056_305617

/-- Calculates the amount of apples used for fruit juice given the harvest and sales information -/
theorem apples_for_juice (total_harvest : ℕ) (restaurant_amount : ℕ) (bag_size : ℕ) 
  (total_sales : ℕ) (price_per_bag : ℕ) : ℕ :=
  let bags_sold := total_sales / price_per_bag
  let apples_sold := bags_sold * bag_size
  total_harvest - (restaurant_amount + apples_sold)

/-- Proves that 90 kg of apples were used for fruit juice given the specific values -/
theorem apples_for_juice_proof : 
  apples_for_juice 405 60 5 408 8 = 90 := by
  sorry

end apples_for_juice_apples_for_juice_proof_l3056_305617


namespace trapezoid_area_l3056_305673

/-- The area of a trapezoid with height h, bases 4h + 2 and 5h is (9h^2 + 2h) / 2 -/
theorem trapezoid_area (h : ℝ) : 
  let base1 := 4 * h + 2
  let base2 := 5 * h
  ((base1 + base2) / 2) * h = (9 * h^2 + 2 * h) / 2 := by
  sorry

end trapezoid_area_l3056_305673


namespace tripod_height_after_break_l3056_305676

theorem tripod_height_after_break (original_leg_length original_height broken_leg_length : ℝ) 
  (h : ℝ) (m n : ℕ) :
  original_leg_length = 6 →
  original_height = 5 →
  broken_leg_length = 4 →
  h = 12 →
  h = m / Real.sqrt n →
  m = 168 →
  n = 169 →
  ⌊m + Real.sqrt n⌋ = 181 :=
by sorry

end tripod_height_after_break_l3056_305676


namespace intersection_nonempty_implies_a_value_l3056_305602

def M (a : ℤ) : Set ℤ := {a, 0}

def N : Set ℤ := {x : ℤ | 2 * x^2 - 5 * x < 0}

theorem intersection_nonempty_implies_a_value (a : ℤ) :
  (M a ∩ N).Nonempty → a = 1 ∨ a = 2 := by
  sorry

end intersection_nonempty_implies_a_value_l3056_305602


namespace area_of_fourth_square_l3056_305678

/-- Given two right triangles PQR and PRS with a common hypotenuse PR,
    where the squares of the sides have areas 25, 64, 49, and an unknown value,
    prove that the area of the square on PS is 138 square units. -/
theorem area_of_fourth_square (PQ PR QR RS PS : ℝ) : 
  PQ^2 = 25 → QR^2 = 49 → RS^2 = 64 → 
  PQ^2 + QR^2 = PR^2 → PR^2 + RS^2 = PS^2 →
  PS^2 = 138 := by
  sorry

end area_of_fourth_square_l3056_305678


namespace cafe_order_combinations_l3056_305670

-- Define the number of menu items
def menu_items : ℕ := 12

-- Define the number of people ordering
def num_people : ℕ := 3

-- Theorem statement
theorem cafe_order_combinations :
  menu_items ^ num_people = 1728 := by
  sorry

end cafe_order_combinations_l3056_305670


namespace probability_heart_then_king_l3056_305627

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Fin 52)

/-- Represents the suit of a card -/
inductive Suit
| Hearts | Diamonds | Clubs | Spades

/-- Represents the rank of a card -/
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- A function to determine if a card is a heart -/
def is_heart (card : Fin 52) : Prop := sorry

/-- A function to determine if a card is a king -/
def is_king (card : Fin 52) : Prop := sorry

/-- The number of hearts in a standard deck -/
def num_hearts : Nat := 13

/-- The number of kings in a standard deck -/
def num_kings : Nat := 4

/-- The probability of drawing a heart as the first card and a king as the second card -/
theorem probability_heart_then_king (d : Deck) :
  (num_hearts / d.cards.val) * (num_kings / (d.cards.val - 1)) = 1 / d.cards.val :=
sorry

end probability_heart_then_king_l3056_305627


namespace parabola_satisfies_equation_l3056_305647

/-- A parabola with vertex at the origin, symmetric about coordinate axes, passing through (2, -3) -/
structure Parabola where
  /-- The parabola passes through the point (2, -3) -/
  passes_through : (2 : ℝ)^2 + (-3 : ℝ)^2 ≠ 0

/-- The equation of the parabola -/
def parabola_equation (p : Parabola) : Prop :=
  (∀ x y : ℝ, y^2 = 9/2 * x) ∨ (∀ x y : ℝ, x^2 = -4/3 * y)

/-- Theorem stating that the parabola satisfies the given equation -/
theorem parabola_satisfies_equation (p : Parabola) : parabola_equation p := by
  sorry

end parabola_satisfies_equation_l3056_305647


namespace equation_implications_l3056_305633

theorem equation_implications (x y : ℝ) (h : x^2 - 2*x*y + 2*y^2 = 1) :
  (abs x ≤ Real.sqrt 2) ∧ (x^2 + 2*y^2 > 1/2) := by
  sorry

end equation_implications_l3056_305633


namespace system_solution_l3056_305609

theorem system_solution (a b : ℤ) :
  (b * (-1) + 2 * 2 = 8) →
  (a * 1 + 3 * 4 = 5) →
  (a = -7 ∧ b = -4) ∧
  ((-7) * 7 + 3 * 18 = 5) ∧
  ((-4) * 7 + 2 * 18 = 8) :=
by sorry

end system_solution_l3056_305609


namespace blueprint_to_actual_length_l3056_305615

/-- Represents the scale of the blueprint in meters per inch -/
def scale : ℝ := 50

/-- Represents the length of the line segment on the blueprint in inches -/
def blueprint_length : ℝ := 7.5

/-- Represents the actual length in meters that the blueprint line segment represents -/
def actual_length : ℝ := blueprint_length * scale

theorem blueprint_to_actual_length : actual_length = 375 := by
  sorry

end blueprint_to_actual_length_l3056_305615


namespace competition_problems_l3056_305677

/-- The total number of problems in the competition. -/
def total_problems : ℕ := 71

/-- The number of problems Lukáš correctly solved. -/
def solved_problems : ℕ := 12

/-- The additional points Lukáš would have gained if he solved the last 12 problems. -/
def additional_points : ℕ := 708

/-- The sum of the first n natural numbers. -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating the total number of problems in the competition. -/
theorem competition_problems :
  (sum_first_n solved_problems) +
  (sum_first_n solved_problems + additional_points) =
  sum_first_n total_problems - sum_first_n (total_problems - solved_problems) :=
by sorry

end competition_problems_l3056_305677


namespace karting_track_routes_l3056_305648

/-- Represents the number of distinct routes ending at point A after n minutes -/
def M : ℕ → ℕ
| 0 => 0
| 1 => 0
| (n+2) => M (n+1) + M n

/-- The karting track problem -/
theorem karting_track_routes : M 10 = 34 := by
  sorry

end karting_track_routes_l3056_305648


namespace triangle_properties_l3056_305656

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  cos B = 4/5 →
  b = 2 →
  (a = 5/3 → A = π/6) ∧
  (∀ a c, a > 0 → c > 0 → (1/2) * a * c * (3/5) ≤ 3) :=
by sorry

end triangle_properties_l3056_305656


namespace bottle_production_l3056_305649

/-- Given that 6 identical machines produce 270 bottles per minute at a constant rate,
    prove that 8 such machines will produce 1440 bottles in 4 minutes. -/
theorem bottle_production (machines_base : ℕ) (bottles_per_minute : ℕ) (machines_new : ℕ) (minutes : ℕ)
    (h1 : machines_base = 6)
    (h2 : bottles_per_minute = 270)
    (h3 : machines_new = 8)
    (h4 : minutes = 4) :
    (machines_new * (bottles_per_minute / machines_base) * minutes) = 1440 :=
by sorry

end bottle_production_l3056_305649


namespace ellipse_parabola_equations_l3056_305696

/-- Given an ellipse and a parabola with specific properties, 
    prove their equations. -/
theorem ellipse_parabola_equations 
  (a b c p : ℝ) 
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : p > 0)
  (h4 : c / a = 1 / 2)  -- eccentricity
  (h5 : a - c = 1 / 2)  -- distance from left focus to directrix
  (h6 : a = p / 2)      -- right vertex is focus of parabola
  : (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 + 4 * y^2 / 3 = 1) ∧
    (∀ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 4 * x) := by
  sorry


end ellipse_parabola_equations_l3056_305696


namespace product_always_even_l3056_305671

theorem product_always_even (a b c : ℤ) : 
  Even ((a - b) * (b - c) * (c - a)) := by
  sorry

end product_always_even_l3056_305671


namespace smallest_multiplier_for_perfect_square_l3056_305667

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_multiplier_for_perfect_square :
  ∃ (k : ℕ), k > 0 ∧ 
  (∃ (m : ℕ), k * y = m^2) ∧
  (∀ (j : ℕ), 0 < j ∧ j < k → ¬∃ (n : ℕ), j * y = n^2) ∧
  k = 6 :=
sorry

end smallest_multiplier_for_perfect_square_l3056_305667


namespace sum_of_five_reals_l3056_305687

theorem sum_of_five_reals (a b c d e : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) (pos_e : 0 < e)
  (eq1 : a + b = c)
  (eq2 : a + b + c = d)
  (eq3 : a + b + c + d = e)
  (c_val : c = 5) : 
  a + b + c + d + e = 40 := by
sorry

end sum_of_five_reals_l3056_305687


namespace largest_base_for_twelve_cubed_l3056_305681

/-- Given a natural number n and a base b, returns the sum of digits of n when represented in base b -/
def sumOfDigits (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Returns true if b is the largest base such that the sum of digits of 12^3 in base b is not 3^2 -/
def isLargestBase (b : ℕ) : Prop :=
  (sumOfDigits (12^3) b ≠ 3^2) ∧
  ∀ k > b, sumOfDigits (12^3) k = 3^2

theorem largest_base_for_twelve_cubed :
  isLargestBase 9 := by sorry

end largest_base_for_twelve_cubed_l3056_305681


namespace second_player_wins_l3056_305645

/-- Represents the state of the game -/
structure GameState :=
  (boxes : Fin 11 → ℕ)

/-- Represents a move in the game -/
structure Move :=
  (skipped : Fin 11)

/-- Applies a move to the game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  { boxes := λ i => if i = move.skipped then state.boxes i else state.boxes i + 1 }

/-- Checks if the game is won -/
def is_won (state : GameState) : Prop :=
  ∃ i, state.boxes i = 21

/-- Represents a strategy for a player -/
def Strategy := GameState → Move

/-- Represents the game play -/
def play (initial_state : GameState) (strategy1 strategy2 : Strategy) : Prop :=
  ∃ (n : ℕ) (states : ℕ → GameState),
    states 0 = initial_state ∧
    (∀ k, states (k+1) = 
      if k % 2 = 0
      then apply_move (states k) (strategy1 (states k))
      else apply_move (states k) (strategy2 (states k))) ∧
    is_won (states (2*n + 1)) ∧ ¬is_won (states (2*n))

/-- The theorem stating that the second player has a winning strategy -/
theorem second_player_wins :
  ∃ (strategy2 : Strategy), ∀ (strategy1 : Strategy),
    play { boxes := λ _ => 0 } strategy1 strategy2 :=
sorry

end second_player_wins_l3056_305645


namespace oil_depths_in_elliptical_tank_l3056_305682

/-- Represents an elliptical oil tank lying horizontally -/
structure EllipticalTank where
  length : ℝ
  majorAxis : ℝ
  minorAxis : ℝ

/-- Calculates the possible oil depths in an elliptical tank -/
def calculateOilDepths (tank : EllipticalTank) (oilSurfaceArea : ℝ) : Set ℝ :=
  sorry

/-- The theorem stating the correct oil depths for the given tank and oil surface area -/
theorem oil_depths_in_elliptical_tank :
  let tank : EllipticalTank := { length := 10, majorAxis := 8, minorAxis := 6 }
  let oilSurfaceArea : ℝ := 48
  calculateOilDepths tank oilSurfaceArea = {1.2, 4.8} := by
  sorry

end oil_depths_in_elliptical_tank_l3056_305682


namespace max_side_length_of_triangle_l3056_305652

theorem max_side_length_of_triangle (a b c : ℕ) : 
  a < b → b < c →  -- Three different integer side lengths
  a + b + c = 24 → -- Perimeter is 24 units
  a + b > c →      -- Triangle inequality
  b + c > a →      -- Triangle inequality
  a + c > b →      -- Triangle inequality
  c ≤ 11 :=        -- Maximum length of any side is 11
by sorry

end max_side_length_of_triangle_l3056_305652


namespace problem_statement_l3056_305690

theorem problem_statement (a b m n c : ℝ) 
  (h1 : a + b = 0) 
  (h2 : m * n = 1) 
  (h3 : |c| = 3) : 
  a + b + m * n - |c| = -2 := by
  sorry

end problem_statement_l3056_305690


namespace quadratic_inequality_l3056_305622

theorem quadratic_inequality (x : ℝ) : x^2 - 6*x + 5 > 0 ↔ x < 1 ∨ x > 5 := by
  sorry

end quadratic_inequality_l3056_305622


namespace exist_integers_satisfying_equation_l3056_305658

theorem exist_integers_satisfying_equation : ∃ (a b : ℤ), a * b * (2 * a + b) = 2015 ∧ a = 13 ∧ b = 5 := by
  sorry

end exist_integers_satisfying_equation_l3056_305658


namespace ellipse_properties_l3056_305683

/-- Ellipse C in the Cartesian coordinate system α -/
def C (b : ℝ) (x y : ℝ) : Prop :=
  0 < b ∧ b < 2 ∧ x^2 / 4 + y^2 / b^2 = 1

/-- Point A is the right vertex of C -/
def A : ℝ × ℝ := (2, 0)

/-- Line l passing through O with non-zero slope -/
def l (m : ℝ) (x y : ℝ) : Prop :=
  m ≠ 0 ∧ y = m * x

/-- P and Q are intersection points of l and C -/
def intersectionPoints (b m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C b x₁ y₁ ∧ C b x₂ y₂ ∧ l m x₁ y₁ ∧ l m x₂ y₂

/-- M and N are intersections of AP, AQ with y-axis -/
def MN (b m : ℝ) : Prop :=
  ∃ (y₁ y₂ : ℝ), (∃ (x₁ y₁' : ℝ), C b x₁ y₁' ∧ l m x₁ y₁' ∧ y₁ = (y₁' / (x₁ - 2)) * (-2)) ∧
                 (∃ (x₂ y₂' : ℝ), C b x₂ y₂' ∧ l m x₂ y₂' ∧ y₂ = (y₂' / (x₂ - 2)) * (-2))

theorem ellipse_properties (b m : ℝ) :
  C b 1 1 → intersectionPoints b m → (∃ (x y : ℝ), C b x y ∧ l m x y ∧ (x^2 + y^2)^(1/2) = 2 * ((x - 2)^2 + y^2)^(1/2)) →
  b^2 = 4/3 ∧ (∀ y₁ y₂ : ℝ, MN b m → y₁ * y₂ = b^2) :=
sorry

end ellipse_properties_l3056_305683


namespace coin_count_l3056_305659

theorem coin_count (total_value : ℕ) (two_dollar_coins : ℕ) : 
  total_value = 402 → two_dollar_coins = 148 → 
  ∃ (one_dollar_coins : ℕ), 
    total_value = 2 * two_dollar_coins + one_dollar_coins ∧
    one_dollar_coins + two_dollar_coins = 254 :=
by
  sorry

end coin_count_l3056_305659


namespace quadratic_roots_condition_l3056_305639

theorem quadratic_roots_condition (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - x + 2 = 0 ∧ a * y^2 - y + 2 = 0) ↔ (a < 1/8 ∧ a ≠ 0) :=
by sorry

end quadratic_roots_condition_l3056_305639


namespace odd_sum_of_squares_l3056_305604

theorem odd_sum_of_squares (n m : ℤ) (h : Odd (n^2 + m^2)) :
  ¬(Even n ∧ Even m) ∧ ¬(Even (n + m)) := by
  sorry

end odd_sum_of_squares_l3056_305604


namespace slope_of_solutions_l3056_305693

/-- The equation that defines the relationship between x and y -/
def equation (x y : ℝ) : Prop := 2 / x + 3 / y = 0

/-- Theorem: The slope of the line determined by any two distinct solutions to the equation is -3/2 -/
theorem slope_of_solutions (x₁ y₁ x₂ y₂ : ℝ) (h₁ : equation x₁ y₁) (h₂ : equation x₂ y₂) (h_dist : (x₁, y₁) ≠ (x₂, y₂)) :
  (y₂ - y₁) / (x₂ - x₁) = -3/2 := by
  sorry

end slope_of_solutions_l3056_305693


namespace doris_erasers_taken_out_l3056_305631

/-- The number of erasers Doris took out of a box -/
def erasers_taken_out (initial : ℕ) (left : ℕ) : ℕ :=
  initial - left

theorem doris_erasers_taken_out :
  let initial := 69
  let left := 15
  erasers_taken_out initial left = 54 := by sorry

end doris_erasers_taken_out_l3056_305631


namespace sin_difference_of_complex_exponentials_l3056_305657

theorem sin_difference_of_complex_exponentials (α β : ℝ) :
  Complex.exp (α * I) = 4/5 + 3/5 * I →
  Complex.exp (β * I) = 12/13 + 5/13 * I →
  Real.sin (α - β) = -16/65 := by
  sorry

end sin_difference_of_complex_exponentials_l3056_305657


namespace solution_set_f_greater_than_5_range_of_m_for_f_geq_g_set_iic_1_equiv_interval_l3056_305650

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2| + 2
def g (m : ℝ) (x : ℝ) : ℝ := m * |x|

-- Theorem for part I
theorem solution_set_f_greater_than_5 :
  {x : ℝ | f x > 5} = {x : ℝ | x < -1 ∨ x > 5} := by sorry

-- Theorem for part II
theorem range_of_m_for_f_geq_g :
  {m : ℝ | ∀ x, f x ≥ g m x} = Set.Iic 1 := by sorry

-- Additional helper theorem to show that Set.Iic 1 is equivalent to (-∞, 1]
theorem set_iic_1_equiv_interval :
  Set.Iic 1 = {m : ℝ | m ≤ 1} := by sorry

end solution_set_f_greater_than_5_range_of_m_for_f_geq_g_set_iic_1_equiv_interval_l3056_305650


namespace table_satisfies_function_l3056_305618

def f (x : ℝ) : ℝ := 100 - 5*x - 5*x^2

theorem table_satisfies_function : 
  (f 0 = 100) ∧ 
  (f 1 = 90) ∧ 
  (f 2 = 70) ∧ 
  (f 3 = 40) ∧ 
  (f 4 = 0) := by
  sorry

end table_satisfies_function_l3056_305618


namespace max_b_letters_l3056_305612

/-- The maximum number of "B" letters that can be formed with 47 sticks -/
theorem max_b_letters (total_sticks : ℕ) (sticks_per_b : ℕ) (sticks_per_v : ℕ)
  (h_total : total_sticks = 47)
  (h_b : sticks_per_b = 4)
  (h_v : sticks_per_v = 5)
  (h_all_used : ∃ (b v : ℕ), total_sticks = b * sticks_per_b + v * sticks_per_v) :
  ∃ (max_b : ℕ), 
    (max_b * sticks_per_b ≤ total_sticks) ∧ 
    (∀ b : ℕ, b * sticks_per_b ≤ total_sticks → b ≤ max_b) ∧
    (∃ v : ℕ, total_sticks = max_b * sticks_per_b + v * sticks_per_v) ∧
    max_b = 8 :=
sorry

end max_b_letters_l3056_305612


namespace max_n_with_2013_trailing_zeros_l3056_305686

/-- Count the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- The maximum value of N such that N! has exactly 2013 trailing zeros -/
theorem max_n_with_2013_trailing_zeros :
  ∀ n : ℕ, n > 8069 → trailingZeros n > 2013 ∧
  trailingZeros 8069 = 2013 :=
by sorry

end max_n_with_2013_trailing_zeros_l3056_305686


namespace prime_divisors_and_totient_l3056_305625

theorem prime_divisors_and_totient (a b c t q : ℕ) (k n : ℕ) 
  (hk : k = c^t) 
  (hn : n = a^k - b^k) 
  (hq : ∃ (p : List ℕ), (∀ x ∈ p, Nat.Prime x) ∧ (List.length p ≥ q) ∧ (∀ x ∈ p, x ∣ k)) :
  (∃ (p : List ℕ), (∀ x ∈ p, Nat.Prime x) ∧ (List.length p ≥ q * t) ∧ (∀ x ∈ p, x ∣ n)) ∧
  (∃ m : ℕ, Nat.totient n = m * 2^(t/2)) := by
  sorry

end prime_divisors_and_totient_l3056_305625
