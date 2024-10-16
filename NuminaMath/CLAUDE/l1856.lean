import Mathlib

namespace NUMINAMATH_CALUDE_fox_speed_l1856_185697

/-- Given a constant speed where 100 kilometers are covered in 120 minutes, 
    prove that the speed in kilometers per hour is 50. -/
theorem fox_speed (distance : ℝ) (time_minutes : ℝ) (speed_km_per_hour : ℝ)
  (h1 : distance = 100)
  (h2 : time_minutes = 120)
  (h3 : speed_km_per_hour = distance / time_minutes * 60) :
  speed_km_per_hour = 50 := by
  sorry

end NUMINAMATH_CALUDE_fox_speed_l1856_185697


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1856_185625

open Set

def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | x ≥ 1}

theorem intersection_A_complement_B :
  A ∩ (𝒰 \ B) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1856_185625


namespace NUMINAMATH_CALUDE_observation_count_l1856_185655

theorem observation_count (original_mean : ℝ) (incorrect_value : ℝ) (correct_value : ℝ) (new_mean : ℝ) : 
  original_mean = 36 →
  incorrect_value = 23 →
  correct_value = 30 →
  new_mean = 36.5 →
  ∃ n : ℕ, n > 0 ∧ (n : ℝ) * new_mean = n * original_mean + (correct_value - incorrect_value) ∧ n = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_observation_count_l1856_185655


namespace NUMINAMATH_CALUDE_milk_drinking_problem_l1856_185666

theorem milk_drinking_problem (initial_milk : ℚ) (rachel_fraction : ℚ) (max_fraction : ℚ) : 
  initial_milk = 3/4 →
  rachel_fraction = 1/2 →
  max_fraction = 1/3 →
  max_fraction * (initial_milk - rachel_fraction * initial_milk) = 1/8 :=
by
  sorry

end NUMINAMATH_CALUDE_milk_drinking_problem_l1856_185666


namespace NUMINAMATH_CALUDE_diagonal_of_square_l1856_185696

theorem diagonal_of_square (side_length : ℝ) (h : side_length = 10) :
  let diagonal := Real.sqrt (2 * side_length ^ 2)
  diagonal = 10 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_diagonal_of_square_l1856_185696


namespace NUMINAMATH_CALUDE_vacation_book_selection_l1856_185653

theorem vacation_book_selection (total_books : ℕ) (books_to_bring : ℕ) (favorite_book : ℕ) :
  total_books = 15 →
  books_to_bring = 3 →
  favorite_book = 1 →
  Nat.choose (total_books - favorite_book) (books_to_bring - favorite_book) = 91 :=
by
  sorry

end NUMINAMATH_CALUDE_vacation_book_selection_l1856_185653


namespace NUMINAMATH_CALUDE_min_value_inequality_l1856_185668

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2*x + y = 2) :
  1/x^2 + 4/y^2 ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + y₀ = 2 ∧ 1/x₀^2 + 4/y₀^2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1856_185668


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1856_185685

/-- An arithmetic sequence with common difference 3 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 3

/-- The first, third, and fourth terms form a geometric sequence -/
def geometric_subsequence (a : ℕ → ℝ) : Prop :=
  (a 3)^2 = a 1 * a 4

/-- Main theorem: If a is an arithmetic sequence with common difference 3
    and its first, third, and fourth terms form a geometric sequence,
    then the second term equals -9 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) 
    (h1 : arithmetic_sequence a) (h2 : geometric_subsequence a) : 
  a 2 = -9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1856_185685


namespace NUMINAMATH_CALUDE_factorization_proof_l1856_185611

theorem factorization_proof (x : ℝ) : x^2 - 4*x + 4 = (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1856_185611


namespace NUMINAMATH_CALUDE_algebraic_expression_values_l1856_185600

theorem algebraic_expression_values (p q : ℝ) :
  (p * 1^3 + q * 1 + 1 = 2023) →
  (p * (-1)^3 + q * (-1) + 1 = -2021) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_values_l1856_185600


namespace NUMINAMATH_CALUDE_wire_length_around_square_field_l1856_185651

theorem wire_length_around_square_field (area : ℝ) (n : ℕ) (wire_length : ℝ) : 
  area = 69696 → n = 15 → wire_length = 15840 → 
  wire_length = n * 4 * Real.sqrt area := by
  sorry

end NUMINAMATH_CALUDE_wire_length_around_square_field_l1856_185651


namespace NUMINAMATH_CALUDE_tiffany_sunscreen_cost_l1856_185667

/-- Calculates the cost of sunscreen for a beach visit given the specified parameters. -/
def sunscreenCost (reapplyInterval : ℕ) (amountPerApplication : ℕ) (bottleSize : ℕ) (bottleCost : ℚ) (visitDuration : ℕ) : ℚ :=
  let applications := visitDuration / reapplyInterval
  let totalAmount := applications * amountPerApplication
  let bottlesNeeded := (totalAmount + bottleSize - 1) / bottleSize  -- Ceiling division
  bottlesNeeded * bottleCost

/-- Theorem stating that the sunscreen cost for Tiffany's beach visit is $7. -/
theorem tiffany_sunscreen_cost :
  sunscreenCost 2 3 12 (7/2) 16 = 7 := by
  sorry

#eval sunscreenCost 2 3 12 (7/2) 16

end NUMINAMATH_CALUDE_tiffany_sunscreen_cost_l1856_185667


namespace NUMINAMATH_CALUDE_olivia_chocolate_sales_l1856_185601

/-- Calculates the money made from selling chocolate bars -/
def money_made (total_bars : ℕ) (bars_left : ℕ) (price_per_bar : ℕ) : ℕ :=
  (total_bars - bars_left) * price_per_bar

/-- Theorem stating that Olivia would make $9 from selling the chocolate bars -/
theorem olivia_chocolate_sales : 
  money_made 7 4 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_olivia_chocolate_sales_l1856_185601


namespace NUMINAMATH_CALUDE_gavins_green_shirts_l1856_185639

theorem gavins_green_shirts (total_shirts : ℕ) (blue_shirts : ℕ) (green_shirts : ℕ) 
  (h1 : total_shirts = 23)
  (h2 : blue_shirts = 6)
  (h3 : green_shirts = total_shirts - blue_shirts) :
  green_shirts = 17 :=
by sorry

end NUMINAMATH_CALUDE_gavins_green_shirts_l1856_185639


namespace NUMINAMATH_CALUDE_hair_cut_ratio_l1856_185629

/-- Given the initial hair length, growth after first cut, second cut length, and final hair length,
    prove that the ratio of the initial hair cut to the original hair length is 1/2. -/
theorem hair_cut_ratio (initial_length growth second_cut final_length : ℝ)
  (h1 : initial_length = 24)
  (h2 : growth = 4)
  (h3 : second_cut = 2)
  (h4 : final_length = 14)
  (h5 : ∃ x, final_length = initial_length - x + growth - second_cut) :
  ∃ x, x / initial_length = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hair_cut_ratio_l1856_185629


namespace NUMINAMATH_CALUDE_area_enclosed_l1856_185638

noncomputable def f (n : ℕ) : ℝ → ℝ := 
  match n with
  | 0 => fun x => |x|
  | k + 1 => fun x => |f k x - (k + 1)|

theorem area_enclosed (n : ℕ) : 
  ∃ (a : ℝ), a > 0 ∧ 
  (∫ (x : ℝ) in -a..a, f n x) = (4 * n^3 + 6 * n^2 - 1 + (-1)^n) / 8 :=
sorry

end NUMINAMATH_CALUDE_area_enclosed_l1856_185638


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1856_185683

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 3 + a 7 - a 10 = -1 →
  a 11 - a 4 = 21 →
  a 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1856_185683


namespace NUMINAMATH_CALUDE_first_tv_width_l1856_185669

/-- Proves that the width of the first TV is 24 inches given the specified conditions. -/
theorem first_tv_width : 
  ∀ (W : ℝ),
  (672 / (W * 16) = 1152 / (48 * 32) + 1) →
  W = 24 := by
sorry

end NUMINAMATH_CALUDE_first_tv_width_l1856_185669


namespace NUMINAMATH_CALUDE_max_trig_ratio_l1856_185624

theorem max_trig_ratio (x : ℝ) : 
  (Real.sin x)^2 + (Real.cos x)^2 = 1 → 
  ((Real.sin x)^4 + (Real.cos x)^4 + 1) / ((Real.sin x)^2 + (Real.cos x)^2 + 1) ≤ 7/4 := by
sorry

end NUMINAMATH_CALUDE_max_trig_ratio_l1856_185624


namespace NUMINAMATH_CALUDE_lines_in_plane_not_intersecting_are_parallel_l1856_185689

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- A line is contained in a plane -/
def contained_in (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Two lines intersect -/
def intersect (l1 l2 : Line3D) : Prop :=
  sorry

theorem lines_in_plane_not_intersecting_are_parallel 
  (α : Plane3D) (a b : Line3D) 
  (ha : contained_in a α) 
  (hb : contained_in b α) 
  (hnot_intersect : ¬ intersect a b) : 
  parallel a b :=
sorry

end NUMINAMATH_CALUDE_lines_in_plane_not_intersecting_are_parallel_l1856_185689


namespace NUMINAMATH_CALUDE_pizza_slices_remaining_l1856_185604

theorem pizza_slices_remaining (initial_slices : ℕ) 
  (breakfast_slices : ℕ) (lunch_slices : ℕ) (snack_slices : ℕ) (dinner_slices : ℕ) :
  initial_slices = 15 →
  breakfast_slices = 4 →
  lunch_slices = 2 →
  snack_slices = 2 →
  dinner_slices = 5 →
  initial_slices - (breakfast_slices + lunch_slices + snack_slices + dinner_slices) = 2 :=
by sorry

end NUMINAMATH_CALUDE_pizza_slices_remaining_l1856_185604


namespace NUMINAMATH_CALUDE_sean_shopping_cost_l1856_185664

-- Define the prices and quantities
def soda_price : ℝ := 1
def soda_quantity : ℕ := 4
def soup_quantity : ℕ := 3
def sandwich_quantity : ℕ := 2
def salad_quantity : ℕ := 1

-- Define price relationships
def soup_price : ℝ := 2 * soda_price
def sandwich_price : ℝ := 4 * soup_price
def salad_price : ℝ := 2 * sandwich_price

-- Define discount and tax rates
def discount_rate : ℝ := 0.1
def tax_rate : ℝ := 0.05

-- Calculate total cost before discount and tax
def total_cost : ℝ :=
  soda_price * soda_quantity +
  soup_price * soup_quantity +
  sandwich_price * sandwich_quantity +
  salad_price * salad_quantity

-- Calculate final cost after discount and tax
def final_cost : ℝ :=
  total_cost * (1 - discount_rate) * (1 + tax_rate)

-- Theorem to prove
theorem sean_shopping_cost :
  final_cost = 39.69 := by sorry

end NUMINAMATH_CALUDE_sean_shopping_cost_l1856_185664


namespace NUMINAMATH_CALUDE_constant_value_l1856_185622

-- Define the function f
def f (x : ℝ) : ℝ := x + 4

-- Define the equation
def equation (x : ℝ) (c : ℝ) : Prop :=
  (3 * f (x - 2)) / f 0 + c = f (2 * x + 1)

-- Theorem statement
theorem constant_value :
  ∃ (c : ℝ), equation 0.4 c ∧ ∀ (x : ℝ), equation x c → x = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_constant_value_l1856_185622


namespace NUMINAMATH_CALUDE_lawn_mowing_earnings_l1856_185619

theorem lawn_mowing_earnings (total_lawns : ℕ) (forgotten_lawns : ℕ) (total_earned : ℕ) :
  total_lawns = 12 →
  forgotten_lawns = 8 →
  total_earned = 36 →
  (total_earned : ℚ) / ((total_lawns - forgotten_lawns) : ℚ) = 9 :=
by sorry

end NUMINAMATH_CALUDE_lawn_mowing_earnings_l1856_185619


namespace NUMINAMATH_CALUDE_chicken_wings_distribution_l1856_185660

theorem chicken_wings_distribution (num_friends : ℕ) (initial_wings : ℕ) (additional_wings : ℕ) :
  num_friends = 4 →
  initial_wings = 9 →
  additional_wings = 7 →
  (initial_wings + additional_wings) % num_friends = 0 →
  (initial_wings + additional_wings) / num_friends = 4 :=
by sorry

end NUMINAMATH_CALUDE_chicken_wings_distribution_l1856_185660


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l1856_185623

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 14 = 56 → Nat.gcd n 14 = 10 → n = 40 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l1856_185623


namespace NUMINAMATH_CALUDE_worker_c_days_l1856_185690

/-- Represents the problem of calculating the number of days worker c worked. -/
theorem worker_c_days (days_a days_b : ℕ) (wage_c : ℕ) (total_earning : ℕ) : 
  days_a = 6 →
  days_b = 9 →
  wage_c = 105 →
  total_earning = 1554 →
  ∃ (days_c : ℕ),
    (3 : ℚ) / 5 * wage_c * days_a + 
    (4 : ℚ) / 5 * wage_c * days_b + 
    wage_c * days_c = total_earning ∧
    days_c = 4 :=
by sorry

end NUMINAMATH_CALUDE_worker_c_days_l1856_185690


namespace NUMINAMATH_CALUDE_geometric_sum_first_six_terms_l1856_185644

theorem geometric_sum_first_six_terms :
  let a₀ : ℚ := 1/2
  let r : ℚ := 1/2
  let n : ℕ := 6
  let S := a₀ * (1 - r^n) / (1 - r)
  S = 63/64 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_first_six_terms_l1856_185644


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l1856_185618

theorem floor_ceil_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(30.3 : ℝ)⌉ = 27 := by sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l1856_185618


namespace NUMINAMATH_CALUDE_clara_cookies_sold_l1856_185677

/-- Calculates the total number of cookies sold by Clara given the number of cookies per box and boxes sold for each type. -/
def total_cookies_sold (cookies_per_box : Fin 3 → ℕ) (boxes_sold : Fin 3 → ℕ) : ℕ :=
  (cookies_per_box 0) * (boxes_sold 0) + 
  (cookies_per_box 1) * (boxes_sold 1) + 
  (cookies_per_box 2) * (boxes_sold 2)

/-- Theorem stating that Clara sells 3320 cookies in total -/
theorem clara_cookies_sold : 
  let cookies_per_box : Fin 3 → ℕ := ![12, 20, 16]
  let boxes_sold : Fin 3 → ℕ := ![50, 80, 70]
  total_cookies_sold cookies_per_box boxes_sold = 3320 := by
sorry


end NUMINAMATH_CALUDE_clara_cookies_sold_l1856_185677


namespace NUMINAMATH_CALUDE_justin_flower_gathering_l1856_185635

def minutes_per_flower : ℕ := 10
def gathering_hours : ℕ := 2
def lost_flowers : ℕ := 3
def classmates : ℕ := 30

def additional_minutes_needed : ℕ :=
  let gathered_flowers := gathering_hours * 60 / minutes_per_flower
  let remaining_flowers := classmates - (gathered_flowers - lost_flowers)
  remaining_flowers * minutes_per_flower

theorem justin_flower_gathering :
  additional_minutes_needed = 210 := by
  sorry

end NUMINAMATH_CALUDE_justin_flower_gathering_l1856_185635


namespace NUMINAMATH_CALUDE_smith_family_seating_arrangements_l1856_185665

/-- Represents a family with parents and children -/
structure Family :=
  (num_parents : Nat)
  (num_children : Nat)

/-- Represents a car with front and back seats -/
structure Car :=
  (front_seats : Nat)
  (back_seats : Nat)

/-- Calculates the number of seating arrangements for a family in a car -/
def seating_arrangements (f : Family) (c : Car) (parent_driver : Bool) : Nat :=
  sorry

/-- The Smith family with 2 parents and 3 children -/
def smith_family : Family :=
  { num_parents := 2, num_children := 3 }

/-- The Smith family car with 2 front seats and 3 back seats -/
def smith_car : Car :=
  { front_seats := 2, back_seats := 3 }

theorem smith_family_seating_arrangements :
  seating_arrangements smith_family smith_car true = 48 := by
  sorry

end NUMINAMATH_CALUDE_smith_family_seating_arrangements_l1856_185665


namespace NUMINAMATH_CALUDE_range_of_f_l1856_185681

-- Define the function f(x) = -x²
def f (x : ℝ) : ℝ := -x^2

-- Define the domain
def domain : Set ℝ := { x | -3 ≤ x ∧ x ≤ 1 }

-- Theorem statement
theorem range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | -9 ≤ y ∧ y ≤ 0 } :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l1856_185681


namespace NUMINAMATH_CALUDE_sum_of_multiples_l1856_185617

theorem sum_of_multiples (n : ℕ) (h : n = 13) : n + n + 2 * n + 4 * n = 104 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l1856_185617


namespace NUMINAMATH_CALUDE_fourth_root_of_409600000_l1856_185636

theorem fourth_root_of_409600000 : (409600000 : ℝ) ^ (1/4 : ℝ) = 80 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_409600000_l1856_185636


namespace NUMINAMATH_CALUDE_mirror_wall_area_ratio_l1856_185609

/-- Proves that the ratio of the area of a square mirror to the area of a rectangular wall is 1:2 -/
theorem mirror_wall_area_ratio (mirror_side : ℝ) (wall_width wall_length : ℝ)
  (h1 : mirror_side = 18)
  (h2 : wall_width = 32)
  (h3 : wall_length = 20.25) :
  (mirror_side^2) / (wall_width * wall_length) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_mirror_wall_area_ratio_l1856_185609


namespace NUMINAMATH_CALUDE_angle_D_value_l1856_185656

-- Define the angles as real numbers (in degrees)
variable (A B C D : ℝ)

-- State the theorem
theorem angle_D_value 
  (h1 : A + B = 180)
  (h2 : C = 2 * D)
  (h3 : A = 100)
  (h4 : B + C + D = 180) :
  D = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_value_l1856_185656


namespace NUMINAMATH_CALUDE_ellipse_through_six_points_l1856_185628

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Check if a point lies on an ellipse with center (h, k), semi-major axis a, and semi-minor axis b -/
def onEllipse (p : Point) (h k a b : ℝ) : Prop :=
  ((p.x - h) ^ 2 / a ^ 2) + ((p.y - k) ^ 2 / b ^ 2) = 1

theorem ellipse_through_six_points :
  let p1 : Point := ⟨-3, 2⟩
  let p2 : Point := ⟨0, 0⟩
  let p3 : Point := ⟨0, 4⟩
  let p4 : Point := ⟨6, 0⟩
  let p5 : Point := ⟨6, 4⟩
  let p6 : Point := ⟨-3, 0⟩
  let points := [p1, p2, p3, p4, p5, p6]
  (∀ (a b c : Point), a ∈ points → b ∈ points → c ∈ points → a ≠ b → b ≠ c → a ≠ c → ¬collinear a b c) →
  ∃ (h k a b : ℝ), 
    a = 6 ∧ 
    b = 1 ∧ 
    (∀ p ∈ points, onEllipse p h k a b) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_through_six_points_l1856_185628


namespace NUMINAMATH_CALUDE_colored_cube_covers_plane_l1856_185659

/-- A cube with colored middle squares on each face -/
structure ColoredCube where
  a : ℕ
  b : ℕ
  c : ℕ

/-- An infinite plane with unit squares -/
def Plane := ℕ × ℕ

/-- A point on the plane is colorable if the cube can land on it with its colored face -/
def isColorable (cube : ColoredCube) (point : Plane) : Prop := sorry

/-- The main theorem: If any two sides of the cube are relatively prime, 
    then every point on the plane is colorable -/
theorem colored_cube_covers_plane (cube : ColoredCube) :
  (Nat.gcd (2 * cube.a + 1) (2 * cube.b + 1) = 1 ∨
   Nat.gcd (2 * cube.b + 1) (2 * cube.c + 1) = 1 ∨
   Nat.gcd (2 * cube.a + 1) (2 * cube.c + 1) = 1) →
  ∀ (point : Plane), isColorable cube point := by
  sorry

end NUMINAMATH_CALUDE_colored_cube_covers_plane_l1856_185659


namespace NUMINAMATH_CALUDE_exploration_writing_ratio_l1856_185674

theorem exploration_writing_ratio :
  let exploring_time : ℝ := 3
  let book_writing_time : ℝ := 0.5
  let total_time : ℝ := 5
  let notes_writing_time : ℝ := total_time - exploring_time - book_writing_time
  (notes_writing_time / exploring_time = 1 / 2) := by
sorry

end NUMINAMATH_CALUDE_exploration_writing_ratio_l1856_185674


namespace NUMINAMATH_CALUDE_inequality_proof_l1856_185663

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  (a * b + b * c + c * a)^2 ≥ 3 * a * b * c * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1856_185663


namespace NUMINAMATH_CALUDE_power_division_negative_x_l1856_185692

theorem power_division_negative_x (x : ℝ) : (-x)^8 / (-x)^4 = x^4 := by sorry

end NUMINAMATH_CALUDE_power_division_negative_x_l1856_185692


namespace NUMINAMATH_CALUDE_cube_difference_l1856_185649

theorem cube_difference (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x - y = 3) (h4 : x + y = 5) : x^3 - y^3 = 63 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l1856_185649


namespace NUMINAMATH_CALUDE_lilliputian_matchboxes_in_gulliver_matchbox_l1856_185679

/-- The scale factor between Gulliver's homeland and Lilliput -/
def scaleFactor : ℕ := 12

/-- The dimensions of a matchbox (length, width, height) -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculate the volume of a matchbox given its dimensions -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- The number of Lilliputian matchboxes that fit in one dimension -/
def fitInOneDimension : ℕ := scaleFactor

theorem lilliputian_matchboxes_in_gulliver_matchbox (g : Dimensions) (l : Dimensions)
    (h_scale : l.length = g.length / scaleFactor ∧ 
               l.width = g.width / scaleFactor ∧ 
               l.height = g.height / scaleFactor) :
    (volume g) / (volume l) = 1728 := by
  sorry

end NUMINAMATH_CALUDE_lilliputian_matchboxes_in_gulliver_matchbox_l1856_185679


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1856_185652

theorem quadratic_equation_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - (k + 2) * x₁ + 2 * k - 1 = 0 ∧
    x₂^2 - (k + 2) * x₂ + 2 * k - 1 = 0) ∧
  (3^2 - (k + 2) * 3 + 2 * k - 1 = 0 → k = 2 ∧ 1^2 - (k + 2) * 1 + 2 * k - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1856_185652


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_l1856_185626

/-- A trapezoid with the given properties -/
structure Trapezoid where
  longer_base : ℝ
  shorter_base : ℝ
  midpoint_segment : ℝ
  longer_base_length : longer_base = 117
  midpoint_segment_length : midpoint_segment = 5
  midpoint_segment_property : midpoint_segment = (longer_base - shorter_base) / 2

/-- Theorem stating that the shorter base of the trapezoid is 107 -/
theorem trapezoid_shorter_base (t : Trapezoid) : t.shorter_base = 107 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_shorter_base_l1856_185626


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l1856_185612

theorem quadratic_roots_sum (α β : ℝ) : 
  α^2 + 2*α - 2024 = 0 → 
  β^2 + 2*β - 2024 = 0 → 
  α^2 + 3*α + β = 2022 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l1856_185612


namespace NUMINAMATH_CALUDE_distance_between_points_l1856_185607

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (10, 8)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 3 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1856_185607


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_greater_than_sin_l1856_185630

theorem negation_of_proposition (p : ℝ → Prop) : 
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_greater_than_sin :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x : ℝ, x ≤ Real.sin x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_greater_than_sin_l1856_185630


namespace NUMINAMATH_CALUDE_factorization_problems_l1856_185687

theorem factorization_problems (m x y : ℝ) : 
  (m^3 - 2*m^2 - 4*m + 8 = (m-2)^2*(m+2)) ∧ 
  (x^2 - 2*x*y + y^2 - 9 = (x-y+3)*(x-y-3)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l1856_185687


namespace NUMINAMATH_CALUDE_solid_circles_in_2006_l1856_185670

def circle_sequence (n : ℕ) : ℕ := n + 1

def total_circles (n : ℕ) : ℕ := (n * (n + 3)) / 2

theorem solid_circles_in_2006 : 
  ∃ n : ℕ, total_circles n ≤ 2006 ∧ total_circles (n + 1) > 2006 ∧ n = 61 :=
sorry

end NUMINAMATH_CALUDE_solid_circles_in_2006_l1856_185670


namespace NUMINAMATH_CALUDE_T_is_three_rays_with_common_point_l1856_185675

/-- The set T of points (x,y) in the coordinate plane satisfying the given conditions -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let x := p.1; let y := p.2
    (x + 3 = 5 ∧ y - 2 ≤ 5) ∨
    (y - 2 = 5 ∧ x + 3 ≤ 5) ∨
    (x + 3 = y - 2 ∧ 5 ≤ x + 3)}

/-- The common point of the three rays -/
def common_point : ℝ × ℝ := (2, 7)

/-- The three rays that form set T -/
def ray1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 2 ∧ p.2 ≤ 7}
def ray2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≤ 2 ∧ p.2 = 7}
def ray3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≥ 2 ∧ p.2 = p.1 + 5}

/-- Theorem stating that T consists of three rays with a common point -/
theorem T_is_three_rays_with_common_point :
  T = ray1 ∪ ray2 ∪ ray3 ∧
  common_point ∈ ray1 ∧ common_point ∈ ray2 ∧ common_point ∈ ray3 :=
sorry

end NUMINAMATH_CALUDE_T_is_three_rays_with_common_point_l1856_185675


namespace NUMINAMATH_CALUDE_f_composition_and_inverse_l1856_185678

noncomputable def f (x : ℝ) : ℝ :=
  if x > 2 then 3 + 1/x
  else if -1 ≤ x ∧ x ≤ 2 then x^2 + 3
  else 2*x + 5

theorem f_composition_and_inverse :
  f (f (f (-3))) = 13/4 ∧ f (Real.sqrt 2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_and_inverse_l1856_185678


namespace NUMINAMATH_CALUDE_playground_area_l1856_185642

theorem playground_area (perimeter : ℝ) (length width : ℝ) : 
  perimeter = 100 → 
  length = 3 * width → 
  2 * length + 2 * width = perimeter → 
  length * width = 468.75 := by
sorry

end NUMINAMATH_CALUDE_playground_area_l1856_185642


namespace NUMINAMATH_CALUDE_range_of_x_l1856_185614

theorem range_of_x (a : ℝ) (h1 : a > 1) :
  {x : ℝ | a^(2*x + 1) > (1/a)^(2*x)} = {x : ℝ | x > -1/4} := by sorry

end NUMINAMATH_CALUDE_range_of_x_l1856_185614


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l1856_185654

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l1856_185654


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1856_185682

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 < 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1856_185682


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1856_185672

-- Define the conditions p and q
def p (x : ℝ) : Prop := (x - 1) * (x - 3) ≤ 0
def q (x : ℝ) : Prop := 2 / (x - 1) ≥ 1

-- Define the set A satisfying condition p
def A : Set ℝ := {x | p x}

-- Define the set B satisfying condition q
def B : Set ℝ := {x | q x}

-- Theorem stating that p is a necessary but not sufficient condition for q
theorem p_necessary_not_sufficient_for_q : 
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) := by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1856_185672


namespace NUMINAMATH_CALUDE_complex_abs_value_l1856_185643

theorem complex_abs_value (z : ℂ) : z = (1 - Complex.I)^2 / (1 + Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_abs_value_l1856_185643


namespace NUMINAMATH_CALUDE_expression_simplification_l1856_185633

theorem expression_simplification :
  (-8 : ℚ) * (18 / 14) * (49 / 27) + 4 / 3 = -52 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1856_185633


namespace NUMINAMATH_CALUDE_ball_distribution_ratio_l1856_185627

/-- The number of balls -/
def n : ℕ := 30

/-- The number of bins -/
def m : ℕ := 6

/-- The probability of one bin having 6 balls, one having 3 balls, and four having 5 balls each -/
noncomputable def p' : ℝ := sorry

/-- The probability of all bins having exactly 5 balls -/
noncomputable def q' : ℝ := sorry

/-- The theorem stating that the ratio of p' to q' is 5 -/
theorem ball_distribution_ratio : p' / q' = 5 := by sorry

end NUMINAMATH_CALUDE_ball_distribution_ratio_l1856_185627


namespace NUMINAMATH_CALUDE_intersection_count_l1856_185676

/-- The number of intersections between the line 3x + 4y = 12 and the circle x^2 + y^2 = 16 -/
def num_intersections : ℕ := 2

/-- The line equation 3x + 4y = 12 -/
def line_equation (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The circle equation x^2 + y^2 = 16 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 16

/-- Theorem stating that the number of intersections between the given line and circle is 2 -/
theorem intersection_count :
  ∃ (p q : ℝ × ℝ),
    line_equation p.1 p.2 ∧ circle_equation p.1 p.2 ∧
    line_equation q.1 q.2 ∧ circle_equation q.1 q.2 ∧
    p ≠ q ∧
    (∀ (r : ℝ × ℝ), line_equation r.1 r.2 ∧ circle_equation r.1 r.2 → r = p ∨ r = q) :=
by sorry

end NUMINAMATH_CALUDE_intersection_count_l1856_185676


namespace NUMINAMATH_CALUDE_equal_pay_implies_hours_constraint_l1856_185657

/-- Represents the payment structure and hours worked for Harry and James -/
structure WorkData where
  x : ℝ  -- hourly rate
  h : ℝ  -- Harry's normal hours
  y : ℝ  -- Harry's overtime hours

/-- The theorem states that if Harry and James were paid the same amount,
    and James worked 41 hours, then h + 2y = 42 -/
theorem equal_pay_implies_hours_constraint (data : WorkData) :
  data.x * data.h + 2 * data.x * data.y = data.x * 40 + 2 * data.x * 1 →
  data.h + 2 * data.y = 42 := by
  sorry

#check equal_pay_implies_hours_constraint

end NUMINAMATH_CALUDE_equal_pay_implies_hours_constraint_l1856_185657


namespace NUMINAMATH_CALUDE_not_equivalent_fraction_l1856_185640

theorem not_equivalent_fraction (x : ℝ) : x = 0.00000325 → x ≠ 1 / 308000000 := by
  sorry

end NUMINAMATH_CALUDE_not_equivalent_fraction_l1856_185640


namespace NUMINAMATH_CALUDE_tangent_circles_radius_l1856_185699

/-- Two circles are tangent if their centers' distance equals the sum or difference of their radii -/
def are_tangent (r₁ r₂ d : ℝ) : Prop := d = r₁ + r₂ ∨ d = |r₁ - r₂|

theorem tangent_circles_radius (r₁ r₂ d : ℝ) (h₁ : r₁ = 2) (h₂ : d = 5) 
  (h₃ : are_tangent r₁ r₂ d) : r₂ = 3 ∨ r₂ = 7 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_radius_l1856_185699


namespace NUMINAMATH_CALUDE_increasing_function_unique_root_l1856_185658

/-- An increasing function on ℝ has exactly one root -/
theorem increasing_function_unique_root (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) :
  ∃! x, f x = 0 :=
sorry

end NUMINAMATH_CALUDE_increasing_function_unique_root_l1856_185658


namespace NUMINAMATH_CALUDE_unique_interior_point_is_median_intersection_l1856_185641

/-- A point with integer coordinates -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle with vertices on lattice points -/
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Predicate to check if a point is inside a triangle -/
def IsInside (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- Predicate to check if a point is on the boundary of a triangle -/
def IsOnBoundary (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- The intersection point of the medians of a triangle -/
def MedianIntersection (t : LatticeTriangle) : LatticePoint := sorry

/-- Main theorem -/
theorem unique_interior_point_is_median_intersection (t : LatticeTriangle) 
  (h1 : ∀ p : LatticePoint, IsOnBoundary p t → (p = t.A ∨ p = t.B ∨ p = t.C))
  (h2 : ∃! O : LatticePoint, IsInside O t) :
  ∃ O : LatticePoint, IsInside O t ∧ O = MedianIntersection t := by
  sorry

end NUMINAMATH_CALUDE_unique_interior_point_is_median_intersection_l1856_185641


namespace NUMINAMATH_CALUDE_angle_B_in_arithmetic_sequence_triangle_l1856_185603

/-- In a triangle ABC where the interior angles A, B, and C form an arithmetic sequence, 
    the measure of angle B is 60°. -/
theorem angle_B_in_arithmetic_sequence_triangle : 
  ∀ (A B C : ℝ),
  (0 < A) ∧ (A < 180) ∧
  (0 < B) ∧ (B < 180) ∧
  (0 < C) ∧ (C < 180) ∧
  (A + B + C = 180) ∧
  (2 * B = A + C) →
  B = 60 := by
sorry

end NUMINAMATH_CALUDE_angle_B_in_arithmetic_sequence_triangle_l1856_185603


namespace NUMINAMATH_CALUDE_sector_area_120_deg_sqrt3_radius_l1856_185602

theorem sector_area_120_deg_sqrt3_radius (π : ℝ) (h_pi : π = Real.pi) : 
  let angle : ℝ := 120 * π / 180
  let radius : ℝ := Real.sqrt 3
  let area : ℝ := 1/2 * angle * radius^2
  area = π := by
sorry

end NUMINAMATH_CALUDE_sector_area_120_deg_sqrt3_radius_l1856_185602


namespace NUMINAMATH_CALUDE_ribbon_segment_length_l1856_185698

theorem ribbon_segment_length :
  let total_length : ℚ := 4/5
  let num_segments : ℕ := 3
  let segment_fraction : ℚ := 1/3
  let segment_length : ℚ := total_length * segment_fraction
  segment_length = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_segment_length_l1856_185698


namespace NUMINAMATH_CALUDE_point_coordinates_l1856_185673

/-- A point in the two-dimensional plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of the two-dimensional plane. -/
def FourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The distance between a point and the x-axis. -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance between a point and the y-axis. -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: A point P in the fourth quadrant with distance 2 to the x-axis
    and distance 3 to the y-axis has coordinates (3, -2). -/
theorem point_coordinates (P : Point)
    (h1 : FourthQuadrant P)
    (h2 : DistanceToXAxis P = 2)
    (h3 : DistanceToYAxis P = 3) :
    P = Point.mk 3 (-2) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1856_185673


namespace NUMINAMATH_CALUDE_volume_ratio_is_one_over_three_root_three_l1856_185606

/-- A right circular cone -/
structure RightCircularCone where
  radius : ℝ
  height : ℝ

/-- A plane cutting the cone -/
structure CuttingPlane where
  tangent_to_base : Bool
  passes_through_midpoint : Bool

/-- The ratio of volumes -/
def volume_ratio (cone : RightCircularCone) (plane : CuttingPlane) : ℝ := 
  sorry

/-- Theorem statement -/
theorem volume_ratio_is_one_over_three_root_three 
  (cone : RightCircularCone) 
  (plane : CuttingPlane) 
  (h1 : plane.tangent_to_base = true) 
  (h2 : plane.passes_through_midpoint = true) : 
  volume_ratio cone plane = 1 / (3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_is_one_over_three_root_three_l1856_185606


namespace NUMINAMATH_CALUDE_carbon_weight_in_C4H8O2_l1856_185684

/-- The molecular weight of the carbon part in C4H8O2 -/
def carbon_weight (atomic_weight : ℝ) (num_atoms : ℕ) : ℝ :=
  atomic_weight * num_atoms

/-- Proof that the molecular weight of the carbon part in C4H8O2 is 48.04 g/mol -/
theorem carbon_weight_in_C4H8O2 :
  let compound_weight : ℝ := 88
  let carbon_atomic_weight : ℝ := 12.01
  let num_carbon_atoms : ℕ := 4
  carbon_weight carbon_atomic_weight num_carbon_atoms = 48.04 := by
  sorry

end NUMINAMATH_CALUDE_carbon_weight_in_C4H8O2_l1856_185684


namespace NUMINAMATH_CALUDE_limit_rational_function_l1856_185647

/-- The limit of (x^2 + 2x - 3) / (x^3 + 4x^2 + 3x) as x approaches -3 is -2/3 -/
theorem limit_rational_function : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x + 3| ∧ |x + 3| < δ → 
    |(x^2 + 2*x - 3) / (x^3 + 4*x^2 + 3*x) + 2/3| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_rational_function_l1856_185647


namespace NUMINAMATH_CALUDE_train_crossing_platforms_l1856_185620

/-- A train crosses two platforms of different lengths. -/
theorem train_crossing_platforms
  (train_length : ℝ)
  (platform1_length : ℝ)
  (platform2_length : ℝ)
  (time1 : ℝ)
  (h1 : train_length = 350)
  (h2 : platform1_length = 100)
  (h3 : platform2_length = 250)
  (h4 : time1 = 15)
  : (train_length + platform2_length) / ((train_length + platform1_length) / time1) = 20 := by
  sorry

#check train_crossing_platforms

end NUMINAMATH_CALUDE_train_crossing_platforms_l1856_185620


namespace NUMINAMATH_CALUDE_min_comparisons_correct_l1856_185637

/-- Represents a set of coins with different weights -/
structure CoinSet (n : ℕ) where
  coins : Fin n → ℝ
  different_weights : ∀ i j, i ≠ j → coins i ≠ coins j

/-- Represents a set of balances, including one faulty balance -/
structure BalanceSet (n : ℕ) where
  balances : Fin n → Bool
  one_faulty : ∃ i, balances i = false

/-- The minimum number of comparisons needed to find the heaviest coin -/
def min_comparisons (n : ℕ) : ℕ := 2 * n - 1

/-- The main theorem: proving the minimum number of comparisons -/
theorem min_comparisons_correct (n : ℕ) (h : n > 2) 
  (coins : CoinSet n) (balances : BalanceSet n) :
  min_comparisons n = 2 * n - 1 :=
sorry

end NUMINAMATH_CALUDE_min_comparisons_correct_l1856_185637


namespace NUMINAMATH_CALUDE_sum_of_diagonals_l1856_185688

/-- A hexagon inscribed in a circle -/
structure InscribedHexagon where
  -- Sides of the hexagon
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  side6 : ℝ
  -- Diagonals from vertex A
  diag1 : ℝ
  diag2 : ℝ
  diag3 : ℝ
  -- Assumption that the hexagon is inscribed in a circle
  inscribed : True

/-- The theorem about the sum of diagonals in a specific inscribed hexagon -/
theorem sum_of_diagonals (h : InscribedHexagon) 
    (h1 : h.side1 = 70)
    (h2 : h.side2 = 90)
    (h3 : h.side3 = 90)
    (h4 : h.side4 = 90)
    (h5 : h.side5 = 90)
    (h6 : h.side6 = 50) :
    h.diag1 + h.diag2 + h.diag3 = 376 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_diagonals_l1856_185688


namespace NUMINAMATH_CALUDE_binary_11011_is_27_l1856_185632

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enum b).foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11011_is_27 :
  binary_to_decimal [true, true, false, true, true] = 27 := by
  sorry

end NUMINAMATH_CALUDE_binary_11011_is_27_l1856_185632


namespace NUMINAMATH_CALUDE_square_from_triangles_even_count_l1856_185691

-- Define the triangle type
structure Triangle :=
  (side1 : ℕ)
  (side2 : ℕ)
  (side3 : ℕ)

-- Define the properties of our specific triangle
def SpecificTriangle : Triangle :=
  { side1 := 3, side2 := 4, side3 := 5 }

-- Define the area of the triangle
def triangleArea (t : Triangle) : ℚ :=
  (t.side1 * t.side2 : ℚ) / 2

-- Define a function to check if a number is even
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Main theorem
theorem square_from_triangles_even_count :
  ∀ n : ℕ, n > 0 →
  (∃ a : ℕ, a > 0 ∧ (a : ℚ)^2 = n * triangleArea SpecificTriangle) →
  isEven n :=
sorry

end NUMINAMATH_CALUDE_square_from_triangles_even_count_l1856_185691


namespace NUMINAMATH_CALUDE_kendra_pens_l1856_185610

/-- Proves that Kendra has 4 packs of pens given the problem conditions -/
theorem kendra_pens (kendra_packs : ℕ) : 
  let tony_packs : ℕ := 2
  let pens_per_pack : ℕ := 3
  let pens_kept_each : ℕ := 2
  let friends_given_pens : ℕ := 14
  kendra_packs * pens_per_pack - pens_kept_each + 
    (tony_packs * pens_per_pack - pens_kept_each) = friends_given_pens →
  kendra_packs = 4 := by
sorry

end NUMINAMATH_CALUDE_kendra_pens_l1856_185610


namespace NUMINAMATH_CALUDE_trapezoid_area_l1856_185694

/-- Represents a trapezoid ABCD with point E at the intersection of diagonals -/
structure Trapezoid :=
  (A B C D E : ℝ × ℝ)

/-- The area of a triangle given its vertices -/
def triangle_area (p q r : ℝ × ℝ) : ℝ := sorry

theorem trapezoid_area (ABCD : Trapezoid) : 
  (ABCD.A.1 = ABCD.B.1) ∧  -- AB is parallel to CD (same x-coordinate)
  (ABCD.C.1 = ABCD.D.1) ∧
  (triangle_area ABCD.A ABCD.B ABCD.E = 60) ∧  -- Area of ABE is 60
  (triangle_area ABCD.A ABCD.D ABCD.E = 30) →  -- Area of ADE is 30
  (triangle_area ABCD.A ABCD.B ABCD.C) + 
  (triangle_area ABCD.A ABCD.C ABCD.D) = 135 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1856_185694


namespace NUMINAMATH_CALUDE_sum_of_factors_40_l1856_185686

theorem sum_of_factors_40 : (Finset.filter (· ∣ 40) (Finset.range 41)).sum id = 90 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_40_l1856_185686


namespace NUMINAMATH_CALUDE_odd_perfect_square_l1856_185621

theorem odd_perfect_square (n : ℕ+) 
  (h : (Finset.sum (Nat.divisors n.val) id) = 2 * n.val + 1) : 
  ∃ (k : ℕ), n.val = 2 * k + 1 ∧ ∃ (m : ℕ), n.val = m ^ 2 :=
sorry

end NUMINAMATH_CALUDE_odd_perfect_square_l1856_185621


namespace NUMINAMATH_CALUDE_rhombus_side_length_l1856_185671

/-- 
A rhombus is a quadrilateral with four equal sides.
The perimeter of a rhombus is the sum of the lengths of all four sides.
-/
structure Rhombus where
  side_length : ℝ
  perimeter : ℝ
  perimeter_eq : perimeter = 4 * side_length

theorem rhombus_side_length (r : Rhombus) (h : r.perimeter = 4) : r.side_length = 1 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l1856_185671


namespace NUMINAMATH_CALUDE_trivia_team_tryouts_l1856_185616

theorem trivia_team_tryouts (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ) :
  not_picked = 17 →
  groups = 8 →
  students_per_group = 6 →
  not_picked + groups * students_per_group = 65 :=
by sorry

end NUMINAMATH_CALUDE_trivia_team_tryouts_l1856_185616


namespace NUMINAMATH_CALUDE_range_of_x_l1856_185613

theorem range_of_x (x : ℝ) : 
  (Real.sqrt ((1 - 2*x)^2) = 2*x - 1) → x ≥ (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_l1856_185613


namespace NUMINAMATH_CALUDE_eating_contest_l1856_185608

/-- Eating contest problem -/
theorem eating_contest (hotdog_weight burger_weight pie_weight : ℕ)
  (jacob_pies noah_burgers mason_hotdogs : ℕ)
  (h1 : hotdog_weight = 2)
  (h2 : burger_weight = 5)
  (h3 : pie_weight = 10)
  (h4 : jacob_pies + 3 = noah_burgers)
  (h5 : mason_hotdogs = 3 * jacob_pies)
  (h6 : mason_hotdogs * hotdog_weight = 30) :
  noah_burgers = 8 := by
  sorry

end NUMINAMATH_CALUDE_eating_contest_l1856_185608


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_product_sum_l1856_185646

theorem consecutive_negative_integers_product_sum (n : ℤ) :
  n < 0 ∧ n * (n + 1) = 2184 → n + (n + 1) = -95 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_product_sum_l1856_185646


namespace NUMINAMATH_CALUDE_sunset_time_l1856_185648

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents the length of a time period in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  deriving Repr

def addTime (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + d.hours * 60 + d.minutes
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

def sunrise : Time := { hours := 6, minutes := 12 }
def daylightLength : Duration := { hours := 12, minutes := 36 }

theorem sunset_time :
  addTime sunrise daylightLength = { hours := 18, minutes := 48 } := by
  sorry

end NUMINAMATH_CALUDE_sunset_time_l1856_185648


namespace NUMINAMATH_CALUDE_ohms_law_application_l1856_185631

/-- Given a constant voltage U, current I inversely proportional to resistance R,
    prove that for I1 = 4A, R1 = 10Ω, and I2 = 5A, the value of R2 is 8Ω. -/
theorem ohms_law_application (U : ℝ) (I1 I2 R1 R2 : ℝ) : 
  U > 0 →  -- Voltage is positive
  I1 > 0 →  -- Current is positive
  I2 > 0 →  -- Current is positive
  R1 > 0 →  -- Resistance is positive
  R2 > 0 →  -- Resistance is positive
  (∀ I R, U = I * R) →  -- Ohm's law: U = IR
  I1 = 4 →
  R1 = 10 →
  I2 = 5 →
  R2 = 8 := by
sorry

end NUMINAMATH_CALUDE_ohms_law_application_l1856_185631


namespace NUMINAMATH_CALUDE_allocation_schemes_eq_90_l1856_185680

/-- The number of ways to distribute 5 college students among 3 freshman classes -/
def allocation_schemes : ℕ :=
  let n_students : ℕ := 5
  let n_classes : ℕ := 3
  let min_per_class : ℕ := 1
  let max_per_class : ℕ := 2
  -- The actual calculation is not implemented, just returning the correct result
  90

/-- Theorem stating that the number of allocation schemes is 90 -/
theorem allocation_schemes_eq_90 : allocation_schemes = 90 := by
  -- The proof is not implemented
  sorry

end NUMINAMATH_CALUDE_allocation_schemes_eq_90_l1856_185680


namespace NUMINAMATH_CALUDE_toms_marbles_pairs_l1856_185662

/-- Represents the set of marbles Tom has --/
structure MarbleSet where
  unique_colors : ℕ
  yellow_count : ℕ
  orange_count : ℕ

/-- Calculates the number of distinct pairs of marbles that can be chosen --/
def distinct_pairs (ms : MarbleSet) : ℕ :=
  let yellow_pairs := if ms.yellow_count ≥ 2 then 1 else 0
  let orange_pairs := if ms.orange_count ≥ 2 then 1 else 0
  let diff_color_pairs := ms.unique_colors.choose 2
  let yellow_other_pairs := ms.unique_colors * ms.yellow_count
  let orange_other_pairs := ms.unique_colors * ms.orange_count
  yellow_pairs + orange_pairs + diff_color_pairs + yellow_other_pairs + orange_other_pairs

/-- Theorem stating that Tom's marble set results in 36 distinct pairs --/
theorem toms_marbles_pairs :
  distinct_pairs { unique_colors := 4, yellow_count := 4, orange_count := 3 } = 36 := by
  sorry

end NUMINAMATH_CALUDE_toms_marbles_pairs_l1856_185662


namespace NUMINAMATH_CALUDE_intersection_trajectory_l1856_185693

/-- The trajectory of the intersection point of two rotating rods -/
theorem intersection_trajectory (a : ℝ) (h : a ≠ 0) :
  ∃ (x y : ℝ), 
    (∃ (b b₁ : ℝ), b * b₁ = a^2 ∧ b ≠ 0 ∧ b₁ ≠ 0) →
    (y = -b / a * (x - a) ∧ y = b₁ / a * (x + a)) →
    x^2 + y^2 = a^2 ∧ -a < x ∧ x < a :=
by sorry

end NUMINAMATH_CALUDE_intersection_trajectory_l1856_185693


namespace NUMINAMATH_CALUDE_seven_digit_number_exists_l1856_185605

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem seven_digit_number_exists : ∃ n : ℕ, 
  (1000000 ≤ n ∧ n < 9000000) ∧ 
  (sum_of_digits n = 53) ∧ 
  (n % 13 = 0) ∧ 
  (n = 8999990) :=
sorry

end NUMINAMATH_CALUDE_seven_digit_number_exists_l1856_185605


namespace NUMINAMATH_CALUDE_election_majority_l1856_185615

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 6900 →
  winning_percentage = 60 / 100 →
  (winning_percentage * total_votes : ℚ).floor - ((1 - winning_percentage) * total_votes : ℚ).floor = 1380 :=
by sorry

end NUMINAMATH_CALUDE_election_majority_l1856_185615


namespace NUMINAMATH_CALUDE_b_initial_investment_l1856_185661

/-- Given A's investment and doubling conditions, proves B's initial investment --/
theorem b_initial_investment 
  (a_initial : ℕ) 
  (a_doubles_after_six_months : Bool) 
  (equal_yearly_investment : Bool) : ℕ :=
by
  -- Assuming a_initial = 3000, a_doubles_after_six_months = true, and equal_yearly_investment = true
  sorry

#check b_initial_investment

end NUMINAMATH_CALUDE_b_initial_investment_l1856_185661


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l1856_185695

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (n : ℕ), n > 0 ∧ (∃ (p₁ p₂ p₃ p₄ : ℕ), 
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0) ∧
  (∀ m : ℕ, m > 0 → (∃ (q₁ q₂ q₃ q₄ : ℕ),
    Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧
    q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
    m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0) → m ≥ n) ∧
  n = 210 :=
sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l1856_185695


namespace NUMINAMATH_CALUDE_song_performance_theorem_l1856_185645

/-- Represents the number of songs performed by each kid -/
structure SongCounts where
  sarah : ℕ
  emily : ℕ
  daniel : ℕ
  oli : ℕ
  chris : ℕ

/-- The total number of songs performed -/
def totalSongs (counts : SongCounts) : ℕ :=
  (counts.sarah + counts.emily + counts.daniel + counts.oli + counts.chris) / 4

theorem song_performance_theorem (counts : SongCounts) :
  counts.chris = 9 →
  counts.sarah = 3 →
  counts.emily > counts.sarah →
  counts.daniel > counts.sarah →
  counts.oli > counts.sarah →
  counts.emily < counts.chris →
  counts.daniel < counts.chris →
  counts.oli < counts.chris →
  totalSongs counts = 6 :=
by sorry

end NUMINAMATH_CALUDE_song_performance_theorem_l1856_185645


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l1856_185650

theorem angle_sum_is_pi_over_two (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β) →
  α + β = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l1856_185650


namespace NUMINAMATH_CALUDE_total_volume_of_four_cubes_l1856_185634

theorem total_volume_of_four_cubes (edge_length : ℝ) (num_cubes : ℕ) : 
  edge_length = 5 → num_cubes = 4 → num_cubes * (edge_length ^ 3) = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_of_four_cubes_l1856_185634
