import Mathlib

namespace beaver_count_l1123_112364

theorem beaver_count (initial_beavers additional_beaver : ℝ) 
  (h1 : initial_beavers = 2.0) 
  (h2 : additional_beaver = 1.0) : 
  initial_beavers + additional_beaver = 3.0 := by
  sorry

end beaver_count_l1123_112364


namespace inequality_proof_l1123_112342

theorem inequality_proof (a b : ℝ) (h1 : a < 0) (h2 : b > 0) (h3 : a + b < 0) :
  -a > b ∧ b > -b ∧ -b > a := by
  sorry

end inequality_proof_l1123_112342


namespace smallest_six_digit_number_divisible_by_3_4_5_l1123_112319

def is_divisible_by (n m : Nat) : Prop := n % m = 0

theorem smallest_six_digit_number_divisible_by_3_4_5 :
  ∀ n : Nat,
    325000 ≤ n ∧ n < 326000 →
    is_divisible_by n 3 ∧ is_divisible_by n 4 ∧ is_divisible_by n 5 →
    325020 ≤ n :=
by sorry

end smallest_six_digit_number_divisible_by_3_4_5_l1123_112319


namespace circle_tangent_theorem_l1123_112309

-- Define the circle
def Circle (a : ℝ) := {(x, y) : ℝ × ℝ | x^2 + y^2 - 2*a*x + 2*y - 1 = 0}

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (-5, a)

-- Define the condition for the tangent lines
def TangentCondition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₂ - y₁) / (x₂ - x₁) + (x₁ + x₂ - 2) / (y₁ + y₂) = 0

theorem circle_tangent_theorem :
  ∀ a : ℝ, ∀ x₁ y₁ x₂ y₂ : ℝ,
    P a ∈ Circle a →
    (x₁, y₁) ∈ Circle a →
    (x₂, y₂) ∈ Circle a →
    TangentCondition x₁ y₁ x₂ y₂ →
    a = 3 ∨ a = -2 := by
  sorry

end circle_tangent_theorem_l1123_112309


namespace total_students_eq_920_l1123_112382

/-- The number of students in the third school -/
def students_third_school : ℕ := 200

/-- The number of students in the second school -/
def students_second_school : ℕ := students_third_school + 40

/-- The number of students in the first school -/
def students_first_school : ℕ := 2 * students_second_school

/-- The total number of students from all three schools -/
def total_students : ℕ := students_first_school + students_second_school + students_third_school

theorem total_students_eq_920 : total_students = 920 := by
  sorry

end total_students_eq_920_l1123_112382


namespace intersection_M_N_l1123_112362

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = Real.sin x}
def N : Set ℝ := {x | x^2 - x - 2 < 0}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x ≤ 1} := by sorry

end intersection_M_N_l1123_112362


namespace sports_club_intersection_l1123_112369

theorem sports_club_intersection (N B T X : ℕ) : 
  N = 30 ∧ B = 18 ∧ T = 19 ∧ (N - (B + T - X) = 2) → X = 9 :=
by
  sorry

end sports_club_intersection_l1123_112369


namespace waiter_tips_l1123_112393

/-- Calculates the total tips earned by a waiter --/
def total_tips (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) : ℕ :=
  (total_customers - non_tipping_customers) * tip_amount

/-- Theorem stating the total tips earned by the waiter --/
theorem waiter_tips : total_tips 7 5 3 = 6 := by
  sorry

end waiter_tips_l1123_112393


namespace oates_reunion_attendees_l1123_112318

/-- The number of people attending the Oates reunion -/
def oates_attendees : ℕ := 50

/-- The number of people attending the Hall reunion -/
def hall_attendees : ℕ := 62

/-- The number of people attending both reunions -/
def both_attendees : ℕ := 12

/-- The total number of guests at the hotel -/
def total_guests : ℕ := 100

theorem oates_reunion_attendees :
  oates_attendees + hall_attendees - both_attendees = total_guests :=
by sorry

end oates_reunion_attendees_l1123_112318


namespace sum_first_four_is_sixty_l1123_112385

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  sum_first_two : a + a * r = 15
  sum_first_six : a * (1 - r^6) / (1 - r) = 93

/-- The sum of the first 4 terms of the geometric sequence is 60 -/
theorem sum_first_four_is_sixty (seq : GeometricSequence) :
  seq.a * (1 - seq.r^4) / (1 - seq.r) = 60 := by
  sorry


end sum_first_four_is_sixty_l1123_112385


namespace glove_selection_ways_l1123_112363

/-- The number of different pairs of gloves -/
def num_pairs : ℕ := 6

/-- The number of gloves to be selected -/
def num_selected : ℕ := 4

/-- The number of matching pairs in the selection -/
def num_matching_pairs : ℕ := 1

/-- The total number of ways to select the gloves -/
def total_ways : ℕ := 240

theorem glove_selection_ways :
  (num_pairs : ℕ) = 6 →
  (num_selected : ℕ) = 4 →
  (num_matching_pairs : ℕ) = 1 →
  (total_ways : ℕ) = 240 := by
  sorry

end glove_selection_ways_l1123_112363


namespace expand_product_l1123_112399

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end expand_product_l1123_112399


namespace smallest_equal_packages_l1123_112333

/-- The number of pencils in each pack -/
def pencils_per_pack : ℕ := 10

/-- The number of pencil sharpeners in each pack -/
def sharpeners_per_pack : ℕ := 14

/-- The smallest number of pencil sharpener packages needed -/
def min_sharpener_packages : ℕ := 5

theorem smallest_equal_packages :
  ∃ (pencil_packs : ℕ),
    pencil_packs * pencils_per_pack = min_sharpener_packages * sharpeners_per_pack ∧
    ∀ (k : ℕ), k < min_sharpener_packages →
      ¬∃ (m : ℕ), m * pencils_per_pack = k * sharpeners_per_pack :=
by sorry

end smallest_equal_packages_l1123_112333


namespace subtract_from_square_l1123_112334

theorem subtract_from_square (n : ℕ) (h : n = 17) : n^2 - n = 272 := by
  sorry

end subtract_from_square_l1123_112334


namespace mailman_problem_l1123_112367

theorem mailman_problem (total_junk_mail : ℕ) (white_mailboxes : ℕ) (red_mailboxes : ℕ) (mail_per_house : ℕ) :
  total_junk_mail = 48 →
  white_mailboxes = 2 →
  red_mailboxes = 3 →
  mail_per_house = 6 →
  white_mailboxes + red_mailboxes + (total_junk_mail - (white_mailboxes + red_mailboxes) * mail_per_house) / mail_per_house = 8 :=
by
  sorry

end mailman_problem_l1123_112367


namespace circle_and_sphere_sum_l1123_112331

theorem circle_and_sphere_sum (c : ℝ) (h : c = 18 * Real.pi) :
  let r := c / (2 * Real.pi)
  (Real.pi * r^2) + (4/3 * Real.pi * r^3) = 1053 * Real.pi :=
by sorry

end circle_and_sphere_sum_l1123_112331


namespace initial_position_proof_l1123_112317

def moves : List Int := [-5, 4, 2, -3, 1]
def final_position : Int := 6

theorem initial_position_proof :
  (moves.foldl (· + ·) final_position) = 7 := by sorry

end initial_position_proof_l1123_112317


namespace triangle_circumcircle_diameter_l1123_112389

theorem triangle_circumcircle_diameter 
  (a b c : ℝ) 
  (ha : a = 25) 
  (hb : b = 39) 
  (hc : c = 40) : 
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  2 * (a * b * c) / (4 * area) = 125 / 3 := by
  sorry

end triangle_circumcircle_diameter_l1123_112389


namespace age_ratio_problem_l1123_112336

theorem age_ratio_problem (sam drew : ℕ) : 
  sam + drew = 54 → sam = 18 → sam * 2 = drew :=
by
  sorry

end age_ratio_problem_l1123_112336


namespace friend_gcd_l1123_112314

theorem friend_gcd (a b : ℕ) (h : ∃ k : ℕ, a * b = k^2) :
  ∃ m : ℕ, a * Nat.gcd a b = m^2 := by
sorry

end friend_gcd_l1123_112314


namespace binomial_consecutive_terms_ratio_l1123_112306

theorem binomial_consecutive_terms_ratio (n k : ℕ) : 
  (∃ (a b c : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    a / b = 1 / 3 ∧ b / c = 3 / 5 ∧
    a / b = Nat.choose n k / Nat.choose n (k + 1) ∧
    b / c = Nat.choose n (k + 1) / Nat.choose n (k + 2)) →
  n + k = 19 :=
by sorry

end binomial_consecutive_terms_ratio_l1123_112306


namespace sum_remainder_by_eight_l1123_112357

theorem sum_remainder_by_eight (n : ℤ) : ((8 - n) + (n + 4)) % 8 = 4 := by
  sorry

end sum_remainder_by_eight_l1123_112357


namespace geometric_sequence_common_ratio_l1123_112327

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  a 2 = 8 →                                 -- Given condition
  a 5 = 64 →                                -- Given condition
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 2 :=
by
  sorry


end geometric_sequence_common_ratio_l1123_112327


namespace parabola_coefficients_l1123_112300

/-- A parabola with vertex (h, k), vertical axis of symmetry, passing through point (x₀, y₀) -/
structure Parabola where
  h : ℝ
  k : ℝ
  x₀ : ℝ
  y₀ : ℝ

/-- The quadratic function representing the parabola -/
def quadratic_function (p : Parabola) (x : ℝ) : ℝ :=
  (x - p.h)^2 + p.k

theorem parabola_coefficients (p : Parabola) 
  (h_vertex : p.h = 2 ∧ p.k = -3)
  (h_point : p.x₀ = 0 ∧ p.y₀ = 1)
  (h_passes : quadratic_function p p.x₀ = p.y₀) :
  ∃ (a b c : ℝ), a = 1 ∧ b = -4 ∧ c = 1 ∧
  ∀ x, quadratic_function p x = a * x^2 + b * x + c :=
sorry

end parabola_coefficients_l1123_112300


namespace fedya_can_keep_below_1000_l1123_112354

/-- Represents the state of the number on the screen -/
structure ScreenNumber where
  value : ℕ
  minutes : ℕ

/-- Increases the number by 102 -/
def increment (n : ScreenNumber) : ScreenNumber :=
  { value := n.value + 102, minutes := n.minutes + 1 }

/-- Rearranges the digits of a number -/
def rearrange (n : ℕ) : ℕ := sorry

/-- Fedya's strategy to keep the number below 1000 -/
def fedya_strategy (n : ScreenNumber) : ScreenNumber :=
  if n.value < 1000 then n else { n with value := rearrange n.value }

/-- Theorem stating that Fedya can always keep the number below 1000 -/
theorem fedya_can_keep_below_1000 :
  ∀ (n : ℕ), n < 1000 →
  ∃ (strategy : ℕ → ScreenNumber),
    (∀ (k : ℕ), (strategy k).value < 1000) ∧
    strategy 0 = { value := 123, minutes := 0 } ∧
    (∀ (k : ℕ), strategy (k + 1) = fedya_strategy (increment (strategy k))) :=
sorry

end fedya_can_keep_below_1000_l1123_112354


namespace train_length_l1123_112346

/-- Calculates the length of a train given its speed and time to pass a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 90 → time_s = 9 → speed_kmh / 3.6 * time_s = 225 := by
  sorry

#check train_length

end train_length_l1123_112346


namespace vasya_can_win_l1123_112379

/-- Represents the state of the water pots -/
structure PotState :=
  (pot3 : Nat)
  (pot5 : Nat)
  (pot7 : Nat)

/-- Represents a move by Vasya -/
inductive VasyaMove
  | FillPot3
  | FillPot5
  | FillPot7
  | TransferPot3ToPot5
  | TransferPot3ToPot7
  | TransferPot5ToPot3
  | TransferPot5ToPot7
  | TransferPot7ToPot3
  | TransferPot7ToPot5

/-- Represents a move by Dima -/
inductive DimaMove
  | EmptyPot3
  | EmptyPot5
  | EmptyPot7

/-- Applies Vasya's move to the current state -/
def applyVasyaMove (state : PotState) (move : VasyaMove) : PotState :=
  sorry

/-- Applies Dima's move to the current state -/
def applyDimaMove (state : PotState) (move : DimaMove) : PotState :=
  sorry

/-- Checks if the game is won (1 liter in any pot) -/
def isGameWon (state : PotState) : Bool :=
  state.pot3 = 1 || state.pot5 = 1 || state.pot7 = 1

/-- Theorem: Vasya can win the game -/
theorem vasya_can_win :
  ∃ (moves : List (VasyaMove × VasyaMove)),
    ∀ (dimaMoves : List DimaMove),
      let finalState := (moves.zip dimaMoves).foldl
        (fun state (vasyaMoves, dimaMove) =>
          let s1 := applyVasyaMove state vasyaMoves.1
          let s2 := applyVasyaMove s1 vasyaMoves.2
          applyDimaMove s2 dimaMove)
        { pot3 := 0, pot5 := 0, pot7 := 0 }
      isGameWon finalState :=
by
  sorry


end vasya_can_win_l1123_112379


namespace liquid_distribution_l1123_112321

theorem liquid_distribution (n : ℕ) (a : ℝ) (h : n ≥ 2) :
  ∃ (x : ℕ → ℝ),
    (∀ k, 1 ≤ k ∧ k ≤ n → x k > 0) ∧
    (∀ k, 2 ≤ k ∧ k ≤ n → (1 - 1/n) * x k + (1/n) * x (k-1) = a) ∧
    ((1 - 1/n) * x 1 + (1/n) * x n = a) ∧
    (x 1 = a * n * (n-2) / (n-1)^2) ∧
    (x 2 = a * (n^2 - 2*n + 2) / (n-1)^2) ∧
    (∀ k, 3 ≤ k ∧ k ≤ n → x k = a) :=
by
  sorry

#check liquid_distribution

end liquid_distribution_l1123_112321


namespace evaluate_expression_l1123_112383

theorem evaluate_expression : (47^2 - 28^2) + 100 = 1525 := by
  sorry

end evaluate_expression_l1123_112383


namespace halfway_between_one_fifth_and_one_third_l1123_112330

theorem halfway_between_one_fifth_and_one_third :
  (1 / 5 : ℚ) / 2 + (1 / 3 : ℚ) / 2 = 4 / 15 := by
  sorry

end halfway_between_one_fifth_and_one_third_l1123_112330


namespace even_function_implies_b_zero_l1123_112377

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = x² + bx -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x

theorem even_function_implies_b_zero (b : ℝ) :
  IsEven (f b) → b = 0 := by sorry

end even_function_implies_b_zero_l1123_112377


namespace lower_limit_of_b_l1123_112368

theorem lower_limit_of_b (a b : ℤ) (h1 : 10 ≤ a ∧ a ≤ 25) (h2 : b < 31) 
  (h3 : (a : ℚ) / b ≤ 4/3) : 19 ≤ b := by
  sorry

end lower_limit_of_b_l1123_112368


namespace f_extrema_l1123_112372

def f (x : ℝ) := 3 * x^4 - 6 * x^2 + 4

theorem f_extrema :
  (∀ x ∈ Set.Icc (-1) 3, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-1) 3, f x = 1) ∧
  (∀ x ∈ Set.Icc (-1) 3, f x ≤ 193) ∧
  (∃ x ∈ Set.Icc (-1) 3, f x = 193) :=
by sorry

end f_extrema_l1123_112372


namespace second_term_of_geometric_series_l1123_112380

/-- Given an infinite geometric series with common ratio 1/4 and sum 48,
    the second term of the sequence is 9. -/
theorem second_term_of_geometric_series :
  ∀ (a : ℝ), -- first term of the series
  let r : ℝ := (1 : ℝ) / 4 -- common ratio
  let S : ℝ := 48 -- sum of the series
  (S = a / (1 - r)) → -- formula for sum of infinite geometric series
  (a * r = 9) -- second term of the sequence
  := by sorry

end second_term_of_geometric_series_l1123_112380


namespace student_distribution_theorem_l1123_112366

/-- The number of ways to distribute students among communities -/
def distribute_students (n_students : ℕ) (n_communities : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the correct number of arrangements -/
theorem student_distribution_theorem :
  distribute_students 4 3 = 36 := by sorry

end student_distribution_theorem_l1123_112366


namespace intersection_point_implies_m_equals_six_l1123_112302

theorem intersection_point_implies_m_equals_six (m : ℕ+) 
  (h : ∃ (x y : ℤ), 13 * x + 11 * y = 700 ∧ y = m * x - 1) : m = 6 := by
  sorry

end intersection_point_implies_m_equals_six_l1123_112302


namespace four_spheres_cover_all_rays_l1123_112371

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a sphere in 3D space
structure Sphere where
  center : Point3D
  radius : ℝ

-- Define a ray in 3D space
structure Ray where
  origin : Point3D
  direction : Point3D

-- Function to check if a ray intersects a sphere
def rayIntersectsSphere (r : Ray) (s : Sphere) : Prop :=
  sorry

-- Theorem statement
theorem four_spheres_cover_all_rays :
  ∃ (lightSource : Point3D) (s₁ s₂ s₃ s₄ : Sphere),
    ∀ (r : Ray),
      r.origin = lightSource →
      rayIntersectsSphere r s₁ ∨
      rayIntersectsSphere r s₂ ∨
      rayIntersectsSphere r s₃ ∨
      rayIntersectsSphere r s₄ :=
sorry

end four_spheres_cover_all_rays_l1123_112371


namespace compute_expression_l1123_112326

theorem compute_expression : 3 * 3^4 - 9^60 / 9^57 = -486 := by sorry

end compute_expression_l1123_112326


namespace additional_capacity_l1123_112349

/-- Represents the number of cars used by the swimming club -/
def num_cars : Nat := 2

/-- Represents the number of vans used by the swimming club -/
def num_vans : Nat := 3

/-- Represents the number of people in each car -/
def people_per_car : Nat := 5

/-- Represents the number of people in each van -/
def people_per_van : Nat := 3

/-- Represents the maximum capacity of each car -/
def max_car_capacity : Nat := 6

/-- Represents the maximum capacity of each van -/
def max_van_capacity : Nat := 8

/-- Theorem stating the number of additional people that could have ridden with the swim team -/
theorem additional_capacity : 
  (num_cars * max_car_capacity + num_vans * max_van_capacity) - 
  (num_cars * people_per_car + num_vans * people_per_van) = 17 := by
  sorry

end additional_capacity_l1123_112349


namespace new_girl_weight_l1123_112301

theorem new_girl_weight (W : ℝ) (new_weight : ℝ) :
  (W - 40 + new_weight) / 20 = W / 20 + 2 →
  new_weight = 80 :=
by
  sorry

end new_girl_weight_l1123_112301


namespace curve_is_hyperbola_l1123_112353

theorem curve_is_hyperbola (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0) →
  ∃ (A B : ℝ), A > 0 ∧ B > 0 ∧
    ∀ (x y : ℝ), b * x^2 + a * y^2 = a * b ↔ x^2 / A - y^2 / B = 1 :=
sorry

end curve_is_hyperbola_l1123_112353


namespace tax_percentage_proof_l1123_112322

def tax_problem (net_income : ℝ) (gross_income : ℝ) (untaxed_amount : ℝ) : Prop :=
  let taxable_income := gross_income - untaxed_amount
  let tax_rate := (gross_income - net_income) / taxable_income
  tax_rate = 0.10

theorem tax_percentage_proof :
  tax_problem 12000 13000 3000 := by
  sorry

end tax_percentage_proof_l1123_112322


namespace max_value_is_27_l1123_112304

/-- Represents the crop types -/
inductive Crop
| Melon
| Fruit
| Vegetable

/-- Represents the problem parameters -/
structure ProblemParams where
  totalLaborers : ℕ
  totalLand : ℕ
  laborRequirement : Crop → ℚ
  cropValue : Crop → ℚ

/-- Represents the allocation of crops to land -/
structure Allocation where
  melonAcres : ℚ
  fruitAcres : ℚ
  vegetableAcres : ℚ

/-- Calculates the total value for a given allocation -/
def totalValue (p : ProblemParams) (a : Allocation) : ℚ :=
  a.melonAcres * p.cropValue Crop.Melon +
  a.fruitAcres * p.cropValue Crop.Fruit +
  a.vegetableAcres * p.cropValue Crop.Vegetable

/-- Checks if an allocation is valid according to the problem constraints -/
def isValidAllocation (p : ProblemParams) (a : Allocation) : Prop :=
  a.melonAcres + a.fruitAcres + a.vegetableAcres = p.totalLand ∧
  a.melonAcres * p.laborRequirement Crop.Melon +
  a.fruitAcres * p.laborRequirement Crop.Fruit +
  a.vegetableAcres * p.laborRequirement Crop.Vegetable = p.totalLaborers

/-- The main theorem stating that the maximum value is 27 million yuan -/
theorem max_value_is_27 (p : ProblemParams)
  (h1 : p.totalLaborers = 20)
  (h2 : p.totalLand = 50)
  (h3 : p.laborRequirement Crop.Melon = 1/2)
  (h4 : p.laborRequirement Crop.Fruit = 1/3)
  (h5 : p.laborRequirement Crop.Vegetable = 1/4)
  (h6 : p.cropValue Crop.Melon = 6/10)
  (h7 : p.cropValue Crop.Fruit = 1/2)
  (h8 : p.cropValue Crop.Vegetable = 3/10) :
  ∃ (a : Allocation), isValidAllocation p a ∧
    ∀ (a' : Allocation), isValidAllocation p a' → totalValue p a' ≤ 27 :=
sorry

end max_value_is_27_l1123_112304


namespace product_constant_percentage_change_l1123_112325

theorem product_constant_percentage_change (x1 y1 x2 y2 : ℝ) :
  x1 * y1 = x2 * y2 ∧ 
  y2 = y1 * (1 - 44.44444444444444 / 100) →
  x2 = x1 * (1 + 80 / 100) :=
by sorry

end product_constant_percentage_change_l1123_112325


namespace square_circle_union_area_l1123_112347

/-- The area of the union of a square with side length 10 and a circle with radius 10 
    centered at one of the square's vertices is equal to 100 + 75π. -/
theorem square_circle_union_area : 
  let square_side : ℝ := 10
  let circle_radius : ℝ := 10
  let square_area := square_side ^ 2
  let circle_area := π * circle_radius ^ 2
  let overlap_area := (1 / 4 : ℝ) * circle_area
  square_area + circle_area - overlap_area = 100 + 75 * π :=
by sorry

end square_circle_union_area_l1123_112347


namespace division_problem_l1123_112310

theorem division_problem (dividend quotient divisor remainder : ℕ) 
  (h1 : remainder = 8)
  (h2 : divisor = 3 * remainder + 3)
  (h3 : dividend = 251)
  (h4 : dividend = divisor * quotient + remainder) :
  ∃ (m : ℕ), divisor = m * quotient ∧ m = 3 := by
sorry

end division_problem_l1123_112310


namespace base4_multiplication_division_l1123_112348

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- Theorem stating that 132₄ × 21₄ ÷ 3₄ = 1122₄ --/
theorem base4_multiplication_division :
  let a := base4ToBase10 [2, 3, 1]  -- 132₄
  let b := base4ToBase10 [1, 2]     -- 21₄
  let c := base4ToBase10 [3]        -- 3₄
  let result := base10ToBase4 ((a * b) / c)
  result = [2, 2, 1, 1] := by sorry

end base4_multiplication_division_l1123_112348


namespace add_2023_minutes_to_midnight_l1123_112370

/-- Represents a time with day, hour, and minute components -/
structure DateTime where
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime and returns the resulting DateTime -/
def addMinutes (start : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- The starting DateTime (midnight on December 31, 2020) -/
def startTime : DateTime :=
  { day := 0, hour := 0, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : Nat := 2023

/-- The expected result DateTime (January 1 at 9:43 AM) -/
def expectedResult : DateTime :=
  { day := 1, hour := 9, minute := 43 }

/-- Theorem stating that adding 2023 minutes to midnight on December 31, 2020,
    results in January 1 at 9:43 AM -/
theorem add_2023_minutes_to_midnight :
  addMinutes startTime minutesToAdd = expectedResult := by
  sorry

end add_2023_minutes_to_midnight_l1123_112370


namespace train_length_l1123_112360

/-- Given a train that crosses a platform in 39 seconds and a signal pole in 18 seconds,
    where the platform is 350 meters long, prove that the length of the train is 300 meters. -/
theorem train_length (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) (platform_length : ℝ)
  (h1 : platform_crossing_time = 39)
  (h2 : pole_crossing_time = 18)
  (h3 : platform_length = 350) :
  let train_length := (platform_length * pole_crossing_time) / (platform_crossing_time - pole_crossing_time)
  train_length = 300 := by sorry

end train_length_l1123_112360


namespace billing_method_comparison_l1123_112339

/-- Cost calculation for Method A -/
def cost_A (x : ℝ) : ℝ := 8 + 0.2 * x

/-- Cost calculation for Method B -/
def cost_B (x : ℝ) : ℝ := 0.3 * x

/-- Theorem comparing billing methods based on call duration -/
theorem billing_method_comparison (x : ℝ) :
  (x < 80 → cost_B x < cost_A x) ∧
  (x = 80 → cost_A x = cost_B x) ∧
  (x > 80 → cost_A x < cost_B x) := by
  sorry

end billing_method_comparison_l1123_112339


namespace toy_bridge_weight_l1123_112388

/-- The weight that a toy bridge must support given the following conditions:
  * There are 6 cans of soda, each containing 12 ounces of soda
  * Each empty can weighs 2 ounces
  * There are 2 additional empty cans
-/
theorem toy_bridge_weight (soda_cans : ℕ) (soda_per_can : ℕ) (empty_can_weight : ℕ) (additional_cans : ℕ) :
  soda_cans = 6 →
  soda_per_can = 12 →
  empty_can_weight = 2 →
  additional_cans = 2 →
  (soda_cans * soda_per_can) + ((soda_cans + additional_cans) * empty_can_weight) = 88 := by
  sorry

end toy_bridge_weight_l1123_112388


namespace inequality_proof_l1123_112308

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 2*(a-1)*(b-1) ≥ 1 := by
  sorry

end inequality_proof_l1123_112308


namespace four_digit_divisibility_sum_l1123_112392

/-- The number of four-digit numbers divisible by 3 -/
def C : ℕ := 3000

/-- The number of four-digit multiples of 7 -/
def D : ℕ := 1286

/-- Theorem stating that the sum of four-digit numbers divisible by 3 and multiples of 7 is 4286 -/
theorem four_digit_divisibility_sum : C + D = 4286 := by
  sorry

end four_digit_divisibility_sum_l1123_112392


namespace geometric_series_ratio_l1123_112351

/-- 
Given an infinite geometric series with first term a and common ratio r,
if the sum of the original series is 81 times the sum of the series
that results when the first four terms are removed, then r = 1/3.
-/
theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) :
  (a / (1 - r)) = 81 * (a * r^4 / (1 - r)) →
  r = 1/3 := by sorry

end geometric_series_ratio_l1123_112351


namespace product_of_numbers_l1123_112386

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 180) : x * y = 72 := by
  sorry

end product_of_numbers_l1123_112386


namespace third_side_length_l1123_112344

theorem third_side_length (a b : ℝ) (h1 : a = 3.14) (h2 : b = 0.67) : 
  ∃ m : ℤ, (m : ℝ) > |a - b| ∧ (m : ℝ) < a + b ∧ m = 3 :=
by sorry

end third_side_length_l1123_112344


namespace katie_miles_run_l1123_112394

/-- Given that Adam ran 125 miles and Adam ran 80 miles more than Katie, prove that Katie ran 45 miles. -/
theorem katie_miles_run (adam_miles : ℕ) (difference : ℕ) (katie_miles : ℕ) 
  (h1 : adam_miles = 125)
  (h2 : adam_miles = katie_miles + difference)
  (h3 : difference = 80) : 
  katie_miles = 45 := by
sorry

end katie_miles_run_l1123_112394


namespace percentage_problem_l1123_112355

theorem percentage_problem (x : ℝ) : (23 / 100) * x = 150 → x = 15000 / 23 := by
  sorry

end percentage_problem_l1123_112355


namespace shirts_total_cost_l1123_112338

/-- Calculates the total cost of shirts with given prices, quantities, discounts, and taxes -/
def totalCost (price1 price2 : ℝ) (quantity1 quantity2 : ℕ) (discount tax : ℝ) : ℝ :=
  quantity1 * (price1 * (1 - discount)) + quantity2 * (price2 * (1 + tax))

/-- Theorem stating that the total cost of the shirts is $82.50 -/
theorem shirts_total_cost :
  totalCost 15 20 3 2 0.1 0.05 = 82.5 := by
  sorry

#eval totalCost 15 20 3 2 0.1 0.05

end shirts_total_cost_l1123_112338


namespace three_fish_thrown_back_l1123_112323

/-- Represents the number of fish caught by each family member and the total number of filets --/
structure FishingTrip where
  ben : Nat
  judy : Nat
  billy : Nat
  jim : Nat
  susie : Nat
  total_filets : Nat

/-- Calculates the number of fish thrown back given a fishing trip --/
def fish_thrown_back (trip : FishingTrip) : Nat :=
  let total_caught := trip.ben + trip.judy + trip.billy + trip.jim + trip.susie
  let kept := trip.total_filets / 2
  total_caught - kept

/-- Theorem stating that for the given fishing trip, 3 fish were thrown back --/
theorem three_fish_thrown_back : 
  let trip := FishingTrip.mk 4 1 3 2 5 24
  fish_thrown_back trip = 3 := by
  sorry

end three_fish_thrown_back_l1123_112323


namespace short_trees_count_l1123_112329

/-- The number of short trees in the park after planting -/
def short_trees_after_planting (initial_short_trees planted_short_trees : ℕ) : ℕ :=
  initial_short_trees + planted_short_trees

/-- Theorem: The number of short trees after planting is 95 -/
theorem short_trees_count : short_trees_after_planting 31 64 = 95 := by
  sorry

end short_trees_count_l1123_112329


namespace reflection_of_circle_center_l1123_112337

/-- Reflects a point (x, y) about the line y = -x --/
def reflectAboutNegativeX (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (8, -3)
  reflectAboutNegativeX original_center = (3, -8) := by
  sorry

end reflection_of_circle_center_l1123_112337


namespace product_one_minus_reciprocals_l1123_112397

theorem product_one_minus_reciprocals : (1 - 1/3) * (1 - 1/4) * (1 - 1/5) = 2/5 := by
  sorry

end product_one_minus_reciprocals_l1123_112397


namespace derivative_x_sin_x_l1123_112305

theorem derivative_x_sin_x (x : ℝ) :
  let f : ℝ → ℝ := λ x => x * Real.sin x
  (deriv f) x = Real.sin x + x * Real.cos x := by
  sorry

end derivative_x_sin_x_l1123_112305


namespace sum_of_coefficients_l1123_112328

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ) :
  (∀ x : ℤ, (1 + 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₂ + a₄ = 121 := by
  sorry

end sum_of_coefficients_l1123_112328


namespace f_extrema_a_range_l1123_112356

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x - 9

-- Theorem for part 1
theorem f_extrema :
  (∀ x ∈ Set.Icc 0 2, f x ≥ -4) ∧
  (∃ x ∈ Set.Icc 0 2, f x = -4) ∧
  (∀ x ∈ Set.Icc 0 2, f x ≤ -3) ∧
  (∃ x ∈ Set.Icc 0 2, f x = -3) :=
sorry

-- Theorem for part 2
theorem a_range :
  ∀ a < 0,
  (∀ x : ℝ, Real.sin x ^ 2 + a * Real.cos x + a^2 ≥ 1 + Real.cos x) →
  a ≤ -2 :=
sorry

end f_extrema_a_range_l1123_112356


namespace catch_up_theorem_l1123_112352

/-- The number of days after which the second student catches up with the first student -/
def catch_up_day : ℕ := 13

/-- The distance walked by the first student each day -/
def first_student_daily_distance : ℕ := 7

/-- The distance walked by the second student on the nth day -/
def second_student_daily_distance (n : ℕ) : ℕ := n

/-- The total distance walked by the first student after n days -/
def first_student_total_distance (n : ℕ) : ℕ :=
  n * first_student_daily_distance

/-- The total distance walked by the second student after n days -/
def second_student_total_distance (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem catch_up_theorem :
  first_student_total_distance catch_up_day = second_student_total_distance catch_up_day :=
by sorry

end catch_up_theorem_l1123_112352


namespace median_squares_sum_l1123_112390

/-- For a triangle with sides a, b, c, medians m_a, m_b, m_c, and circumcircle diameter D,
    the sum of squares of medians equals 3/4 of the sum of squares of sides plus 3/4 of the square of the diameter. -/
theorem median_squares_sum (a b c m_a m_b m_c D : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_m_a : 0 < m_a) (h_pos_m_b : 0 < m_b) (h_pos_m_c : 0 < m_c)
  (h_pos_D : 0 < D)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_median_a : 4 * m_a^2 + a^2 = 2 * b^2 + 2 * c^2)
  (h_median_b : 4 * m_b^2 + b^2 = 2 * c^2 + 2 * a^2)
  (h_median_c : 4 * m_c^2 + c^2 = 2 * a^2 + 2 * b^2)
  (h_D : D ≥ max a (max b c)) :
  m_a^2 + m_b^2 + m_c^2 = 3/4 * (a^2 + b^2 + c^2) + 3/4 * D^2 :=
sorry

end median_squares_sum_l1123_112390


namespace probability_of_red_ball_l1123_112311

/-- The number of red balls in the bag -/
def num_red_balls : ℕ := 4

/-- The number of green balls in the bag -/
def num_green_balls : ℕ := 5

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_red_balls + num_green_balls

/-- The probability of drawing a red ball from the bag -/
def prob_red_ball : ℚ := num_red_balls / total_balls

theorem probability_of_red_ball :
  prob_red_ball = 4 / 9 := by sorry

end probability_of_red_ball_l1123_112311


namespace complex_square_root_l1123_112303

theorem complex_square_root (z : ℂ) : z^2 = 3 - 4*I → z = 1 - 2*I ∨ z = -1 + 2*I := by
  sorry

end complex_square_root_l1123_112303


namespace telescope_visual_range_l1123_112374

/-- Given a telescope that increases the visual range by 66.67% to reach 150 kilometers,
    prove that the initial visual range without the telescope is 90 kilometers. -/
theorem telescope_visual_range (initial_range : ℝ) : 
  (initial_range + initial_range * (2/3) = 150) → initial_range = 90 := by
  sorry

end telescope_visual_range_l1123_112374


namespace gift_box_volume_l1123_112376

/-- The volume of a rectangular box. -/
def boxVolume (length width height : ℝ) : ℝ :=
  length * width * height

/-- Theorem: The volume of a gift box with dimensions 9 cm wide, 4 cm long, and 7 cm high is 252 cubic centimeters. -/
theorem gift_box_volume :
  boxVolume 4 9 7 = 252 := by
  sorry

end gift_box_volume_l1123_112376


namespace two_books_different_genres_l1123_112324

/-- The number of ways to choose two books of different genres -/
def choose_two_books (mystery fantasy biography : ℕ) : ℕ :=
  mystery * fantasy + mystery * biography + fantasy * biography

/-- Theorem: Given 4 mystery novels, 3 fantasy novels, and 3 biographies,
    the number of ways to choose two books of different genres is 33 -/
theorem two_books_different_genres :
  choose_two_books 4 3 3 = 33 := by
  sorry

end two_books_different_genres_l1123_112324


namespace hidden_dots_count_l1123_112378

/-- The sum of numbers on a single die -/
def die_sum : ℕ := 1 + 2 + 3 + 4 + 5 + 6

/-- The total number of dots on three dice -/
def total_dots : ℕ := 3 * die_sum

/-- The sum of visible numbers on the dice -/
def visible_dots : ℕ := 1 + 1 + 2 + 3 + 4 + 5 + 6

/-- The number of hidden dots on the dice -/
def hidden_dots : ℕ := total_dots - visible_dots

theorem hidden_dots_count : hidden_dots = 41 := by
  sorry

end hidden_dots_count_l1123_112378


namespace hot_tea_sales_average_l1123_112332

/-- Represents the linear relationship between temperature and cups of hot tea sold -/
structure HotDrinkSales where
  slope : ℝ
  intercept : ℝ

/-- Calculates the average cups of hot tea sold given average temperature -/
def average_sales (model : HotDrinkSales) (avg_temp : ℝ) : ℝ :=
  model.slope * avg_temp + model.intercept

theorem hot_tea_sales_average (model : HotDrinkSales) (avg_temp : ℝ) 
    (h1 : model.slope = -2)
    (h2 : model.intercept = 58)
    (h3 : avg_temp = 12) :
    average_sales model avg_temp = 34 := by
  sorry

#check hot_tea_sales_average

end hot_tea_sales_average_l1123_112332


namespace chennys_friends_l1123_112395

theorem chennys_friends (initial_candies : ℕ) (additional_candies : ℕ) (candies_per_friend : ℕ) :
  initial_candies = 10 →
  additional_candies = 4 →
  candies_per_friend = 2 →
  (initial_candies + additional_candies) / candies_per_friend = 7 :=
by
  sorry

end chennys_friends_l1123_112395


namespace range_of_a_l1123_112340

-- Define the function f(x) = |x+2| - |x-3|
def f (x : ℝ) : ℝ := |x + 2| - |x - 3|

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∃ x, f x ≤ a) → a ∈ Set.Ici (-5) :=
by
  sorry

end range_of_a_l1123_112340


namespace investment_problem_investment_problem_proof_l1123_112381

/-- The investment problem -/
theorem investment_problem (a_investment : ℕ) (b_join_time : ℚ) (profit_ratio : ℚ × ℚ) : ℕ :=
  let a_investment := 27000
  let b_join_time := 7.5
  let profit_ratio := (2, 1)
  let total_months := 12
  let b_investment := a_investment * (total_months / (total_months - b_join_time)) * (profit_ratio.2 / profit_ratio.1)
  36000

/-- Proof of the investment problem -/
theorem investment_problem_proof : investment_problem 27000 (15/2) (2, 1) = 36000 := by
  sorry

end investment_problem_investment_problem_proof_l1123_112381


namespace rectangular_solid_length_l1123_112341

/-- The length of a rectangular solid with given dimensions and surface area -/
theorem rectangular_solid_length
  (width : ℝ) (depth : ℝ) (surface_area : ℝ)
  (h_width : width = 9)
  (h_depth : depth = 6)
  (h_surface_area : surface_area = 408)
  (h_formula : surface_area = 2 * length * width + 2 * length * depth + 2 * width * depth) :
  length = 10 :=
by sorry

end rectangular_solid_length_l1123_112341


namespace min_y_value_l1123_112335

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 10*x + 36*y) : 
  ∀ z : ℝ, (∃ w : ℝ, w^2 + z^2 = 10*w + 36*z) → y ≤ z → -7 ≤ y :=
sorry

end min_y_value_l1123_112335


namespace max_sum_of_complex_product_l1123_112343

/-- The maximum sum of real and imaginary parts of the product of two specific complex functions of θ -/
theorem max_sum_of_complex_product :
  let z1 (θ : ℝ) := (8 + Complex.I) * Real.sin θ + (7 + 4 * Complex.I) * Real.cos θ
  let z2 (θ : ℝ) := (1 + 8 * Complex.I) * Real.sin θ + (4 + 7 * Complex.I) * Real.cos θ
  ∃ (θ : ℝ), ∀ (φ : ℝ), (z1 θ * z2 θ).re + (z1 θ * z2 θ).im ≥ (z1 φ * z2 φ).re + (z1 φ * z2 φ).im ∧
  (z1 θ * z2 θ).re + (z1 θ * z2 θ).im = 125 :=
by
  sorry


end max_sum_of_complex_product_l1123_112343


namespace compound_interest_calculation_l1123_112315

/-- Given an investment with the following properties:
  * Initial investment: $8000
  * Interest rate: y% per annum
  * Time period: 2 years
  * Simple interest earned: $800
Prove that the compound interest earned is $820 -/
theorem compound_interest_calculation (initial_investment : ℝ) (y : ℝ) (time : ℝ) 
  (simple_interest : ℝ) (h1 : initial_investment = 8000)
  (h2 : time = 2) (h3 : simple_interest = 800) 
  (h4 : simple_interest = initial_investment * y * time / 100) :
  initial_investment * ((1 + y / 100) ^ time - 1) = 820 := by
  sorry

end compound_interest_calculation_l1123_112315


namespace same_terminal_side_l1123_112365

theorem same_terminal_side (k : ℤ) : 
  -330 = k * 360 + 30 :=
sorry

end same_terminal_side_l1123_112365


namespace f_properties_l1123_112384

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 - a*x + a)

theorem f_properties (a : ℝ) (h : a > 2) :
  let f' := deriv (f a)
  ∃ (S₁ S₂ S₃ : Set ℝ),
    (f' 0 = a) ∧
    (S₁ = Set.Iio 0) ∧
    (S₂ = Set.Ioi (a - 2)) ∧
    (S₃ = Set.Ioo 0 (a - 2)) ∧
    (StrictMonoOn (f a) S₁) ∧
    (StrictMonoOn (f a) S₂) ∧
    (StrictAntiOn (f a) S₃) :=
by sorry

end f_properties_l1123_112384


namespace set_relationship_theorem_l1123_112361

def A : Set ℝ := {-1, 2}

def B (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 = 2}

def whale_swallowing (X Y : Set ℝ) : Prop := X ⊆ Y ∨ Y ⊆ X

def moth_eating (X Y : Set ℝ) : Prop := 
  (∃ x, x ∈ X ∩ Y) ∧ ¬(X ⊆ Y) ∧ ¬(Y ⊆ X)

theorem set_relationship_theorem : 
  {a : ℝ | a ≥ 0 ∧ (whale_swallowing A (B a) ∨ moth_eating A (B a))} = {0, 1/2, 2} := by
  sorry

end set_relationship_theorem_l1123_112361


namespace tips_fraction_of_income_l1123_112316

/-- Represents the income structure of a waitress -/
structure WaitressIncome where
  salary : ℚ
  tips : ℚ

/-- Calculates the total income of a waitress -/
def totalIncome (w : WaitressIncome) : ℚ :=
  w.salary + w.tips

/-- Theorem: If a waitress's tips are 7/4 of her salary, then 7/11 of her income comes from tips -/
theorem tips_fraction_of_income (w : WaitressIncome) 
  (h : w.tips = (7 : ℚ) / 4 * w.salary) : 
  w.tips / totalIncome w = (7 : ℚ) / 11 := by
  sorry

end tips_fraction_of_income_l1123_112316


namespace consecutive_integers_product_l1123_112358

theorem consecutive_integers_product (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a * b * c * d * e = 15120 →
  e = 9 :=
by sorry

end consecutive_integers_product_l1123_112358


namespace airplane_travel_time_l1123_112320

/-- Proves that an airplane traveling 3600 km against the wind in 5 hours,
    with a still air speed of 810 km/h, takes 4 hours to travel the same distance with the wind. -/
theorem airplane_travel_time
  (distance : ℝ)
  (time_against : ℝ)
  (speed_still : ℝ)
  (h_distance : distance = 3600)
  (h_time_against : time_against = 5)
  (h_speed_still : speed_still = 810)
  : ∃ (wind_speed : ℝ),
    (distance / (speed_still - wind_speed) = time_against) ∧
    (distance / (speed_still + wind_speed) = 4) :=
by
  sorry


end airplane_travel_time_l1123_112320


namespace negation_equivalence_l1123_112307

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x ≤ -1 ∨ x ≥ 2) ↔ (∀ x : ℝ, -1 < x ∧ x < 2) := by sorry

end negation_equivalence_l1123_112307


namespace problem_statement_l1123_112313

-- Define proposition p
def p : Prop := ∀ x : ℝ, (|x| = x ↔ x > 0)

-- Define proposition q
def q : Prop := (¬∃ x₀ : ℝ, x₀^2 - x₀ > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0)

-- Theorem to prove
theorem problem_statement : ¬(p ∧ q) := by
  sorry

end problem_statement_l1123_112313


namespace parking_probability_theorem_l1123_112312

/-- Represents the parking fee structure and probabilities for a business district parking lot. -/
structure ParkingLot where
  base_fee : ℕ := 6  -- Base fee for first hour
  hourly_fee : ℕ := 8  -- Fee for each additional hour
  max_hours : ℕ := 4  -- Maximum parking duration
  prob_A_1to2 : ℚ := 1/3  -- Probability A parks between 1-2 hours
  prob_A_over14 : ℚ := 5/12  -- Probability A pays over 14 yuan

/-- Calculates the probability of various parking scenarios. -/
def parking_probabilities (lot : ParkingLot) : ℚ × ℚ :=
  let prob_A_6yuan := 1 - (lot.prob_A_1to2 + lot.prob_A_over14)
  let prob_total_36yuan := 1/4  -- Given equal probability for each time interval
  (prob_A_6yuan, prob_total_36yuan)

/-- Theorem stating the probabilities of specific parking scenarios. -/
theorem parking_probability_theorem (lot : ParkingLot) :
  parking_probabilities lot = (1/4, 1/4) := by sorry

/-- Verifies that the calculated probabilities match the expected values. -/
example (lot : ParkingLot) : 
  parking_probabilities lot = (1/4, 1/4) := by sorry

end parking_probability_theorem_l1123_112312


namespace system_solution_l1123_112345

theorem system_solution :
  ∃! (x y : ℚ), 2 * x - 3 * y = 1 ∧ (y + 1) / 4 + 1 = (x + 2) / 3 ∧ x = 3 ∧ y = 5/3 := by
  sorry

end system_solution_l1123_112345


namespace bookshelf_problem_l1123_112396

theorem bookshelf_problem (x : ℕ) 
  (h1 : (4 * x : ℚ) / (5 * x + 35 + 6 * x + 4 * x) = 22 / 100) : 
  4 * x = 44 := by
  sorry

end bookshelf_problem_l1123_112396


namespace candidate_vote_difference_l1123_112375

theorem candidate_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 6000 → 
  candidate_percentage = 35/100 →
  (total_votes : ℚ) * (1 - candidate_percentage) - (total_votes : ℚ) * candidate_percentage = 1800 :=
by sorry

end candidate_vote_difference_l1123_112375


namespace complex_equation_solution_l1123_112391

theorem complex_equation_solution (i : ℂ) (z : ℂ) 
  (h1 : i * i = -1) 
  (h2 : i * z = (1 - 2*i)^2) : 
  z = -4 + 3*i := by
  sorry

end complex_equation_solution_l1123_112391


namespace even_function_range_l1123_112373

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 2

theorem even_function_range (a b : ℝ) :
  (∀ x ∈ Set.Icc (1 + a) 2, f a b x = f a b (-x)) →
  (∃ y ∈ Set.Icc (1 + a) 2, -y ∈ Set.Icc (1 + a) 2) →
  (Set.range (f a b) = Set.Icc (-10) 2) :=
sorry

end even_function_range_l1123_112373


namespace clouddale_total_rainfall_l1123_112398

/-- Calculates the total annual rainfall given the average monthly rainfall -/
def annual_rainfall (average_monthly : ℝ) : ℝ := average_monthly * 12

/-- Represents the rainfall data for Clouddale -/
structure ClouddaleRainfall where
  avg_2003 : ℝ  -- Average monthly rainfall in 2003
  increase_rate : ℝ  -- Percentage increase in 2004

/-- Theorem stating the total rainfall for both years in Clouddale -/
theorem clouddale_total_rainfall (data : ClouddaleRainfall) 
  (h1 : data.avg_2003 = 45)
  (h2 : data.increase_rate = 0.05) : 
  (annual_rainfall data.avg_2003 = 540) ∧ 
  (annual_rainfall (data.avg_2003 * (1 + data.increase_rate)) = 567) := by
  sorry

#eval annual_rainfall 45
#eval annual_rainfall (45 * 1.05)

end clouddale_total_rainfall_l1123_112398


namespace calculation_proof_l1123_112387

theorem calculation_proof : 3^2 + Real.sqrt 25 - (64 : ℝ)^(1/3) + abs (-9) = 19 := by
  sorry

end calculation_proof_l1123_112387


namespace complex_equation_implication_l1123_112350

theorem complex_equation_implication (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (a + b * i) * i = 1 + 2 * i →
  a - b = 3 := by
  sorry

end complex_equation_implication_l1123_112350


namespace smallest_s_for_arithmetic_progression_l1123_112359

open Real

theorem smallest_s_for_arithmetic_progression (β : ℝ) (s : ℝ) :
  0 < β ∧ β < π / 2 →
  (∃ d : ℝ, arcsin (sin (3 * β)) + d = arcsin (sin (5 * β)) ∧
            arcsin (sin (5 * β)) + d = arcsin (sin (10 * β)) ∧
            arcsin (sin (10 * β)) + d = arcsin (sin (s * β))) →
  s ≥ 12 :=
by sorry

end smallest_s_for_arithmetic_progression_l1123_112359
