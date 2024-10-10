import Mathlib

namespace triangle_height_l1388_138837

theorem triangle_height (a b : ℝ) (α : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_angle : 0 < α ∧ α < π) :
  let c := Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos α))
  let h := (a * b * Real.sin α) / c
  0 < h ∧ h < a ∧ h < b ∧ h * c = a * b * Real.sin α :=
by sorry

end triangle_height_l1388_138837


namespace crowdfunding_highest_level_l1388_138856

/-- Represents the financial backing levels and backers for a crowdfunding campaign -/
structure CrowdfundingCampaign where
  lowest_level : ℕ
  second_level : ℕ
  highest_level : ℕ
  lowest_backers : ℕ
  second_backers : ℕ
  highest_backers : ℕ

/-- Theorem stating the conditions and the result to be proven -/
theorem crowdfunding_highest_level 
  (campaign : CrowdfundingCampaign)
  (level_relation : campaign.second_level = 10 * campaign.lowest_level ∧ 
                    campaign.highest_level = 10 * campaign.second_level)
  (backers : campaign.lowest_backers = 10 ∧ 
             campaign.second_backers = 3 ∧ 
             campaign.highest_backers = 2)
  (total_raised : campaign.lowest_backers * campaign.lowest_level + 
                  campaign.second_backers * campaign.second_level + 
                  campaign.highest_backers * campaign.highest_level = 12000) :
  campaign.highest_level = 5000 := by
  sorry


end crowdfunding_highest_level_l1388_138856


namespace proportion_not_true_l1388_138896

theorem proportion_not_true (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : 3 * a = 5 * b) :
  ¬(a / b = 3 / 5) := by
  sorry

end proportion_not_true_l1388_138896


namespace fraction_sum_equality_l1388_138863

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (55 - c) = 8) :
  6 / (30 - a) + 14 / (70 - b) + 11 / (55 - c) = 11 / 5 := by
  sorry

end fraction_sum_equality_l1388_138863


namespace circle_max_min_sum_l1388_138859

theorem circle_max_min_sum (x y : ℝ) :
  (x - 1)^2 + (y + 2)^2 = 4 →
  ∃ (S_max S_min : ℝ),
    (∀ x' y', (x' - 1)^2 + (y' + 2)^2 = 4 → 2*x' + y' ≤ S_max) ∧
    (∀ x' y', (x' - 1)^2 + (y' + 2)^2 = 4 → 2*x' + y' ≥ S_min) ∧
    S_max = 4 + 2*Real.sqrt 5 ∧
    S_min = 4 - 2*Real.sqrt 5 :=
by sorry

end circle_max_min_sum_l1388_138859


namespace cupcake_distribution_l1388_138867

theorem cupcake_distribution (total_cupcakes : ℕ) (total_children : ℕ) 
  (ratio_1 ratio_2 ratio_3 : ℕ) :
  total_cupcakes = 144 →
  total_children = 12 →
  ratio_1 = 3 →
  ratio_2 = 2 →
  ratio_3 = 1 →
  total_children % 3 = 0 →
  let total_ratio := ratio_1 + ratio_2 + ratio_3
  let cupcakes_per_part := total_cupcakes / total_ratio
  let group_3_cupcakes := ratio_3 * cupcakes_per_part
  let children_per_group := total_children / 3
  group_3_cupcakes / children_per_group = 6 :=
by sorry

end cupcake_distribution_l1388_138867


namespace liza_final_balance_l1388_138862

/-- Calculates the final balance in Liza's account after a series of transactions --/
def calculate_final_balance (initial_balance rent paycheck electricity internet phone additional_deposit : ℚ) : ℚ :=
  let balance_after_rent := initial_balance - rent
  let balance_after_paycheck := balance_after_rent + paycheck
  let balance_after_bills := balance_after_paycheck - electricity - internet
  let grocery_spending := balance_after_bills * (20 / 100)
  let balance_after_groceries := balance_after_bills - grocery_spending
  let interest := balance_after_groceries * (2 / 100)
  let balance_after_interest := balance_after_groceries + interest
  let final_balance := balance_after_interest - phone + additional_deposit
  final_balance

/-- Theorem stating that Liza's final account balance is $1562.528 --/
theorem liza_final_balance :
  calculate_final_balance 800 450 1500 117 100 70 300 = 1562.528 := by
  sorry

end liza_final_balance_l1388_138862


namespace intersection_dot_product_l1388_138883

/-- Given a line Ax + By + C = 0 intersecting the circle x^2 + y^2 = 9 at points P and Q,
    where A^2, C^2, and B^2 form an arithmetic sequence, prove that OP · PQ = -1 -/
theorem intersection_dot_product 
  (A B C : ℝ) 
  (P Q : ℝ × ℝ) 
  (h_line : ∀ x y, A * x + B * y + C = 0 ↔ (x, y) = P ∨ (x, y) = Q)
  (h_circle : P.1^2 + P.2^2 = 9 ∧ Q.1^2 + Q.2^2 = 9)
  (h_arithmetic : 2 * C^2 = A^2 + B^2)
  (h_distinct : P ≠ Q) :
  (P.1 * (Q.1 - P.1) + P.2 * (Q.2 - P.2) : ℝ) = -1 := by
  sorry

end intersection_dot_product_l1388_138883


namespace absolute_value_inequality_l1388_138830

theorem absolute_value_inequality (x : ℝ) : 
  ‖‖x - 2‖ - 1‖ ≤ 1 ↔ 0 ≤ x ∧ x ≤ 4 := by sorry

end absolute_value_inequality_l1388_138830


namespace square_diff_equality_l1388_138870

theorem square_diff_equality : 1003^2 - 997^2 - 1001^2 + 999^2 = 8000 := by
  sorry

end square_diff_equality_l1388_138870


namespace correct_answers_range_l1388_138809

/-- Represents the scoring system and conditions of the test --/
structure TestScoring where
  total_questions : Nat
  correct_points : Int
  wrong_points : Int
  min_score : Int

/-- Represents Xiaoyu's test result --/
structure TestResult (scoring : TestScoring) where
  correct_answers : Nat
  no_missed_questions : correct_answers ≤ scoring.total_questions

/-- Calculates the total score based on the number of correct answers --/
def calculate_score (scoring : TestScoring) (result : TestResult scoring) : Int :=
  result.correct_answers * scoring.correct_points + 
  (scoring.total_questions - result.correct_answers) * scoring.wrong_points

/-- Theorem stating the range of possible values for correct answers --/
theorem correct_answers_range (scoring : TestScoring) 
  (h_total : scoring.total_questions = 25)
  (h_correct : scoring.correct_points = 4)
  (h_wrong : scoring.wrong_points = -2)
  (h_min_score : scoring.min_score = 70) :
  ∀ (result : TestResult scoring), 
    calculate_score scoring result ≥ scoring.min_score →
    (20 : Nat) ≤ result.correct_answers ∧ result.correct_answers ≤ 25 := by
  sorry

end correct_answers_range_l1388_138809


namespace comic_book_stacks_result_l1388_138848

/-- The number of ways to stack comic books with given constraints -/
def comic_book_stacks (spiderman_count archie_count garfield_count : ℕ) : ℕ :=
  spiderman_count.factorial * archie_count.factorial * garfield_count.factorial * 2

/-- Theorem stating the number of ways to stack the comic books -/
theorem comic_book_stacks_result : comic_book_stacks 7 6 5 = 91612800 := by
  sorry

end comic_book_stacks_result_l1388_138848


namespace inequality_proof_l1388_138860

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + b + c ≥ a * b * c) : 
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 * a * b * c := by
  sorry

end inequality_proof_l1388_138860


namespace green_peaches_per_basket_l1388_138849

theorem green_peaches_per_basket (num_baskets : ℕ) (red_per_basket : ℕ) (total_peaches : ℕ) :
  num_baskets = 11 →
  red_per_basket = 10 →
  total_peaches = 308 →
  ∃ green_per_basket : ℕ, 
    green_per_basket * num_baskets + red_per_basket * num_baskets = total_peaches ∧
    green_per_basket = 18 := by
  sorry

end green_peaches_per_basket_l1388_138849


namespace speedster_convertibles_l1388_138828

theorem speedster_convertibles (total : ℕ) (speedsters : ℕ) (convertibles : ℕ) :
  speedsters = (2 : ℚ) / 3 * total →
  total - speedsters = 50 →
  convertibles = (4 : ℚ) / 5 * speedsters →
  convertibles = 80 :=
by
  sorry

end speedster_convertibles_l1388_138828


namespace arithmetic_mean_of_sequence_l1388_138869

theorem arithmetic_mean_of_sequence : 
  let start : ℕ := 3
  let count : ℕ := 60
  let sequence := fun (n : ℕ) => start + n - 1
  let sum := (count * (sequence 1 + sequence count)) / 2
  (sum : ℚ) / count = 32.5 := by
sorry

end arithmetic_mean_of_sequence_l1388_138869


namespace expression_evaluation_l1388_138832

theorem expression_evaluation :
  let x : ℚ := 1/25
  let y : ℚ := -25
  x * (x + 2*y) - (x + 1)^2 + 2*x = -3 := by sorry

end expression_evaluation_l1388_138832


namespace modulus_of_pure_imaginary_z_l1388_138831

/-- If z = (x^2 - 1) + (x - 1)i where x is a real number and z is a pure imaginary number, then |z| = 2 -/
theorem modulus_of_pure_imaginary_z (x : ℝ) (z : ℂ) 
  (h1 : z = Complex.mk (x^2 - 1) (x - 1))
  (h2 : z.re = 0) : Complex.abs z = 2 := by
  sorry

end modulus_of_pure_imaginary_z_l1388_138831


namespace inequality_proof_l1388_138853

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) :
  a + b + c ≤ (a^2 + b^2)/(2*c) + (b^2 + c^2)/(2*a) + (c^2 + a^2)/(2*b) ∧
  (a^2 + b^2)/(2*c) + (b^2 + c^2)/(2*a) + (c^2 + a^2)/(2*b) ≤ a^3/(b*c) + b^3/(c*a) + c^3/(a*b) :=
by
  sorry

end inequality_proof_l1388_138853


namespace car_travel_time_l1388_138894

theorem car_travel_time (actual_speed : ℝ) (actual_time : ℝ) 
  (h1 : actual_speed > 0) 
  (h2 : actual_time > 0) 
  (h3 : actual_speed * actual_time = (4/5 * actual_speed) * (actual_time + 15)) : 
  actual_time = 60 := by
sorry

end car_travel_time_l1388_138894


namespace negative_division_l1388_138822

theorem negative_division (a b : ℤ) (ha : a = -300) (hb : b = -50) :
  a / b = 6 := by
  sorry

end negative_division_l1388_138822


namespace f_derivative_at_negative_one_l1388_138807

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + 6

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

theorem f_derivative_at_negative_one (a b : ℝ) :
  f' a b 1 = 2 → f' a b (-1) = -2 := by
  sorry

end f_derivative_at_negative_one_l1388_138807


namespace infinitely_many_solutions_l1388_138800

theorem infinitely_many_solutions (b : ℝ) : 
  (∀ x : ℝ, 3 * (5 + b * x) = 18 * x + 15) → b = 6 := by
  sorry

end infinitely_many_solutions_l1388_138800


namespace unique_divisible_by_19_l1388_138842

/-- Converts a base 7 number of the form 52x3 to its decimal equivalent --/
def base7ToDecimal (x : ℕ) : ℕ := 5 * 7^3 + 2 * 7^2 + x * 7 + 3

/-- Checks if a number is a valid base 7 digit --/
def isBase7Digit (x : ℕ) : Prop := x ≤ 6

theorem unique_divisible_by_19 :
  ∃! x : ℕ, isBase7Digit x ∧ (base7ToDecimal x) % 19 = 0 :=
sorry

end unique_divisible_by_19_l1388_138842


namespace digit_product_of_24_l1388_138838

theorem digit_product_of_24 :
  ∀ x y : ℕ,
  x < 10 ∧ y < 10 →  -- Ensures x and y are single digits
  10 * x + y = 24 →  -- The number is 24
  10 * x + y + 18 = 10 * y + x →  -- When 18 is added, digits are reversed
  x * y = 8 :=  -- Product of digits is 8
by
  sorry

end digit_product_of_24_l1388_138838


namespace line_points_l1388_138898

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem line_points : 
  let p1 : Point := ⟨8, 16⟩
  let p2 : Point := ⟨0, -8⟩
  let p3 : Point := ⟨4, 4⟩
  let p4 : Point := ⟨2, 0⟩
  let p5 : Point := ⟨9, 19⟩
  let p6 : Point := ⟨-1, -9⟩
  let p7 : Point := ⟨-2, -10⟩
  collinear p1 p2 p3 ∧ 
  collinear p1 p2 p5 ∧ 
  ¬collinear p1 p2 p4 ∧ 
  ¬collinear p1 p2 p6 ∧ 
  ¬collinear p1 p2 p7 :=
by sorry

end line_points_l1388_138898


namespace g_is_even_f_periodic_4_l1388_138877

-- Define the real-valued function f
variable (f : ℝ → ℝ)

-- Define g in terms of f
def g (x : ℝ) : ℝ := f x + f (-x)

-- Theorem 1: g is an even function
theorem g_is_even : ∀ x : ℝ, g f x = g f (-x) := by sorry

-- Theorem 2: f is periodic with period 4 if it's odd and f(x+2) is odd
theorem f_periodic_4 (h1 : ∀ x : ℝ, f (-x) = -f x) 
                     (h2 : ∀ x : ℝ, f (-(x+2)) = -f (x+2)) : 
  ∀ x : ℝ, f (x + 4) = f x := by sorry

end g_is_even_f_periodic_4_l1388_138877


namespace replace_section_breaks_loop_l1388_138889

/-- Represents a railway section type -/
inductive SectionType
| Type1
| Type2

/-- Represents a railway configuration -/
structure RailwayConfig where
  type1Count : ℕ
  type2Count : ℕ

/-- Checks if a railway configuration forms a valid closed loop -/
def isValidClosedLoop (config : RailwayConfig) : Prop :=
  config.type1Count = config.type2Count

/-- Represents the operation of replacing a type 1 section with a type 2 section -/
def replaceSection (config : RailwayConfig) : RailwayConfig :=
  { type1Count := config.type1Count - 1,
    type2Count := config.type2Count + 1 }

/-- Main theorem: If a configuration forms a valid closed loop, 
    replacing a type 1 section with a type 2 section makes it impossible to form a closed loop -/
theorem replace_section_breaks_loop (config : RailwayConfig) :
  isValidClosedLoop config → ¬isValidClosedLoop (replaceSection config) := by
  sorry

end replace_section_breaks_loop_l1388_138889


namespace sunset_time_calculation_l1388_138801

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Adds two Times together -/
def addTime (t1 t2 : Time) : Time :=
  let totalMinutes := t1.hours * 60 + t1.minutes + t2.hours * 60 + t2.minutes
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

/-- Converts 24-hour format to 12-hour format -/
def to12HourFormat (t : Time) : Time :=
  if t.hours ≥ 12 then
    { hours := if t.hours = 12 then 12 else t.hours - 12, minutes := t.minutes }
  else
    { hours := if t.hours = 0 then 12 else t.hours, minutes := t.minutes }

theorem sunset_time_calculation (sunrise : Time) (daylight : Time) : 
  to12HourFormat (addTime sunrise daylight) = { hours := 7, minutes := 40 } :=
  sorry

end sunset_time_calculation_l1388_138801


namespace cosine_sine_identity_l1388_138821

theorem cosine_sine_identity : 
  Real.cos (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.sin (160 * π / 180) * Real.sin (10 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end cosine_sine_identity_l1388_138821


namespace expand_binomials_l1388_138814

theorem expand_binomials (x y : ℝ) : (x + 5) * (3 * y + 15) = 3 * x * y + 15 * x + 15 * y + 75 := by
  sorry

end expand_binomials_l1388_138814


namespace hyperbola_asymptote_slope_l1388_138892

theorem hyperbola_asymptote_slope (x y : ℝ) :
  (x^2 / 49 - y^2 / 36 = 4) →
  ∃ (m : ℝ), m > 0 ∧ (y = m * x ∨ y = -m * x) ∧ m = 6/7 :=
by sorry

end hyperbola_asymptote_slope_l1388_138892


namespace wendy_camera_pictures_l1388_138875

/-- Represents the number of pictures in Wendy's photo upload scenario -/
structure WendyPictures where
  phone : ℕ
  albums : ℕ
  per_album : ℕ

/-- The number of pictures Wendy uploaded from her camera -/
def camera_pictures (w : WendyPictures) : ℕ :=
  w.albums * w.per_album - w.phone

/-- Theorem stating the number of pictures Wendy uploaded from her camera -/
theorem wendy_camera_pictures :
  ∀ (w : WendyPictures),
    w.phone = 22 →
    w.albums = 4 →
    w.per_album = 6 →
    camera_pictures w = 2 := by
  sorry

end wendy_camera_pictures_l1388_138875


namespace square_intersection_inverse_squares_sum_l1388_138880

/-- Given a unit square ABCD and a point E on side CD, prove that if F is the intersection
    of line AE and BC, then 1/|AE|^2 + 1/|AF|^2 = 1. -/
theorem square_intersection_inverse_squares_sum (A B C D E F : ℝ × ℝ) : 
  -- Square ABCD has side length 1
  A = (0, 1) ∧ B = (1, 1) ∧ C = (1, 0) ∧ D = (0, 0) →
  -- E lies on CD
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ E = (x, 0) →
  -- F is the intersection of AE and BC
  F = (1, 0) →
  -- Then 1/|AE|^2 + 1/|AF|^2 = 1
  1 / (Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2))^2 + 
  1 / (Real.sqrt ((F.1 - A.1)^2 + (F.2 - A.2)^2))^2 = 1 := by
  sorry


end square_intersection_inverse_squares_sum_l1388_138880


namespace negative_sixty_four_to_four_thirds_l1388_138843

theorem negative_sixty_four_to_four_thirds (x : ℝ) : x = (-64)^(4/3) → x = 256 := by
  sorry

end negative_sixty_four_to_four_thirds_l1388_138843


namespace range_of_m_l1388_138817

/-- The function f(x) = -x^2 + 2x + 5 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 5

/-- Theorem stating the range of m given the conditions on f -/
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ≤ 6) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 6) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ 5) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 5) →
  m ∈ Set.Icc 1 2 :=
by sorry

end range_of_m_l1388_138817


namespace stamps_from_other_countries_l1388_138886

def total_stamps : ℕ := 500
def chinese_percent : ℚ := 40 / 100
def us_percent : ℚ := 25 / 100
def japanese_percent : ℚ := 15 / 100
def british_percent : ℚ := 10 / 100

theorem stamps_from_other_countries :
  total_stamps * (1 - (chinese_percent + us_percent + japanese_percent + british_percent)) = 50 := by
  sorry

end stamps_from_other_countries_l1388_138886


namespace sum_of_integers_l1388_138895

theorem sum_of_integers (x y z w : ℤ) 
  (eq1 : x - y + z = 7)
  (eq2 : y - z + w = 8)
  (eq3 : z - w + x = 4)
  (eq4 : w - x + y = 3) :
  x + y + z + w = 22 := by
sorry

end sum_of_integers_l1388_138895


namespace right_triangle_trig_l1388_138808

/-- Given a right-angled triangle ABC where ∠A = 90° and tan C = 2,
    prove that cos C = √5/5 and sin C = 2√5/5 -/
theorem right_triangle_trig (A B C : ℝ) (h1 : A = Real.pi / 2) (h2 : Real.tan C = 2) :
  Real.cos C = Real.sqrt 5 / 5 ∧ Real.sin C = 2 * Real.sqrt 5 / 5 := by
  sorry

end right_triangle_trig_l1388_138808


namespace fraction_simplification_l1388_138834

theorem fraction_simplification : (2468 * 2468) / (2468 + 2468) = 1234 := by
  sorry

end fraction_simplification_l1388_138834


namespace min_distance_point_to_tangent_l1388_138802

/-- The minimum distance between a point on the line x - y - 6 = 0 and 
    its tangent point on the circle (x-1)^2 + (y-1)^2 = 4 is √14 -/
theorem min_distance_point_to_tangent (x y : ℝ) : 
  let circle := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 4}
  let line := {p : ℝ × ℝ | p.1 - p.2 - 6 = 0}
  ∃ (M N : ℝ × ℝ), M ∈ line ∧ N ∈ circle ∧ 
    (∀ (M' N' : ℝ × ℝ), M' ∈ line → N' ∈ circle → 
      (M'.1 - N'.1)^2 + (M'.2 - N'.2)^2 ≥ (M.1 - N.1)^2 + (M.2 - N.2)^2) ∧
    (M.1 - N.1)^2 + (M.2 - N.2)^2 = 14 := by
  sorry

end min_distance_point_to_tangent_l1388_138802


namespace smallest_z_for_cube_equation_l1388_138806

theorem smallest_z_for_cube_equation : 
  (∃ (w x y z : ℕ), 
    w < x ∧ x < y ∧ y < z ∧
    w + 1 = x ∧ x + 1 = y ∧ y + 1 = z ∧
    w^3 + x^3 + y^3 = 2 * z^3) ∧
  (∀ (w x y z : ℕ),
    w < x → x < y → y < z →
    w + 1 = x → x + 1 = y → y + 1 = z →
    w^3 + x^3 + y^3 = 2 * z^3 →
    z ≥ 6) :=
by sorry

end smallest_z_for_cube_equation_l1388_138806


namespace ping_pong_tournament_l1388_138881

theorem ping_pong_tournament (n : ℕ) (k : ℕ) : 
  (∀ subset : Finset (Fin n), subset.card = n - 2 → Nat.choose subset.card 2 = 3^k) →
  n = 5 :=
sorry

end ping_pong_tournament_l1388_138881


namespace max_primes_in_table_l1388_138852

/-- A number in the table is either prime or the product of two primes -/
inductive TableNumber
  | prime : Nat → TableNumber
  | product : Nat → Nat → TableNumber

/-- Definition of the table -/
def Table := Fin 80 → Fin 80 → TableNumber

/-- Predicate to check if two TableNumbers are not coprime -/
def not_coprime : TableNumber → TableNumber → Prop :=
  sorry

/-- Predicate to check if all numbers in the table are distinct -/
def all_distinct (t : Table) : Prop :=
  sorry

/-- Predicate to check if for any number, there's another number in the same row or column that's not coprime -/
def has_not_coprime_neighbor (t : Table) : Prop :=
  sorry

/-- Count the number of prime numbers in the table -/
def count_primes (t : Table) : Nat :=
  sorry

/-- The main theorem -/
theorem max_primes_in_table :
  ∀ t : Table,
    all_distinct t →
    has_not_coprime_neighbor t →
    count_primes t ≤ 4266 :=
  sorry

end max_primes_in_table_l1388_138852


namespace expression_evaluation_l1388_138872

theorem expression_evaluation :
  36 + (150 / 15) + (12^2 * 5) - 300 - (270 / 9) = 436 := by
  sorry

end expression_evaluation_l1388_138872


namespace multiply_negative_with_absolute_value_l1388_138882

theorem multiply_negative_with_absolute_value : (-3.6 : ℝ) * |(-2 : ℝ)| = -7.2 := by
  sorry

end multiply_negative_with_absolute_value_l1388_138882


namespace lidia_remaining_money_l1388_138865

/-- Calculates the remaining money after Lidia's app purchase --/
def remaining_money (productivity_apps : ℕ) (productivity_cost : ℚ)
                    (gaming_apps : ℕ) (gaming_cost : ℚ)
                    (lifestyle_apps : ℕ) (lifestyle_cost : ℚ)
                    (initial_money : ℚ) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_cost := productivity_apps * productivity_cost +
                    gaming_apps * gaming_cost +
                    lifestyle_apps * lifestyle_cost
  let discounted_cost := total_cost * (1 - discount_rate)
  let final_cost := discounted_cost * (1 + tax_rate)
  initial_money - final_cost

/-- Theorem stating that Lidia will be left with $6.16 after her app purchase --/
theorem lidia_remaining_money :
  remaining_money 5 4 7 5 3 3 66 (15/100) (10/100) = (616/100) :=
sorry


end lidia_remaining_money_l1388_138865


namespace triangle_third_side_l1388_138890

/-- Given a triangle with sides b, c, and x, where the area S = 0.4bc, 
    prove that the third side x satisfies the equation: x² = b² + c² ± 1.2bc -/
theorem triangle_third_side (b c x : ℝ) (h : b > 0 ∧ c > 0 ∧ x > 0) :
  (0.4 * b * c)^2 = (1/16) * (4 * b^2 * c^2 - (b^2 + c^2 - x^2)^2) →
  x^2 = b^2 + c^2 + 1.2 * b * c ∨ x^2 = b^2 + c^2 - 1.2 * b * c :=
by sorry

end triangle_third_side_l1388_138890


namespace triangle_is_isosceles_triangle_area_l1388_138888

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def isIsosceles (t : Triangle) : Prop :=
  t.b * Real.cos t.C = t.a * (Real.cos t.B)^2 + t.b * Real.cos t.A * Real.cos t.B

def hasSpecificProperties (t : Triangle) : Prop :=
  isIsosceles t ∧ Real.cos t.A = 7/8 ∧ t.a + t.b + t.c = 5

-- State the theorems
theorem triangle_is_isosceles (t : Triangle) (h : isIsosceles t) : 
  t.B = t.C := by sorry

theorem triangle_area (t : Triangle) (h : hasSpecificProperties t) :
  (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 15 / 4 := by sorry

end triangle_is_isosceles_triangle_area_l1388_138888


namespace min_occupied_seats_for_150_l1388_138813

/-- Given a row of seats, returns the minimum number of occupied seats required
    to ensure that the next person must sit next to someone -/
def min_occupied_seats (total_seats : ℕ) : ℕ :=
  (total_seats - 2) / 4 + 1

theorem min_occupied_seats_for_150 :
  min_occupied_seats 150 = 37 := by
  sorry

end min_occupied_seats_for_150_l1388_138813


namespace manuscript_typing_cost_l1388_138839

theorem manuscript_typing_cost : 
  let total_pages : ℕ := 100
  let pages_revised_once : ℕ := 30
  let pages_revised_twice : ℕ := 20
  let pages_not_revised : ℕ := total_pages - pages_revised_once - pages_revised_twice
  let initial_typing_cost : ℕ := 5
  let revision_cost : ℕ := 3
  let total_cost : ℕ := 
    total_pages * initial_typing_cost + 
    pages_revised_once * revision_cost + 
    pages_revised_twice * revision_cost * 2
  total_cost = 710 := by
sorry

end manuscript_typing_cost_l1388_138839


namespace cube_volume_ratio_l1388_138818

theorem cube_volume_ratio : 
  let cube_volume (edge : ℚ) : ℚ := edge ^ 3
  let cube1_edge : ℚ := 4
  let cube2_edge : ℚ := 10
  (cube_volume cube1_edge) / (cube_volume cube2_edge) = 8 / 125 := by
sorry

end cube_volume_ratio_l1388_138818


namespace linear_function_proof_l1388_138815

def is_linear (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x, f x = a * x + b

theorem linear_function_proof (f : ℝ → ℝ) 
  (h_linear : is_linear f) 
  (h_composite : ∀ x, f (f x) = 4 * x - 1) 
  (h_specific : f 3 = -5) : 
  ∀ x, f x = -2 * x + 1 := by
sorry

end linear_function_proof_l1388_138815


namespace age_sum_from_product_l1388_138836

theorem age_sum_from_product (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 144 → a + b + c = 16 := by
  sorry

end age_sum_from_product_l1388_138836


namespace a_10_equals_1000_l1388_138861

def a (n : ℕ) : ℕ :=
  let first_odd := 2 * n - 1
  let last_odd := first_odd + 2 * (n - 1)
  n * (first_odd + last_odd) / 2

theorem a_10_equals_1000 : a 10 = 1000 := by
  sorry

end a_10_equals_1000_l1388_138861


namespace carrot_sticks_total_l1388_138891

theorem carrot_sticks_total (before_dinner after_dinner total : ℕ) 
  (h1 : before_dinner = 22)
  (h2 : after_dinner = 15)
  (h3 : total = before_dinner + after_dinner) :
  total = 37 := by sorry

end carrot_sticks_total_l1388_138891


namespace bobby_total_pieces_l1388_138820

def total_pieces_eaten (initial_candy : ℕ) (initial_chocolate : ℕ) (initial_licorice : ℕ) 
                       (additional_candy : ℕ) (additional_chocolate : ℕ) : ℕ :=
  (initial_candy + additional_candy) + (initial_chocolate + additional_chocolate) + initial_licorice

theorem bobby_total_pieces : 
  total_pieces_eaten 33 14 7 4 5 = 63 := by
  sorry

end bobby_total_pieces_l1388_138820


namespace cube_root_unity_sum_l1388_138899

theorem cube_root_unity_sum (ω : ℂ) : 
  ω^3 = 1 → ((-1 + Complex.I * Real.sqrt 3) / 2)^8 + ((-1 - Complex.I * Real.sqrt 3) / 2)^8 = -1 := by
  sorry

end cube_root_unity_sum_l1388_138899


namespace A_equiv_B_l1388_138874

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 = 1 ∧ 2 * p.1 - p.2 = 2}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 1 ∧ p.2 = 0}

-- Theorem stating that A and B are equivalent
theorem A_equiv_B : A = B := by sorry

end A_equiv_B_l1388_138874


namespace hanging_spheres_mass_ratio_l1388_138835

/-- Given two hanging spheres with masses m₁ and m₂, where the tension in the upper thread (T_B)
    is three times the tension in the lower thread (T_H), prove that the ratio m₁/m₂ = 2. -/
theorem hanging_spheres_mass_ratio
  (m₁ m₂ : ℝ) -- masses of the spheres
  (T_B T_H : ℝ) -- tensions in the upper and lower threads
  (h1 : T_B = 3 * T_H) -- condition: upper tension is 3 times lower tension
  (h2 : T_H = m₂ * (9.8 : ℝ)) -- force balance for bottom sphere
  (h3 : T_B = m₁ * (9.8 : ℝ) + T_H) -- force balance for top sphere
  : m₁ / m₂ = 2 := by
  sorry

end hanging_spheres_mass_ratio_l1388_138835


namespace balls_distribution_proof_l1388_138845

def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

theorem balls_distribution_proof :
  distribute_balls 10 4 = 84 := by
  sorry

end balls_distribution_proof_l1388_138845


namespace commute_time_sum_of_squares_l1388_138826

theorem commute_time_sum_of_squares 
  (x y : ℝ) 
  (avg_eq : (x + y + 10 + 11 + 9) / 5 = 10) 
  (var_eq : ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2) : 
  x^2 + y^2 = 208 := by
sorry

end commute_time_sum_of_squares_l1388_138826


namespace triangle_sides_l1388_138866

theorem triangle_sides (a b c : ℚ) : 
  a + b + c = 24 →
  a + 2*b = 2*c →
  a = (1/2) * b →
  a = 16/3 ∧ b = 32/3 ∧ c = 8 := by
sorry

end triangle_sides_l1388_138866


namespace excircle_incircle_relation_l1388_138829

/-- Given a triangle ABC with inscribed circle radius r, excircle radii r_a, r_b, r_c,
    and semiperimeter p, prove that (r_a * r_b * r_c) / r = p^2 -/
theorem excircle_incircle_relation (r r_a r_b r_c p : ℝ) : r > 0 → r_a > 0 → r_b > 0 → r_c > 0 → p > 0 →
  (r_a * r_b * r_c) / r = p^2 := by sorry

end excircle_incircle_relation_l1388_138829


namespace sqrt_expression_equals_two_l1388_138840

theorem sqrt_expression_equals_two :
  Real.sqrt 72 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 - abs (2 - Real.sqrt 6) = 2 := by
  sorry

end sqrt_expression_equals_two_l1388_138840


namespace johns_money_left_l1388_138879

theorem johns_money_left (initial_amount : ℚ) (snack_fraction : ℚ) (necessity_fraction : ℚ) : 
  initial_amount = 20 →
  snack_fraction = 1/5 →
  necessity_fraction = 3/4 →
  let remaining_after_snacks := initial_amount - (initial_amount * snack_fraction)
  let final_amount := remaining_after_snacks - (remaining_after_snacks * necessity_fraction)
  final_amount = 4 := by
  sorry

end johns_money_left_l1388_138879


namespace unique_solution_quadratic_l1388_138804

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, 3 * x^2 + m * x + 36 = 0) ↔ m = 12 * Real.sqrt 3 :=
sorry

end unique_solution_quadratic_l1388_138804


namespace intersection_point_correct_l1388_138803

/-- Represents a 2D vector --/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Represents a parametric line in 2D --/
structure ParametricLine where
  origin : Vector2D
  direction : Vector2D

/-- The first line --/
def line1 : ParametricLine :=
  { origin := { x := 1, y := 2 },
    direction := { x := -2, y := 4 } }

/-- The second line --/
def line2 : ParametricLine :=
  { origin := { x := 3, y := 5 },
    direction := { x := 1, y := 3 } }

/-- Calculates a point on a parametric line given a parameter t --/
def pointOnLine (line : ParametricLine) (t : ℝ) : Vector2D :=
  { x := line.origin.x + t * line.direction.x,
    y := line.origin.y + t * line.direction.y }

/-- The intersection point of the two lines --/
def intersectionPoint : Vector2D :=
  { x := 1.2, y := 1.6 }

/-- Theorem stating that the calculated intersection point is correct --/
theorem intersection_point_correct :
  ∃ t u : ℝ, pointOnLine line1 t = intersectionPoint ∧ pointOnLine line2 u = intersectionPoint :=
sorry


end intersection_point_correct_l1388_138803


namespace right_triangle_leg_sum_range_l1388_138885

theorem right_triangle_leg_sum_range (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  x^2 + y^2 = 5 → Real.sqrt 5 < x + y ∧ x + y ≤ Real.sqrt 10 := by
  sorry

end right_triangle_leg_sum_range_l1388_138885


namespace expand_product_l1388_138805

theorem expand_product (x : ℝ) : (5*x + 3) * (2*x^2 + 4) = 10*x^3 + 6*x^2 + 20*x + 12 := by
  sorry

end expand_product_l1388_138805


namespace elise_initial_dog_food_l1388_138825

/-- The amount of dog food Elise already had -/
def initial_amount : ℕ := sorry

/-- The amount of dog food in the first bag Elise bought -/
def first_bag : ℕ := 15

/-- The amount of dog food in the second bag Elise bought -/
def second_bag : ℕ := 10

/-- The total amount of dog food Elise has after buying -/
def total_amount : ℕ := 40

theorem elise_initial_dog_food : initial_amount = 15 :=
  sorry

end elise_initial_dog_food_l1388_138825


namespace book_reading_ratio_l1388_138811

/-- Given the number of books read by Candice, Amanda, Kara, and Patricia in a Book Tournament, 
    prove the ratio of books read by Kara to Amanda. -/
theorem book_reading_ratio 
  (candice amanda kara patricia : ℕ) 
  (x : ℚ) 
  (h1 : candice = 3 * amanda) 
  (h2 : candice = 18) 
  (h3 : kara = x * amanda) 
  (h4 : patricia = 7 * kara) : 
  (kara : ℚ) / amanda = x := by
  sorry

end book_reading_ratio_l1388_138811


namespace money_left_after_trip_l1388_138864

def initial_savings : ℕ := 6000
def flight_cost : ℕ := 1200
def hotel_cost : ℕ := 800
def food_cost : ℕ := 3000

theorem money_left_after_trip :
  initial_savings - (flight_cost + hotel_cost + food_cost) = 1000 := by
  sorry

end money_left_after_trip_l1388_138864


namespace tailor_buttons_count_l1388_138884

/-- The number of buttons purchased by a tailor -/
theorem tailor_buttons_count : 
  let green : ℕ := 90
  let yellow : ℕ := green + 10
  let blue : ℕ := green - 5
  let red : ℕ := 2 * (yellow + blue)
  green + yellow + blue + red = 645 := by sorry

end tailor_buttons_count_l1388_138884


namespace positive_expression_l1388_138854

theorem positive_expression (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) :
  5 * a^2 - 6 * a * b + 5 * b^2 > 0 := by
  sorry

end positive_expression_l1388_138854


namespace overlapping_squares_diagonal_l1388_138893

theorem overlapping_squares_diagonal (small_side large_side : ℝ) 
  (h1 : small_side = 1) 
  (h2 : large_side = 7) : 
  Real.sqrt ((small_side + large_side)^2 + (large_side - small_side)^2) = 10 := by
  sorry

end overlapping_squares_diagonal_l1388_138893


namespace martha_initial_pantry_bottles_l1388_138878

/-- The number of bottles of juice Martha initially had in the pantry -/
def initial_pantry_bottles : ℕ := sorry

/-- The number of bottles of juice Martha initially had in the refrigerator -/
def initial_fridge_bottles : ℕ := 4

/-- The number of bottles of juice Martha bought during the week -/
def bought_bottles : ℕ := 5

/-- The number of bottles of juice Martha and her family drank during the week -/
def drunk_bottles : ℕ := 3

/-- The number of bottles of juice left at the end of the week -/
def remaining_bottles : ℕ := 10

theorem martha_initial_pantry_bottles :
  initial_pantry_bottles = 4 :=
by sorry

end martha_initial_pantry_bottles_l1388_138878


namespace quadratic_rewrite_l1388_138833

theorem quadratic_rewrite :
  ∃ (p q r : ℤ), 
    (∀ x, 8 * x^2 - 24 * x - 56 = (p * x + q)^2 + r) ∧
    p * q = -12 := by
  sorry

end quadratic_rewrite_l1388_138833


namespace cube_equals_self_mod_thousand_l1388_138827

theorem cube_equals_self_mod_thousand (n : ℕ) :
  100 ≤ n ∧ n ≤ 999 →
  (n^3 % 1000 = n) ↔ (n = 376 ∨ n = 625) := by
  sorry

end cube_equals_self_mod_thousand_l1388_138827


namespace weekly_average_expenditure_l1388_138824

/-- The average expenditure for a week given the average expenditures for two parts of the week -/
theorem weekly_average_expenditure 
  (first_three_days_avg : ℝ) 
  (next_four_days_avg : ℝ) 
  (h1 : first_three_days_avg = 350)
  (h2 : next_four_days_avg = 420) :
  (3 * first_three_days_avg + 4 * next_four_days_avg) / 7 = 390 := by
sorry

end weekly_average_expenditure_l1388_138824


namespace camel_traveler_water_ratio_l1388_138841

/-- The amount of water the traveler drank in ounces -/
def traveler_amount : ℕ := 32

/-- The number of ounces in a gallon -/
def ounces_per_gallon : ℕ := 128

/-- The total number of gallons they drank -/
def total_gallons : ℕ := 2

/-- The ratio of the amount of water the camel drank to the amount the traveler drank -/
def camel_to_traveler_ratio : ℚ := 7

theorem camel_traveler_water_ratio :
  (total_gallons * ounces_per_gallon - traveler_amount) / traveler_amount = camel_to_traveler_ratio :=
by sorry

end camel_traveler_water_ratio_l1388_138841


namespace salary_sum_l1388_138871

theorem salary_sum (average_salary : ℕ) (num_people : ℕ) (known_salary : ℕ) :
  average_salary = 9000 →
  num_people = 5 →
  known_salary = 9000 →
  (num_people * average_salary) - known_salary = 36000 := by
  sorry

end salary_sum_l1388_138871


namespace second_white_given_first_white_probability_l1388_138847

/-- Represents the color of a ball -/
inductive Color
| White
| Black

/-- Represents the pocket containing balls -/
structure Pocket where
  white : Nat
  black : Nat

/-- Represents the result of two consecutive draws -/
structure TwoDraws where
  first : Color
  second : Color

/-- Calculates the probability of drawing a white ball on the second draw
    given that the first ball drawn is white -/
def probSecondWhiteGivenFirstWhite (p : Pocket) : Rat :=
  if p.white > 0 then
    (p.white - 1) / (p.white + p.black - 1)
  else
    0

theorem second_white_given_first_white_probability 
  (p : Pocket) (h1 : p.white = 3) (h2 : p.black = 2) :
  probSecondWhiteGivenFirstWhite p = 1/2 := by
  sorry

end second_white_given_first_white_probability_l1388_138847


namespace internet_fee_calculation_l1388_138851

/-- The fixed monthly fee for Anna's internet service -/
def fixed_fee : ℝ := sorry

/-- The variable fee per hour of usage for Anna's internet service -/
def variable_fee : ℝ := sorry

/-- Anna's internet usage in November (in hours) -/
def november_usage : ℝ := sorry

/-- Anna's bill for November -/
def november_bill : ℝ := 20.60

/-- Anna's bill for December -/
def december_bill : ℝ := 33.20

theorem internet_fee_calculation :
  (fixed_fee + variable_fee * november_usage = november_bill) ∧
  (fixed_fee + variable_fee * (3 * november_usage) = december_bill) →
  fixed_fee = 14.30 := by
sorry

end internet_fee_calculation_l1388_138851


namespace envelope_counting_time_l1388_138855

/-- Represents the time in seconds to count a given number of envelopes -/
def count_time (envelopes : ℕ) : ℕ :=
  10 * ((100 - envelopes) / 10)

theorem envelope_counting_time :
  (count_time 60 = 40) ∧ (count_time 90 = 10) :=
sorry

end envelope_counting_time_l1388_138855


namespace jerry_tickets_l1388_138873

def ticket_calculation (initial_tickets spent_tickets additional_tickets : ℕ) : ℕ :=
  initial_tickets - spent_tickets + additional_tickets

theorem jerry_tickets :
  ticket_calculation 4 2 47 = 49 := by
  sorry

end jerry_tickets_l1388_138873


namespace sum_2018_is_1009_l1388_138810

/-- An arithmetic sequence with first term 1 and common difference -1/2017 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  1 - (n - 1 : ℚ) / 2017

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℚ :=
  n * (arithmetic_sequence 1 + arithmetic_sequence n) / 2

/-- Theorem stating that the sum of the first 2018 terms is 1009 -/
theorem sum_2018_is_1009 : S 2018 = 1009 := by
  sorry

end sum_2018_is_1009_l1388_138810


namespace coffee_order_total_cost_l1388_138857

def drip_coffee_price : ℝ := 2.25
def drip_coffee_quantity : ℕ := 2

def espresso_price : ℝ := 3.50
def espresso_quantity : ℕ := 1

def latte_price : ℝ := 4.00
def latte_quantity : ℕ := 2

def vanilla_syrup_price : ℝ := 0.50
def vanilla_syrup_quantity : ℕ := 1

def cold_brew_price : ℝ := 2.50
def cold_brew_quantity : ℕ := 2

def cappuccino_price : ℝ := 3.50
def cappuccino_quantity : ℕ := 1

theorem coffee_order_total_cost :
  drip_coffee_price * drip_coffee_quantity +
  espresso_price * espresso_quantity +
  latte_price * latte_quantity +
  vanilla_syrup_price * vanilla_syrup_quantity +
  cold_brew_price * cold_brew_quantity +
  cappuccino_price * cappuccino_quantity = 25.00 := by
  sorry

end coffee_order_total_cost_l1388_138857


namespace borrowed_sheets_theorem_l1388_138812

/-- Represents a set of algebra notes -/
structure AlgebraNotes where
  total_pages : ℕ
  total_sheets : ℕ
  borrowed_sheets : ℕ

/-- Calculates the average page number of remaining sheets -/
def average_page_number (notes : AlgebraNotes) : ℚ :=
  let remaining_sheets := notes.total_sheets - notes.borrowed_sheets
  let sum_of_remaining_pages := (notes.total_pages * (notes.total_pages + 1)) / 2 -
    (notes.borrowed_sheets * 2 * (notes.borrowed_sheets * 2 + 1)) / 2
  sum_of_remaining_pages / (2 * remaining_sheets)

/-- Main theorem: The average page number of remaining sheets is 31 when 20 sheets are borrowed -/
theorem borrowed_sheets_theorem (notes : AlgebraNotes)
  (h1 : notes.total_pages = 80)
  (h2 : notes.total_sheets = 40)
  (h3 : notes.borrowed_sheets = 20) :
  average_page_number notes = 31 := by
  sorry

end borrowed_sheets_theorem_l1388_138812


namespace line_parallel_plane_l1388_138897

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (not_subset : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_plane 
  (a b : Line) (α : Plane)
  (h1 : parallel_line a b)
  (h2 : parallel_line_plane b α)
  (h3 : not_subset a α) :
  parallel_line_plane a α :=
sorry

end line_parallel_plane_l1388_138897


namespace part_1_part_2_l1388_138823

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x + 1|

-- Define the function g
def g (x : ℝ) : ℝ := f 2 x - |x + 1|

-- Theorem for part 1
theorem part_1 : ∃ (a : ℝ), (∀ x, f a x ≤ 3 ↔ -2 ≤ x ∧ x ≤ 1) ∧ a = 2 := by sorry

-- Theorem for part 2
theorem part_2 : ∃ (min_value : ℝ), (∀ x, g x ≥ min_value) ∧ min_value = -1/2 := by sorry

end part_1_part_2_l1388_138823


namespace complement_S_union_T_l1388_138850

def S : Set ℝ := {x | x > -2}
def T : Set ℝ := {x | x^2 + 3*x - 4 ≤ 0}

theorem complement_S_union_T : (Set.univ \ S) ∪ T = {x : ℝ | x ≤ 1} := by sorry

end complement_S_union_T_l1388_138850


namespace largest_area_triangle_l1388_138844

/-- Represents a triangle in 2D space -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Checks if a point is an internal point of a line segment -/
def isInternalPoint (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop := sorry

/-- Checks if two triangles are similar -/
def areSimilar (T1 T2 : Triangle) : Prop := sorry

/-- Calculates the area of a triangle -/
def triangleArea (T : Triangle) : ℝ := sorry

/-- Represents a circle in 2D space -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Represents an arc of a circle -/
structure Arc :=
  (circle : Circle)
  (startAngle endAngle : ℝ)

/-- Finds the intersection point of two circles -/
def circleIntersection (c1 c2 : Circle) : Option (ℝ × ℝ) := sorry

/-- Calculates the distance between two points -/
def distance (P1 P2 : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem largest_area_triangle
  (A₀B₀C₀ : Triangle)
  (A'B'C' : Triangle)
  (k_a k_c : Circle)
  (i_a i_c : Arc) :
  ∀ (ABC : Triangle),
    isInternalPoint A₀B₀C₀.C ABC.A ABC.B →
    isInternalPoint A₀B₀C₀.A ABC.B ABC.C →
    isInternalPoint A₀B₀C₀.B ABC.C ABC.A →
    areSimilar ABC A'B'C' →
    (∃ (M : ℝ × ℝ), circleIntersection k_a k_c = some M) →
    (∀ (ABC' : Triangle),
      isInternalPoint A₀B₀C₀.C ABC'.A ABC'.B →
      isInternalPoint A₀B₀C₀.A ABC'.B ABC'.C →
      isInternalPoint A₀B₀C₀.B ABC'.C ABC'.A →
      areSimilar ABC' A'B'C' →
      (∃ (M' : ℝ × ℝ), circleIntersection k_a k_c = some M') →
      distance M ABC.C + distance M ABC.A ≥ distance M' ABC'.C + distance M' ABC'.A) →
    ∀ (ABC' : Triangle),
      isInternalPoint A₀B₀C₀.C ABC'.A ABC'.B →
      isInternalPoint A₀B₀C₀.A ABC'.B ABC'.C →
      isInternalPoint A₀B₀C₀.B ABC'.C ABC'.A →
      areSimilar ABC' A'B'C' →
      triangleArea ABC ≥ triangleArea ABC' :=
by
  sorry

end largest_area_triangle_l1388_138844


namespace shop_monthly_rent_l1388_138819

/-- The monthly rent of a rectangular shop given its dimensions and annual rent per square foot -/
def monthly_rent (length width annual_rent_per_sqft : ℕ) : ℕ :=
  length * width * annual_rent_per_sqft / 12

/-- Proof that the monthly rent of a shop with given dimensions is 3600 -/
theorem shop_monthly_rent :
  monthly_rent 18 20 120 = 3600 := by
  sorry

end shop_monthly_rent_l1388_138819


namespace max_points_in_plane_max_points_in_space_l1388_138876

/-- A point in a Euclidean space -/
structure Point (n : Nat) where
  coords : Fin n → ℝ

/-- Checks if three points form an obtuse angle -/
def is_obtuse_angle (n : Nat) (p1 p2 p3 : Point n) : Prop :=
  sorry -- Definition of obtuse angle check

/-- A configuration of points in a Euclidean space -/
structure PointConfiguration (n : Nat) where
  dim : Nat -- dimension of the space (2 for plane, 3 for space)
  points : Fin n → Point dim
  no_obtuse_angles : ∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k → 
    ¬ is_obtuse_angle dim (points i) (points j) (points k)

/-- The maximum number of points in a plane configuration without obtuse angles -/
theorem max_points_in_plane :
  (∃ (c : PointConfiguration 4), c.dim = 2) ∧
  (∀ (n : Nat), n > 4 → ¬ ∃ (c : PointConfiguration n), c.dim = 2) :=
sorry

/-- The maximum number of points in a space configuration without obtuse angles -/
theorem max_points_in_space :
  (∃ (c : PointConfiguration 8), c.dim = 3) ∧
  (∀ (n : Nat), n > 8 → ¬ ∃ (c : PointConfiguration n), c.dim = 3) :=
sorry

end max_points_in_plane_max_points_in_space_l1388_138876


namespace tangent_line_max_a_l1388_138816

/-- Given a real number a, if there exists a common tangent line to the curves y = x^2 and y = a ln x for x > 0, then a ≤ 2e -/
theorem tangent_line_max_a (a : ℝ) : 
  (∃ (x : ℝ), x > 0 ∧ 
    (∃ (m : ℝ), (2 * x = a / x) ∧ 
      (x^2 = a * Real.log x + m))) → 
  a ≤ 2 * Real.exp 1 :=
sorry

end tangent_line_max_a_l1388_138816


namespace min_value_of_a_is_two_l1388_138887

/-- Given an equation with parameter a and two real solutions, 
    prove that the minimum value of a is 2 -/
theorem min_value_of_a_is_two (a : ℝ) (x₁ x₂ : ℝ) : 
  (9 * x₁ - (4 + a) * 3 * x₁ + 4 = 0) ∧ 
  (9 * x₂ - (4 + a) * 3 * x₂ + 4 = 0) ∧ 
  (x₁ ≠ x₂) →
  ∀ b : ℝ, (∃ y₁ y₂ : ℝ, (9 * y₁ - (4 + b) * 3 * y₁ + 4 = 0) ∧ 
                         (9 * y₂ - (4 + b) * 3 * y₂ + 4 = 0) ∧ 
                         (y₁ ≠ y₂)) →
  b ≥ 2 := by
  sorry

end min_value_of_a_is_two_l1388_138887


namespace cubic_inequality_false_l1388_138858

theorem cubic_inequality_false (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  ¬(a^3 > b^3) := by
sorry

end cubic_inequality_false_l1388_138858


namespace car_down_payment_sharing_l1388_138846

def down_payment : ℕ := 3500
def individual_payment : ℕ := 1167

theorem car_down_payment_sharing :
  (down_payment + 2) / individual_payment = 3 :=
sorry

end car_down_payment_sharing_l1388_138846


namespace jen_age_proof_l1388_138868

/-- Jen's age when her son was born -/
def jen_age_at_birth : ℕ := 25

/-- Jen's son's current age -/
def son_current_age : ℕ := 16

/-- Jen's current age -/
def jen_current_age : ℕ := 3 * son_current_age - 7

theorem jen_age_proof : jen_current_age = 41 := by
  sorry

end jen_age_proof_l1388_138868
