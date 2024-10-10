import Mathlib

namespace smallest_multiple_of_all_up_to_ten_l922_92252

def is_multiple_of_all (n : ℕ) : Prop :=
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → x ∣ n

theorem smallest_multiple_of_all_up_to_ten :
  ∃ n : ℕ, is_multiple_of_all n ∧ ∀ m : ℕ, is_multiple_of_all m → n ≤ m :=
sorry

end smallest_multiple_of_all_up_to_ten_l922_92252


namespace ellipse_intersection_property_l922_92201

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 / 4 = 1

/-- Definition of the line l passing through P(x₁, y₁) -/
def line_l (x₁ y₁ x y : ℝ) : Prop :=
  4 * x₁ * x + 9 * y₁ * y = 36

/-- Theorem statement -/
theorem ellipse_intersection_property :
  ∀ (x₁ y₁ : ℝ),
  is_on_ellipse x₁ y₁ →
  ∃ (M_x M_y M'_x M'_y : ℝ),
    line_l x₁ y₁ M_x M_y ∧
    line_l x₁ y₁ M'_x M'_y ∧
    M_x = 3 ∧
    M'_x = -3 ∧
    (M_y^2 + 9) * (M'_y^2 + 9) = 36 ∧
    ∀ (N_x N_y N'_x N'_y : ℝ),
      line_l x₁ y₁ N_x N_y →
      line_l x₁ y₁ N'_x N'_y →
      N_x = 3 →
      N'_x = -3 →
      6 * (|N_y| + |N'_y|) ≥ 72 :=
sorry

end ellipse_intersection_property_l922_92201


namespace cylinder_section_area_l922_92277

/-- Represents a cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Represents a plane passing through two points on the top rim of a cylinder and its axis -/
structure CuttingPlane where
  cylinder : Cylinder
  arcAngle : ℝ  -- Angle of the arc PQ in radians

/-- Area of the new section formed when a plane cuts the cylinder -/
def newSectionArea (plane : CuttingPlane) : ℝ := sorry

theorem cylinder_section_area
  (c : Cylinder)
  (p : CuttingPlane)
  (h1 : c.radius = 5)
  (h2 : c.height = 10)
  (h3 : p.cylinder = c)
  (h4 : p.arcAngle = 5 * π / 6)  -- 150° in radians
  : newSectionArea p = 48 * π :=
sorry

end cylinder_section_area_l922_92277


namespace smallest_possible_d_l922_92279

theorem smallest_possible_d : ∃ d : ℝ, d > 0 ∧ 
  (5 * Real.sqrt 2)^2 + (d + 4)^2 = (4 * d)^2 ∧ 
  ∀ d' : ℝ, d' > 0 → (5 * Real.sqrt 2)^2 + (d' + 4)^2 = (4 * d')^2 → d ≤ d' :=
by sorry

end smallest_possible_d_l922_92279


namespace distribute_balls_theorem_l922_92211

/-- The number of ways to distribute 4 different balls into 3 labeled boxes, with no box left empty -/
def distributeWays : ℕ := 36

/-- The number of ways to choose 2 balls from 4 different balls -/
def chooseTwo : ℕ := 6

/-- The number of ways to permute 3 groups -/
def permuteThree : ℕ := 6

theorem distribute_balls_theorem :
  distributeWays = chooseTwo * permuteThree := by sorry

end distribute_balls_theorem_l922_92211


namespace right_triangle_hypotenuse_l922_92293

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 40 →
  (1/2) * a * b = 30 →
  a^2 + b^2 = c^2 →
  c = 18.5 := by
sorry

end right_triangle_hypotenuse_l922_92293


namespace valid_field_area_is_189_l922_92264

/-- Represents a rectangular sports field with posts -/
structure SportsField where
  total_posts : ℕ
  post_distance : ℕ
  long_side_posts : ℕ
  short_side_posts : ℕ

/-- Checks if the field configuration is valid according to the problem conditions -/
def is_valid_field (field : SportsField) : Prop :=
  field.total_posts = 24 ∧
  field.post_distance = 3 ∧
  field.long_side_posts = 2 * field.short_side_posts ∧
  2 * (field.long_side_posts + field.short_side_posts - 2) = field.total_posts

/-- Calculates the area of the field given its configuration -/
def field_area (field : SportsField) : ℕ :=
  (field.short_side_posts - 1) * field.post_distance * 
  (field.long_side_posts - 1) * field.post_distance

/-- Theorem stating that a valid field configuration results in an area of 189 square yards -/
theorem valid_field_area_is_189 (field : SportsField) :
  is_valid_field field → field_area field = 189 := by
  sorry

#check valid_field_area_is_189

end valid_field_area_is_189_l922_92264


namespace fraction_product_theorem_l922_92289

theorem fraction_product_theorem :
  (7 / 5 : ℚ) * (8 / 12 : ℚ) * (21 / 15 : ℚ) * (16 / 24 : ℚ) * 
  (35 / 25 : ℚ) * (20 / 30 : ℚ) * (49 / 35 : ℚ) * (32 / 48 : ℚ) = 38416 / 50625 := by
  sorry

end fraction_product_theorem_l922_92289


namespace max_peak_consumption_theorem_l922_92275

/-- Represents the electricity pricing and consumption parameters for a household. -/
structure ElectricityParams where
  originalPrice : ℝ
  peakPrice : ℝ
  offPeakPrice : ℝ
  totalConsumption : ℝ
  savingsPercentage : ℝ

/-- Calculates the maximum peak hour consumption given electricity parameters. -/
def maxPeakConsumption (params : ElectricityParams) : ℝ := by
  sorry

/-- Theorem stating the maximum peak hour consumption for the given scenario. -/
theorem max_peak_consumption_theorem (params : ElectricityParams) 
  (h1 : params.originalPrice = 0.52)
  (h2 : params.peakPrice = 0.55)
  (h3 : params.offPeakPrice = 0.35)
  (h4 : params.totalConsumption = 200)
  (h5 : params.savingsPercentage = 0.1) :
  maxPeakConsumption params = 118 := by
  sorry

end max_peak_consumption_theorem_l922_92275


namespace ceiling_squared_negative_fraction_l922_92271

theorem ceiling_squared_negative_fraction : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end ceiling_squared_negative_fraction_l922_92271


namespace sum_of_m_and_n_l922_92268

theorem sum_of_m_and_n (m n : ℕ) (hm : m > 1) (hn : n > 1) 
  (h : 2005^2 + m^2 = 2004^2 + n^2) : m + n = 211 := by
  sorry

end sum_of_m_and_n_l922_92268


namespace first_four_eq_last_four_l922_92248

/-- A finite sequence of 0s and 1s with special properties -/
def SpecialSequence : Type :=
  {s : List Bool // 
    (∀ i j, i ≠ j → i + 5 ≤ s.length → j + 5 ≤ s.length → 
      (List.take 5 (List.drop i s) ≠ List.take 5 (List.drop j s))) ∧
    (¬∀ i j, i ≠ j → i + 5 ≤ (s ++ [true]).length → j + 5 ≤ (s ++ [true]).length → 
      (List.take 5 (List.drop i (s ++ [true])) ≠ List.take 5 (List.drop j (s ++ [true])))) ∧
    (¬∀ i j, i ≠ j → i + 5 ≤ (s ++ [false]).length → j + 5 ≤ (s ++ [false]).length → 
      (List.take 5 (List.drop i (s ++ [false])) ≠ List.take 5 (List.drop j (s ++ [false]))))}

/-- The theorem stating that the first 4 digits are the same as the last 4 digits -/
theorem first_four_eq_last_four (s : SpecialSequence) : 
  List.take 4 s.val = List.take 4 (List.reverse s.val) := by
  sorry

end first_four_eq_last_four_l922_92248


namespace third_derivative_at_negative_one_l922_92267

/-- Given a function f where f(x) = e^(-x) + 2f''(0)x, prove that f'''(-1) = 2 - e -/
theorem third_derivative_at_negative_one (f : ℝ → ℝ) (h : ∀ x, f x = Real.exp (-x) + 2 * (deriv^[2] f 0) * x) :
  (deriv^[3] f) (-1) = 2 - Real.exp 1 := by
  sorry

end third_derivative_at_negative_one_l922_92267


namespace business_join_time_l922_92288

/-- Represents the number of months in a year -/
def months_in_year : ℕ := 12

/-- Represents A's investment in Rupees -/
def investment_A : ℕ := 36000

/-- Represents B's investment in Rupees -/
def investment_B : ℕ := 54000

/-- Represents the ratio of A's profit share to B's profit share -/
def profit_ratio : ℚ := 2 / 1

theorem business_join_time (x : ℕ) : 
  (investment_A * months_in_year : ℚ) / (investment_B * (months_in_year - x)) = profit_ratio →
  x = 8 :=
by sorry

end business_join_time_l922_92288


namespace jakes_weight_l922_92239

/-- Given the weights of Mildred, Carol, and Jake, prove Jake's weight -/
theorem jakes_weight (mildred_weight : ℕ) (carol_weight : ℕ) (jake_weight : ℕ) 
  (h1 : mildred_weight = 59)
  (h2 : carol_weight = mildred_weight + 9)
  (h3 : jake_weight = 2 * carol_weight) : 
  jake_weight = 136 := by
  sorry

end jakes_weight_l922_92239


namespace cube_plus_135002_l922_92226

theorem cube_plus_135002 (n : ℤ) : 
  (n = 149 ∨ n = -151) → n^3 + 135002 = (n + 1)^3 := by
  sorry

end cube_plus_135002_l922_92226


namespace second_number_value_l922_92265

theorem second_number_value (x y z : ℚ) 
  (sum_eq : x + y + z = 120)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 7 / 9) :
  y = 672 / 17 := by
sorry

end second_number_value_l922_92265


namespace anna_phone_chargers_l922_92208

/-- The number of phone chargers Anna has -/
def phone_chargers : ℕ := sorry

/-- The number of laptop chargers Anna has -/
def laptop_chargers : ℕ := sorry

/-- The total number of chargers Anna has -/
def total_chargers : ℕ := 24

theorem anna_phone_chargers :
  (laptop_chargers = 5 * phone_chargers) →
  (phone_chargers + laptop_chargers = total_chargers) →
  phone_chargers = 4 := by
sorry

end anna_phone_chargers_l922_92208


namespace max_y_coordinate_cos_2theta_l922_92218

/-- The maximum y-coordinate of a point on the curve r = cos 2θ in polar coordinates -/
theorem max_y_coordinate_cos_2theta : 
  let r : ℝ → ℝ := λ θ ↦ Real.cos (2 * θ)
  let x : ℝ → ℝ := λ θ ↦ r θ * Real.cos θ
  let y : ℝ → ℝ := λ θ ↦ r θ * Real.sin θ
  ∃ (θ_max : ℝ), ∀ (θ : ℝ), y θ ≤ y θ_max ∧ y θ_max = 3 * Real.sqrt 3 / 4 :=
sorry

end max_y_coordinate_cos_2theta_l922_92218


namespace troll_count_l922_92206

/-- The number of creatures at the table -/
def total_creatures : ℕ := 60

/-- The number of trolls at the table -/
def num_trolls : ℕ := 20

/-- The number of elves who made a mistake -/
def mistake_elves : ℕ := 2

theorem troll_count :
  ∀ t : ℕ,
  t = num_trolls →
  (∃ x : ℕ,
    x ∈ ({2, 4, 6} : Set ℕ) ∧
    3 * t + x = total_creatures + 4 ∧
    t + (total_creatures - t) = total_creatures ∧
    t - mistake_elves = (total_creatures - t) - x / 2) :=
by sorry

end troll_count_l922_92206


namespace min_value_theorem_l922_92273

def is_arithmetic_geometric (a : ℕ → ℝ) : Prop :=
  ∃ (d q : ℝ), q > 0 ∧ ∀ n, a (n + 1) = a n * q + d

theorem min_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  is_arithmetic_geometric a →
  (∀ n, a n > 0) →
  a 7 = a 6 + 2 * a 5 →
  a m * a n = 4 * (a 1) ^ 2 →
  (1 : ℝ) / m + 4 / n ≥ 9 / 4 :=
by sorry

end min_value_theorem_l922_92273


namespace homework_problem_solution_l922_92230

theorem homework_problem_solution :
  ∃ (a b c d : ℤ),
    a ≤ -1 ∧ b ≤ -1 ∧ c ≤ -1 ∧ d ≤ -1 ∧
    -a - b = -a * b ∧
    c * d = -182 * (1 / (-c - d)) :=
by sorry

end homework_problem_solution_l922_92230


namespace anns_age_l922_92243

theorem anns_age (A B : ℕ) : 
  A + B = 52 → 
  B = (2 * B - A / 3) → 
  A = 39 := by sorry

end anns_age_l922_92243


namespace tan_theta_half_l922_92284

theorem tan_theta_half (θ : Real) (h : (1 + Real.cos (2 * θ)) / Real.sin (2 * θ) = 2) : 
  Real.tan θ = 1 / 2 := by
  sorry

end tan_theta_half_l922_92284


namespace grace_mowing_hours_l922_92204

/-- Represents the rates and hours worked by Grace in her landscaping business -/
structure LandscapingWork where
  mowing_rate : ℕ
  weeding_rate : ℕ
  mulching_rate : ℕ
  weeding_hours : ℕ
  mulching_hours : ℕ
  total_earnings : ℕ

/-- Calculates the number of hours spent mowing lawns given the landscaping work details -/
def mowing_hours (work : LandscapingWork) : ℕ :=
  (work.total_earnings - (work.weeding_rate * work.weeding_hours + work.mulching_rate * work.mulching_hours)) / work.mowing_rate

/-- Theorem stating that Grace spent 63 hours mowing lawns in September -/
theorem grace_mowing_hours :
  let work : LandscapingWork := {
    mowing_rate := 6,
    weeding_rate := 11,
    mulching_rate := 9,
    weeding_hours := 9,
    mulching_hours := 10,
    total_earnings := 567
  }
  mowing_hours work = 63 := by sorry

end grace_mowing_hours_l922_92204


namespace ribbon_length_reduction_l922_92281

theorem ribbon_length_reduction (original_length : ℝ) (ratio_original : ℝ) (ratio_new : ℝ) (new_length : ℝ) : 
  original_length = 55 →
  ratio_original = 11 →
  ratio_new = 7 →
  new_length = (original_length * ratio_new) / ratio_original →
  new_length = 35 := by
sorry

end ribbon_length_reduction_l922_92281


namespace waiter_shift_earnings_l922_92234

/-- Calculates the waiter's earnings during a shift --/
def waiter_earnings (total_customers : ℕ) 
                    (three_dollar_tippers : ℕ) 
                    (four_fifty_tippers : ℕ) 
                    (non_tippers : ℕ) 
                    (tip_pool_contribution : ℚ) 
                    (meal_cost : ℚ) : ℚ :=
  (3 * three_dollar_tippers + 4.5 * four_fifty_tippers) - tip_pool_contribution - meal_cost

theorem waiter_shift_earnings :
  waiter_earnings 15 6 4 5 10 5 = 21 := by
  sorry

end waiter_shift_earnings_l922_92234


namespace cash_refund_per_bottle_l922_92245

/-- The number of bottles of kombucha Henry drinks per month -/
def bottles_per_month : ℕ := 15

/-- The cost of each bottle of kombucha in dollars -/
def bottle_cost : ℚ := 3

/-- The number of bottles Henry can buy with his cash refund after 1 year -/
def bottles_from_refund : ℕ := 6

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- Theorem: The cash refund per bottle is $0.10 -/
theorem cash_refund_per_bottle :
  (bottles_from_refund * bottle_cost) / (bottles_per_month * months_in_year) = 1/10 := by
  sorry

end cash_refund_per_bottle_l922_92245


namespace log_equation_solution_l922_92232

theorem log_equation_solution (p q : ℝ) (h1 : p > q) (h2 : q > 0) :
  Real.log p + Real.log q = Real.log (p - q) ↔ p = q / (1 - q) ∧ q < 1 :=
by sorry

end log_equation_solution_l922_92232


namespace brother_age_l922_92295

theorem brother_age (man_age brother_age : ℕ) : 
  man_age = brother_age + 12 →
  man_age + 2 = 2 * (brother_age + 2) →
  brother_age = 10 := by
sorry

end brother_age_l922_92295


namespace initial_balloons_l922_92247

theorem initial_balloons (lost_balloons current_balloons : ℕ) 
  (h1 : lost_balloons = 2)
  (h2 : current_balloons = 7) : 
  current_balloons + lost_balloons = 9 := by
  sorry

end initial_balloons_l922_92247


namespace pure_imaginary_solutions_l922_92202

theorem pure_imaginary_solutions (x : ℂ) :
  (x^4 - 4*x^3 + 6*x^2 - 40*x - 64 = 0) ∧ (∃ k : ℝ, x = k * Complex.I) ↔
  x = Complex.I * Real.sqrt 10 ∨ x = -Complex.I * Real.sqrt 10 := by
  sorry

end pure_imaginary_solutions_l922_92202


namespace third_number_in_systematic_sampling_l922_92227

/-- Systematic sampling function that returns the nth number drawn -/
def systematicSample (totalStudents : Nat) (sampleSize : Nat) (firstDrawn : Nat) (n : Nat) : Nat :=
  firstDrawn + (n - 1) * (totalStudents / sampleSize)

theorem third_number_in_systematic_sampling
  (totalStudents : Nat)
  (sampleSize : Nat)
  (firstPartEnd : Nat)
  (firstDrawn : Nat)
  (h1 : totalStudents = 1000)
  (h2 : sampleSize = 50)
  (h3 : firstPartEnd = 20)
  (h4 : firstDrawn = 15)
  (h5 : firstDrawn ≤ firstPartEnd) :
  systematicSample totalStudents sampleSize firstDrawn 3 = 55 := by
sorry

#eval systematicSample 1000 50 15 3

end third_number_in_systematic_sampling_l922_92227


namespace M_equals_set_l922_92286

def M (x y z : ℝ) : Set ℝ :=
  { w | ∃ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
    w = (x / abs x) + (y / abs y) + (z / abs z) + (abs (x * y * z) / (x * y * z)) }

theorem M_equals_set (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  M x y z = {4, -4, 0} := by
  sorry

end M_equals_set_l922_92286


namespace exists_quadrilateral_equal_angle_tangents_l922_92250

/-- Represents a quadrilateral with four interior angles -/
structure Quadrilateral where
  α : Real
  β : Real
  γ : Real
  δ : Real
  sum_360 : α + β + γ + δ = 2 * Real.pi

/-- The theorem stating the existence of a quadrilateral with equal angle tangents -/
theorem exists_quadrilateral_equal_angle_tangents :
  ∃ q : Quadrilateral, Real.tan q.α = Real.tan q.β ∧ Real.tan q.α = Real.tan q.γ ∧ Real.tan q.α = Real.tan q.δ :=
by sorry

end exists_quadrilateral_equal_angle_tangents_l922_92250


namespace family_income_problem_l922_92296

theorem family_income_problem (x : ℝ) : 
  (4 * x - 1178) / 3 = 650 → x = 782 := by
  sorry

end family_income_problem_l922_92296


namespace grade_improvement_l922_92224

/-- Represents the distribution of grades --/
structure GradeDistribution where
  a : ℕ  -- number of 1's
  b : ℕ  -- number of 2's
  c : ℕ  -- number of 3's
  d : ℕ  -- number of 4's
  e : ℕ  -- number of 5's

/-- Calculates the average grade --/
def averageGrade (g : GradeDistribution) : ℚ :=
  (g.a + 2 * g.b + 3 * g.c + 4 * g.d + 5 * g.e) / (g.a + g.b + g.c + g.d + g.e)

/-- Represents the change in grade distribution after changing 1's to 3's --/
def changeGrades (g : GradeDistribution) : GradeDistribution :=
  { a := 0, b := g.b, c := g.c + g.a, d := g.d, e := g.e }

theorem grade_improvement (g : GradeDistribution) :
  averageGrade g < 3 → averageGrade (changeGrades g) ≤ 4 := by
  sorry


end grade_improvement_l922_92224


namespace rock_band_fuel_cost_l922_92287

theorem rock_band_fuel_cost (x : ℝ) :
  (2 * (0.5 * x + 100) + 2 * (0.75 * x + 100) = 550) →
  x = 60 := by
  sorry

end rock_band_fuel_cost_l922_92287


namespace quadratic_root_m_value_l922_92266

theorem quadratic_root_m_value : ∀ m : ℝ, 
  (1 : ℝ)^2 + m * 1 - 6 = 0 → m = 5 := by
  sorry

end quadratic_root_m_value_l922_92266


namespace complex_multiplication_l922_92231

/-- Given two complex numbers z₁ and z₂, prove that their product is equal to the specified result. -/
theorem complex_multiplication (z₁ z₂ : ℂ) : 
  z₁ = 1 - 3*I → z₂ = 6 - 8*I → z₁ * z₂ = -18 - 26*I := by
  sorry


end complex_multiplication_l922_92231


namespace billys_songs_l922_92270

theorem billys_songs (total_songs : ℕ) (can_play : ℕ) (to_learn : ℕ) :
  total_songs = 52 →
  can_play = 24 →
  to_learn = 28 →
  can_play = total_songs - to_learn :=
by sorry

end billys_songs_l922_92270


namespace log7_10_approximation_l922_92209

-- Define the approximations given in the problem
def log10_2_approx : ℝ := 0.301
def log10_5_approx : ℝ := 0.699

-- Define a tolerance for approximation
def tolerance : ℝ := 0.001

-- Theorem statement
theorem log7_10_approximation :
  let log10_7 := log10_5_approx + log10_2_approx
  abs (Real.log 10 / Real.log 7 - 33 / 10) < tolerance := by
  sorry

end log7_10_approximation_l922_92209


namespace remainder_of_eighteen_divided_by_seven_l922_92215

theorem remainder_of_eighteen_divided_by_seven : ∃ k : ℤ, 18 = 7 * k + 4 := by
  sorry

end remainder_of_eighteen_divided_by_seven_l922_92215


namespace geometric_progression_ratio_l922_92297

theorem geometric_progression_ratio (x y z r : ℝ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z →
  ∃ (a : ℝ), a ≠ 0 ∧ 
    x * (y - z) = a ∧
    y * (z - x) = a * r ∧
    z * (y - x) = a * r^2 →
  r^2 - r + 1 = 0 :=
by sorry

end geometric_progression_ratio_l922_92297


namespace prob_three_even_out_of_six_l922_92246

/-- The probability of rolling an even number on a fair 12-sided die -/
def prob_even : ℚ := 1 / 2

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The number of dice we want to show even numbers -/
def target_even : ℕ := 3

/-- The probability of exactly three out of six fair 12-sided dice showing an even number -/
theorem prob_three_even_out_of_six :
  (Nat.choose num_dice target_even : ℚ) * prob_even ^ num_dice = 5 / 16 := by
  sorry

end prob_three_even_out_of_six_l922_92246


namespace expression_value_l922_92283

theorem expression_value : (3^2 - 3) + (4^2 - 4) - (5^2 - 5) = -2 := by
  sorry

end expression_value_l922_92283


namespace second_number_proof_l922_92213

theorem second_number_proof (x y z : ℝ) 
  (sum_eq : x + y + z = 120)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 4 / 7)
  (x_pos : x > 0) (y_pos : y > 0) (z_pos : z > 0) : 
  y = 240 / 7 := by
  sorry

end second_number_proof_l922_92213


namespace multiplication_puzzle_l922_92222

theorem multiplication_puzzle :
  ∀ P Q R : ℕ,
    P ≠ Q → P ≠ R → Q ≠ R →
    P < 10 → Q < 10 → R < 10 →
    (100 * P + 10 * P + Q) * Q = 1000 * R + 100 * Q + 50 + Q →
    P + Q + R = 17 := by
  sorry

end multiplication_puzzle_l922_92222


namespace professors_arrangement_count_l922_92259

/-- The number of ways to arrange professors among students. -/
def arrange_professors (num_students : ℕ) (num_professors : ℕ) : ℕ :=
  Nat.descFactorial (num_students - 1) num_professors

/-- Theorem stating that arranging 3 professors among 6 students results in 60 possibilities. -/
theorem professors_arrangement_count :
  arrange_professors 6 3 = 60 := by
  sorry

end professors_arrangement_count_l922_92259


namespace coeff_x_cubed_expansion_l922_92255

/-- The coefficient of x^3 in the expansion of (x^2 - x + 1)^10 -/
def coeff_x_cubed : ℤ := -210

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

theorem coeff_x_cubed_expansion :
  coeff_x_cubed = binomial 10 8 * binomial 2 1 * (-1) + binomial 10 7 * binomial 3 3 * (-1) :=
sorry

end coeff_x_cubed_expansion_l922_92255


namespace max_daily_profit_daily_profit_correct_l922_92229

/-- Represents the daily profit function for a store selling an item --/
def daily_profit (x : ℕ) : ℝ :=
  if x ≤ 30 then -x^2 + 54*x + 640
  else -40*x + 2560

/-- Theorem stating the maximum daily profit and the day it occurs --/
theorem max_daily_profit :
  ∃ (max_profit : ℝ) (max_day : ℕ),
    max_profit = 1369 ∧ 
    max_day = 27 ∧
    (∀ x : ℕ, 1 ≤ x ∧ x ≤ 60 → daily_profit x ≤ max_profit) ∧
    daily_profit max_day = max_profit :=
  sorry

/-- Cost price of the item --/
def cost_price : ℝ := 30

/-- Selling price function --/
def selling_price (x : ℕ) : ℝ :=
  if x ≤ 30 then 0.5 * x + 35
  else 50

/-- Quantity sold function --/
def quantity_sold (x : ℕ) : ℝ := 128 - 2 * x

/-- Verifies that the daily_profit function is correct --/
theorem daily_profit_correct (x : ℕ) (h : 1 ≤ x ∧ x ≤ 60) :
  daily_profit x = (selling_price x - cost_price) * quantity_sold x :=
  sorry

end max_daily_profit_daily_profit_correct_l922_92229


namespace rectangular_to_polar_conversion_l922_92262

theorem rectangular_to_polar_conversion :
  ∀ (x y r θ : ℝ),
    x = -Real.sqrt 3 →
    y = Real.sqrt 3 →
    r > 0 →
    0 ≤ θ ∧ θ < 2 * Real.pi →
    r = 3 ∧ θ = 3 * Real.pi / 4 →
    x = -r * Real.cos θ ∧
    y = r * Real.sin θ :=
by sorry

end rectangular_to_polar_conversion_l922_92262


namespace line_plane_relationship_l922_92207

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the contained relation between a line and a plane
variable (contained_in : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel_lines : Line → Line → Prop)

-- Define the skew relation between two lines
variable (skew_lines : Line → Line → Prop)

-- Theorem statement
theorem line_plane_relationship (a b : Line) (α : Plane) 
  (h1 : parallel_line_plane a α) 
  (h2 : contained_in b α) :
  parallel_lines a b ∨ skew_lines a b :=
sorry

end line_plane_relationship_l922_92207


namespace perpendicular_vectors_m_value_l922_92240

/-- Given two planar vectors a and b, where a is perpendicular to b,
    prove that the value of m in a = (m, m-1) and b = (1, 2) is 2/3. -/
theorem perpendicular_vectors_m_value :
  ∀ (m : ℝ),
  let a : ℝ × ℝ := (m, m - 1)
  let b : ℝ × ℝ := (1, 2)
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- dot product = 0 for perpendicular vectors
  m = 2/3 := by
sorry

end perpendicular_vectors_m_value_l922_92240


namespace fraction_comparison_l922_92276

theorem fraction_comparison : 
  (2.00000000004 / ((1.00000000004)^2 + 2.00000000004)) < 
  (2.00000000002 / ((1.00000000002)^2 + 2.00000000002)) := by
sorry

end fraction_comparison_l922_92276


namespace exists_1992_gon_l922_92256

/-- A convex polygon with n sides that is circumscribable about a circle -/
structure CircumscribablePolygon (n : ℕ) where
  sides : Fin n → ℝ
  convex : sorry
  circumscribable : sorry

/-- The condition that the side lengths are 1, 2, 3, ..., n in some order -/
def valid_side_lengths (n : ℕ) (p : CircumscribablePolygon n) : Prop :=
  ∃ (σ : Equiv (Fin n) (Fin n)), ∀ i, p.sides i = (σ i).val + 1

/-- The main theorem stating the existence of a 1992-sided circumscribable polygon
    with side lengths 1, 2, 3, ..., 1992 in some order -/
theorem exists_1992_gon :
  ∃ (p : CircumscribablePolygon 1992), valid_side_lengths 1992 p :=
sorry

end exists_1992_gon_l922_92256


namespace cost_per_foot_metal_roofing_l922_92251

/-- Calculates the cost per foot of metal roofing --/
theorem cost_per_foot_metal_roofing (total_required : ℕ) (free_provided : ℕ) (cost_remaining : ℕ) :
  total_required = 300 →
  free_provided = 250 →
  cost_remaining = 400 →
  (cost_remaining : ℚ) / (total_required - free_provided : ℚ) = 8 := by
  sorry

#check cost_per_foot_metal_roofing

end cost_per_foot_metal_roofing_l922_92251


namespace f_properties_l922_92216

-- Define the properties of function f
def is_additive (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

def is_negative_for_positive (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → f x < 0

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- State the theorem
theorem f_properties (f : ℝ → ℝ) 
  (h1 : is_additive f) (h2 : is_negative_for_positive f) : 
  is_odd f ∧ is_decreasing f := by
  sorry

end f_properties_l922_92216


namespace digit_A_is_zero_l922_92261

def is_divisible_by (n m : ℕ) : Prop := n % m = 0

theorem digit_A_is_zero (A : ℕ) (h1 : A < 10) 
  (h2 : is_divisible_by (353808 * 10 + A) 2)
  (h3 : is_divisible_by (353808 * 10 + A) 3)
  (h4 : is_divisible_by (353808 * 10 + A) 5)
  (h5 : is_divisible_by (353808 * 10 + A) 6)
  (h6 : is_divisible_by (353808 * 10 + A) 9) : 
  A = 0 := by
  sorry

end digit_A_is_zero_l922_92261


namespace same_solution_implies_c_equals_nine_l922_92272

theorem same_solution_implies_c_equals_nine (x c : ℝ) :
  (3 * x + 5 = 4) ∧ (c * x + 6 = 3) → c = 9 :=
by sorry

end same_solution_implies_c_equals_nine_l922_92272


namespace add_and_round_to_nearest_ten_l922_92282

def round_to_nearest_ten (n : ℤ) : ℤ :=
  10 * ((n + 5) / 10)

theorem add_and_round_to_nearest_ten : round_to_nearest_ten (58 + 29) = 90 := by
  sorry

end add_and_round_to_nearest_ten_l922_92282


namespace max_cups_in_kitchen_l922_92254

theorem max_cups_in_kitchen (a b : ℕ) : 
  (a.choose 2) * (b.choose 3) = 1200 → a + b ≤ 29 :=
by sorry

end max_cups_in_kitchen_l922_92254


namespace quadratic_function_inequality_l922_92285

theorem quadratic_function_inequality (a b : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^2 + a*x + b
  |f 1| + 2 * |f 2| + |f 3| ≥ 2 := by
  sorry

end quadratic_function_inequality_l922_92285


namespace rectangular_box_side_area_l922_92291

theorem rectangular_box_side_area 
  (l w h : ℝ) 
  (front_top : w * h = 0.5 * (l * w))
  (top_side : l * w = 1.5 * (l * h))
  (volume : l * w * h = 3000) :
  l * h = 200 :=
by sorry

end rectangular_box_side_area_l922_92291


namespace complex_square_simplification_l922_92260

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (4 + 3 * i)^2 = 7 + 24 * i :=
by sorry

end complex_square_simplification_l922_92260


namespace shirt_selection_theorem_l922_92278

/-- The number of shirts of each color in the drawer -/
def shirts : Finset (Nat × Nat) := {(4, 1), (7, 2), (9, 3)}

/-- The total number of shirts in the drawer -/
def total_shirts : Nat := 20

/-- The minimum number of shirts to select to ensure n shirts of the same color -/
def min_select (n : Nat) : Nat :=
  if n ≤ 4 then 3 * (n - 1) + 1
  else min (3 * (n - 1) + 1) total_shirts

/-- Theorem stating the minimum number of shirts to select for each case -/
theorem shirt_selection_theorem :
  (min_select 4 = 10) ∧
  (min_select 5 = 13) ∧
  (min_select 6 = 16) ∧
  (min_select 7 = 17) ∧
  (min_select 8 = 19) ∧
  (min_select 9 = 20) := by
  sorry

end shirt_selection_theorem_l922_92278


namespace estate_value_l922_92210

/-- Represents the distribution of Mr. T's estate -/
structure EstateDistribution where
  total : ℝ
  wife_share : ℝ
  daughter1_share : ℝ
  daughter2_share : ℝ
  son_share : ℝ
  gardener_share : ℝ

/-- Defines the conditions of Mr. T's estate distribution -/
def valid_distribution (e : EstateDistribution) : Prop :=
  -- Two daughters and son received 3/4 of the estate
  e.daughter1_share + e.daughter2_share + e.son_share = 3/4 * e.total ∧
  -- Daughters shared their portion in the ratio of 5:3
  e.daughter1_share / e.daughter2_share = 5/3 ∧
  -- Wife received thrice as much as the son
  e.wife_share = 3 * e.son_share ∧
  -- Gardener received $600
  e.gardener_share = 600 ∧
  -- Sum of wife and gardener's shares was 1/4 of the estate
  e.wife_share + e.gardener_share = 1/4 * e.total ∧
  -- Total is sum of all shares
  e.total = e.wife_share + e.daughter1_share + e.daughter2_share + e.son_share + e.gardener_share

/-- Theorem stating that Mr. T's estate value is $2400 -/
theorem estate_value (e : EstateDistribution) (h : valid_distribution e) : e.total = 2400 :=
  sorry


end estate_value_l922_92210


namespace sqrt_seven_sixth_power_l922_92292

theorem sqrt_seven_sixth_power : (Real.sqrt 7) ^ 6 = 343 := by sorry

end sqrt_seven_sixth_power_l922_92292


namespace distance_difference_l922_92228

-- Define the distances
def john_distance : ℝ := 0.7
def nina_distance : ℝ := 0.4

-- Theorem statement
theorem distance_difference : john_distance - nina_distance = 0.3 := by
  sorry

end distance_difference_l922_92228


namespace people_with_banners_l922_92290

/-- Given a stadium with a certain number of seats, prove that the number of people
    holding banners is equal to the number of attendees minus the number of empty seats. -/
theorem people_with_banners (total_seats attendees empty_seats : ℕ) :
  total_seats = 92 →
  attendees = 47 →
  empty_seats = 45 →
  attendees - empty_seats = 2 :=
by
  sorry

end people_with_banners_l922_92290


namespace victors_lives_l922_92203

theorem victors_lives (lost : ℕ) (diff : ℕ) (current : ℕ) : 
  lost = 14 → diff = 12 → lost - current = diff → current = 2 := by
  sorry

end victors_lives_l922_92203


namespace a_value_proof_l922_92217

/-- The function f(x) = ax³ + 3x² + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x

theorem a_value_proof (a : ℝ) : f_derivative a (-1) = -12 → a = -2 := by
  sorry

end a_value_proof_l922_92217


namespace rectangle_perimeter_after_increase_l922_92235

/-- Given a rectangle with width 10 meters and original area 150 square meters,
    if its length is increased such that the new area is 4/3 times the original area,
    then the new perimeter is 60 meters. -/
theorem rectangle_perimeter_after_increase (original_length : ℝ) (new_length : ℝ) : 
  original_length * 10 = 150 →
  new_length * 10 = 150 * (4/3) →
  2 * (new_length + 10) = 60 :=
by sorry

end rectangle_perimeter_after_increase_l922_92235


namespace triangle_problem_l922_92233

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- b(cos A - 2cos C) = (2c - a)cos B
  b * (Real.cos A - 2 * Real.cos C) = (2 * c - a) * Real.cos B →
  -- Part I: Prove c/a = 2
  c / a = 2 ∧
  -- Part II: If cos B = 1/4 and perimeter = 5, prove b = 2
  (Real.cos B = 1/4 ∧ a + b + c = 5 → b = 2) := by
sorry

end triangle_problem_l922_92233


namespace unique_integer_solution_l922_92299

theorem unique_integer_solution : ∃! (x : ℕ), x > 0 ∧ (3 * x)^2 - x = 2016 :=
by
  -- The proof goes here
  sorry

end unique_integer_solution_l922_92299


namespace calculation_proof_l922_92269

theorem calculation_proof : (4 - Real.sqrt 3) ^ 0 - 3 * Real.tan (π / 3) - (-1/2)⁻¹ + Real.sqrt 12 = 3 - Real.sqrt 3 := by
  sorry

end calculation_proof_l922_92269


namespace units_digit_problem_l922_92205

theorem units_digit_problem : ∃ n : ℕ, (8 * 14 * 1986 + 8^2) % 10 = 6 ∧ n * 10 ≤ (8 * 14 * 1986 + 8^2) ∧ (8 * 14 * 1986 + 8^2) < (n + 1) * 10 := by
  sorry

end units_digit_problem_l922_92205


namespace triangle_area_l922_92219

theorem triangle_area (A B C : ℝ) (a : ℝ) (h1 : a = 2) (h2 : C = π/4) (h3 : Real.tan (B/2) = 1/2) :
  (1/2) * a * (Real.sin C) * (8 * Real.sqrt 2 / 7) = 8/7 := by
  sorry

end triangle_area_l922_92219


namespace trapezoid_base_ratio_l922_92241

/-- A trapezoid with bases a and b, where a > b, and its midsegment is divided into three equal parts by the diagonals. -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  h : a > b

/-- The ratio of the bases of a trapezoid with the given properties is 2:1 -/
theorem trapezoid_base_ratio (t : Trapezoid) : t.a / t.b = 2 := by
  sorry

end trapezoid_base_ratio_l922_92241


namespace z_sixth_power_l922_92237

theorem z_sixth_power (z : ℂ) : z = (-Real.sqrt 3 + Complex.I) / 2 → z^6 = -1 := by
  sorry

end z_sixth_power_l922_92237


namespace PQRS_equals_nine_l922_92212

theorem PQRS_equals_nine :
  let P : ℝ := Real.sqrt 2010 + Real.sqrt 2007
  let Q : ℝ := -Real.sqrt 2010 - Real.sqrt 2007
  let R : ℝ := Real.sqrt 2010 - Real.sqrt 2007
  let S : ℝ := Real.sqrt 2007 - Real.sqrt 2010
  P * Q * R * S = 9 := by
  sorry

end PQRS_equals_nine_l922_92212


namespace square_area_is_121_l922_92238

/-- A square in a 2D coordinate system --/
structure Square where
  x : ℝ
  y : ℝ

/-- The area of a square --/
def square_area (s : Square) : ℝ :=
  (20 - 9) ^ 2

/-- Theorem: The area of the given square is 121 square units --/
theorem square_area_is_121 (s : Square) : square_area s = 121 := by
  sorry

end square_area_is_121_l922_92238


namespace sum_of_altitudes_triangle_l922_92253

/-- The sum of altitudes of a triangle formed by the line 8x + 3y = 48 and the coordinate axes -/
theorem sum_of_altitudes_triangle (x y : ℝ) (h : 8 * x + 3 * y = 48) :
  let x_intercept : ℝ := 48 / 8
  let y_intercept : ℝ := 48 / 3
  let hypotenuse : ℝ := Real.sqrt (x_intercept^2 + y_intercept^2)
  let altitude_to_hypotenuse : ℝ := 96 / hypotenuse
  x_intercept + y_intercept + altitude_to_hypotenuse = (22 * Real.sqrt 292 + 96) / Real.sqrt 292 := by
sorry

end sum_of_altitudes_triangle_l922_92253


namespace calories_per_slice_l922_92221

/-- Given a pizza with 8 slices, where half the pizza contains 1200 calories,
    prove that each slice contains 300 calories. -/
theorem calories_per_slice (total_slices : ℕ) (eaten_fraction : ℚ) (total_calories : ℕ) :
  total_slices = 8 →
  eaten_fraction = 1/2 →
  total_calories = 1200 →
  (total_calories : ℚ) / (eaten_fraction * total_slices) = 300 := by
sorry

end calories_per_slice_l922_92221


namespace max_reciprocal_sum_l922_92257

theorem max_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y^2 = 1) :
  (1 / x + 1 / y) ≤ 2 * Real.sqrt 2 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀^2 + y₀^2 = 1 ∧ 1 / x₀ + 1 / y₀ = 2 * Real.sqrt 2 :=
sorry

end max_reciprocal_sum_l922_92257


namespace N_equals_one_l922_92280

theorem N_equals_one :
  let N := (Real.sqrt (Real.sqrt 5 + 2) + Real.sqrt (Real.sqrt 5 - 2)) / Real.sqrt (Real.sqrt 5 + 1) - Real.sqrt (3 - 2 * Real.sqrt 2)
  N = 1 := by sorry

end N_equals_one_l922_92280


namespace fruit_stand_problem_l922_92236

/-- Proves that the price of each apple is $0.90 given the conditions of the fruit stand problem -/
theorem fruit_stand_problem (total_cost : ℝ) (total_fruits : ℕ) (banana_price : ℝ)
  (h_total_cost : total_cost = 6.50)
  (h_total_fruits : total_fruits = 9)
  (h_banana_price : banana_price = 0.70) :
  ∃ (apple_price : ℝ) (num_apples : ℕ),
    apple_price = 0.90 ∧
    num_apples + (total_fruits - num_apples) = total_fruits ∧
    apple_price * num_apples + banana_price * (total_fruits - num_apples) = total_cost :=
by
  sorry

#check fruit_stand_problem

end fruit_stand_problem_l922_92236


namespace abs_value_sum_and_diff_l922_92223

theorem abs_value_sum_and_diff (a b : ℝ) :
  (abs a = 5 ∧ abs b = 3) →
  ((a > 0 ∧ b < 0) → a + b = 2) ∧
  (abs (a + b) = a + b → (a - b = 2 ∨ a - b = 8)) :=
by sorry

end abs_value_sum_and_diff_l922_92223


namespace rectangle_area_theorem_l922_92244

/-- Rectangle with known side length and area -/
structure Rectangle1 where
  side : ℝ
  area : ℝ

/-- Rectangle similar to Rectangle1 with known diagonal -/
structure Rectangle2 where
  diagonal : ℝ

/-- The area of Rectangle2 given the properties of Rectangle1 -/
def area_rectangle2 (r1 : Rectangle1) (r2 : Rectangle2) : ℝ :=
  sorry

theorem rectangle_area_theorem (r1 : Rectangle1) (r2 : Rectangle2) :
  r1.side = 3 ∧ r1.area = 18 ∧ r2.diagonal = 20 → area_rectangle2 r1 r2 = 160 :=
sorry

end rectangle_area_theorem_l922_92244


namespace sqrt_difference_equals_negative_two_m_l922_92263

theorem sqrt_difference_equals_negative_two_m (m n : ℝ) (h1 : n < m) (h2 : m < 0) :
  Real.sqrt (m^2 + 2*m*n + n^2) - Real.sqrt (m^2 - 2*m*n + n^2) = -2*m := by
  sorry

end sqrt_difference_equals_negative_two_m_l922_92263


namespace line_segment_proportion_l922_92249

-- Define the line segments as real numbers (representing their lengths in cm)
def a : ℝ := 1
def b : ℝ := 4
def c : ℝ := 2

-- Define the proportion relationship
def are_proportional (a b c d : ℝ) : Prop := a * d = b * c

-- State the theorem
theorem line_segment_proportion :
  ∀ d : ℝ, are_proportional a b c d → d = 8 :=
by sorry

end line_segment_proportion_l922_92249


namespace total_money_l922_92294

theorem total_money (mark carolyn dave : ℚ) 
  (h1 : mark = 4/5)
  (h2 : carolyn = 2/5)
  (h3 : dave = 1/2) :
  mark + carolyn + dave = 17/10 := by
  sorry

end total_money_l922_92294


namespace sqrt_equation_solution_l922_92214

theorem sqrt_equation_solution : 
  ∃! z : ℝ, Real.sqrt (10 + 3 * z) = 13 := by sorry

end sqrt_equation_solution_l922_92214


namespace trig_expression_equals_half_l922_92242

theorem trig_expression_equals_half : 
  2 * Real.cos (30 * π / 180) - Real.tan (60 * π / 180) + Real.sin (45 * π / 180) * Real.cos (45 * π / 180) = 1/2 := by
  sorry

end trig_expression_equals_half_l922_92242


namespace systematic_sampling_questionnaire_C_l922_92225

/-- Systematic sampling problem -/
theorem systematic_sampling_questionnaire_C (total_population : ℕ) 
  (sample_size : ℕ) (first_number : ℕ) : 
  total_population = 960 →
  sample_size = 32 →
  first_number = 9 →
  (960 - 750) / (960 / 32) = 7 :=
by sorry

end systematic_sampling_questionnaire_C_l922_92225


namespace all_pairs_divisible_by_seven_l922_92274

-- Define the type for pairs on the board
def BoardPair := ℤ × ℤ

-- Define the property that 2a - b is divisible by 7
def DivisibleBySeven (p : BoardPair) : Prop :=
  ∃ k : ℤ, 2 * p.1 - p.2 = 7 * k

-- Define the set of all pairs that can appear on the board
inductive ValidPair : BoardPair → Prop where
  | initial : ValidPair (1, 2)
  | negate (a b : ℤ) : ValidPair (a, b) → ValidPair (-a, -b)
  | rotate (a b : ℤ) : ValidPair (a, b) → ValidPair (-b, a + b)
  | add (a b c d : ℤ) : ValidPair (a, b) → ValidPair (c, d) → ValidPair (a + c, b + d)

-- Theorem statement
theorem all_pairs_divisible_by_seven :
  ∀ p : BoardPair, ValidPair p → DivisibleBySeven p :=
  sorry

end all_pairs_divisible_by_seven_l922_92274


namespace y_intercept_of_line_l922_92220

/-- The y-intercept of the line 3x - 4y = 12 is -3 -/
theorem y_intercept_of_line (x y : ℝ) : 3 * x - 4 * y = 12 → x = 0 → y = -3 := by
  sorry

end y_intercept_of_line_l922_92220


namespace locus_equation_l922_92200

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The locus of centers of circles externally tangent to C₁ and internally tangent to C₃ -/
def locus_of_centers (C₁ C₃ : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ r : ℝ,
    (Circle.mk p r).radius + C₁.radius = Real.sqrt ((p.1 - C₁.center.1)^2 + (p.2 - C₁.center.2)^2) ∧
    C₃.radius - (Circle.mk p r).radius = Real.sqrt ((p.1 - C₃.center.1)^2 + (p.2 - C₃.center.2)^2)}

theorem locus_equation (C₁ C₃ : Circle)
  (h₁ : C₁.center = (0, 0) ∧ C₁.radius = 2)
  (h₃ : C₃.center = (3, 0) ∧ C₃.radius = 5) :
  locus_of_centers C₁ C₃ = {p : ℝ × ℝ | p.1^2 + 7*p.2^2 - 34*p.1 - 57 = 0} :=
sorry

end locus_equation_l922_92200


namespace median_in_65_interval_l922_92258

/-- Represents a score interval with its lower bound and frequency -/
structure ScoreInterval :=
  (lower_bound : ℕ)
  (frequency : ℕ)

/-- Finds the interval containing the median score -/
def find_median_interval (intervals : List ScoreInterval) : Option ℕ :=
  let total_students := intervals.foldl (fun acc i => acc + i.frequency) 0
  let median_position := (total_students + 1) / 2
  let rec find_interval (acc : ℕ) (remaining : List ScoreInterval) : Option ℕ :=
    match remaining with
    | [] => none
    | i :: is =>
        if acc + i.frequency ≥ median_position then
          some i.lower_bound
        else
          find_interval (acc + i.frequency) is
  find_interval 0 intervals

theorem median_in_65_interval (score_data : List ScoreInterval) :
  score_data = [
    ⟨80, 20⟩, ⟨75, 15⟩, ⟨70, 10⟩, ⟨65, 25⟩, ⟨60, 15⟩, ⟨55, 15⟩
  ] →
  find_median_interval score_data = some 65 :=
by sorry

end median_in_65_interval_l922_92258


namespace fraction_of_juices_consumed_l922_92298

/-- Represents the fraction of juices consumed at a summer picnic -/
theorem fraction_of_juices_consumed (total_people : ℕ) (soda_cans : ℕ) (water_bottles : ℕ) (juice_bottles : ℕ)
  (soda_drinkers : ℚ) (water_drinkers : ℚ) (total_recyclables : ℕ) :
  total_people = 90 →
  soda_cans = 50 →
  water_bottles = 50 →
  juice_bottles = 50 →
  soda_drinkers = 1/2 →
  water_drinkers = 1/3 →
  total_recyclables = 115 →
  (juice_bottles - (total_recyclables - (soda_drinkers * total_people + water_drinkers * total_people))) / juice_bottles = 4/5 := by
  sorry

end fraction_of_juices_consumed_l922_92298
