import Mathlib

namespace inscribed_squares_ratio_l3728_372802

/-- A square inscribed in a right triangle with one vertex at the right angle -/
def square_in_triangle_vertex (a b c : ℝ) (x : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ (b - x) / x = a / b

/-- A square inscribed in a right triangle with one side on the hypotenuse -/
def square_in_triangle_hypotenuse (a b c : ℝ) (y : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ y / c = (b - y) / a

theorem inscribed_squares_ratio :
  ∀ (x y : ℝ),
    square_in_triangle_vertex 5 12 13 x →
    square_in_triangle_hypotenuse 6 8 10 y →
    x / y = 216 / 85 := by
  sorry

end inscribed_squares_ratio_l3728_372802


namespace ticket_price_possibilities_l3728_372834

def is_valid_price (y : ℕ) : Prop :=
  y > 0 ∧ 90 % y = 0 ∧ 100 % y = 0

theorem ticket_price_possibilities :
  ∃! (n : ℕ), n > 0 ∧ (∃ (S : Finset ℕ), S.card = n ∧ ∀ y ∈ S, is_valid_price y) :=
sorry

end ticket_price_possibilities_l3728_372834


namespace log3_graph_properties_l3728_372845

-- Define the logarithm function base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- Define the graph of y = log₃(x)
def graph_log3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = log3 p.1 ∧ p.1 > 0}

-- Define the x-axis and y-axis
def x_axis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 0}
def y_axis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0}

-- Theorem statement
theorem log3_graph_properties :
  (∃ p, p ∈ graph_log3 ∩ x_axis) ∧
  (graph_log3 ∩ y_axis = ∅) :=
by sorry

end log3_graph_properties_l3728_372845


namespace ratio_a_to_b_l3728_372807

theorem ratio_a_to_b (a b : ℚ) (h : (12*a - 5*b) / (17*a - 3*b) = 4/7) : a/b = 23/16 := by
  sorry

end ratio_a_to_b_l3728_372807


namespace probability_one_red_ball_l3728_372803

def total_balls : ℕ := 12
def red_balls : ℕ := 3
def black_balls : ℕ := 4
def white_balls : ℕ := 5
def drawn_balls : ℕ := 2

theorem probability_one_red_ball :
  (red_balls * (black_balls + white_balls)) / (total_balls.choose drawn_balls) = 9 / 22 := by
  sorry

end probability_one_red_ball_l3728_372803


namespace consecutive_numbers_sum_l3728_372805

theorem consecutive_numbers_sum (x : ℕ) :
  (∃ y : ℕ, 0 ≤ y ∧ y ≤ 9 ∧
    (List.sum (List.filter (λ i => i ≠ x + y) [x, x+1, x+2, x+3, x+4, x+5, x+6, x+7, x+8, x+9]) = 2002)) →
  x = 218 ∧ 
  List.filter (λ i => i ≠ 223) [x, x+1, x+2, x+3, x+4, x+5, x+6, x+7, x+8, x+9] = 
    [218, 219, 220, 221, 222, 224, 225, 226, 227] := by
  sorry

end consecutive_numbers_sum_l3728_372805


namespace money_difference_l3728_372820

/-- Calculates the difference between final and initial amounts given monetary transactions --/
theorem money_difference (initial chores birthday neighbor candy lost : ℕ) : 
  initial = 2 →
  chores = 5 →
  birthday = 10 →
  neighbor = 7 →
  candy = 3 →
  lost = 2 →
  (initial + chores + birthday + neighbor - candy - lost) - initial = 17 := by
  sorry

end money_difference_l3728_372820


namespace sum_of_quotient_digits_l3728_372886

def dividend : ℕ := 111111
def divisor : ℕ := 3

def quotient : ℕ := dividend / divisor

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

theorem sum_of_quotient_digits :
  sum_of_digits quotient = 20 := by sorry

end sum_of_quotient_digits_l3728_372886


namespace ellipse_left_vertex_l3728_372819

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
def ellipse_conditions (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ b = 4 ∧ (3 : ℝ)^2 = a^2 - b^2

-- Theorem statement
theorem ellipse_left_vertex (a b : ℝ) :
  ellipse_conditions a b →
  ∃ (x y : ℝ), ellipse a b x y ∧ x = -5 ∧ y = 0 :=
sorry

end ellipse_left_vertex_l3728_372819


namespace smallest_multiple_of_seven_l3728_372800

def is_valid_abc (a b c : ℕ) : Prop :=
  a > 3 ∧ b > 3 ∧ c > 3 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

def form_number (a b c : ℕ) : ℕ :=
  100000 * a + 10000 * b + 1000 * c + 321

theorem smallest_multiple_of_seven :
  ∀ a b c : ℕ,
    is_valid_abc a b c →
    form_number a b c ≥ 468321 ∨ ¬(form_number a b c % 7 = 0) :=
by sorry

end smallest_multiple_of_seven_l3728_372800


namespace work_completion_time_l3728_372887

theorem work_completion_time (a_time b_time b_remaining : ℚ) 
  (ha : a_time = 45)
  (hb : b_time = 40)
  (hc : b_remaining = 23) : 
  let x := (b_time * b_remaining * a_time - a_time * b_time) / (a_time * b_time + a_time * b_remaining)
  x = 9 := by sorry

end work_completion_time_l3728_372887


namespace complex_sixth_root_of_negative_sixteen_l3728_372885

theorem complex_sixth_root_of_negative_sixteen :
  ∀ z : ℂ, z^6 = -16 ↔ z = Complex.I * 2 ∨ z = Complex.I * (-2) := by
  sorry

end complex_sixth_root_of_negative_sixteen_l3728_372885


namespace initial_apples_correct_l3728_372856

/-- The number of apples the cafeteria had initially -/
def initial_apples : ℕ := 23

/-- The number of apples used for lunch -/
def apples_used : ℕ := 20

/-- The number of apples bought -/
def apples_bought : ℕ := 6

/-- The number of apples remaining after transactions -/
def remaining_apples : ℕ := 9

/-- Theorem stating that the initial number of apples is correct -/
theorem initial_apples_correct : 
  initial_apples - apples_used + apples_bought = remaining_apples := by
  sorry

end initial_apples_correct_l3728_372856


namespace correct_sums_count_l3728_372866

theorem correct_sums_count (total : ℕ) (correct : ℕ) (incorrect : ℕ) : 
  total = 75 → 
  incorrect = 2 * correct → 
  total = correct + incorrect →
  correct = 25 := by
sorry

end correct_sums_count_l3728_372866


namespace smallest_m_plus_n_l3728_372830

/-- The function f(x) = arcsin(log_m(nx)) has a domain that is a closed interval of length 1/1007 --/
def domain_length (m n : ℕ) : ℚ :=
  (m^2 - 1 : ℚ) / (m * n)

/-- The theorem stating the smallest possible value of m + n --/
theorem smallest_m_plus_n :
  ∃ (m n : ℕ),
    m > 1 ∧
    domain_length m n = 1/1007 ∧
    ∀ (m' n' : ℕ), m' > 1 → domain_length m' n' = 1/1007 → m + n ≤ m' + n' ∧
    m + n = 19099 :=
sorry

end smallest_m_plus_n_l3728_372830


namespace f_satisfies_conditions_l3728_372882

/-- A function satisfying the given functional equation -/
def f : ℝ → ℝ := fun _ ↦ 1

/-- The main theorem stating that f satisfies the required conditions -/
theorem f_satisfies_conditions :
  (∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y) ∧ (f 0 = 1) :=
by sorry

end f_satisfies_conditions_l3728_372882


namespace initial_boys_count_l3728_372857

theorem initial_boys_count (t : ℕ) : 
  t > 0 →  -- Ensure the group is non-empty
  (t / 2 : ℚ) = (t : ℚ) * (1 / 2 : ℚ) →  -- Initially 50% boys
  ((t / 2 - 4 : ℚ) / (t + 2 : ℚ) = (2 / 5 : ℚ)) →  -- After changes, 40% boys
  t / 2 = 24 := by  -- Initial number of boys is 24
sorry

end initial_boys_count_l3728_372857


namespace arrange_six_books_two_pairs_l3728_372874

/-- The number of ways to arrange books with some identical copies -/
def arrange_books (total : ℕ) (identical_pairs : ℕ) (unique : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial 2 ^ identical_pairs)

/-- Theorem: Arranging 6 books with 2 identical pairs and 2 unique books -/
theorem arrange_six_books_two_pairs : arrange_books 6 2 2 = 180 := by
  sorry

end arrange_six_books_two_pairs_l3728_372874


namespace harkamal_payment_l3728_372867

/-- The amount Harkamal paid to the shopkeeper -/
def total_amount (grape_quantity grape_rate mango_quantity mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem: Given the conditions, Harkamal paid 1010 to the shopkeeper -/
theorem harkamal_payment : total_amount 8 70 9 50 = 1010 := by
  sorry

end harkamal_payment_l3728_372867


namespace ellipse_properties_l3728_372837

-- Define the ellipse
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

-- Define a point on the ellipse
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

-- Define the foci
def foci (e : Ellipse) : ℝ × ℝ := sorry

-- Theorem statement
theorem ellipse_properties (e : Ellipse) (A : PointOnEllipse e)
  (h_A : A.x = 1 ∧ A.y = 1)
  (h_foci_dist : let (F1, F2) := foci e
                 Real.sqrt ((A.x - F1)^2 + (A.y - F1)^2) +
                 Real.sqrt ((A.x - F2)^2 + (A.y - F2)^2) = 4) :
  (e.a = 2 ∧ e.b^2 = 4/3) ∧
  (∀ x y, x + 3*y - 4 = 0 ↔ x^2/4 + 3*y^2/4 = 1) ∧
  (∀ C D : PointOnEllipse e,
    let k₁ := (C.y - A.y) / (C.x - A.x)
    let k₂ := (D.y - A.y) / (D.x - A.x)
    k₁ * k₂ = -1 →
    (D.y - C.y) / (D.x - C.x) = 1/3) :=
by sorry

end ellipse_properties_l3728_372837


namespace fashion_markup_l3728_372870

theorem fashion_markup (original_price : ℝ) (markup1 markup2 markup3 : ℝ) 
  (h1 : markup1 = 0.35)
  (h2 : markup2 = 0.25)
  (h3 : markup3 = 0.45) :
  let price1 := original_price * (1 + markup1)
  let price2 := price1 * (1 + markup2)
  let final_price := price2 * (1 + markup3)
  (final_price - original_price) / original_price * 100 = 144.69 := by
sorry

end fashion_markup_l3728_372870


namespace cone_volume_from_half_sector_l3728_372844

theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) : 
  let circumference := π * r
  let base_radius := circumference / (2 * π)
  let slant_height := r
  let cone_height := Real.sqrt (slant_height^2 - base_radius^2)
  let volume := (1/3) * π * base_radius^2 * cone_height
  volume = 9 * π * Real.sqrt 3 := by
  sorry

end cone_volume_from_half_sector_l3728_372844


namespace buckingham_palace_visitors_l3728_372871

theorem buckingham_palace_visitors :
  let current_day_visitors : ℕ := 132
  let previous_day_visitors : ℕ := 274
  let total_visitors : ℕ := 406
  let days_considered : ℕ := 2
  current_day_visitors + previous_day_visitors = total_visitors →
  days_considered = 2 :=
by
  sorry

end buckingham_palace_visitors_l3728_372871


namespace parabola_focus_l3728_372814

/-- The parabola is defined by the equation x^2 = 20y -/
def parabola (x y : ℝ) : Prop := x^2 = 20 * y

/-- The focus of a parabola with equation x^2 = 4py has coordinates (0, p) -/
def is_focus (x y p : ℝ) : Prop := x = 0 ∧ y = p

/-- Theorem: The focus of the parabola x^2 = 20y has coordinates (0, 5) -/
theorem parabola_focus :
  ∃ (x y : ℝ), parabola x y ∧ is_focus x y 5 :=
sorry

end parabola_focus_l3728_372814


namespace tile_problem_l3728_372863

theorem tile_problem (n : ℕ) (total_tiles : ℕ) : 
  (total_tiles = n^2 + 64) ∧ (total_tiles = (n+1)^2 - 25) → total_tiles = 2000 := by
sorry

end tile_problem_l3728_372863


namespace yadav_clothes_transport_expenditure_l3728_372892

/-- Represents Mr Yadav's monthly finances --/
structure YadavFinances where
  salary : ℝ
  consumable_percentage : ℝ
  clothes_transport_percentage : ℝ
  yearly_savings : ℝ

/-- Calculates the monthly amount spent on clothes and transport --/
def monthly_clothes_transport (y : YadavFinances) : ℝ :=
  y.salary * (1 - y.consumable_percentage) * y.clothes_transport_percentage

/-- Theorem stating the amount spent on clothes and transport --/
theorem yadav_clothes_transport_expenditure (y : YadavFinances) 
  (h1 : y.consumable_percentage = 0.6)
  (h2 : y.clothes_transport_percentage = 0.5)
  (h3 : y.yearly_savings = 19008)
  (h4 : y.yearly_savings = 12 * (y.salary * (1 - y.consumable_percentage) * (1 - y.clothes_transport_percentage))) :
  monthly_clothes_transport y = 1584 := by
  sorry

#eval monthly_clothes_transport { salary := 7920, consumable_percentage := 0.6, clothes_transport_percentage := 0.5, yearly_savings := 19008 }

end yadav_clothes_transport_expenditure_l3728_372892


namespace line_through_hyperbola_points_l3728_372818

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2/4 - y^2/2 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := 2*x + 8*y + 7 = 0

/-- Theorem stating that the line passing through two points on the given hyperbola
    with midpoint (1/2, -1) has the equation 2x + 8y + 7 = 0 -/
theorem line_through_hyperbola_points :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola x₁ y₁ ∧
    hyperbola x₂ y₂ ∧
    (x₁ + x₂)/2 = 1/2 ∧
    (y₁ + y₂)/2 = -1 ∧
    (∀ (x y : ℝ), (x - x₁)*(y₂ - y₁) = (y - y₁)*(x₂ - x₁) ↔ line x y) :=
by sorry

end line_through_hyperbola_points_l3728_372818


namespace regression_lines_intersect_l3728_372839

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The point (s, t) represents the average values of x and y -/
structure AveragePoint where
  s : ℝ
  t : ℝ

/-- Theorem: Two regression lines with the same average point intersect at that point -/
theorem regression_lines_intersect (t₁ t₂ : RegressionLine) (avg : AveragePoint) :
  (avg.s * t₁.slope + t₁.intercept = avg.t) →
  (avg.s * t₂.slope + t₂.intercept = avg.t) →
  ∃ (x y : ℝ), x = avg.s ∧ y = avg.t ∧ 
    y = x * t₁.slope + t₁.intercept ∧
    y = x * t₂.slope + t₂.intercept := by
  sorry


end regression_lines_intersect_l3728_372839


namespace solve_equation_l3728_372838

theorem solve_equation (x y : ℝ) : y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end solve_equation_l3728_372838


namespace mean_height_of_players_l3728_372894

def heights : List ℕ := [145, 149, 151, 151, 157, 158, 163, 163, 164, 167, 168, 169, 170, 175]

def total_players : ℕ := heights.length

def sum_of_heights : ℕ := heights.sum

theorem mean_height_of_players :
  (sum_of_heights : ℚ) / (total_players : ℚ) = 160.714 := by sorry

end mean_height_of_players_l3728_372894


namespace zoey_holidays_l3728_372898

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of holidays Zoey takes per month -/
def holidays_per_month : ℕ := 2

/-- The total number of holidays Zoey takes in a year -/
def total_holidays : ℕ := months_in_year * holidays_per_month

theorem zoey_holidays : total_holidays = 24 := by
  sorry

end zoey_holidays_l3728_372898


namespace a_eq_b_sufficient_not_necessary_for_a_sq_eq_b_sq_l3728_372862

theorem a_eq_b_sufficient_not_necessary_for_a_sq_eq_b_sq :
  (∃ a b : ℝ, a = b → a^2 = b^2) ∧
  (∃ a b : ℝ, a^2 = b^2 ∧ a ≠ b) :=
by sorry

end a_eq_b_sufficient_not_necessary_for_a_sq_eq_b_sq_l3728_372862


namespace intersection_of_A_and_B_l3728_372895

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end intersection_of_A_and_B_l3728_372895


namespace C₁_C₂_intersections_l3728_372828

/-- The polar coordinate equation of curve C₁ -/
def C₁_polar (ρ θ : ℝ) : Prop := ρ - 2 * Real.cos θ = 0

/-- The rectangular coordinate equation of curve C₁ -/
def C₁_rect (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

/-- The equation of curve C₂ -/
def C₂ (x y m : ℝ) : Prop := 2*x - y - 2*m - 1 = 0

/-- The condition for C₁ and C₂ to have two distinct intersection points -/
def has_two_intersections (m : ℝ) : Prop :=
  (1 - Real.sqrt 5) / 2 < m ∧ m < (1 + Real.sqrt 5) / 2

theorem C₁_C₂_intersections (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    C₁_rect x₁ y₁ ∧ C₁_rect x₂ y₂ ∧
    C₂ x₁ y₁ m ∧ C₂ x₂ y₂ m) ↔
  has_two_intersections m :=
sorry

end C₁_C₂_intersections_l3728_372828


namespace reciprocal_of_negative_two_l3728_372891

theorem reciprocal_of_negative_two :
  ((-2 : ℝ)⁻¹ : ℝ) = -1/2 := by sorry

end reciprocal_of_negative_two_l3728_372891


namespace sine_fraction_equals_three_l3728_372840

theorem sine_fraction_equals_three (d : ℝ) (h : d = π / 7) :
  (3 * Real.sin (2 * d) * Real.sin (4 * d) * Real.sin (6 * d) * Real.sin (8 * d) * Real.sin (10 * d)) /
  (Real.sin d * Real.sin (2 * d) * Real.sin (3 * d) * Real.sin (4 * d) * Real.sin (5 * d)) = 3 := by
  sorry

end sine_fraction_equals_three_l3728_372840


namespace line_segments_proportion_l3728_372896

theorem line_segments_proportion : 
  let a : ℝ := 1
  let b : ℝ := 2
  let c : ℝ := 2
  let d : ℝ := 4
  (a / b = c / d) := by sorry

end line_segments_proportion_l3728_372896


namespace xyz_sum_equals_96_l3728_372813

theorem xyz_sum_equals_96 
  (x y z : ℝ) 
  (hpos_x : x > 0) 
  (hpos_y : y > 0) 
  (hpos_z : z > 0)
  (eq1 : x^2 + x*y + y^2 = 108)
  (eq2 : y^2 + y*z + z^2 = 64)
  (eq3 : z^2 + x*z + x^2 = 172) : 
  x*y + y*z + x*z = 96 := by
sorry

end xyz_sum_equals_96_l3728_372813


namespace intersection_implies_difference_l3728_372808

def set_A (a : ℝ) : Set (ℝ × ℝ) := {p | p.2 = a * p.1 + 6}
def set_B : Set (ℝ × ℝ) := {p | p.2 = 5 * p.1 - 3}

theorem intersection_implies_difference (a b : ℝ) :
  (1, b) ∈ set_A a ∩ set_B → a - b = -6 := by
  sorry

end intersection_implies_difference_l3728_372808


namespace inequality_proof_l3728_372815

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^2014 + b^2014 + c^2014 + a*b*c = 4) :
  (a^2013 + b^2013 - c)/c^2013 + (b^2013 + c^2013 - a)/a^2013 + (c^2013 + a^2013 - b)/b^2013 
  ≥ a^2012 + b^2012 + c^2012 := by
sorry

end inequality_proof_l3728_372815


namespace a_8_equals_16_l3728_372833

def sequence_property (a : ℕ+ → ℕ) : Prop :=
  ∀ p q : ℕ+, a (p + q) = a p * a q

theorem a_8_equals_16 (a : ℕ+ → ℕ) (h1 : sequence_property a) (h2 : a 2 = 2) :
  a 8 = 16 := by
  sorry

end a_8_equals_16_l3728_372833


namespace banana_groups_indeterminate_l3728_372849

theorem banana_groups_indeterminate 
  (total_bananas : ℕ) 
  (total_oranges : ℕ) 
  (orange_groups : ℕ) 
  (oranges_per_group : ℕ) 
  (h1 : total_bananas = 142) 
  (h2 : total_oranges = 356) 
  (h3 : orange_groups = 178) 
  (h4 : oranges_per_group = 2) 
  (h5 : total_oranges = orange_groups * oranges_per_group) : 
  ∀ (banana_groups : ℕ), ¬ (∃ (bananas_per_group : ℕ), total_bananas = banana_groups * bananas_per_group) :=
by sorry

end banana_groups_indeterminate_l3728_372849


namespace student_marks_difference_l3728_372860

/-- Given a student's marks in physics, chemistry, and mathematics,
    prove that the total marks exceed the physics marks by 140,
    given that the average of chemistry and mathematics marks is 70. -/
theorem student_marks_difference 
  (P C M : ℕ)  -- Marks in Physics, Chemistry, and Mathematics
  (h_avg : (C + M) / 2 = 70)  -- Average of Chemistry and Mathematics is 70
  : (P + C + M) - P = 140 := by
  sorry

end student_marks_difference_l3728_372860


namespace basketball_volleyball_problem_l3728_372859

/-- Given the conditions of the basketball and volleyball purchase problem,
    prove the prices of the balls and the minimum total cost. -/
theorem basketball_volleyball_problem
  (basketball_price volleyball_price : ℕ)
  (total_balls min_cost : ℕ) :
  (3 * basketball_price + volleyball_price = 360) →
  (5 * basketball_price + 3 * volleyball_price = 680) →
  (total_balls = 100) →
  (∀ x y, x + y = total_balls → x ≥ 3 * y → 
    basketball_price * x + volleyball_price * y ≥ min_cost) →
  (basketball_price = 100 ∧ 
   volleyball_price = 60 ∧
   min_cost = 9000) :=
by sorry

end basketball_volleyball_problem_l3728_372859


namespace identify_participants_with_2k_minus_3_questions_l3728_372826

/-- Represents the type of participant -/
inductive Participant
| Chemist
| Alchemist

/-- Represents the state of the identification process -/
structure IdentificationState where
  participants : Nat
  chemists : Nat
  alchemists : Nat
  questions_asked : Nat

/-- The main theorem stating that 2k-3 questions are sufficient -/
theorem identify_participants_with_2k_minus_3_questions 
  (k : Nat) 
  (h : k > 0) 
  (more_chemists : ∃ (c a : Nat), c > a ∧ c + a = k) :
  ∃ (strategy : IdentificationState → Participant), 
    (∀ (state : IdentificationState), 
      state.participants = k → 
      state.chemists > state.alchemists → 
      state.questions_asked ≤ 2 * k - 3 → 
      (∀ p, strategy state = p → 
        (p = Participant.Chemist → state.chemists > 0) ∧ 
        (p = Participant.Alchemist → state.alchemists > 0))) :=
sorry

end identify_participants_with_2k_minus_3_questions_l3728_372826


namespace committee_probability_grammar_club_committee_probability_l3728_372832

/-- The probability of selecting a committee with at least one boy and one girl -/
theorem committee_probability (total : ℕ) (boys : ℕ) (girls : ℕ) (committee_size : ℕ) :
  total = boys + girls →
  total = 25 →
  boys = 15 →
  girls = 10 →
  committee_size = 5 →
  (Nat.choose total committee_size - (Nat.choose boys committee_size + Nat.choose girls committee_size)) / 
  Nat.choose total committee_size = 195 / 208 := by
  sorry

/-- Main theorem stating the probability for the specific case -/
theorem grammar_club_committee_probability :
  (Nat.choose 25 5 - (Nat.choose 15 5 + Nat.choose 10 5)) / Nat.choose 25 5 = 195 / 208 := by
  sorry

end committee_probability_grammar_club_committee_probability_l3728_372832


namespace part_i_part_ii_l3728_372855

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := Real.log x + 1 + 2 / x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 1 / x - 2 / (x^2)

-- Theorem for part I
theorem part_i :
  (∃ a : ℝ, a > 0 ∧ 
    (∀ x : ℝ, x > 0 → (f x - f a = -1/a * (x - a)) → (x = 0 → f x = 4)) ∧
    (∀ x : ℝ, x ∈ Set.Ioo 0 2 → f' x < 0) ∧
    (∀ x : ℝ, x > 2 → f' x > 0)) :=
sorry

-- Theorem for part II
theorem part_ii :
  (∃ k : ℤ, k = 7 ∧
    (∀ x : ℝ, x > 1 → 2 * f x > k * (1 - 1/x)) ∧
    (∀ m : ℤ, m > k → ∃ x : ℝ, x > 1 ∧ 2 * f x ≤ m * (1 - 1/x))) :=
sorry

end part_i_part_ii_l3728_372855


namespace value_of_x_l3728_372835

theorem value_of_x :
  ∀ (x a b c d : ℤ),
    x = a + 7 →
    a = b + 9 →
    b = c + 15 →
    c = d + 25 →
    d = 60 →
    x = 116 := by
  sorry

end value_of_x_l3728_372835


namespace sum_of_absolute_coefficients_l3728_372822

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 3^5 := by
  sorry

end sum_of_absolute_coefficients_l3728_372822


namespace apples_left_after_pie_l3728_372817

theorem apples_left_after_pie (initial_apples : Real) (anita_contribution : Real) (pie_requirement : Real) :
  initial_apples = 10.0 →
  anita_contribution = 5.0 →
  pie_requirement = 4.0 →
  initial_apples + anita_contribution - pie_requirement = 11.0 := by
  sorry

end apples_left_after_pie_l3728_372817


namespace gcd_problem_l3728_372847

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, a = 2345 * k) :
  Int.gcd (a^2 + 10*a + 25) (a + 5) = a + 5 := by
  sorry

end gcd_problem_l3728_372847


namespace policeman_can_catch_gangster_l3728_372865

/-- Represents a point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square --/
structure Square where
  sideLength : ℝ
  center : Point
  
/-- Represents the policeman --/
structure Policeman where
  position : Point
  speed : ℝ

/-- Represents the gangster --/
structure Gangster where
  position : Point
  speed : ℝ

/-- A function to check if a point is on the edge of a square --/
def isOnEdge (p : Point) (s : Square) : Prop :=
  (p.x = s.center.x - s.sideLength / 2 ∨ p.x = s.center.x + s.sideLength / 2) ∨
  (p.y = s.center.y - s.sideLength / 2 ∨ p.y = s.center.y + s.sideLength / 2)

/-- The main theorem --/
theorem policeman_can_catch_gangster 
  (s : Square) 
  (p : Policeman) 
  (g : Gangster) 
  (h1 : s.sideLength > 0)
  (h2 : p.position = s.center)
  (h3 : isOnEdge g.position s)
  (h4 : p.speed = g.speed / 2)
  (h5 : g.speed > 0) :
  ∃ (t : ℝ) (pFinal gFinal : Point), 
    t ≥ 0 ∧
    isOnEdge pFinal s ∧
    isOnEdge gFinal s ∧
    ∃ (edge : Set Point), 
      edge.Subset {p | isOnEdge p s} ∧
      pFinal ∈ edge ∧
      gFinal ∈ edge :=
by sorry

end policeman_can_catch_gangster_l3728_372865


namespace art_department_probability_l3728_372881

theorem art_department_probability : 
  let total_students : ℕ := 4
  let students_per_grade : ℕ := 2
  let selected_students : ℕ := 2
  let different_grade_selections : ℕ := students_per_grade * students_per_grade
  let total_selections : ℕ := Nat.choose total_students selected_students
  (different_grade_selections : ℚ) / total_selections = 2 / 3 := by
sorry

end art_department_probability_l3728_372881


namespace divisor_congruence_l3728_372890

theorem divisor_congruence (p n d : ℕ) : 
  Prime p → d ∣ ((n + 1)^p - n^p) → d ≡ 1 [MOD p] := by sorry

end divisor_congruence_l3728_372890


namespace exactly_two_balls_distribution_l3728_372806

-- Define the number of balls and boxes
def num_balls : ℕ := 5
def num_boxes : ℕ := 3

-- Define the function to calculate the number of ways to distribute balls
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  (n.choose 2) * k * (k ^ (n - 2))

-- Theorem statement
theorem exactly_two_balls_distribution :
  distribute_balls num_balls num_boxes = 810 :=
sorry

end exactly_two_balls_distribution_l3728_372806


namespace m_range_l3728_372824

def A : Set ℝ := {x | 1/2 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 1}

theorem m_range (m : ℝ) (h : A ∪ B m = A) : 1/2 ≤ m ∧ m ≤ 1 := by
  sorry

end m_range_l3728_372824


namespace expected_rainfall_is_19_25_l3728_372827

/-- Weather forecast for a single day -/
structure DailyForecast where
  prob_sun : ℝ
  prob_rain3 : ℝ
  prob_rain8 : ℝ
  sum_to_one : prob_sun + prob_rain3 + prob_rain8 = 1

/-- Calculate expected rainfall for a single day -/
def expected_daily_rainfall (f : DailyForecast) : ℝ :=
  f.prob_sun * 0 + f.prob_rain3 * 3 + f.prob_rain8 * 8

/-- Weather forecast for the week -/
def weekly_forecast : DailyForecast :=
  { prob_sun := 0.3
    prob_rain3 := 0.35
    prob_rain8 := 0.35
    sum_to_one := by norm_num }

/-- Number of days in the forecast -/
def num_days : ℕ := 5

/-- Expected total rainfall for the week -/
def expected_total_rainfall : ℝ :=
  (expected_daily_rainfall weekly_forecast) * num_days

theorem expected_rainfall_is_19_25 :
  expected_total_rainfall = 19.25 := by sorry

end expected_rainfall_is_19_25_l3728_372827


namespace triangle_equality_l3728_372816

theorem triangle_equality (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_equation : (c - b) / a + (a - c) / b + (b - a) / c = 0) :
  a = b ∨ b = c ∨ c = a :=
sorry

end triangle_equality_l3728_372816


namespace x_is_integer_l3728_372811

theorem x_is_integer (x : ℝ) (h1 : ∃ n : ℤ, x^3 - x = n) (h2 : ∃ m : ℤ, x^4 - x = m) : ∃ k : ℤ, x = k := by
  sorry

end x_is_integer_l3728_372811


namespace arithmetic_sequence_common_difference_l3728_372888

/-- An arithmetic sequence with sum of first 5 terms equal to 15 and second term equal to 5 has common difference -2 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ)  -- arithmetic sequence
  (S : ℕ → ℝ)  -- sum function
  (h1 : S 5 = 15)  -- sum of first 5 terms is 15
  (h2 : a 2 = 5)   -- second term is 5
  (h3 : ∀ n, S n = n * (a 1 + a n) / 2)  -- sum formula for arithmetic sequence
  (h4 : ∀ n, a (n + 1) = a n + d)  -- definition of arithmetic sequence
  : d = -2 := by
  sorry

end arithmetic_sequence_common_difference_l3728_372888


namespace complex_equation_solution_l3728_372836

theorem complex_equation_solution (a : ℝ) :
  (2 + a * Complex.I) / (1 + Real.sqrt 2 * Complex.I) = -(Real.sqrt 2) * Complex.I →
  a = Real.sqrt 2 := by
sorry

end complex_equation_solution_l3728_372836


namespace seven_boys_without_calculators_l3728_372889

/-- Represents the number of boys in Miss Parker's class who did not bring calculators. -/
def boys_without_calculators (total_students : ℕ) (boys_in_class : ℕ) (students_with_calculators : ℕ) (girls_with_calculators : ℕ) : ℕ :=
  boys_in_class - (students_with_calculators - girls_with_calculators)

/-- Theorem stating that 7 boys in Miss Parker's class did not bring calculators. -/
theorem seven_boys_without_calculators :
  boys_without_calculators 24 18 26 15 = 7 := by
  sorry

end seven_boys_without_calculators_l3728_372889


namespace marble_probability_difference_l3728_372829

/-- The number of red marbles in the box -/
def red : ℕ := 500

/-- The number of black marbles in the box -/
def black : ℕ := 700

/-- The number of blue marbles in the box -/
def blue : ℕ := 800

/-- The total number of marbles in the box -/
def total : ℕ := red + black + blue

/-- The probability of drawing two marbles of the same color -/
noncomputable def Ps : ℚ := 
  (red * (red - 1) + black * (black - 1) + blue * (blue - 1)) / (total * (total - 1))

/-- The probability of drawing two marbles of different colors -/
noncomputable def Pd : ℚ := 
  (red * black + red * blue + black * blue) * 2 / (total * (total - 1))

/-- Theorem stating that the absolute difference between Ps and Pd is 31/100 -/
theorem marble_probability_difference : |Ps - Pd| = 31 / 100 := by sorry

end marble_probability_difference_l3728_372829


namespace brown_eyed_brunettes_l3728_372883

theorem brown_eyed_brunettes (total : ℕ) (blondes : ℕ) (brunettes : ℕ) (blue_eyed : ℕ) (brown_eyed : ℕ) (blue_eyed_blondes : ℕ) :
  total = 60 →
  blondes + brunettes = total →
  blue_eyed + brown_eyed = total →
  brunettes = 35 →
  blue_eyed_blondes = 20 →
  brown_eyed = 22 →
  brown_eyed - (blondes - blue_eyed_blondes) = 17 :=
by sorry

end brown_eyed_brunettes_l3728_372883


namespace negation_of_universal_proposition_l3728_372854

-- Define a parallelogram
def Parallelogram : Type := sorry

-- Define the property of having equal diagonals
def has_equal_diagonals (p : Parallelogram) : Prop := sorry

-- Define the property of diagonals bisecting each other
def diagonals_bisect_each_other (p : Parallelogram) : Prop := sorry

-- State the theorem
theorem negation_of_universal_proposition :
  (¬ ∀ p : Parallelogram, has_equal_diagonals p ∧ diagonals_bisect_each_other p) ↔
  (∃ p : Parallelogram, ¬(has_equal_diagonals p ∧ diagonals_bisect_each_other p)) :=
by sorry

end negation_of_universal_proposition_l3728_372854


namespace a_eq_3_sufficient_not_necessary_l3728_372825

/-- Two lines in the plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be parallel -/
def parallel (l₁ l₂ : Line2D) : Prop :=
  l₁.a * l₂.b = l₂.a * l₁.b

/-- The first line: ax - 5y - 1 = 0 -/
def l₁ (a : ℝ) : Line2D :=
  { a := a, b := -5, c := -1 }

/-- The second line: 3x - (a+2)y + 4 = 0 -/
def l₂ (a : ℝ) : Line2D :=
  { a := 3, b := -(a+2), c := 4 }

/-- The statement that a = 3 is a sufficient but not necessary condition for l₁ ∥ l₂ -/
theorem a_eq_3_sufficient_not_necessary :
  (∃ (a : ℝ), a ≠ 3 ∧ parallel (l₁ a) (l₂ a)) ∧
  (parallel (l₁ 3) (l₂ 3)) := by
  sorry

end a_eq_3_sufficient_not_necessary_l3728_372825


namespace fourth_grade_classroom_count_l3728_372850

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 22

/-- The number of pet hamsters in each classroom -/
def hamsters_per_classroom : ℕ := 3

/-- The number of pet guinea pigs in each classroom -/
def guinea_pigs_per_classroom : ℕ := 1

/-- The difference between the total number of students and the total number of pets -/
def student_pet_difference : ℕ := 90

theorem fourth_grade_classroom_count :
  num_classrooms * students_per_classroom - 
  num_classrooms * (hamsters_per_classroom + guinea_pigs_per_classroom) = 
  student_pet_difference := by sorry

end fourth_grade_classroom_count_l3728_372850


namespace expression_equality_l3728_372873

theorem expression_equality (a x : ℝ) (h : a^(2*x) = Real.sqrt 2 - 1) :
  (a^(3*x) + a^(-3*x)) / (a^x + a^(-x)) = 2 * Real.sqrt 2 - 1 := by
  sorry

end expression_equality_l3728_372873


namespace equation_solution_l3728_372852

theorem equation_solution :
  ∀ x : ℚ, (x + 4) / (x - 3) = (x - 2) / (x + 2) ↔ x = -2/11 := by
  sorry

end equation_solution_l3728_372852


namespace alice_shoe_probability_l3728_372899

/-- Represents the number of pairs for each color of shoes --/
structure ShoePairs where
  black : Nat
  brown : Nat
  white : Nat
  gray : Nat

/-- Calculates the probability of picking two shoes of the same color
    with one being left and one being right --/
def probability_same_color_different_feet (pairs : ShoePairs) : Rat :=
  let total_shoes := 2 * (pairs.black + pairs.brown + pairs.white + pairs.gray)
  let prob_black := (2 * pairs.black) * pairs.black / (total_shoes * (total_shoes - 1))
  let prob_brown := (2 * pairs.brown) * pairs.brown / (total_shoes * (total_shoes - 1))
  let prob_white := (2 * pairs.white) * pairs.white / (total_shoes * (total_shoes - 1))
  let prob_gray := (2 * pairs.gray) * pairs.gray / (total_shoes * (total_shoes - 1))
  prob_black + prob_brown + prob_white + prob_gray

theorem alice_shoe_probability :
  probability_same_color_different_feet ⟨7, 4, 3, 1⟩ = 25 / 145 := by
  sorry

end alice_shoe_probability_l3728_372899


namespace rectangular_box_volume_l3728_372801

theorem rectangular_box_volume (x : ℕ) (h : x > 0) :
  let volume := 10 * x^3
  (volume = 60 ∨ volume = 80 ∨ volume = 100 ∨ volume = 120 ∨ volume = 200) →
  volume = 80 :=
by sorry

end rectangular_box_volume_l3728_372801


namespace polynomial_factorization_l3728_372872

theorem polynomial_factorization (x : ℝ) : x^4 + 256 = (x^2 - 8*x + 16) * (x^2 + 8*x + 16) := by
  sorry

end polynomial_factorization_l3728_372872


namespace regression_lines_common_point_l3728_372864

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The average point of a dataset -/
structure AveragePoint where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a regression line -/
def pointOnLine (l : RegressionLine) (p : AveragePoint) : Prop :=
  p.y = l.slope * p.x + l.intercept

theorem regression_lines_common_point 
  (l₁ l₂ : RegressionLine) (avg : AveragePoint) : 
  pointOnLine l₁ avg ∧ pointOnLine l₂ avg := by
  sorry

#check regression_lines_common_point

end regression_lines_common_point_l3728_372864


namespace bd_production_l3728_372823

/-- Represents the total production of all workshops -/
def total_production : ℕ := 2800

/-- Represents the total number of units sampled for quality inspection -/
def total_sampled : ℕ := 140

/-- Represents the number of units sampled from workshops A and C combined -/
def ac_sampled : ℕ := 60

/-- Theorem stating that the total production from workshops B and D is 1600 units -/
theorem bd_production : 
  total_production - (ac_sampled * (total_production / total_sampled)) = 1600 := by
  sorry

end bd_production_l3728_372823


namespace opposite_numbers_equation_product_l3728_372810

theorem opposite_numbers_equation_product : ∀ x : ℤ, 
  (3 * x - 2 * (-x) = 30) → (x * (-x) = -36) := by
  sorry

end opposite_numbers_equation_product_l3728_372810


namespace arithmetic_sequence_third_term_l3728_372879

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_third_term (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 1 + a 2 + a 3 + a 4 + a 5 = 20 →
  a 3 = 4 := by
sorry

end arithmetic_sequence_third_term_l3728_372879


namespace monomial_difference_l3728_372851

theorem monomial_difference (m n : ℤ) : 
  (∃ (a : ℝ) (p q : ℤ), ∀ (x y : ℝ), 9 * x^(m-2) * y^2 - (-3 * x^3 * y^(n+1)) = a * x^p * y^q) → 
  n - m = -4 :=
by sorry

end monomial_difference_l3728_372851


namespace polynomial_multiplication_l3728_372893

variables (x y : ℝ)

theorem polynomial_multiplication :
  (2 * x^25 - 5 * x^8 + 2 * x * y^3 - 9) * (3 * x^7) =
  6 * x^32 - 15 * x^15 + 6 * x^8 * y^3 - 27 * x^7 := by sorry

end polynomial_multiplication_l3728_372893


namespace division_simplification_l3728_372869

theorem division_simplification (x y : ℝ) (h : y ≠ 0) :
  (-2 * x^2 * y + y) / y = -2 * x^2 + 1 := by
  sorry

end division_simplification_l3728_372869


namespace billion_yuan_scientific_notation_l3728_372841

/-- Represents the value of 209.6 billion yuan in standard form -/
def billion_yuan : ℝ := 209.6 * (10^9)

/-- Represents the scientific notation of 209.6 billion yuan -/
def scientific_notation : ℝ := 2.096 * (10^10)

/-- Theorem stating that the standard form equals the scientific notation -/
theorem billion_yuan_scientific_notation : billion_yuan = scientific_notation := by
  sorry

end billion_yuan_scientific_notation_l3728_372841


namespace product_nonnegative_proof_l3728_372842

theorem product_nonnegative_proof :
  -- Original proposition
  (∀ x y : ℝ, x ≥ 0 ∧ y ≥ 0 → x * y ≥ 0) ∧
  -- Contrapositive is true
  (∀ x y : ℝ, x * y < 0 → x < 0 ∨ y < 0) ∧
  -- Converse is false
  ¬(∀ x y : ℝ, x * y ≥ 0 → x ≥ 0 ∧ y ≥ 0) ∧
  -- Negation is false
  ¬(∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ x * y < 0) := by
sorry

end product_nonnegative_proof_l3728_372842


namespace sequence_is_arithmetic_progression_l3728_372821

theorem sequence_is_arithmetic_progression 
  (a : ℕ → ℝ) 
  (h : ∀ m n : ℕ, |a m + a n - a (m + n)| ≤ 1 / (m + n : ℝ)) : 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d := by
  sorry

end sequence_is_arithmetic_progression_l3728_372821


namespace parabola_and_line_equations_l3728_372858

/-- Parabola with focus F and point (3,m) on it -/
structure Parabola where
  p : ℝ
  m : ℝ
  h_p_pos : p > 0
  h_on_parabola : m^2 = 2 * p * 3
  h_distance_to_focus : Real.sqrt ((3 - p/2)^2 + m^2) = 4

/-- Line passing through focus F and intersecting parabola at A and B -/
structure IntersectingLine (E : Parabola) where
  k : ℝ  -- slope of the line
  h_midpoint : ∃ (y_A y_B : ℝ), y_A^2 = 4 * (k * y_A + 1) ∧
                                 y_B^2 = 4 * (k * y_B + 1) ∧
                                 (y_A + y_B) / 2 = -1

/-- Main theorem -/
theorem parabola_and_line_equations (E : Parabola) (l : IntersectingLine E) :
  (E.p = 2 ∧ ∀ x y, y^2 = 2 * E.p * x ↔ y^2 = 4 * x) ∧
  (l.k = -1/2 ∧ ∀ x y, y = l.k * (x - 1) ↔ 2 * x + y - 2 = 0) := by
  sorry

end parabola_and_line_equations_l3728_372858


namespace square_difference_301_299_l3728_372831

theorem square_difference_301_299 : 301^2 - 299^2 = 1200 := by
  sorry

end square_difference_301_299_l3728_372831


namespace translator_selection_ways_l3728_372878

-- Define the staff members
def total_staff : ℕ := 7
def english_only : ℕ := 3
def japanese_only : ℕ := 2
def bilingual : ℕ := 2

-- Define the required translators
def english_translators : ℕ := 3
def japanese_translators : ℕ := 2

-- Define the function to calculate the number of ways to select translators
def select_translators : ℕ := 27

-- Theorem statement
theorem translator_selection_ways :
  select_translators = 27 :=
sorry

end translator_selection_ways_l3728_372878


namespace number_of_blue_balls_l3728_372877

/-- The number of blue balls originally in the box -/
def B : ℕ := sorry

/-- The number of red balls originally in the box -/
def R : ℕ := sorry

/-- Theorem stating the number of blue balls originally in the box -/
theorem number_of_blue_balls : 
  B = R + 17 ∧ 
  (B + 57) + (R + 18) - 44 = 502 → 
  B = 244 := by sorry

end number_of_blue_balls_l3728_372877


namespace perpendicular_bisector_focus_condition_l3728_372880

/-- A point on a parabola -/
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : y = 2 * x^2

/-- The perpendicular bisector of two points passes through the focus of the parabola -/
def perpendicular_bisector_passes_through_focus (A B : PointOnParabola) : Prop :=
  let midpoint := ((A.x + B.x) / 2, (A.y + B.y) / 2)
  let slope := if A.x = B.x then 0 else (B.y - A.y) / (B.x - A.x)
  let perp_slope := if slope = 0 then 0 else -1 / slope
  ∃ (t : ℝ), midpoint.1 + t * perp_slope = 0 ∧ midpoint.2 + t = 1/8

/-- Theorem: The perpendicular bisector passes through the focus iff x₁ + x₂ = 0 -/
theorem perpendicular_bisector_focus_condition (A B : PointOnParabola) :
  perpendicular_bisector_passes_through_focus A B ↔ A.x + B.x = 0 := by
  sorry

/-- The equation of the perpendicular bisector when x₁ = 1 and x₂ = -3 -/
def perpendicular_bisector_equation (A B : PointOnParabola) (h₁ : A.x = 1) (h₂ : B.x = -3) : 
  ∃ (a b c : ℝ), a * A.x + b * A.y + c = 0 ∧ a * B.x + b * B.y + c = 0 ∧ (a, b, c) = (1, -4, 41) := by
  sorry

end perpendicular_bisector_focus_condition_l3728_372880


namespace upper_bound_necessary_not_sufficient_l3728_372843

variable {α : Type*} [PartialOrder α]
variable (I : Set α) (f : α → ℝ) (M : ℝ)

def is_upper_bound (f : α → ℝ) (M : ℝ) (I : Set α) : Prop :=
  ∀ x ∈ I, f x ≤ M

def is_maximum (f : α → ℝ) (M : ℝ) (I : Set α) : Prop :=
  (is_upper_bound f M I) ∧ (∃ x ∈ I, f x = M)

theorem upper_bound_necessary_not_sufficient :
  (is_upper_bound f M I → is_maximum f M I) ∧
  ¬(is_maximum f M I → is_upper_bound f M I) :=
sorry

end upper_bound_necessary_not_sufficient_l3728_372843


namespace power_function_through_point_l3728_372846

/-- A power function passing through (2, 1/8) has exponent -3 -/
theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, x > 0 → f x = x^α) →  -- f is a power function for positive x
  f 2 = 1/8 →                 -- f passes through (2, 1/8)
  α = -3 := by
sorry


end power_function_through_point_l3728_372846


namespace troy_beef_purchase_l3728_372804

/-- Represents the problem of determining the amount of beef Troy buys -/
theorem troy_beef_purchase 
  (veg_pounds : ℝ) 
  (veg_price : ℝ) 
  (beef_price_multiplier : ℝ) 
  (total_cost : ℝ) 
  (h1 : veg_pounds = 6)
  (h2 : veg_price = 2)
  (h3 : beef_price_multiplier = 3)
  (h4 : total_cost = 36) :
  ∃ (beef_pounds : ℝ), 
    beef_pounds * (veg_price * beef_price_multiplier) + veg_pounds * veg_price = total_cost ∧ 
    beef_pounds = 4 := by
  sorry

end troy_beef_purchase_l3728_372804


namespace simplify_fraction_l3728_372897

theorem simplify_fraction : (45 : ℚ) / 75 = 3 / 5 := by sorry

end simplify_fraction_l3728_372897


namespace decimal_point_problem_l3728_372848

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 / x) : x = Real.sqrt 30 / 100 := by
  sorry

end decimal_point_problem_l3728_372848


namespace sqrt_14400_l3728_372875

theorem sqrt_14400 : Real.sqrt 14400 = 120 := by
  sorry

end sqrt_14400_l3728_372875


namespace factorial_division_equality_l3728_372876

theorem factorial_division_equality : (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end factorial_division_equality_l3728_372876


namespace complex_equation_solution_l3728_372853

theorem complex_equation_solution :
  ∀ z : ℂ, (1 + Complex.I) * z = -1 + 5 * Complex.I → z = 2 + 3 * Complex.I := by
  sorry

end complex_equation_solution_l3728_372853


namespace restaurant_cooks_count_l3728_372884

theorem restaurant_cooks_count :
  ∀ (C W : ℕ),
  (C : ℚ) / W = 3 / 10 →
  (C : ℚ) / (W + 12) = 3 / 14 →
  C = 9 :=
by
  sorry

end restaurant_cooks_count_l3728_372884


namespace sum_of_y_coords_on_y_axis_l3728_372809

-- Define the circle
def circle_center : ℝ × ℝ := (-6, 2)
def circle_radius : ℝ := 10

-- Define a point on the circle
def point_on_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 = circle_radius^2

-- Define a point on the y-axis
def point_on_y_axis (p : ℝ × ℝ) : Prop :=
  p.1 = 0

-- Theorem statement
theorem sum_of_y_coords_on_y_axis :
  ∃ (p1 p2 : ℝ × ℝ),
    point_on_circle p1 ∧ point_on_y_axis p1 ∧
    point_on_circle p2 ∧ point_on_y_axis p2 ∧
    p1 ≠ p2 ∧
    p1.2 + p2.2 = 4 := by
  sorry

end sum_of_y_coords_on_y_axis_l3728_372809


namespace coefficient_x_squared_in_product_l3728_372861

theorem coefficient_x_squared_in_product : 
  let p₁ : Polynomial ℤ := 2 * X^3 + 4 * X^2 + 5 * X - 3
  let p₂ : Polynomial ℤ := 6 * X^2 - 5 * X + 1
  (p₁ * p₂).coeff 2 = -39 := by
  sorry

end coefficient_x_squared_in_product_l3728_372861


namespace tangent_line_curve1_tangent_lines_curve2_l3728_372868

-- Define the curves
def curve1 (x : ℝ) : ℝ := x^3 + x^2 + 1
def curve2 (x : ℝ) : ℝ := x^2

-- Define the points
def P1 : ℝ × ℝ := (-1, 1)
def P2 : ℝ × ℝ := (3, 5)

-- Theorem for the first curve
theorem tangent_line_curve1 :
  ∃ (k m : ℝ), k * P1.1 + m * P1.2 + 2 = 0 ∧
  ∀ x y, y = curve1 x → k * x + m * y + 2 = 0 → x = P1.1 ∧ y = P1.2 :=
sorry

-- Theorem for the second curve
theorem tangent_lines_curve2 :
  ∃ (k1 m1 k2 m2 : ℝ),
  (k1 * P2.1 + m1 * P2.2 + 1 = 0 ∧ k2 * P2.1 + m2 * P2.2 + 25 = 0) ∧
  (∀ x y, y = curve2 x → (k1 * x + m1 * y + 1 = 0 ∨ k2 * x + m2 * y + 25 = 0) → x = P2.1 ∧ y = P2.2) ∧
  (k1 = 2 ∧ m1 = -1) ∧ (k2 = 10 ∧ m2 = -1) :=
sorry

end tangent_line_curve1_tangent_lines_curve2_l3728_372868


namespace solve_equation_l3728_372812

theorem solve_equation (r : ℚ) : 4 * (r - 10) = 3 * (3 - 3 * r) + 9 → r = 58 / 13 := by
  sorry

end solve_equation_l3728_372812
