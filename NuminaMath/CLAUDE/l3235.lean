import Mathlib

namespace package_volume_calculation_l3235_323529

/-- Proves that the total volume needed to package the collection is 3,060,000 cubic inches -/
theorem package_volume_calculation (box_length box_width box_height : ℕ) 
  (cost_per_box total_cost : ℚ) : 
  box_length = 20 →
  box_width = 20 →
  box_height = 15 →
  cost_per_box = 7/10 →
  total_cost = 357 →
  (box_length * box_width * box_height) * (total_cost / cost_per_box) = 3060000 :=
by sorry

end package_volume_calculation_l3235_323529


namespace two_numbers_sum_product_l3235_323535

theorem two_numbers_sum_product (S P : ℝ) :
  ∃ (x y : ℝ), x + y = S ∧ x * y = P →
  ((x = (S + Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4*P)) / 2) ∨
   (x = (S - Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4*P)) / 2)) :=
by sorry

end two_numbers_sum_product_l3235_323535


namespace ascending_four_digit_difference_l3235_323505

/-- Represents a four-digit number where each subsequent digit is 1 greater than the previous one -/
structure AscendingFourDigitNumber where
  first_digit : ℕ
  constraint : first_digit ≤ 6

/-- Calculates the value of the four-digit number -/
def value (n : AscendingFourDigitNumber) : ℕ :=
  1000 * n.first_digit + 100 * (n.first_digit + 1) + 10 * (n.first_digit + 2) + (n.first_digit + 3)

/-- Calculates the value of the reversed four-digit number -/
def reverse_value (n : AscendingFourDigitNumber) : ℕ :=
  1000 * (n.first_digit + 3) + 100 * (n.first_digit + 2) + 10 * (n.first_digit + 1) + n.first_digit

/-- The main theorem stating that the difference between the reversed number and the original number is always 3087 -/
theorem ascending_four_digit_difference (n : AscendingFourDigitNumber) :
  reverse_value n - value n = 3087 := by
  sorry

end ascending_four_digit_difference_l3235_323505


namespace disk_contains_origin_l3235_323582

theorem disk_contains_origin (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ a b c : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₄ > 0) (h₃ : y₁ > 0) (h₄ : y₂ > 0)
  (h₅ : x₂ < 0) (h₆ : x₃ < 0) (h₇ : y₃ < 0) (h₈ : y₄ < 0)
  (h₉ : (x₁ - a)^2 + (y₁ - b)^2 ≤ c^2)
  (h₁₀ : (x₂ - a)^2 + (y₂ - b)^2 ≤ c^2)
  (h₁₁ : (x₃ - a)^2 + (y₃ - b)^2 ≤ c^2)
  (h₁₂ : (x₄ - a)^2 + (y₄ - b)^2 ≤ c^2) :
  a^2 + b^2 ≤ c^2 := by
sorry

end disk_contains_origin_l3235_323582


namespace complement_of_A_in_U_l3235_323552

def U : Set ℤ := {x | x^2 < 9}
def A : Set ℤ := {-2, 2}

theorem complement_of_A_in_U :
  U \ A = {-1, 0, 1} := by sorry

end complement_of_A_in_U_l3235_323552


namespace power_sum_equality_l3235_323585

theorem power_sum_equality : 3^(3+4+5) + (3^3 + 3^4 + 3^5) = 531792 := by
  sorry

end power_sum_equality_l3235_323585


namespace vector_subtraction_l3235_323528

/-- Given two vectors a and b in ℝ², prove that their difference is equal to a specific vector. -/
theorem vector_subtraction (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (3, 1)) :
  b - a = (2, -1) := by sorry

end vector_subtraction_l3235_323528


namespace lcm_of_72_108_126_156_l3235_323510

theorem lcm_of_72_108_126_156 : Nat.lcm 72 (Nat.lcm 108 (Nat.lcm 126 156)) = 19656 := by
  sorry

end lcm_of_72_108_126_156_l3235_323510


namespace tom_bought_ten_candies_l3235_323500

/-- Calculates the number of candy pieces Tom bought -/
def candy_bought (initial : ℕ) (from_friend : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial + from_friend)

/-- Theorem stating that Tom bought 10 pieces of candy -/
theorem tom_bought_ten_candies : candy_bought 2 7 19 = 10 := by
  sorry

end tom_bought_ten_candies_l3235_323500


namespace average_of_five_numbers_l3235_323541

theorem average_of_five_numbers (numbers : Fin 5 → ℝ) 
  (sum_of_three : ∃ (i j k : Fin 5), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ numbers i + numbers j + numbers k = 48)
  (avg_of_two : ∃ (l m : Fin 5), l ≠ m ∧ (numbers l + numbers m) / 2 = 26) :
  (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4) / 5 = 20 := by
sorry

end average_of_five_numbers_l3235_323541


namespace new_car_sticker_price_l3235_323508

/-- Calculates the sticker price of a new car based on given conditions --/
theorem new_car_sticker_price 
  (old_car_value : ℝ)
  (old_car_sale_percentage : ℝ)
  (new_car_purchase_percentage : ℝ)
  (out_of_pocket : ℝ)
  (h1 : old_car_value = 20000)
  (h2 : old_car_sale_percentage = 0.8)
  (h3 : new_car_purchase_percentage = 0.9)
  (h4 : out_of_pocket = 11000)
  : ∃ (sticker_price : ℝ), 
    sticker_price * new_car_purchase_percentage - old_car_value * old_car_sale_percentage = out_of_pocket ∧ 
    sticker_price = 30000 := by
  sorry

end new_car_sticker_price_l3235_323508


namespace right_triangle_similarity_x_values_l3235_323596

theorem right_triangle_similarity_x_values :
  let segments : Finset ℝ := {1, 9, 5, x}
  ∃ (AB CD : ℝ) (a b c d : ℝ),
    AB ∈ segments ∧ CD ∈ segments ∧
    a ∈ segments ∧ b ∈ segments ∧ c ∈ segments ∧ d ∈ segments ∧
    a^2 + b^2 = AB^2 ∧ c^2 + d^2 = CD^2 ∧
    a / c = b / d ∧
    x > 0 →
    (∃! (s : Finset ℝ), s.card = 3 ∧ ∀ y ∈ s, ∃ (AB CD a b c d : ℝ),
      AB ∈ segments ∧ CD ∈ segments ∧
      a ∈ segments ∧ b ∈ segments ∧ c ∈ segments ∧ d ∈ segments ∧
      a^2 + b^2 = AB^2 ∧ c^2 + d^2 = CD^2 ∧
      a / c = b / d ∧
      y = x) :=
by
  sorry

end right_triangle_similarity_x_values_l3235_323596


namespace square_root_of_nine_l3235_323570

theorem square_root_of_nine (x : ℝ) : x^2 = 9 → (x = 3 ∨ x = -3) := by
  sorry

end square_root_of_nine_l3235_323570


namespace axis_of_symmetry_is_correct_l3235_323568

/-- The quadratic function f(x) = -2(x-3)^2 + 1 -/
def f (x : ℝ) : ℝ := -2 * (x - 3)^2 + 1

/-- The axis of symmetry of f(x) -/
def axis_of_symmetry : ℝ := 3

/-- Theorem: The axis of symmetry of f(x) = -2(x-3)^2 + 1 is x = 3 -/
theorem axis_of_symmetry_is_correct :
  ∀ x : ℝ, f (axis_of_symmetry + x) = f (axis_of_symmetry - x) :=
sorry

end axis_of_symmetry_is_correct_l3235_323568


namespace lettuce_calories_l3235_323599

/-- Calculates the calories in lettuce given the conditions of Jackson's meal -/
theorem lettuce_calories (
  pizza_crust : ℝ)
  (pizza_cheese : ℝ)
  (salad_dressing : ℝ)
  (total_calories_consumed : ℝ)
  (h1 : pizza_crust = 600)
  (h2 : pizza_cheese = 400)
  (h3 : salad_dressing = 210)
  (h4 : total_calories_consumed = 330) :
  let pizza_pepperoni := pizza_crust / 3
  let total_pizza := pizza_crust + pizza_pepperoni + pizza_cheese
  let pizza_consumed := total_pizza / 5
  let salad_consumed := total_calories_consumed - pizza_consumed
  let total_salad := salad_consumed * 4
  let lettuce := (total_salad - salad_dressing) / 3
  lettuce = 50 := by sorry

end lettuce_calories_l3235_323599


namespace gumball_probability_l3235_323534

/-- Given a jar with pink and blue gumballs, where the probability of drawing two blue
    gumballs in a row with replacement is 36/49, the probability of drawing a pink gumball
    is 1/7. -/
theorem gumball_probability (blue pink : ℝ) : 
  blue + pink = 1 →
  blue * blue = 36 / 49 →
  pink = 1 / 7 := by
  sorry

end gumball_probability_l3235_323534


namespace circle_center_coordinates_l3235_323525

/-- The center of a circle satisfying given conditions -/
theorem circle_center_coordinates :
  ∃ (x y : ℝ),
    (x - 2*y = 0) ∧
    (3*x - 4*y = 20) ∧
    (x = 20 ∧ y = 10) := by
  sorry

end circle_center_coordinates_l3235_323525


namespace a_6_equals_448_l3235_323515

/-- Given a sequence {a_n} where S_n is the sum of its first n terms -/
def S (n : ℕ) : ℕ := n * 2^(n+1)

/-- The nth term of the sequence -/
def a (n : ℕ) : ℕ := S n - S (n-1)

/-- The 6th term of the sequence equals 448 -/
theorem a_6_equals_448 : a 6 = 448 := by sorry

end a_6_equals_448_l3235_323515


namespace chickens_per_coop_l3235_323564

/-- Given a farm with chicken coops, prove that the number of chickens per coop is as stated. -/
theorem chickens_per_coop
  (total_coops : ℕ)
  (total_chickens : ℕ)
  (h_coops : total_coops = 9)
  (h_chickens : total_chickens = 540) :
  total_chickens / total_coops = 60 := by
  sorry

end chickens_per_coop_l3235_323564


namespace cube_surface_area_increase_l3235_323572

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) : 
  let original_area := 6 * L^2
  let new_edge_length := 1.3 * L
  let new_area := 6 * new_edge_length^2
  (new_area - original_area) / original_area * 100 = 69 := by
  sorry

end cube_surface_area_increase_l3235_323572


namespace pet_store_dogs_l3235_323567

/-- Given a ratio of cats to dogs and the number of cats, calculate the number of dogs -/
def calculate_dogs (cat_ratio : ℕ) (dog_ratio : ℕ) (num_cats : ℕ) : ℕ :=
  (num_cats / cat_ratio) * dog_ratio

/-- Theorem: Given the ratio of cats to dogs is 3:4 and there are 18 cats, there are 24 dogs -/
theorem pet_store_dogs : calculate_dogs 3 4 18 = 24 := by
  sorry

end pet_store_dogs_l3235_323567


namespace cultural_group_members_l3235_323549

theorem cultural_group_members :
  ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 200 ∧ n % 7 = 4 ∧ n % 11 = 6 ∧
  (n = 116 ∨ n = 193) :=
by sorry

end cultural_group_members_l3235_323549


namespace sum_of_x_and_y_is_2018_l3235_323557

theorem sum_of_x_and_y_is_2018 (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) 
  (h : x^4 - 2018*x^3 - 2018*y^2*x = y^4 - 2018*y^3 - 2018*y*x^2) : 
  x + y = 2018 := by sorry

end sum_of_x_and_y_is_2018_l3235_323557


namespace xyz_value_l3235_323504

theorem xyz_value (x y z : ℝ) 
  (sum_eq : x + y + z = 2)
  (sum_sq_eq : x^2 + y^2 + z^2 = 14)
  (sum_cube_eq : x^3 + y^3 + z^3 = 17) :
  x * y * z = -7 := by
  sorry

end xyz_value_l3235_323504


namespace edmund_normal_chores_l3235_323544

/-- The number of chores Edmund normally has to do in a week -/
def normal_chores : ℕ := sorry

/-- The number of chores Edmund does per day -/
def chores_per_day : ℕ := 4

/-- The number of days Edmund works -/
def work_days : ℕ := 14

/-- The total amount Edmund earns -/
def total_earnings : ℕ := 64

/-- The payment per extra chore -/
def payment_per_chore : ℕ := 2

theorem edmund_normal_chores :
  normal_chores = 12 :=
by sorry

end edmund_normal_chores_l3235_323544


namespace photo_perimeter_is_23_l3235_323539

/-- Represents a rectangular photograph with a border -/
structure BorderedPhoto where
  width : ℝ
  length : ℝ
  borderWidth : ℝ

/-- Calculates the total area of a bordered photograph -/
def totalArea (photo : BorderedPhoto) : ℝ :=
  (photo.width + 2 * photo.borderWidth) * (photo.length + 2 * photo.borderWidth)

/-- Calculates the perimeter of the photograph without the border -/
def photoPerimeter (photo : BorderedPhoto) : ℝ :=
  2 * (photo.width + photo.length)

theorem photo_perimeter_is_23 (photo : BorderedPhoto) (m : ℝ) :
  photo.borderWidth = 2 →
  totalArea photo = m →
  totalArea { photo with borderWidth := 4 } = m + 94 →
  photoPerimeter photo = 23 := by
  sorry

end photo_perimeter_is_23_l3235_323539


namespace remaining_money_l3235_323594

def initial_amount : ℕ := 760
def ticket_price : ℕ := 300
def hotel_price : ℕ := ticket_price / 2

theorem remaining_money :
  initial_amount - (ticket_price + hotel_price) = 310 := by
  sorry

end remaining_money_l3235_323594


namespace two_complex_roots_iff_m_values_l3235_323569

/-- The equation (x / (x+2)) + (x / (x+3)) = mx has exactly two complex roots
    if and only if m is equal to 0, 2i, or -2i. -/
theorem two_complex_roots_iff_m_values (m : ℂ) : 
  (∃! (r₁ r₂ : ℂ), ∀ (x : ℂ), x ≠ -2 ∧ x ≠ -3 →
    (x / (x + 2) + x / (x + 3) = m * x) ↔ (x = r₁ ∨ x = r₂)) ↔
  (m = 0 ∨ m = 2*I ∨ m = -2*I) :=
sorry

end two_complex_roots_iff_m_values_l3235_323569


namespace mass_of_man_l3235_323571

/-- The mass of a man who causes a boat to sink by a certain amount in water. -/
theorem mass_of_man (boat_length boat_breadth boat_sink_depth water_density : ℝ) 
  (h1 : boat_length = 3)
  (h2 : boat_breadth = 2)
  (h3 : boat_sink_depth = 0.02)
  (h4 : water_density = 1000) : 
  boat_length * boat_breadth * boat_sink_depth * water_density = 120 := by
  sorry

end mass_of_man_l3235_323571


namespace cl35_properties_neutron_calculation_electron_proton_equality_l3235_323540

/-- Represents an atom with its atomic properties -/
structure Atom where
  protons : ℕ
  mass_number : ℕ
  neutrons : ℕ
  electrons : ℕ

/-- Cl-35 atom -/
def cl35 : Atom :=
  { protons := 17,
    mass_number := 35,
    neutrons := 35 - 17,
    electrons := 17 }

/-- Theorem stating the properties of Cl-35 -/
theorem cl35_properties :
  cl35.protons = 17 ∧
  cl35.mass_number = 35 ∧
  cl35.neutrons = 18 ∧
  cl35.electrons = 17 := by
  sorry

/-- Theorem stating the relationship between neutrons, mass number, and protons -/
theorem neutron_calculation (a : Atom) :
  a.neutrons = a.mass_number - a.protons := by
  sorry

/-- Theorem stating the relationship between electrons and protons -/
theorem electron_proton_equality (a : Atom) :
  a.electrons = a.protons := by
  sorry

end cl35_properties_neutron_calculation_electron_proton_equality_l3235_323540


namespace traffic_light_change_probability_l3235_323581

/-- Represents the duration of a traffic light cycle in seconds -/
def cycle_duration : ℕ := 95

/-- Represents the number of color changes in a cycle -/
def color_changes : ℕ := 3

/-- Represents the duration of each color change in seconds -/
def change_duration : ℕ := 5

/-- Represents the duration of the observation interval in seconds -/
def observation_interval : ℕ := 5

/-- The probability of observing a color change during a random observation interval -/
theorem traffic_light_change_probability :
  (color_changes * change_duration : ℚ) / cycle_duration = 3 / 19 := by sorry

end traffic_light_change_probability_l3235_323581


namespace f_satisfies_data_points_l3235_323573

def f (x : ℝ) : ℝ := 240 - 60 * x

theorem f_satisfies_data_points : 
  (f 0 = 240) ∧ 
  (f 1 = 180) ∧ 
  (f 2 = 120) ∧ 
  (f 3 = 60) ∧ 
  (f 4 = 0) := by
  sorry

end f_satisfies_data_points_l3235_323573


namespace eighth_term_is_21_l3235_323574

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem eighth_term_is_21 :
  fibonacci 7 = 21 ∧ fibonacci 8 = 34 ∧ fibonacci 9 = 55 :=
by sorry

end eighth_term_is_21_l3235_323574


namespace infinite_solutions_l3235_323563

theorem infinite_solutions (a b : ℝ) :
  (∀ x, 4 * (3 * x - a) = 3 * (4 * x + b)) ↔ b = -4/3 * a := by
  sorry

end infinite_solutions_l3235_323563


namespace investment_amount_correct_l3235_323588

/-- Calculates the investment amount in T-shirt printing equipment -/
def calculate_investment (cost_per_shirt : ℚ) (selling_price : ℚ) (break_even_point : ℕ) : ℚ :=
  selling_price * break_even_point - cost_per_shirt * break_even_point

/-- Proves that the investment amount is correct -/
theorem investment_amount_correct (cost_per_shirt : ℚ) (selling_price : ℚ) (break_even_point : ℕ) :
  calculate_investment cost_per_shirt selling_price break_even_point = 1411 :=
by
  have h1 : cost_per_shirt = 3 := by sorry
  have h2 : selling_price = 20 := by sorry
  have h3 : break_even_point = 83 := by sorry
  sorry

#eval calculate_investment 3 20 83

end investment_amount_correct_l3235_323588


namespace probability_centrally_symmetric_shape_l3235_323577

/-- Represents the shapes on the cards -/
inductive Shape
  | Circle
  | Rectangle
  | EquilateralTriangle
  | RegularPentagon

/-- Determines if a shape is centrally symmetric -/
def isCentrallySymmetric (s : Shape) : Bool :=
  match s with
  | Shape.Circle => true
  | Shape.Rectangle => true
  | Shape.EquilateralTriangle => false
  | Shape.RegularPentagon => false

/-- The set of all shapes -/
def allShapes : List Shape :=
  [Shape.Circle, Shape.Rectangle, Shape.EquilateralTriangle, Shape.RegularPentagon]

/-- Theorem: The probability of randomly selecting a centrally symmetric shape is 1/2 -/
theorem probability_centrally_symmetric_shape :
  (allShapes.filter isCentrallySymmetric).length / allShapes.length = 1 / 2 := by
  sorry


end probability_centrally_symmetric_shape_l3235_323577


namespace regular_polygon_exterior_angle_l3235_323509

theorem regular_polygon_exterior_angle (n : ℕ) (exterior_angle : ℝ) : 
  (n > 2) → (exterior_angle = 30 * Real.pi / 180) → (n * exterior_angle = 2 * Real.pi) → n = 12 := by
  sorry

end regular_polygon_exterior_angle_l3235_323509


namespace tangent_circle_to_sphere_reasoning_l3235_323583

/-- Represents the type of reasoning used in geometric analogies --/
inductive ReasoningType
  | Inductive
  | Deductive
  | Analogical
  | Other

/-- Represents a geometric property in 2D --/
structure Property2D where
  statement : String

/-- Represents a geometric property in 3D --/
structure Property3D where
  statement : String

/-- The property of tangent lines to circles in 2D --/
def tangentLineCircle : Property2D :=
  { statement := "When a line is tangent to a circle, the line connecting the center of the circle to the tangent point is perpendicular to the line" }

/-- The property of tangent planes to spheres in 3D --/
def tangentPlaneSphere : Property3D :=
  { statement := "When a plane is tangent to a sphere, the line connecting the center of the sphere to the tangent point is perpendicular to the plane" }

/-- The theorem stating that the reasoning used to extend the 2D property to 3D is analogical --/
theorem tangent_circle_to_sphere_reasoning :
  (∃ (p2d : Property2D) (p3d : Property3D), p2d = tangentLineCircle ∧ p3d = tangentPlaneSphere) →
  (∃ (r : ReasoningType), r = ReasoningType.Analogical) :=
by sorry

end tangent_circle_to_sphere_reasoning_l3235_323583


namespace intersection_of_M_and_P_l3235_323532

-- Define the sets M and P
def M : Set ℝ := {x | ∃ y, y = Real.log (x - 3) ∧ x > 3}
def P : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

-- State the theorem
theorem intersection_of_M_and_P : M ∩ P = {x | 3 < x ∧ x ≤ 4} := by sorry

end intersection_of_M_and_P_l3235_323532


namespace toms_average_score_l3235_323537

theorem toms_average_score (subjects_sem1 subjects_sem2 : ℕ)
  (avg_score_sem1 avg_score_5_sem2 avg_score_all : ℚ) :
  subjects_sem1 = 3 →
  subjects_sem2 = 7 →
  avg_score_sem1 = 85 →
  avg_score_5_sem2 = 78 →
  avg_score_all = 80 →
  (subjects_sem1 * avg_score_sem1 + 5 * avg_score_5_sem2 + 2 * ((subjects_sem1 + subjects_sem2) * avg_score_all - subjects_sem1 * avg_score_sem1 - 5 * avg_score_5_sem2) / 2) / (subjects_sem1 + subjects_sem2) = avg_score_all :=
by sorry

end toms_average_score_l3235_323537


namespace circle_and_point_position_l3235_323598

-- Define the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - 18/5)^2 + y^2 = 569/25

-- Define the points
def point_A : ℝ × ℝ := (1, 4)
def point_B : ℝ × ℝ := (3, 2)
def point_P : ℝ × ℝ := (2, 4)

-- Define what it means for a point to be on the circle
def on_circle (p : ℝ × ℝ) : Prop :=
  circle_equation p.1 p.2

-- Define what it means for a point to be inside the circle
def inside_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - 18/5)^2 + p.2^2 < 569/25

-- Theorem statement
theorem circle_and_point_position :
  (on_circle point_A) ∧ 
  (on_circle point_B) ∧ 
  (inside_circle point_P) := by
  sorry

end circle_and_point_position_l3235_323598


namespace roots_when_m_zero_m_value_when_product_41_perimeter_of_isosceles_triangle_l3235_323561

-- Define the quadratic equation
def quadratic_eq (m : ℝ) (x : ℝ) : Prop :=
  x^2 - 2*(m+2)*x + m^2 = 0

-- Define the roots of the equation
def roots (m : ℝ) (x₁ x₂ : ℝ) : Prop :=
  quadratic_eq m x₁ ∧ quadratic_eq m x₂ ∧ x₁ ≠ x₂

-- Theorem for part 1
theorem roots_when_m_zero :
  roots 0 0 4 :=
sorry

-- Theorem for part 2
theorem m_value_when_product_41 :
  ∀ x₁ x₂ : ℝ, roots 9 x₁ x₂ → (x₁ - 2) * (x₂ - 2) = 41 :=
sorry

-- Define an isosceles triangle
def isosceles_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a = b ∨ b = c ∨ a = c)

-- Theorem for part 3
theorem perimeter_of_isosceles_triangle :
  ∀ m x₁ x₂ : ℝ, 
    roots m x₁ x₂ → 
    isosceles_triangle 9 x₁ x₂ → 
    x₁ + x₂ + 9 = 19 :=
sorry

end roots_when_m_zero_m_value_when_product_41_perimeter_of_isosceles_triangle_l3235_323561


namespace tank_capacity_l3235_323555

/-- Represents a cylindrical water tank -/
structure WaterTank where
  capacity : ℝ
  
/-- The tank is 24% full when it contains 72 liters -/
def condition1 (tank : WaterTank) : Prop :=
  0.24 * tank.capacity = 72

/-- The tank is 60% full when it contains 180 liters -/
def condition2 (tank : WaterTank) : Prop :=
  0.60 * tank.capacity = 180

/-- The theorem stating the total capacity of the tank -/
theorem tank_capacity (tank : WaterTank) 
  (h1 : condition1 tank) (h2 : condition2 tank) : 
  tank.capacity = 300 := by
  sorry

end tank_capacity_l3235_323555


namespace eq1_solution_eq2_solution_eq3_solution_eq4_solution_eq5_solution_l3235_323579

-- Equation 1: 3x^2 - 15 = 0
theorem eq1_solution (x : ℝ) : x = Real.sqrt 5 ∨ x = -Real.sqrt 5 ↔ 3 * x^2 - 15 = 0 := by sorry

-- Equation 2: x^2 - 8x + 15 = 0
theorem eq2_solution (x : ℝ) : x = 3 ∨ x = 5 ↔ x^2 - 8*x + 15 = 0 := by sorry

-- Equation 3: x^2 - 6x + 7 = 0
theorem eq3_solution (x : ℝ) : x = 3 + Real.sqrt 2 ∨ x = 3 - Real.sqrt 2 ↔ x^2 - 6*x + 7 = 0 := by sorry

-- Equation 4: 2x^2 - 6x + 1 = 0
theorem eq4_solution (x : ℝ) : x = (3 + Real.sqrt 7) / 2 ∨ x = (3 - Real.sqrt 7) / 2 ↔ 2*x^2 - 6*x + 1 = 0 := by sorry

-- Equation 5: (2x^2 + 3x)^2 - 4(2x^2 + 3x) - 5 = 0
theorem eq5_solution (x : ℝ) : x = -5/2 ∨ x = 1 ∨ x = -1/2 ∨ x = -1 ↔ (2*x^2 + 3*x)^2 - 4*(2*x^2 + 3*x) - 5 = 0 := by sorry

end eq1_solution_eq2_solution_eq3_solution_eq4_solution_eq5_solution_l3235_323579


namespace x_percent_of_z_l3235_323546

theorem x_percent_of_z (x y z : ℝ) (h1 : x = 1.30 * y) (h2 : y = 0.60 * z) : 
  x = 0.78 * z := by sorry

end x_percent_of_z_l3235_323546


namespace probability_of_28_l3235_323545

/-- Represents a die with a specific face configuration -/
structure Die :=
  (faces : List ℕ)
  (blank_faces : ℕ)

/-- The first die configuration -/
def die1 : Die :=
  { faces := List.range 18, blank_faces := 1 }

/-- The second die configuration -/
def die2 : Die :=
  { faces := (List.range 7) ++ (List.range' 9 20), blank_faces := 1 }

/-- Calculates the probability of a specific sum when rolling two dice -/
def probability_of_sum (d1 d2 : Die) (target_sum : ℕ) : ℚ :=
  sorry

theorem probability_of_28 :
  probability_of_sum die1 die2 28 = 1 / 40 := by sorry

end probability_of_28_l3235_323545


namespace problem_solution_l3235_323593

theorem problem_solution : 
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁^3 - 3*x₁*y₁^2 = 2010) ∧ (y₁^3 - 3*x₁^2*y₁ = 2006) ∧
    (x₂^3 - 3*x₂*y₂^2 = 2010) ∧ (y₂^3 - 3*x₂^2*y₂ = 2006) ∧
    (x₃^3 - 3*x₃*y₃^2 = 2010) ∧ (y₃^3 - 3*x₃^2*y₃ = 2006) ∧
    ((1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 996/1005) :=
by sorry

end problem_solution_l3235_323593


namespace bicycle_sale_price_l3235_323565

def price_store_p : ℝ := 200

def regular_price_store_q : ℝ := price_store_p * 1.15

def sale_price_store_q : ℝ := regular_price_store_q * 0.9

theorem bicycle_sale_price : sale_price_store_q = 207 := by sorry

end bicycle_sale_price_l3235_323565


namespace proportion_solution_l3235_323503

theorem proportion_solution : 
  ∀ x : ℚ, (2 : ℚ) / 5 = (4 : ℚ) / 3 / x → x = 10 / 3 := by
  sorry

end proportion_solution_l3235_323503


namespace school_election_votes_l3235_323589

theorem school_election_votes (bob_votes : ℕ) (total_votes : ℕ) 
  (h1 : bob_votes = 48)
  (h2 : bob_votes = (2 : ℕ) * total_votes / (5 : ℕ)) :
  total_votes = 120 := by
  sorry

end school_election_votes_l3235_323589


namespace greatest_n_perfect_cube_l3235_323522

def sumOfSquares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def productOfSums (n : ℕ) : ℕ := 
  (sumOfSquares n) * (sumOfSquares (2 * n) - sumOfSquares n)

def isPerfectCube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem greatest_n_perfect_cube : 
  ∀ n : ℕ, n ≤ 2050 → 
    (isPerfectCube (productOfSums n) → n ≤ 2016) ∧ 
    (isPerfectCube (productOfSums 2016)) := by
  sorry

end greatest_n_perfect_cube_l3235_323522


namespace smallest_divisible_by_11_ending_in_9_l3235_323512

def is_smallest_divisible_by_11_ending_in_9 (n : ℕ) : Prop :=
  n > 0 ∧ 
  n % 10 = 9 ∧ 
  n % 11 = 0 ∧ 
  ∀ m : ℕ, m > 0 → m % 10 = 9 → m % 11 = 0 → m ≥ n

theorem smallest_divisible_by_11_ending_in_9 : 
  is_smallest_divisible_by_11_ending_in_9 99 := by
  sorry

end smallest_divisible_by_11_ending_in_9_l3235_323512


namespace worker_c_completion_time_l3235_323556

/-- The time it takes for worker c to complete a job alone, given the work rates of combinations of workers. -/
theorem worker_c_completion_time 
  (ab_rate : ℚ)  -- Rate at which workers a and b complete the job together
  (abc_rate : ℚ) -- Rate at which workers a, b, and c complete the job together
  (h1 : ab_rate = 1 / 15)  -- a and b finish the job in 15 days
  (h2 : abc_rate = 1 / 5)  -- a, b, and c finish the job in 5 days
  : (1 : ℚ) / (abc_rate - ab_rate) = 15 / 2 := by
  sorry


end worker_c_completion_time_l3235_323556


namespace relationship_abc_l3235_323531

theorem relationship_abc : 
  let a := Real.sin (15 * π / 180) * Real.cos (15 * π / 180)
  let b := Real.cos (π / 6) ^ 2 - Real.sin (π / 6) ^ 2
  let c := Real.tan (30 * π / 180) / (1 - Real.tan (30 * π / 180) ^ 2)
  a < b ∧ b < c := by sorry

end relationship_abc_l3235_323531


namespace medal_award_scenario_l3235_323521

/-- The number of ways to award medals in a specific race scenario -/
def medal_award_ways (total_sprinters : ℕ) (italian_sprinters : ℕ) : ℕ :=
  let non_italian_sprinters := total_sprinters - italian_sprinters
  italian_sprinters * non_italian_sprinters * (non_italian_sprinters - 1)

/-- Theorem stating the number of ways to award medals in the given scenario -/
theorem medal_award_scenario : medal_award_ways 10 4 = 120 := by
  sorry

end medal_award_scenario_l3235_323521


namespace return_speed_calculation_l3235_323543

/-- Calculates the return speed given the distance, outbound speed, and total time for a round trip -/
theorem return_speed_calculation (distance : ℝ) (outbound_speed : ℝ) (total_time : ℝ) :
  distance = 19.999999999999996 →
  outbound_speed = 25 →
  total_time = 5 + 48 / 60 →
  4 = distance / (total_time - distance / outbound_speed) := by
  sorry

end return_speed_calculation_l3235_323543


namespace oplus_problem_l3235_323538

-- Define the operation ⊕
def oplus (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y - a * b

-- State the theorem
theorem oplus_problem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : oplus a b 1 2 = 3) (h2 : oplus a b 2 3 = 6) :
  oplus a b 3 4 = 9 := by sorry

end oplus_problem_l3235_323538


namespace g_properties_l3235_323527

/-- Given a function f(x) = a - b cos(x) with maximum value 5/2 and minimum value -1/2,
    we define g(x) = -4a sin(bx) and prove its properties. -/
theorem g_properties (a b : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : f = fun x ↦ a - b * Real.cos x)
  (hmax : ∀ x, f x ≤ 5/2)
  (hmin : ∀ x, -1/2 ≤ f x)
  (hg : g = fun x ↦ -4 * a * Real.sin (b * x)) :
  (∃ x, g x = 4) ∧
  (∃ x, g x = -4) ∧
  (∃ T > 0, ∀ x, g (x + T) = g x ∧ ∀ S, 0 < S → S < T → ∃ y, g (y + S) ≠ g y) ∧
  (∀ x, -4 ≤ g x ∧ g x ≤ 4) :=
by sorry

end g_properties_l3235_323527


namespace roots_of_equation_l3235_323597

theorem roots_of_equation (x : ℝ) : (x + 1)^2 = 0 ↔ x = -1 := by
  sorry

end roots_of_equation_l3235_323597


namespace no_real_solutions_l3235_323516

theorem no_real_solutions : ¬∃ (x : ℝ), 7 * (4 * x + 3) - 4 = -3 * (2 - 9 * x^2) := by
  sorry

end no_real_solutions_l3235_323516


namespace set_operations_and_subset_l3235_323524

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | 3 ≤ x ∧ x < 8}

-- Define set B
def B : Set ℝ := {x | 2 < x ∧ x ≤ 6}

-- Define set C
def C (a : ℝ) : Set ℝ := {x | x ≥ a}

theorem set_operations_and_subset :
  (A ∩ B = {x | 3 ≤ x ∧ x ≤ 6}) ∧
  (A ∪ B = {x | 2 < x ∧ x < 8}) ∧
  (Aᶜ = {x | x < 3 ∨ x ≥ 8}) ∧
  (∀ a : ℝ, A ⊆ C a → a ≤ 3) :=
by sorry

end set_operations_and_subset_l3235_323524


namespace triangle_cosine_problem_l3235_323576

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b / cos B = c / cos C and cos A = 2/3, then cos B = √6 / 6 -/
theorem triangle_cosine_problem (a b c : ℝ) (A B C : ℝ) :
  b / Real.cos B = c / Real.cos C →
  Real.cos A = 2/3 →
  Real.cos B = Real.sqrt 6 / 6 := by
  sorry

end triangle_cosine_problem_l3235_323576


namespace harriet_speed_l3235_323584

/-- Harriet's round trip between A-ville and B-town -/
theorem harriet_speed (total_time : ℝ) (time_to_b : ℝ) (speed_from_b : ℝ) :
  total_time = 5 →
  time_to_b = 3 →
  speed_from_b = 150 →
  ∃ (distance : ℝ) (speed_to_b : ℝ),
    distance = speed_from_b * (total_time - time_to_b) ∧
    distance = speed_to_b * time_to_b ∧
    speed_to_b = 100 := by
  sorry


end harriet_speed_l3235_323584


namespace line_plane_perpendicular_parallel_l3235_323501

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular_parallel 
  (b c : Line) (α β : Plane) :
  perpendicular c β → parallel c α → plane_perpendicular α β :=
by sorry

end line_plane_perpendicular_parallel_l3235_323501


namespace four_values_with_2001_l3235_323530

/-- Represents a sequence where each term after the first two is defined by the previous two terms. -/
def SpecialSequence (x : ℝ) : ℕ → ℝ
  | 0 => x
  | 1 => 2000
  | (n + 2) => SpecialSequence x n * SpecialSequence x (n + 1) - 1

/-- The set of positive real numbers x such that 2001 appears in the special sequence starting with x. -/
def SequencesWith2001 : Set ℝ :=
  {x : ℝ | x > 0 ∧ ∃ n : ℕ, SpecialSequence x n = 2001}

theorem four_values_with_2001 :
  ∃ (S : Finset ℝ), S.card = 4 ∧ (∀ x ∈ SequencesWith2001, x ∈ S) ∧ (∀ x ∈ S, x ∈ SequencesWith2001) :=
sorry

end four_values_with_2001_l3235_323530


namespace constant_term_in_system_of_equations_l3235_323551

theorem constant_term_in_system_of_equations :
  ∀ (x y k : ℝ),
  (7 * x + y = 19) →
  (x + 3 * y = k) →
  (2 * x + y = 5) →
  k = 15 := by
sorry

end constant_term_in_system_of_equations_l3235_323551


namespace largest_perfect_square_factor_of_1800_l3235_323526

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem largest_perfect_square_factor_of_1800 :
  ∃ (n : ℕ), is_perfect_square n ∧ is_factor n 1800 ∧
  ∀ (m : ℕ), is_perfect_square m → is_factor m 1800 → m ≤ n :=
sorry

end largest_perfect_square_factor_of_1800_l3235_323526


namespace variance_transformation_l3235_323553

variable {n : ℕ}
variable (a : Fin n → ℝ)

def variance (x : Fin n → ℝ) : ℝ := sorry

def transformed_sample (a : Fin n → ℝ) : Fin n → ℝ := 
  fun i => 3 * a i + (if i.val = n - 1 then 2 else 1)

theorem variance_transformation (h : variance a = 3) : 
  variance (transformed_sample a) = 27 := by sorry

end variance_transformation_l3235_323553


namespace parallel_planes_counterexample_l3235_323513

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (not_parallel : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_counterexample 
  (a b : Line) (α β γ : Plane) : 
  ¬ (∀ (a b : Line) (α β γ : Plane), 
    (subset a α ∧ subset b α ∧ not_parallel a β ∧ not_parallel b β) 
    → ¬(parallel α β)) :=
sorry

end parallel_planes_counterexample_l3235_323513


namespace bad_carrots_count_l3235_323511

theorem bad_carrots_count (haley_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ)
  (h1 : haley_carrots = 39)
  (h2 : mom_carrots = 38)
  (h3 : good_carrots = 64) :
  haley_carrots + mom_carrots - good_carrots = 13 :=
by
  sorry

end bad_carrots_count_l3235_323511


namespace find_divisor_l3235_323587

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 919 →
  quotient = 17 →
  remainder = 11 →
  dividend = divisor * quotient + remainder →
  divisor = 53 := by
sorry

end find_divisor_l3235_323587


namespace sequence_property_l3235_323517

theorem sequence_property (a b c : ℝ) 
  (h1 : (4 * b) ^ 2 = 3 * a * 5 * c)  -- geometric sequence condition
  (h2 : 2 / b = 1 / a + 1 / c)        -- arithmetic sequence condition
  : a / c + c / a = 34 / 15 := by
  sorry

end sequence_property_l3235_323517


namespace inscribed_squares_ratio_l3235_323502

/-- Given two right triangles with sides 3, 4, and 5, where one triangle has a square
    inscribed with a vertex at the right angle (side length x) and the other has a square
    inscribed with a side on the hypotenuse (side length y), prove that x/y = 37/35 -/
theorem inscribed_squares_ratio (x y : ℝ) : 
  (∃ (a b c d : ℝ), 
    a^2 + b^2 = c^2 ∧ a = 3 ∧ b = 4 ∧ c = 5 ∧
    x^2 = a * b - (a - x) * (b - x) ∧
    y * (a + b) = c * y) →
  x / y = 37 / 35 := by sorry

end inscribed_squares_ratio_l3235_323502


namespace x_plus_one_is_square_l3235_323591

def x : ℕ := (1 + 2) * (1 + 2^2) * (1 + 2^4) * (1 + 2^8) * (1 + 2^16) * (1 + 2^32) * (1 + 2^64) * (1 + 2^128) * (1 + 2^256)

theorem x_plus_one_is_square (x : ℕ := x) : x + 1 = 2^512 := by
  sorry

end x_plus_one_is_square_l3235_323591


namespace angle_in_second_quadrant_implies_complement_in_first_quadrant_l3235_323533

/-- If the terminal side of angle α is in the second quadrant, then π - α is in the first quadrant -/
theorem angle_in_second_quadrant_implies_complement_in_first_quadrant (α : Real) : 
  (∃ k : ℤ, 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) → 
  (∃ m : ℤ, 2 * m * π < π - α ∧ π - α < 2 * m * π + π / 2) :=
by sorry

end angle_in_second_quadrant_implies_complement_in_first_quadrant_l3235_323533


namespace handshake_theorem_l3235_323586

/-- The number of handshakes for each student in a class where every two students shake hands once. -/
def handshakes_per_student (n : ℕ) : ℕ := n - 1

/-- The total number of handshakes in a class where every two students shake hands once. -/
def total_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a class of 57 students, if every two students shake hands with each other once, 
    then each student shakes hands 56 times, and the total number of handshakes is (57 × 56) / 2. -/
theorem handshake_theorem :
  handshakes_per_student 57 = 56 ∧ total_handshakes 57 = (57 * 56) / 2 := by
  sorry

end handshake_theorem_l3235_323586


namespace share_division_l3235_323559

/-- Given a total sum to be divided among three people A, B, and C, where
    3 times A's share equals 4 times B's share equals 7 times C's share,
    prove that C's share is 84 when the total sum is 427. -/
theorem share_division (total : ℕ) (a b c : ℚ)
  (h_total : total = 427)
  (h_sum : a + b + c = total)
  (h_prop : 3 * a = 4 * b ∧ 4 * b = 7 * c) :
  c = 84 := by
  sorry

end share_division_l3235_323559


namespace regular_polygon_interior_angle_sum_l3235_323542

theorem regular_polygon_interior_angle_sum (n : ℕ) (h1 : n > 2) : 
  (360 / 24 : ℝ) = n → (180 * (n - 2) : ℝ) = 2340 := by
  sorry

end regular_polygon_interior_angle_sum_l3235_323542


namespace soccer_league_games_l3235_323578

/-- The total number of games played in a soccer league. -/
def totalGames (n : ℕ) (g : ℕ) : ℕ :=
  n * (n - 1) * g / 2

/-- Theorem: In a league with 12 teams, where each team plays 4 games with each other team,
    the total number of games played is 264. -/
theorem soccer_league_games :
  totalGames 12 4 = 264 := by
  sorry

end soccer_league_games_l3235_323578


namespace expression_result_l3235_323506

theorem expression_result : (3.242 * 12) / 100 = 0.38904 := by
  sorry

end expression_result_l3235_323506


namespace factorial_ratio_l3235_323595

theorem factorial_ratio : (50 : ℕ).factorial / (48 : ℕ).factorial = 2450 := by
  sorry

end factorial_ratio_l3235_323595


namespace charles_reading_days_l3235_323566

/-- Represents the number of pages Charles reads each day -/
def daily_pages : List Nat := [7, 12, 10, 6]

/-- The total number of pages in the book -/
def total_pages : Nat := 96

/-- Calculates the number of days needed to finish the book -/
def days_to_finish (pages : List Nat) (total : Nat) : Nat :=
  let pages_read := pages.sum
  let remaining := total - pages_read
  let weekdays := pages.length
  let average_daily := (pages_read + remaining - 1) / weekdays
  weekdays + (remaining + average_daily - 1) / average_daily

theorem charles_reading_days :
  days_to_finish daily_pages total_pages = 11 := by
  sorry

#eval days_to_finish daily_pages total_pages

end charles_reading_days_l3235_323566


namespace min_square_area_l3235_323580

/-- A monic quartic polynomial with integer coefficients -/
structure MonicQuarticPolynomial where
  p : ℤ
  q : ℤ
  r : ℤ
  s : ℤ

/-- The roots of a polynomial form a square on the complex plane -/
def roots_form_square (poly : MonicQuarticPolynomial) : Prop :=
  sorry

/-- The area of the square formed by the roots of a polynomial -/
def square_area (poly : MonicQuarticPolynomial) : ℝ :=
  sorry

/-- The minimum possible area of a square formed by the roots of a monic quartic polynomial
    with integer coefficients is 2 -/
theorem min_square_area (poly : MonicQuarticPolynomial) 
  (h : roots_form_square poly) : 
  ∃ (min_area : ℝ), min_area = 2 ∧ ∀ (p : MonicQuarticPolynomial), 
  roots_form_square p → square_area p ≥ min_area :=
sorry

end min_square_area_l3235_323580


namespace triangle_is_equilateral_l3235_323554

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- The angle at vertex A of a triangle -/
def angleA (t : Triangle) : ℝ := sorry

/-- Check if a triangle is acute -/
def isAcute (t : Triangle) : Prop := sorry

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop := sorry

/-- The region G formed by points P inside the triangle satisfying PA ≤ PB and PA ≤ PC -/
def regionG (t : Triangle) : Set Point :=
  {p : Point | isInside p t ∧ distance p t.A ≤ distance p t.B ∧ distance p t.A ≤ distance p t.C}

/-- The area of region G -/
def areaG (t : Triangle) : ℝ := sorry

/-- Check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Main theorem -/
theorem triangle_is_equilateral (t : Triangle) :
  isAcute t →
  angleA t = π / 3 →
  areaG t = (1 / 3) * triangleArea t →
  isEquilateral t := by sorry

end triangle_is_equilateral_l3235_323554


namespace wizard_elixir_combinations_l3235_323575

/-- Represents the number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- Represents the number of enchanted stones available. -/
def num_stones : ℕ := 6

/-- Represents the number of herbs that are incompatible with one specific stone. -/
def incompatible_herbs : ℕ := 3

/-- Represents the number of valid combinations for the wizard's elixir. -/
def valid_combinations : ℕ := num_herbs * num_stones - incompatible_herbs

theorem wizard_elixir_combinations :
  valid_combinations = 21 :=
by sorry

end wizard_elixir_combinations_l3235_323575


namespace function_extrema_l3235_323536

/-- The function f(x) = 1 + 3x - x³ has a minimum value of -1 and a maximum value of 3. -/
theorem function_extrema :
  ∃ (a b : ℝ), (∀ x : ℝ, 1 + 3 * x - x^3 ≥ a) ∧
                (∃ x : ℝ, 1 + 3 * x - x^3 = a) ∧
                (∀ x : ℝ, 1 + 3 * x - x^3 ≤ b) ∧
                (∃ x : ℝ, 1 + 3 * x - x^3 = b) ∧
                a = -1 ∧ b = 3 := by
  sorry

end function_extrema_l3235_323536


namespace one_weighing_sufficient_l3235_323558

/-- Represents the types of balls -/
inductive BallType
| Aluminum
| Duralumin

/-- The total number of balls -/
def totalBalls : ℕ := 2000

/-- The number of balls in each group -/
def groupSize : ℕ := 1000

/-- The weight of an aluminum ball in grams -/
def aluminumWeight : ℚ := 10

/-- The weight of a duralumin ball in grams -/
def duraluminWeight : ℚ := 9.9

/-- A function that returns the weight of a ball given its type -/
def ballWeight (t : BallType) : ℚ :=
  match t with
  | BallType.Aluminum => aluminumWeight
  | BallType.Duralumin => duraluminWeight

/-- Represents a group of balls -/
structure BallGroup where
  aluminum : ℕ
  duralumin : ℕ

/-- The total weight of a group of balls -/
def groupWeight (g : BallGroup) : ℚ :=
  g.aluminum * aluminumWeight + g.duralumin * duraluminWeight

/-- Theorem stating that it's possible to separate the balls into two groups
    with equal size but different weights using one weighing -/
theorem one_weighing_sufficient :
  ∃ (g1 g2 : BallGroup),
    g1.aluminum + g1.duralumin = groupSize ∧
    g2.aluminum + g2.duralumin = groupSize ∧
    g1.aluminum + g2.aluminum = groupSize ∧
    g1.duralumin + g2.duralumin = groupSize ∧
    groupWeight g1 ≠ groupWeight g2 :=
  sorry

end one_weighing_sufficient_l3235_323558


namespace robie_chocolates_l3235_323560

theorem robie_chocolates (initial_bags : ℕ) : 
  (initial_bags - 2 + 3 = 4) → initial_bags = 3 := by
  sorry

end robie_chocolates_l3235_323560


namespace square_boundary_product_l3235_323523

theorem square_boundary_product : 
  ∀ (b₁ b₂ : ℝ),
  (∀ x y : ℝ, (y = 3 ∨ y = 7 ∨ x = -1 ∨ x = b₁) → 
    (y = 3 ∨ y = 7 ∨ x = -1 ∨ x = b₂) → 
    (0 ≤ x ∧ x ≤ 4 ∧ 3 ≤ y ∧ y ≤ 7)) →
  (b₁ * b₂ = -15) :=
by sorry

end square_boundary_product_l3235_323523


namespace unique_room_setup_l3235_323520

/-- Represents the number of people, stools, and chairs in a room -/
structure RoomSetup where
  people : ℕ
  stools : ℕ
  chairs : ℕ

/-- Checks if a given room setup satisfies all conditions -/
def isValidSetup (setup : RoomSetup) : Prop :=
  2 * setup.people + 3 * setup.stools + 4 * setup.chairs = 32 ∧
  setup.people > setup.stools ∧
  setup.people > setup.chairs ∧
  setup.people < setup.stools + setup.chairs

/-- The theorem stating that there is only one valid room setup -/
theorem unique_room_setup :
  ∃! setup : RoomSetup, isValidSetup setup ∧ 
    setup.people = 5 ∧ setup.stools = 2 ∧ setup.chairs = 4 := by
  sorry


end unique_room_setup_l3235_323520


namespace four_noncoplanar_points_determine_four_planes_l3235_323590

-- Define a type for points in 3D space
def Point3D := ℝ × ℝ × ℝ

-- Define a function to check if four points are non-coplanar
def nonCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Define a function to count the number of planes determined by four points
def countPlanes (p1 p2 p3 p4 : Point3D) : ℕ := sorry

-- Theorem statement
theorem four_noncoplanar_points_determine_four_planes 
  (p1 p2 p3 p4 : Point3D) 
  (h : nonCoplanar p1 p2 p3 p4) : 
  countPlanes p1 p2 p3 p4 = 4 := by sorry

end four_noncoplanar_points_determine_four_planes_l3235_323590


namespace chess_pawns_remaining_l3235_323547

theorem chess_pawns_remaining (initial_pawns : ℕ) 
  (kennedy_lost : ℕ) (riley_lost : ℕ) : 
  initial_pawns = 8 → kennedy_lost = 4 → riley_lost = 1 →
  (initial_pawns - kennedy_lost) + (initial_pawns - riley_lost) = 11 :=
by
  sorry

end chess_pawns_remaining_l3235_323547


namespace largest_c_for_g_range_two_l3235_323550

/-- The quadratic function g(x) = x^2 - 6x + c -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

/-- Theorem: The largest value of c for which 2 is in the range of g(x) is 11 -/
theorem largest_c_for_g_range_two :
  ∀ c : ℝ, (∃ x : ℝ, g c x = 2) ↔ c ≤ 11 :=
sorry

end largest_c_for_g_range_two_l3235_323550


namespace xyz_product_l3235_323518

theorem xyz_product (x y z : ℕ+) 
  (h1 : x + 2*y = z) 
  (h2 : x^2 - 4*y^2 + z^2 = 310) : 
  x*y*z = 11935 ∨ x*y*z = 2015 := by
  sorry

end xyz_product_l3235_323518


namespace sequence_3_9_729_arithmetic_and_geometric_l3235_323519

/-- A sequence is arithmetic if the difference between consecutive terms is constant -/
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- A sequence is geometric if the ratio between consecutive terms is constant -/
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) / a n = r

theorem sequence_3_9_729_arithmetic_and_geometric :
  ∃ (a g : ℕ → ℝ),
    is_arithmetic a ∧ is_geometric g ∧
    (∃ i j k : ℕ, a i = 3 ∧ a j = 9 ∧ a k = 729) ∧
    (∃ x y z : ℕ, g x = 3 ∧ g y = 9 ∧ g z = 729) := by
  sorry

end sequence_3_9_729_arithmetic_and_geometric_l3235_323519


namespace smallest_non_nine_divisible_by_999_l3235_323592

/-- Checks if a natural number contains the digit 9 --/
def containsNine (n : ℕ) : Prop :=
  ∃ (k : ℕ), n / (10^k) % 10 = 9

/-- Checks if a natural number is divisible by 999 --/
def divisibleBy999 (n : ℕ) : Prop :=
  n % 999 = 0

theorem smallest_non_nine_divisible_by_999 :
  ∀ n : ℕ, n > 0 → divisibleBy999 n → ¬containsNine n → n ≥ 112 :=
sorry

end smallest_non_nine_divisible_by_999_l3235_323592


namespace equal_roots_quadratic_l3235_323562

/-- Given a quadratic equation x^2 + 2x + k = 0 with two equal real roots, prove that k = 1 -/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 + 2*y + k = 0 → y = x) → 
  k = 1 := by
sorry

end equal_roots_quadratic_l3235_323562


namespace products_sum_bounds_l3235_323548

def CircularArray (α : Type) := Fin 999 → α

def CircularProduct (arr : CircularArray Int) (start : Fin 999) : Int :=
  (List.range 10).foldl (λ acc i => acc * arr ((start + i) % 999)) 1

def SumOfProducts (arr : CircularArray Int) : Int :=
  (List.range 999).foldl (λ acc i => acc + CircularProduct arr i) 0

theorem products_sum_bounds 
  (arr : CircularArray Int) 
  (h1 : ∀ i, arr i = 1 ∨ arr i = -1) 
  (h2 : ∃ i j, arr i ≠ arr j) : 
  -997 ≤ SumOfProducts arr ∧ SumOfProducts arr ≤ 995 :=
sorry

end products_sum_bounds_l3235_323548


namespace second_replaced_man_age_is_35_l3235_323514

/-- The age of the second replaced man in a group replacement scenario -/
def second_replaced_man_age (initial_count : ℕ) (age_increase : ℕ) 
  (replaced_count : ℕ) (first_replaced_age : ℕ) (new_men_avg_age : ℕ) : ℕ :=
  47 - (initial_count * age_increase)

/-- Theorem stating the age of the second replaced man is 35 -/
theorem second_replaced_man_age_is_35 :
  second_replaced_man_age 12 1 2 21 34 = 35 := by
  sorry

end second_replaced_man_age_is_35_l3235_323514


namespace festival_attendance_l3235_323507

theorem festival_attendance (total_students : ℕ) (festival_attendees : ℕ) 
  (h_total : total_students = 1500)
  (h_attendees : festival_attendees = 820) :
  ∃ (girls boys : ℕ),
    girls + boys = total_students ∧
    (3 * girls) / 4 + (2 * boys) / 5 = festival_attendees ∧
    (3 * girls) / 4 = 471 := by
  sorry

end festival_attendance_l3235_323507
